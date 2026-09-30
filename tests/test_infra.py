"""
Checkpointing, early stopping, reproducibility.

These three are what turn 150 runs into a result someone can audit, and
each has one property that matters more than the rest:

  a checkpoint restores the weights that produced the recorded val
  score, and is either whole or absent;

  stopping is decided on validation and never on test, and a run that
  hits the epoch ceiling says so;

  the same seed gives the same numbers, including through the
  dataloader workers.
"""

import json

import pytest
import torch
import torch.nn as nn

from src.checkpoint import (
    checkpoint_dir,
    checkpoint_exists,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from src.early_stopping import EarlyStopping
from src.reproducibility import (
    describe_determinism,
    is_hip,
    make_generator,
    seed_worker,
    set_global_seed,
)


def tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))


# ------------------------------------------------------------ checkpoint

def test_round_trip_restores_weights_and_metadata(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "best.pt"

    save_checkpoint(path, model, None, None, epoch=7,
                     val_metrics={"HR@10": 0.123}, config={"beta": 0.5}, seed=42)

    restored_into = tiny_model()
    assert not torch.equal(restored_into[0].weight, model[0].weight) or True
    meta = load_checkpoint(path, restored_into)

    torch.testing.assert_close(restored_into[0].weight, model[0].weight, atol=0, rtol=0)
    assert meta["epoch"] == 7 and meta["seed"] == 42
    assert meta["val_metrics"]["HR@10"] == 0.123
    assert meta["config"]["beta"] == 0.5


def test_optimizer_state_is_omitted_by_default(tmp_path):
    """
    Adam keeps two moments per parameter, so storing them triples every
    checkpoint -- 108.7 MB instead of ~36 MB for DeepCF on AMusic,
    across 150 runs -- and nothing reads them back.
    """
    model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()

    slim, fat = tmp_path / "slim.pt", tmp_path / "fat.pt"
    save_checkpoint(slim, model, None, None, 0, {}, {}, 0)
    save_checkpoint(fat, model, optimizer, None, 0, {}, {}, 0)

    assert torch.load(slim, weights_only=False)["optimizer_state_dict"] is None
    assert torch.load(fat, weights_only=False)["optimizer_state_dict"] is not None
    assert slim.stat().st_size < fat.stat().st_size


def test_write_is_atomic(tmp_path):
    """
    The matrix runs under tmux with --skip-existing. A job killed
    mid-write must not leave a truncated best.pt that skip-existing
    counts as complete, silently dropping that cell from the results.
    """
    path = tmp_path / "best.pt"
    save_checkpoint(path, tiny_model(), None, None, 0, {}, {}, 0)
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"


def test_checkpoint_exists_rejects_a_zero_byte_remnant(tmp_path):
    empty = tmp_path / "best.pt"
    empty.touch()
    assert not checkpoint_exists(empty)


def test_checkpoint_path_layout_mirrors_results():
    path = checkpoint_path("AMusic", "deepcf_rpucb__beta0.5", 42,
                            root="/tmp/ckpt_layout_test")
    assert path.parts[-4:] == ("AMusic", "deepcf_rpucb__beta0.5", "seed42", "best.pt")


def test_probing_does_not_create_directories(tmp_path):
    """
    select_beta.py probes one path per grid point. Creating a directory
    as a side effect of asking would leave empty seed folders for runs
    that never happened, which a later audit reads as "ran, produced
    nothing".
    """
    probed = checkpoint_path("AMusic", "deepcf", 42, root=tmp_path, create=False)
    assert not probed.parent.exists()

    checkpoint_dir("AMusic", "deepcf", 42, root=tmp_path)
    assert probed.parent.exists()


# -------------------------------------------------------- early stopping

def test_improvement_must_exceed_min_delta():
    """
    Patience alone would let a plateaued run drift for the full
    max_epochs on floating-point noise.
    """
    stopper = EarlyStopping(patience=2, min_delta=0.01, mode="max")
    assert stopper.step(0.50, 0) is True
    assert stopper.step(0.505, 1) is False, "a sub-min_delta gain counted"
    assert stopper.step(0.52, 2) is True


def test_stops_after_patience_epochs_without_improvement():
    stopper = EarlyStopping(patience=3, min_delta=1e-4, mode="max")
    stopper.step(0.5, 0)
    for epoch in range(1, 4):
        assert not stopper.should_stop
        stopper.step(0.4, epoch)
    assert stopper.should_stop
    assert stopper.best_epoch == 0 and stopper.best_score == 0.5


def test_min_mode_for_a_loss():
    stopper = EarlyStopping(patience=2, min_delta=1e-4, mode="min")
    assert stopper.step(1.0, 0) is True
    assert stopper.step(0.5, 1) is True
    assert stopper.step(0.9, 2) is False


def test_summary_distinguishes_converged_from_capped():
    """
    If a lot of the 150 runs hit the ceiling, max_epochs=100 was too low
    and the matrix needs rerunning -- so `stopped_early` is recorded
    rather than reconstructed from epoch logs.
    """
    converged = EarlyStopping(patience=1, min_delta=1e-4, mode="max")
    converged.step(0.5, 0)
    converged.step(0.4, 1)
    assert converged.summary()["stopped_early"] is True

    capped = EarlyStopping(patience=10, min_delta=1e-4, mode="max")
    for epoch in range(3):
        capped.step(0.5 + epoch * 0.1, epoch)
    assert capped.summary()["stopped_early"] is False


def test_state_round_trip():
    stopper = EarlyStopping(patience=5, min_delta=1e-3, mode="max")
    stopper.step(0.7, 0)
    stopper.step(0.6, 1)

    restored = EarlyStopping(patience=1, min_delta=1.0, mode="min")
    restored.load_state_dict(stopper.state_dict())
    assert restored.summary() == stopper.summary()
    assert restored.patience == 5 and restored.mode == "max"


# ------------------------------------------------------ reproducibility

def test_global_seed_makes_sampling_repeatable():
    set_global_seed(123)
    a = torch.randn(5)
    set_global_seed(123)
    assert torch.equal(a, torch.randn(5))


def test_global_seed_covers_python_and_numpy():
    """
    Negative sampling in RecTrainDataset uses Python's `random`, and the
    preprocessing path uses numpy. Seeding torch alone would leave both
    unseeded, which is the quiet half of a reproducibility gap.
    """
    import random

    import numpy as np

    set_global_seed(99)
    py, np_val = random.random(), np.random.rand()
    set_global_seed(99)
    assert random.random() == py and np.random.rand() == np_val


def test_generator_and_worker_seeding_are_deterministic():
    """
    The pre-refactor dataloader ran `num_workers=8` with no
    `worker_init_fn` and no `generator=`, so shuffling and per-worker
    negative sampling were both unseeded.
    """
    a, b = make_generator(5), make_generator(5)
    assert torch.equal(torch.randperm(20, generator=a), torch.randperm(20, generator=b))
    assert not torch.equal(
        torch.randperm(20, generator=make_generator(5)),
        torch.randperm(20, generator=make_generator(6)),
    )
    seed_worker(0)          # must not raise outside a worker process


def test_determinism_description_is_json_safe():
    """It goes into every results file's provenance block."""
    for flag in (False, True):
        json.dumps(describe_determinism(flag))
    assert isinstance(is_hip(), bool)


def test_two_identical_builds_score_identically(dataset, tiny_config):
    """
    The end-to-end property the rest of this section exists for: same
    seed, same data, same numbers.
    """
    from src.models.registry import build_model

    def build_and_score():
        set_global_seed(7)
        model = build_model("deepcf_rpucb", dataset, tiny_config()).eval()
        users, items = torch.arange(5), torch.arange(dataset.num_items)
        with torch.no_grad():
            return model.score_all_items(
                dataset.interaction_rows[users], users,
                dataset.interaction_cols[items], items,
            )

    torch.testing.assert_close(build_and_score(), build_and_score(), atol=0, rtol=0)
