"""
Checkpoint writing and restoration.

Gameplan v8 §6: `checkpoints/{dataset}/{model}/seed{seed}/best.pt`, holding
state dicts, epoch, val metrics, resolved config and seed. One per run,
corresponding exactly to the reported test row.

That last clause is the point. The pre-refactor code reported
`max(HR) over epochs`, `max(NDCG) over epochs`, ... independently, so the
published row was an envelope over epochs rather than the behaviour of any
one model. Here the restored `best.pt` *is* the model that gets evaluated on
test, once, and every metric in the results JSON comes from that single
forward pass.

Note on size, corrected against measurement. §10 assumed the DCM family
would produce the largest checkpoints. It does not: the DeepCF family
carries MLPs over the raw interaction profile, so its parameter count
scales with num_items, and the first-light run wrote 108.7 MB for DeepCF
on AMusic against 6.9 MB for DCM. Estimated across the matrix that is
about 9.4 GB with optimizer state and 3.1 GB without, which fits `/home`'s
19 GB either way -- so the `/mnt/prof-*` question is about headroom for
tuning and robustness runs rather than about the main matrix.

Optimizer and scheduler state are optional and off by default
(`checkpoint_optimizer_state` in base.yaml). Adam keeps two moments per
parameter, so storing them triples the file, and nothing reads them back:
`--skip-existing` resumes at run granularity, and a run killed mid-training
restarts from epoch 0. Turn it on only alongside an actual mid-run resume
path.

`checkpoint_root` stays a config key rather than a hardcoded path, so
relocating the tree is a config change rather than a code edit.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def checkpoint_dir(
    dataset: str,
    model: str,
    seed: int,
    root: str | os.PathLike = "checkpoints",
    create: bool = True,
) -> Path:
    """
    `{root}/{dataset}/{model}/seed{seed}/`, created on demand.

    `create=False` is for read-only callers -- `select_beta.py` probes one
    path per grid point to see whether that run produced a checkpoint, and
    creating a directory as a side effect of asking would leave the
    tuning tree littered with empty seed folders for grid points that
    never ran, which is exactly the kind of thing a later audit reads as
    "this ran and produced nothing".
    """
    path = Path(root) / dataset / model / f"seed{seed}"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path(
    dataset: str,
    model: str,
    seed: int,
    root: str | os.PathLike = "checkpoints",
    filename: str = "best.pt",
    create: bool = True,
) -> Path:
    return checkpoint_dir(dataset, model, seed, root, create=create) / filename


def save_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    epoch: int,
    val_metrics: dict,
    config: dict,
    seed: int,
    early_stopping_state: dict | None = None,
) -> None:
    """
    Write one checkpoint, atomically.

    Atomic because the matrix runs under tmux with `--skip-existing`: a run
    killed mid-write would otherwise leave a truncated `best.pt` that
    `--skip-existing` counts as complete, silently dropping that cell from
    the results. Writing to a temp file and renaming means the final path
    either does not exist or is a whole checkpoint.

    `config` is the *resolved* config -- the merged base + dataset + model
    dict, not a path to a YAML. Storing the resolution rather than the
    inputs is what lets the provenance audit verify what actually ran, and
    is the same reason the results JSON carries it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Passing optimizer=None / scheduler=None is the default path: see the
    # module docstring for why their state is not worth 3x the disk.
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "val_metrics": val_metrics,
        "config": config,
        "seed": seed,
        "early_stopping": early_stopping_state,
        "torch_version": torch.__version__,
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """
    Restore into `model` (and optionally optimizer/scheduler) in place.

    Returns the payload minus the tensor-heavy state dicts, so the caller can
    read `epoch` / `val_metrics` / `seed` for logging without holding a
    second copy of the weights in memory.

    `weights_only=False` is required here: the payload contains the resolved
    config dict and metric dicts, not just tensors. These files are produced
    by this codebase, so that is safe -- but it means never point this at a
    checkpoint from an untrusted source.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model_state_dict"], strict=strict)

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return {
        "epoch": payload["epoch"],
        "val_metrics": payload["val_metrics"],
        "config": payload["config"],
        "seed": payload["seed"],
        "early_stopping": payload.get("early_stopping"),
        "torch_version": payload.get("torch_version"),
    }


def checkpoint_exists(path: str | os.PathLike) -> bool:
    """
    Used by `--skip-existing`. Requires a non-empty file, so a zero-byte
    remnant from a crashed job is not mistaken for a finished run.
    """
    p = Path(path)
    return p.is_file() and p.stat().st_size > 0