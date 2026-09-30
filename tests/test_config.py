"""
Config resolution and its whitelists.

The pre-refactor repo had `beta: 0.05` sitting in `citeulike.yaml` while
every other dataset ran at the default. Nothing about that was visible
in the results: one dataset was quietly running a twentieth of the
exploration scale, and "RP-UCB helps more as sparsity increases" would
have been partly measuring it.

Deleting the line fixes the instance. The whitelists fix the class, so
they are what gets tested here -- along with the real config files,
because a whitelist nobody's files satisfy is not protection.
"""

import pytest
import yaml

from src.config import (
    ConfigError,
    available_datasets,
    available_models,
    load_config,
    resolve_seeds,
)
from src.data.preprocess.common import EXPECTED_STATS
from src.models import MATRIX_MODELS


def write_configs(root, base=None, dataset=None, model=None,
                  dataset_name="toy", model_name="toy_model"):
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "base.yaml").write_text(yaml.safe_dump(base or {
        "embed_dim": 8, "n_fields": 4, "attn_heads": 2, "K": 3,
        "rl_layers": [16, 8], "beta": 1.0, "base_seed": 42,
    }))
    (root / "datasets" / f"{dataset_name}.yaml").write_text(
        yaml.safe_dump(dataset if dataset is not None
                       else {"dataset": dataset_name, "data_path": "data/toy"}))
    (root / "models" / f"{model_name}.yaml").write_text(
        yaml.safe_dump(model if model is not None else {"model": model_name}))
    return root


# ------------------------------------------------------- the actual bug

def test_dataset_file_cannot_set_beta(tmp_path):
    """The citeulike bug, as a class rather than an instance."""
    root = write_configs(tmp_path, dataset={
        "dataset": "toy", "data_path": "data/toy", "beta": 0.05})
    with pytest.raises(ConfigError, match="may not set"):
        load_config("toy", "toy_model", root=root)


@pytest.mark.parametrize("key", ["lr", "max_epochs", "embed_dim", "mask_init",
                                  "num_negatives", "patience"])
def test_dataset_file_cannot_set_shared_hyperparameters(tmp_path, key):
    root = write_configs(tmp_path, dataset={
        "dataset": "toy", "data_path": "data/toy", key: 1})
    with pytest.raises(ConfigError, match="may not set"):
        load_config("toy", "toy_model", root=root)


def test_model_file_cannot_set_optimiser_settings(tmp_path):
    root = write_configs(tmp_path, model={"model": "toy_model", "lr": 0.1})
    with pytest.raises(ConfigError, match="may not set"):
        load_config("toy", "toy_model", root=root)


def test_model_file_may_set_beta_and_structural_switches(tmp_path):
    root = write_configs(tmp_path, model={
        "model": "toy_model", "beta": 0.5, "head_mode": "concat",
        "mask_granularity": "shared", "adaptive_k": False, "K": 1})
    config = load_config("toy", "toy_model", root=root)
    assert config["beta"] == 0.5 and config["K"] == 1


def test_cli_overrides_are_not_whitelisted(tmp_path):
    """
    An override typed on the command line is a deliberate act; a file
    that quietly persists is not. main.py's --tag requirement is what
    keeps the deliberate act from landing on the main matrix's path.
    """
    root = write_configs(tmp_path)
    config = load_config("toy", "toy_model", overrides={"lr": 0.5, "beta": 0.25}, root=root)
    assert config["lr"] == 0.5 and config["beta"] == 0.25


def test_merge_order_is_base_then_dataset_then_model_then_cli(tmp_path):
    root = write_configs(
        tmp_path,
        base={"embed_dim": 8, "n_fields": 4, "attn_heads": 2, "K": 9,
              "beta": 1.0, "base_seed": 42},
        dataset={"dataset": "toy", "data_path": "data/toy", "batch_size": 64},
        model={"model": "toy_model", "K": 3, "beta": 0.5},
    )
    config = load_config("toy", "toy_model", root=root)
    assert config["K"] == 3 and config["batch_size"] == 64
    assert load_config("toy", "toy_model", overrides={"K": 1}, root=root)["K"] == 1


# ------------------------------------------------------- the validations

def test_filename_and_declared_name_must_agree(tmp_path):
    root = write_configs(tmp_path, dataset={"dataset": "other", "data_path": "d"})
    with pytest.raises(ConfigError, match="filename and contents must agree"):
        load_config("toy", "toy_model", root=root)


def test_rl_layers_must_end_at_embed_dim(tmp_path):
    """
    The d-dim mask is applied to the CFNet-rl branch output as well as
    the ML embeddings, so a mismatch is a shape error at the first
    masked forward -- better caught at load time.
    """
    root = write_configs(tmp_path, base={
        "embed_dim": 8, "n_fields": 4, "attn_heads": 2, "K": 3,
        "rl_layers": [16, 32], "beta": 1.0, "base_seed": 42})
    with pytest.raises(ConfigError, match="must equal embed_dim"):
        load_config("toy", "toy_model", root=root)


@pytest.mark.parametrize("bad", [{"n_fields": 3}, {"attn_heads": 3}])
def test_divisibility_constraints(tmp_path, bad):
    base = {"embed_dim": 8, "n_fields": 4, "attn_heads": 2, "K": 3,
            "rl_layers": [16, 8], "beta": 1.0, "base_seed": 42}
    base.update(bad)
    with pytest.raises(ConfigError, match="divisible|must be divisible"):
        load_config("toy", "toy_model", root=write_configs(tmp_path, base=base))


def test_expected_stats_drift_is_caught(tmp_path):
    """
    §4's table lives in two places -- the dataset config for readability
    and common.py for the gate -- so the pairing is checked on every
    load instead of being left to drift.
    """
    root = write_configs(tmp_path, dataset={
        "dataset": "AMusic", "data_path": "data/AMusic",
        "expected_stats": {"users": 1776, "items": 12926},
    }, dataset_name="AMusic")
    with pytest.raises(ConfigError, match="drifted"):
        load_config("AMusic", "toy_model", root=root)


def test_expected_stats_for_an_unknown_dataset_is_refused(tmp_path):
    root = write_configs(tmp_path, dataset={
        "dataset": "toy", "data_path": "d", "expected_stats": {"users": 1}})
    with pytest.raises(ConfigError, match="no entry in EXPECTED_STATS"):
        load_config("toy", "toy_model", root=root)


def test_seeds_resolve_from_base_seed():
    assert resolve_seeds({"base_seed": 42}, runs=3) == [42, 43, 44]
    assert resolve_seeds({"base_seed": 42}, explicit="7,9") == [7, 9]


# ------------------------------------------------------- the real files

def test_every_real_config_pair_resolves():
    """
    A whitelist the repo's own files violate is not protection, and a
    matrix that dies on config resolution at run 137 is worse than one
    that dies now.
    """
    for dataset in available_datasets():
        for model in available_models():
            config = load_config(dataset, model)
            assert config["dataset"] == dataset and config["model"] == model


def test_every_registry_key_has_a_config_file():
    assert set(available_models()) == set(MATRIX_MODELS), (
        f"configs/models and MATRIX_MODELS disagree: "
        f"{set(available_models()) ^ set(MATRIX_MODELS)}"
    )


def test_every_dataset_in_expected_stats_has_a_config():
    assert set(EXPECTED_STATS) <= set(available_datasets())


def test_no_real_dataset_config_sets_a_modelling_key():
    """The direct regression test for the citeulike override."""
    from src.config import CONFIG_ROOT, DATASET_ALLOWED_KEYS
    for dataset in available_datasets():
        keys = set(yaml.safe_load((CONFIG_ROOT / "datasets" / f"{dataset}.yaml").read_text()))
        assert keys <= DATASET_ALLOWED_KEYS, f"{dataset}: {keys - DATASET_ALLOWED_KEYS}"


def test_beta_is_only_set_for_masked_models():
    """
    An unmasked control with a beta line implies the value does
    something. select_beta.py writes to masked models only, so the two
    have to agree about which those are.
    """
    from src.config import CONFIG_ROOT
    from src.tuning import MASKED_MODELS
    masked = {m for ms in MASKED_MODELS.values() for m in ms}
    for model in available_models():
        raw = yaml.safe_load((CONFIG_ROOT / "models" / f"{model}.yaml").read_text())
        assert ("beta" in raw) == (model in masked), (
            f"{model}: beta in config = {'beta' in raw}, masked = {model in masked}"
        )
