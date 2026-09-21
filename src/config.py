"""
Config resolution: base.yaml -> datasets/<name>.yaml -> models/<key>.yaml
-> CLI overrides.

The point of this module is not the merge, which is trivial. It is the
whitelists.

The pre-refactor repo had `beta: 0.05` sitting in `citeulike.yaml` while
every other dataset ran at the default. Nothing about that was visible in
the results: one dataset was quietly running a twentieth of the
exploration scale, and any conclusion drawn about "RP-UCB helps more as
sparsity increases" would have been partly measuring that. §7 calls for
killing it specifically.

Deleting the line fixes this instance. Whitelisting the keys each layer
may set fixes the class -- a dataset file that tries to set a modelling
hyperparameter now raises at load time rather than silently winning the
merge. Which layer owns which key is the actual protocol decision (§5:
"Shared fixed ...; only beta tuned, on validation only"), so it belongs
in code where it is enforced, not in a comment where it is aspirational.

The fully resolved dict is what gets written into every checkpoint and
every results JSON (§6), so an audit reads what ran rather than
re-deriving this merge.
"""

from pathlib import Path

import yaml

from .data.preprocess.common import EXPECTED_STATS

CONFIG_ROOT = Path("configs")

# A dataset file describes the data and how much of it fits in a batch.
# It may not touch model capacity, the mask, or the optimiser.
#
# `expected_stats` is documentation that is checked rather than trusted:
# _validate cross-references it against EXPECTED_STATS in
# data/preprocess/common.py, which is what the Phase 1 gate enforces. Two
# copies of §4's table exist because both are genuinely useful in place --
# the gate needs it without importing config machinery, the dataset file
# needs it to be readable -- so the pairing is checked on every load
# instead of left to drift.
DATASET_ALLOWED_KEYS = {
    "dataset",
    "data_path",
    "batch_size",
    "expected_stats",
}

# A model file selects a variant and carries the one hyperparameter §3 says
# is tuned per family. Structural switches (head_mode, mask_granularity,
# adaptive_k, K) live here because they define which of the ten models this
# is, not how it is trained.
MODEL_ALLOWED_KEYS = {
    "model",
    "beta",
    "head_mode",
    "mask_granularity",
    "adaptive_k",
    "K",
}


class ConfigError(ValueError):
    pass


def _load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _check_allowed(layer_name, path, config, allowed):
    offenders = sorted(set(config) - allowed)
    if offenders:
        raise ConfigError(
            f"{path}: {layer_name} config may not set {offenders}.\n"
            f"  Allowed here: {sorted(allowed)}\n"
            f"  Everything else belongs in configs/base.yaml, so that all runs "
            f"share it and no single {layer_name} can diverge silently.\n"
            f"  (This is the check that would have caught the pre-refactor "
            f"`citeulike.yaml: beta: 0.05` override.)"
        )


def available_datasets(root=CONFIG_ROOT):
    return sorted(p.stem for p in (Path(root) / "datasets").glob("*.yaml"))


def available_models(root=CONFIG_ROOT):
    return sorted(p.stem for p in (Path(root) / "models").glob("*.yaml"))


def load_config(dataset, model, overrides=None, root=CONFIG_ROOT):
    """
    Resolve one (dataset, model) pair into a single flat dict.

    Args:
        dataset: stem of configs/datasets/<dataset>.yaml
        model:   stem of configs/models/<model>.yaml
        overrides: dict applied last, for CLI flags. Not whitelisted --
            an explicit command-line override is a deliberate act by
            whoever typed it, unlike a file that quietly persists.
    """
    root = Path(root)

    base = _load_yaml(root / "base.yaml")

    dataset_cfg = _load_yaml(root / "datasets" / f"{dataset}.yaml")
    _check_allowed("dataset", root / "datasets" / f"{dataset}.yaml",
                   dataset_cfg, DATASET_ALLOWED_KEYS)

    model_cfg = _load_yaml(root / "models" / f"{model}.yaml")
    _check_allowed("model", root / "models" / f"{model}.yaml",
                   model_cfg, MODEL_ALLOWED_KEYS)

    resolved = dict(base)
    resolved.update(dataset_cfg)
    resolved.update(model_cfg)
    if overrides:
        resolved.update({k: v for k, v in overrides.items() if v is not None})

    _validate(resolved, dataset, model)
    return resolved


def _validate(config, dataset, model):
    if config.get("dataset") != dataset:
        raise ConfigError(
            f"configs/datasets/{dataset}.yaml declares dataset="
            f"{config.get('dataset')!r}; filename and contents must agree"
        )
    if config.get("model") != model:
        raise ConfigError(
            f"configs/models/{model}.yaml declares model={config.get('model')!r}; "
            f"filename and contents must agree"
        )

    rl_layers = config.get("rl_layers")
    embed_dim = config.get("embed_dim")
    if rl_layers and rl_layers[-1] != embed_dim:
        raise ConfigError(
            f"rl_layers[-1]={rl_layers[-1]} must equal embed_dim={embed_dim}: the "
            f"d-dim RP-UCB mask is applied to the CFNet-rl branch output as well "
            f"as the ML embeddings (see models/deepcf_rpucb.py)"
        )

    embed_dim = config.get("embed_dim")
    n_fields = config.get("n_fields")
    if embed_dim and n_fields and embed_dim % n_fields:
        raise ConfigError(
            f"embed_dim={embed_dim} must be divisible by n_fields={n_fields} "
            f"(LiteDHEN splits the embedding into equal field chunks)"
        )

    attn_heads = config.get("attn_heads")
    if embed_dim and attn_heads and embed_dim % attn_heads:
        raise ConfigError(
            f"embed_dim={embed_dim} must be divisible by attn_heads={attn_heads}"
        )

    expected = config.get("expected_stats")
    if expected is not None:
        canonical = EXPECTED_STATS.get(dataset)
        if canonical is None:
            raise ConfigError(
                f"configs/datasets/{dataset}.yaml declares expected_stats but "
                f"{dataset!r} has no entry in EXPECTED_STATS "
                f"(src/data/preprocess/common.py); the Phase 1 gate could not "
                f"check this dataset"
            )
        drift = {
            key: (expected[key], canonical[key])
            for key in canonical
            if key in expected and expected[key] != canonical[key]
        }
        if drift:
            details = "; ".join(
                f"{k}: config says {a}, common.py says {b}" for k, (a, b) in drift.items()
            )
            raise ConfigError(
                f"configs/datasets/{dataset}.yaml expected_stats has drifted from "
                f"EXPECTED_STATS in src/data/preprocess/common.py -- {details}. "
                f"One of the two is wrong about §4; fix both to agree."
            )

    if config.get("model") == "dcm_rpucb_kd":
        # concat mode masks a K*d-wide vector, so K*d must also satisfy the
        # tower's divisibility constraints if anything downstream re-splits it.
        wide = config["K"] * config["embed_dim"]
        if wide % config["n_fields"]:
            raise ConfigError(
                f"K*embed_dim={wide} must be divisible by n_fields="
                f"{config['n_fields']} for dcm_rpucb_kd"
            )


def resolve_seeds(config, runs=None, explicit=None):
    """base_seed .. base_seed+runs-1, or an explicit list."""
    if explicit:
        return [int(s) for s in str(explicit).split(",")]
    runs = runs if runs is not None else 3
    return [config.get("base_seed", 42) + i for i in range(runs)]


def dataset_is_available(dataset, root=CONFIG_ROOT):
    """
    Whether `dataset`'s rating files are actually on disk.

    Separate from config resolution: lastfm and AToy have complete config
    files but no data yet, and `--all` should report that as a skip rather
    than dying on a FileNotFoundError partway through a 150-run matrix.
    """
    try:
        cfg = _load_yaml(Path(root) / "datasets" / f"{dataset}.yaml")
    except ConfigError:
        return False, f"no configs/datasets/{dataset}.yaml"

    data_path = Path(cfg.get("data_path", ""))
    missing = [
        str(data_path / name)
        for name in ("train.rating", "test.rating")
        if not (data_path / name).is_file()
    ]
    if missing:
        return False, f"missing {', '.join(missing)}"
    return True, None