"""
Result persistence, aggregation, the 5-metric table, and significance
tests (§7).

Storage layout: `results/{dataset}/{model}/seed{seed}.json`, mirroring
`checkpoints/{dataset}/{model}/seed{seed}/best.pt`, so `--skip-existing`
works at seed granularity.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Config keys that define *which experiment* a results file represents.
# `check_existing_run` compares these before honouring --skip-existing, so a
# stored result can never be mistaken for one produced under different
# settings. The motivating case: the Phase 5 dry run writes seed-42 results
# with max_epochs=2, and the full run's first seed is also 42 -- without this,
# --skip-existing would keep 2-epoch numbers for a third of the matrix and say
# nothing.
#
# Deliberately excluded: git SHA (code edits between runs are normal),
# device, and the throughput knobs (user_batch_size, item_encode_chunk,
# pair_chunk, num_workers, popularity_reference) -- none of them change
# what is being measured.
RUN_IDENTITY_KEYS = (
    "dataset", "model", "seed",
    "embed_dim", "K", "head_mode", "mask_granularity", "adaptive_k",
    "rl_layers", "ml_layers", "max_hist_len", "routing_iters", "label_aware_power",
    "beta", "gamma_init", "mask_init", "lambda_l1", "loss_override",
    "lr", "batch_size", "num_negatives",
    "max_epochs", "patience", "min_delta",
    "early_stopping_metric", "val_subsample_size",
    "tie_policy", "tie_atol",
)


class StaleResultError(RuntimeError):
    """An existing results file was produced under different settings."""


# Warn once per path, not once per read: print_results_table alone reads
# every cell five times over, and a wall of the same warning trains the
# reader to scroll past the thing it is there to make visible.
_WARNED_UNREADABLE = set()

ALL_MODELS = [
    "deepcf", "deepcf_rpucb", "deepcf_rpucb_attn",
    "mind", "mind_rpucb_multi", "mind_rpucb",
    "dcm", "dcm_rpucb_multi", "dcm_rpucb_kd", "dcm_rpucb_d",
]

ALL_DATASETS = ["ml-1m", "lastfm", "citeulike-a", "AMusic", "AToy"]

ALL_METRICS = ["HR@10", "HR@100", "NDCG@10", "Coverage@10", "ILD@10"]

MODEL_DISPLAY = {
    "deepcf":            "DeepCF",
    "deepcf_rpucb":      "DeepCF + RP-UCB",
    "deepcf_rpucb_attn": "DeepCF + RP-UCB + Attn",
    "mind":              "MIND",
    "mind_rpucb_multi":  "MIND + RP-UCB (multi)",
    "mind_rpucb":        "MIND + RP-UCB (K=1)",
    "dcm":               "Pinterest DCM (K=7)",
    "dcm_rpucb_multi":   "DCM + RP-UCB (7 heads)",
    "dcm_rpucb_kd":      "DCM + RP-UCB (K*d)",
    "dcm_rpucb_d":       "DCM + RP-UCB (d)",
    "dcm_rpucb_shared":  "DCM + RP-UCB (shared mask)",
}

# §2's "comparison structure" table, encoded directly.
COMPARISONS = [
    ("deepcf",           "deepcf_rpucb",      "pure mask effect (DeepCF)"),
    ("deepcf_rpucb",     "deepcf_rpucb_attn", "pure attention effect (DeepCF)"),
    ("deepcf",           "deepcf_rpucb_attn", "combined mask+attention (DeepCF)"),
    ("mind",             "mind_rpucb_multi",  "pure mask effect (MIND)"),
    ("mind_rpucb_multi", "mind_rpucb",        "architecture+capacity effect (MIND)"),
    ("mind",             "mind_rpucb",        "headline efficiency (MIND)"),
    ("dcm",              "dcm_rpucb_multi",   "pure mask effect (DCM)"),
    ("dcm_rpucb_multi",  "dcm_rpucb_kd",      "pure architecture effect (DCM)"),
    ("dcm_rpucb_kd",     "dcm_rpucb_d",       "capacity effect (DCM)"),
    ("dcm",              "dcm_rpucb_kd",      "headline: K heads vs 1 gated embedding (DCM)"),
    ("dcm",              "dcm_rpucb_d",       "RP-UCB vs multi-interest at 1/7 capacity (DCM)"),
]


def run_slug(model_name, tag=None):
    """
    Directory name for one run variant: `deepcf_rpucb`, or
    `deepcf_rpucb__beta0.5` when tagged.

    Tags keep non-default runs -- the beta sweep, the parity-start
    mask_init check, the BPR robustness pass, the optimistic-tie
    sensitivity check -- in their own directories. Without them every
    variant writes to the main-matrix path and either overwrites a real
    result or gets skipped as though it were one. Used for both the
    results and the checkpoint tree so the two stay in step.
    """
    return f"{model_name}__{tag}" if tag else model_name


def _seed_path(dataset_name, model_name, seed, root, tag=None):
    return Path(root) / dataset_name / run_slug(model_name, tag) / f"seed{seed}.json"


def run_result_exists(dataset_name, model_name, seed, root="results", tag=None):
    p = _seed_path(dataset_name, model_name, seed, root, tag)
    return p.is_file() and p.stat().st_size > 0


def check_existing_run(config, dataset_name, model_name, seed, root="results", tag=None):
    """
    Decide whether --skip-existing may skip this run.

    Returns True to skip. Raises StaleResultError if a results file exists
    but was produced under different settings -- refusing rather than
    overwriting, because silently discarding a completed run is as bad as
    silently keeping a stale one. The caller is told to either use a --tag
    or move the old results aside.
    """
    path = _seed_path(dataset_name, model_name, seed, root, tag)
    if not (path.is_file() and path.stat().st_size > 0):
        return False

    try:
        with open(path) as f:
            stored = json.load(f).get("config", {})
    except (OSError, json.JSONDecodeError):
        return False  # unreadable: treat as absent and re-run

    differences = {
        key: (stored.get(key), config.get(key))
        for key in RUN_IDENTITY_KEYS
        if key in stored and stored.get(key) != config.get(key)
    }
    if differences:
        detail = "\n    ".join(
            f"{k}: stored {old!r}, current {new!r}" for k, (old, new) in differences.items()
        )
        raise StaleResultError(
            f"{path} already exists but was produced under different settings:\n"
            f"    {detail}\n"
            f"  Refusing to skip it and refusing to overwrite it. Either pass "
            f"--tag NAME to write this variant somewhere else, or move the old "
            f"results directory aside."
        )
    return True


def save_run_result(result, dataset_name, model_name, seed, config=None,
                     provenance=None, root="results", tag=None):
    """`result` is train_model's return value."""
    path = _seed_path(dataset_name, model_name, seed, root, tag)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(result)
    payload["dataset"] = dataset_name
    payload["model"] = model_name
    payload["seed"] = seed
    payload["tag"] = tag
    if config is not None:
        payload["config"] = config
    if provenance is not None:
        payload["provenance"] = provenance

    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)

    hr10 = result["test_metrics"].get("HR@10", float("nan"))
    print(f"Saved {path} (test HR@10={hr10:.4f})")


def split_slug(slug):
    """
    Inverse of `run_slug`: `deepcf_rpucb__beta0.5` -> ('deepcf_rpucb',
    'beta0.5'), `deepcf_rpucb` -> ('deepcf_rpucb', None).

    Used when walking a results tree from the outside -- aggregate.py's
    status pass and the Phase 7 provenance audit both need to read what a
    directory *is* rather than construct the name of one they already
    know. `__` is safe as the separator because no model key contains it.
    """
    if "__" in slug:
        model, tag = slug.split("__", 1)
        return model, tag
    return slug, None


def available_tags(root="results"):
    """Every tag present under `root`, from directory names."""
    tags = set()
    for p in Path(root).glob("*/*"):
        if p.is_dir() and "__" in p.name:
            tags.add(p.name.split("__", 1)[1])
    return sorted(tags)


def iter_result_files(root="results"):
    """
    Every results JSON under `root`, as (dataset, model, tag, seed, path).

    Walks the tree rather than taking a model list, so a run that landed
    somewhere unexpected -- a typo'd tag, a model key that is no longer in
    ALL_MODELS -- shows up instead of being invisible to a lookup keyed on
    what we expected to find.
    """
    for path in sorted(Path(root).glob("*/*/seed*.json")):
        model, tag = split_slug(path.parent.name)
        try:
            seed = int(path.stem[len("seed"):])
        except ValueError:
            continue
        yield path.parent.parent.name, model, tag, seed, path


def load_all_results(root="results", tag=None):
    """
    Every results record under `root`, with `tag=None` meaning untagged
    main-matrix runs only -- not "any tag". A status pass that silently
    folded the beta sweep in with the matrix would report 8 seeds for
    cells that have 3.
    """
    out = []
    for dataset, model, found_tag, seed, path in iter_result_files(root):
        if found_tag != tag:
            continue
        try:
            with open(path) as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            out.append({"dataset": dataset, "model": model, "tag": found_tag,
                        "seed": seed, "path": str(path), "unreadable": str(exc)})
            continue
        record.setdefault("dataset", dataset)
        record.setdefault("model", model)
        record.setdefault("seed", seed)
        record["path"] = str(path)
        out.append(record)
    return out


def _load_seed_results(dataset_name, model_name, root="results", tag=None):
    """
    Every seed file for one cell.

    Unreadable files are skipped with a warning rather than raised on. A
    truncated JSON anywhere under `results/` used to abort the entire
    aggregation, so a single bad file -- a copy from worker1 interrupted
    mid-transfer, a disk that filled between the temp write and the
    rename -- cost the tables for all 150 runs. Skipping keeps the other
    149 readable; the warning and `aggregate.py --status` are what stop
    it from being silent. `check_existing_run` already treats an
    unreadable file as absent, so this is the same convention.
    """
    d = Path(root) / dataset_name / run_slug(model_name, tag)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("seed*.json")):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            if str(p) not in _WARNED_UNREADABLE:
                _WARNED_UNREADABLE.add(str(p))
                print(f"[warning] skipping unreadable {p}: {exc}", file=sys.stderr)
    return out


def aggregate_results(dataset_name, model_name, root="results", tag=None):
    """Mean/std of each of the 5 test metrics across however many seed
    files exist. Returns None if none exist yet."""
    runs = _load_seed_results(dataset_name, model_name, root, tag)
    if not runs:
        return None

    out = {"num_seeds": len(runs)}
    for metric in ALL_METRICS:
        values = [r["test_metrics"][metric] for r in runs if metric in r["test_metrics"]]
        if values:
            out[f"mean_{metric}"] = float(np.mean(values))
            out[f"std_{metric}"] = float(np.std(values))
    return out


def print_results_table(models=ALL_MODELS, datasets=ALL_DATASETS,
                         metrics=ALL_METRICS, root="results", tag=None):
    """One table per metric, rows=models, columns=datasets. Missing cells
    print as '---' so a partially-run matrix stays legible."""
    for metric in metrics:
        header = f"{'Model':<28}" + "".join(f"| {d:<16}" for d in datasets)
        label = f"{metric} (mean ± std across seeds)"
        print(f"\n--- {label}{'' if tag is None else f'  [tag: {tag}]'} ---")
        print(header)
        print("-" * len(header))

        for model in models:
            row = f"{MODEL_DISPLAY.get(model, model):<28}"
            for dataset in datasets:
                agg = aggregate_results(dataset, model, root, tag)
                if agg and f"mean_{metric}" in agg:
                    cell = f"{agg[f'mean_{metric}']:.4f}±{agg[f'std_{metric}']:.4f} (n={agg['num_seeds']})"
                else:
                    cell = "---"
                row += f"| {cell:<16}"
            print(row)


def paired_ttest(dataset_name, model_a, model_b, metric="HR@10", root="results", tag=None):
    """
    Paired t-test on `metric`, paired by seed. Only seeds present for
    BOTH models are used; returns None below 2 pairs. With the locked
    protocol's 3 seeds this has very little power even at n=3 -- treat
    p-values as suggestive, not confirmatory.
    """
    runs_a = {r["seed"]: r["test_metrics"].get(metric)
              for r in _load_seed_results(dataset_name, model_a, root, tag)}
    runs_b = {r["seed"]: r["test_metrics"].get(metric)
              for r in _load_seed_results(dataset_name, model_b, root, tag)}

    common = sorted(set(runs_a) & set(runs_b))
    if len(common) < 2:
        return None

    a = np.array([runs_a[s] for s in common])
    b = np.array([runs_b[s] for s in common])
    t_stat, p_value = stats.ttest_rel(b, a)

    return {
        "n": len(common), "seeds": common,
        "mean_baseline": float(a.mean()), "mean_variant": float(b.mean()),
        "delta": float(b.mean() - a.mean()),
        "t_stat": float(t_stat), "p_value": float(p_value),
    }


def run_all_significance_tests(datasets=ALL_DATASETS, metrics=("HR@10", "NDCG@10"),
                                root="results", tag=None):
    """Runs every §2 comparison, on every dataset, for the given metrics."""
    results = []
    for baseline, variant, label in COMPARISONS:
        for dataset in datasets:
            for metric in metrics:
                r = paired_ttest(dataset, baseline, variant, metric, root, tag)
                if r is None:
                    continue
                r.update({"dataset": dataset, "metric": metric, "comparison": label,
                          "baseline": baseline, "variant": variant})
                results.append(r)
                sig = "*" if r["p_value"] < 0.05 else " "
                print(
                    f"{sig} {dataset:<12} {metric:<10} {label:<45} "
                    f"delta={r['delta']:+.4f}  p={r['p_value']:.4f}  n={r['n']}"
                )
    return results