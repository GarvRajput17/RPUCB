"""
Result persistence, aggregation, the 5-metric table, and significance
tests (§7).

Storage layout: `results/{dataset}/{model}/seed{seed}.json`, mirroring
`checkpoints/{dataset}/{model}/seed{seed}/best.pt`, so `--skip-existing`
works at seed granularity.
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

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


def _seed_path(dataset_name, model_name, seed, root):
    return Path(root) / dataset_name / model_name / f"seed{seed}.json"


def run_result_exists(dataset_name, model_name, seed, root="results"):
    p = _seed_path(dataset_name, model_name, seed, root)
    return p.is_file() and p.stat().st_size > 0


def save_run_result(result, dataset_name, model_name, seed, config=None,
                     provenance=None, root="results"):
    """`result` is train_model's return value."""
    path = _seed_path(dataset_name, model_name, seed, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(result)
    payload["dataset"] = dataset_name
    payload["model"] = model_name
    payload["seed"] = seed
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


def _load_seed_results(dataset_name, model_name, root="results"):
    d = Path(root) / dataset_name / model_name
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("seed*.json")):
        with open(p) as f:
            out.append(json.load(f))
    return out


def aggregate_results(dataset_name, model_name, root="results"):
    """Mean/std of each of the 5 test metrics across however many seed
    files exist. Returns None if none exist yet."""
    runs = _load_seed_results(dataset_name, model_name, root)
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
                         metrics=ALL_METRICS, root="results"):
    """One table per metric, rows=models, columns=datasets. Missing cells
    print as '---' so a partially-run matrix stays legible."""
    for metric in metrics:
        header = f"{'Model':<28}" + "".join(f"| {d:<16}" for d in datasets)
        print(f"\n--- {metric} (mean ± std across seeds) ---")
        print(header)
        print("-" * len(header))

        for model in models:
            row = f"{MODEL_DISPLAY.get(model, model):<28}"
            for dataset in datasets:
                agg = aggregate_results(dataset, model, root)
                if agg and f"mean_{metric}" in agg:
                    cell = f"{agg[f'mean_{metric}']:.4f}±{agg[f'std_{metric}']:.4f} (n={agg['num_seeds']})"
                else:
                    cell = "---"
                row += f"| {cell:<16}"
            print(row)


def paired_ttest(dataset_name, model_a, model_b, metric="HR@10", root="results"):
    """
    Paired t-test on `metric`, paired by seed. Only seeds present for
    BOTH models are used; returns None below 2 pairs. With the locked
    protocol's 3 seeds this has very little power even at n=3 -- treat
    p-values as suggestive, not confirmatory.
    """
    runs_a = {r["seed"]: r["test_metrics"].get(metric)
              for r in _load_seed_results(dataset_name, model_a, root)}
    runs_b = {r["seed"]: r["test_metrics"].get(metric)
              for r in _load_seed_results(dataset_name, model_b, root)}

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
                                root="results"):
    """Runs every §2 comparison, on every dataset, for the given metrics."""
    results = []
    for baseline, variant, label in COMPARISONS:
        for dataset in datasets:
            for metric in metrics:
                r = paired_ttest(dataset, baseline, variant, metric, root)
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