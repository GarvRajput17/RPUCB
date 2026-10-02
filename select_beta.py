"""
Beta selection, Phase 5.5 (§3/§5: "only beta tuned, on validation only").

Run after `experiments/tune_beta.sh` has produced the sweep:

    python select_beta.py                        # report only
    python select_beta.py --write                # and commit the values

What it does, and why it is not just `max(val_metrics_at_best)` over the
results files:

  1. When `val_subsample_size` is set, per-epoch validation is a subsample.
     Under full-catalog evaluation that yields few HR@10 hits, so
     neighbouring grid points differ by less than the noise.

  2. `val_metrics_at_best` is a *maximum over epochs* of that estimate, so
     it is biased upward, and biased more for the noisiest settings.

So each grid point's best checkpoint is re-scored once on the **full**
validation set, and selection runs on that.

With `val_subsample_size: null` -- the current setting -- per-epoch
validation already is the full set, so the re-score should reproduce
`val_metrics_at_best` to rounding. That makes it a check rather than a
correction: a grid point whose re-score disagrees with its own recorded
score has a checkpoint that does not reproduce what training saw, and
every "subsample disagreement" in the report is then a reproducibility
problem, not noise. The max-over-epochs bias in point 2 remains either
way; no re-scoring of the chosen checkpoint can remove it.

The choice itself is arithmetic and lives in `src/tuning.py`; this file
is the part that needs a GPU.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from src.checkpoint import checkpoint_path, load_checkpoint
from src.config import available_datasets, dataset_is_available, load_config
from src.data.dataset import RecDataset
from src.evaluate import evaluate_split
from src.losses import LOSS_FAMILY
from src.models.registry import build_model
from src.reproducibility import set_global_seed
from src.tuning import (
    DEFAULT_GRID,
    DEFAULT_SWEEP_MODELS,
    FAMILIES,
    apply_config_updates,
    beta_tag,
    config_updates,
    family_of,
    parse_grid,
    render_report,
    score_key,
    select_all,
    witness_selections,
)
from src.utils import run_slug


def assert_families_match_losses():
    """
    `src/tuning.py` groups models by key prefix and stays torch-free;
    `src/losses.py` groups them by loss. Beta is tuned per family, so the
    two groupings must be the same partition. Checked here rather than
    trusted, because this is the one place that has both.
    """
    by_prefix = {}
    for model, loss in LOSS_FAMILY.items():
        by_prefix.setdefault(family_of(model), set()).add(loss)

    for family, losses in by_prefix.items():
        if len(losses) != 1:
            raise SystemExit(
                f"family {family!r} spans more than one loss {sorted(losses)}; "
                f"src/tuning.py's per-prefix grouping no longer matches "
                f"src/losses.py's LOSS_FAMILY, so 'tuned per family' is ambiguous."
            )
    seen = [next(iter(v)) for v in by_prefix.values()]
    if len(set(seen)) != len(seen):
        raise SystemExit(
            f"two families share a loss {seen}; the per-family beta would be "
            f"tuned twice for the same objective."
        )


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def load_cache(path):
    if not Path(path).is_file():
        return {}
    with open(path) as f:
        return json.load(f)


def save_json(path, payload):
    """Atomic, like every other artifact this codebase writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def read_run_result(results_root, dataset, model, beta, seed):
    """The sweep run's own results file, for its subsampled best-epoch val."""
    path = (Path(results_root) / dataset / run_slug(model, beta_tag(beta))
            / f"seed{seed}.json")
    if not path.is_file():
        return None, path
    try:
        with open(path) as f:
            return json.load(f), path
    except (OSError, json.JSONDecodeError):
        return None, path


def reevaluate(dataset_obj, dataset_name, model_name, beta, seed, config,
               checkpoint_root, device, resident):
    """
    Score one grid point's best checkpoint on the whole validation set.

    The checkpoint's stored config is checked against what we think we
    are loading before anything is scored. A tag that does not match the
    beta inside the file would otherwise produce a complete, plausible
    selection report built on mislabelled runs -- the failure mode this
    refactor keeps closing elsewhere, and there is no reason to leave it
    open here.
    """
    ckpt = checkpoint_path(dataset_name, run_slug(model_name, beta_tag(beta)), seed,
                            root=checkpoint_root, create=False)
    if not (ckpt.is_file() and ckpt.stat().st_size > 0):
        return None, f"no checkpoint at {ckpt}"

    model = build_model(model_name, dataset_obj, config).to(device)
    restored = load_checkpoint(ckpt, model, map_location=device)
    model.to(device)

    stored = restored["config"]
    for key, expected in (("model", model_name), ("dataset", dataset_name),
                          ("beta", beta)):
        if stored.get(key) != expected:
            return None, (f"{ckpt} holds {key}={stored.get(key)!r}, expected "
                          f"{expected!r} -- the tag and the checkpoint disagree")
    if restored["seed"] != seed:
        return None, (f"{ckpt} was trained at seed {restored['seed']}, not {seed}; "
                      f"the validation carve differs, so the scores are not comparable")

    metrics = evaluate_split(
        model, dataset_obj, "val", device,
        max_users=None,                      # the whole point: no subsample
        interaction_rows_gpu=resident["rows"],
        interaction_cols_gpu=resident["cols"],
        user_batch_size=config.get("user_batch_size", 64),
        item_chunk_size=config.get("item_encode_chunk", 8192),
        pair_chunk=config.get("pair_chunk", 262144),
        tie_policy=config.get("tie_policy", "mid"),
        tie_atol=config.get("tie_atol", 1e-6),
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"metrics": metrics, "best_epoch": restored["epoch"],
            "checkpoint": str(ckpt)}, None


def main():
    parser = argparse.ArgumentParser(description="Select beta per family on full validation")
    parser.add_argument("--results-root", default="results_tuning",
                         help="where experiments/tune_beta.sh wrote the sweep")
    parser.add_argument("--checkpoint-root", default="checkpoints_tuning")
    parser.add_argument("--grid", default=",".join(f"{b:g}" for b in DEFAULT_GRID))
    parser.add_argument("--models", default=",".join(DEFAULT_SWEEP_MODELS),
                         help="the masked models that were swept")
    parser.add_argument("--datasets", default=None,
                         help="default: every dataset with rating files on disk")
    parser.add_argument("--seed", type=int, default=None,
                         help="the sweep's single seed; default base_seed from base.yaml")
    parser.add_argument("--metric", default=None,
                         help="drives the choice; default is early_stopping_metric from "
                              "the config, so beta is selected by the same criterion "
                              "that selected each checkpoint. The others are witnesses.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--refresh", action="store_true",
                         help="re-score grid points already in the cache")
    parser.add_argument("--cache", default=None,
                         help="default: <results-root>/beta_val_scores.json")
    parser.add_argument("--report", default=None,
                         help="default: <results-root>/beta_selection.json")
    parser.add_argument("--write", action="store_true",
                         help="write the chosen values into configs/models/*.yaml")
    args = parser.parse_args()

    assert_families_match_losses()

    grid = parse_grid(args.grid)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cache_path = args.cache or str(Path(args.results_root) / "beta_val_scores.json")
    report_path = args.report or str(Path(args.results_root) / "beta_selection.json")

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = [d for d in available_datasets() if dataset_is_available(d)[0]]
    if not datasets:
        parser.error("no dataset has its rating files on disk")

    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    print(f"Using device: {device}")
    print(f"grid={[f'{b:g}' for b in grid]}  models={models}  datasets={datasets}")

    resolved = load_config(datasets[0], models[0])
    seed = args.seed
    if seed is None:
        seed = int(resolved["base_seed"])
        print(f"seed: {seed} (base_seed)")
    if args.metric is None:
        # Selecting beta on a different metric from the one that chose each
        # run's checkpoint would mix two criteria in one decision.
        args.metric = resolved.get("early_stopping_metric", "HR@10")
        print(f"selection metric: {args.metric} (early_stopping_metric)")

    cache = load_cache(cache_path)
    missing = []

    # One dataset at a time: RecDataset carries the interaction matrices,
    # and both of them stay resident on the device for the length of the
    # dataset's grid so that 5 betas x N models do not re-transfer them.
    for dataset_name in datasets:
        wanted = [
            (model, beta) for model in models for beta in grid
            if args.refresh or score_key(dataset_name, model, beta) not in cache
        ]
        if not wanted:
            print(f"[{dataset_name}] all grid points cached")
            continue

        probe = load_config(dataset_name, models[0])
        set_global_seed(seed, deterministic=False)
        dataset_obj = RecDataset(probe["data_path"], probe["num_negatives"], seed=seed)
        resident = {"rows": dataset_obj.interaction_rows.to(device),
                    "cols": dataset_obj.interaction_cols.to(device)}
        num_val_users = len(dataset_obj.get_val_data())
        print(f"[{dataset_name}] {num_val_users:,} validation users, "
              f"{len(wanted)} grid points to score")

        for model_name, beta in wanted:
            config = load_config(dataset_name, model_name, overrides={"beta": beta})
            config["seed"] = seed

            t0 = time.perf_counter()
            scored, problem = reevaluate(
                dataset_obj, dataset_name, model_name, beta, seed, config,
                args.checkpoint_root, device, resident,
            )
            if problem:
                print(f"  [missing] {model_name} {beta_tag(beta)}: {problem}")
                missing.append({"dataset": dataset_name, "model": model_name,
                                 "beta": beta, "problem": problem})
                continue

            run, run_path = read_run_result(args.results_root, dataset_name,
                                             model_name, beta, seed)
            cache[score_key(dataset_name, model_name, beta)] = {
                "dataset": dataset_name,
                "model": model_name,
                "beta": beta,
                "tag": beta_tag(beta),
                "seed": seed,
                "full_val": scored["metrics"],
                "subsample_at_best": (run or {}).get("val_metrics_at_best"),
                "val_subsample_size": (run or {}).get("config", {}).get("val_subsample_size"),
                "num_val_users": num_val_users,
                "best_epoch": scored["best_epoch"],
                "epochs_run": ((run or {}).get("timing") or {}).get("epochs_run"),
                "stopped_early": ((run or {}).get("stopping") or {}).get("stopped_early"),
                "checkpoint": scored["checkpoint"],
                "results_file": str(run_path),
                "reeval_seconds": round(time.perf_counter() - t0, 2),
                "git_sha": git_sha(),
            }
            # Written after every grid point, not at the end: on AToy's
            # 33,953-item catalog a full-validation pass is not cheap, and
            # a crash 60 points in should not cost all 60.
            save_json(cache_path, cache)
            value = scored["metrics"].get(args.metric, float("nan"))
            print(f"  {model_name:<18} {beta_tag(beta):<9} "
                  f"full-val {args.metric}={value:.4f}  "
                  f"({cache[score_key(dataset_name, model_name, beta)]['reeval_seconds']}s)")

        del dataset_obj, resident
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not cache:
        print("\nNothing scored. Has experiments/tune_beta.sh run?")
        sys.exit(1)

    selections = select_all(cache, grid, args.metric)
    witnesses = witness_selections(cache, grid)
    print(render_report(selections, witnesses))

    updates = config_updates(selections)
    print("\n=== config changes ===")
    for update in updates:
        if update.get("problem"):
            print(f"  ! {update['path']}: {update['problem']}")
        else:
            arrow = "unchanged" if update["old"] == update["new"] else \
                    f"{update['old']:g} -> {update['new']:g}"
            print(f"  {update['path']:<40} beta: {arrow}")

    save_json(report_path, {
        "metric": args.metric,
        "grid": grid,
        "seed": seed,
        "models_swept": models,
        "datasets": datasets,
        "selections": selections,
        "witness_metrics": witnesses,
        "config_updates": updates,
        "missing": missing,
        "git_sha": git_sha(),
        "written": False,
    })

    if args.write:
        written = apply_config_updates(updates)
        print(f"\nWrote {len(written)} config files. Commit them before Phase 6 -- "
              f"the full run's provenance audit expects one code version, and the "
              f"beta values are part of it.")
        report = json.load(open(report_path))
        report["written"] = True
        save_json(report_path, report)
    else:
        print("\nReport only. Re-run with --write to commit these values.")

    print(f"\nscores: {cache_path}\nreport: {report_path}")
    if missing:
        print(f"{len(missing)} grid points had no usable checkpoint -- "
              f"rows containing them were excluded from selection, not averaged over.")


if __name__ == "__main__":
    main()