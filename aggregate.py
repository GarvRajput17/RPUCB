"""
Read a results tree. Runs nothing.

Until now the five-metric tables and the paired t-tests only appeared at
the end of `main.py --all`, which means the only way to look at 150
finished results files was to start the matrix again and let
`--skip-existing` walk it. That is fine when everything is present and
useless when it is not -- and "not" is the normal state for most of
Phase 6.

    python aggregate.py                       # tables + significance tests
    python aggregate.py --status              # coverage and health
    python aggregate.py --tag beta0.5         # one tagged variant
    python aggregate.py --json findings.json  # machine-readable dump

`--status` is the one to run while the matrix is going. It answers the
two questions §10 says to watch for: which cells are missing or failed,
and how many runs hit the `max_epochs` ceiling instead of stopping early
-- if that count is high, 100 epochs was too low and the matrix needs
rerunning, which is much cheaper to learn on day one than in Phase 7.
"""

import argparse
import json
from collections import defaultdict

from src.utils import (
    ALL_DATASETS,
    ALL_METRICS,
    ALL_MODELS,
    COMPARISONS,
    MODEL_DISPLAY,
    aggregate_results,
    available_tags,
    iter_result_files,
    load_all_results,
    print_results_table,
    run_all_significance_tests,
)


def status_report(root, tag, datasets, models):
    """
    Coverage and health of a results tree.

    Deliberately reports on what is *on disk*, including cells nobody
    asked for: a run written under a misspelled tag or a retired model
    key is invisible to a lookup keyed on the expected matrix, and those
    are exactly the runs whose absence from the tables would otherwise go
    unexplained.
    """
    records = load_all_results(root, tag)
    expected = {(d, m) for d in datasets for m in models}

    seeds = defaultdict(list)
    broken_seeds = set()
    unreadable, capped, not_stopped, no_test_metrics = [], [], [], []
    epochs, gpu_seconds, peak_memory = [], 0.0, []

    for record in records:
        key = (record["dataset"], record["model"])
        seeds[key].append(record["seed"])

        if record.get("unreadable"):
            # Still counted in the coverage row -- the file is there, and a
            # cell that looks full is the thing worth flagging -- but marked
            # so "42,43,44,99" is not read as a fourth seed.
            broken_seeds.add((key, record["seed"]))
            unreadable.append(record)
            continue
        if "test_metrics" not in record:
            no_test_metrics.append(record)

        timing = record.get("timing") or {}
        stopping = record.get("stopping") or {}
        config = record.get("config") or {}

        ran = timing.get("epochs_run")
        if ran is not None:
            epochs.append(ran)
            gpu_seconds += (timing.get("train_seconds_total") or 0.0)
            gpu_seconds += (timing.get("val_seconds_total") or 0.0)
            gpu_seconds += (timing.get("test_seconds") or 0.0)
            cap = config.get("max_epochs")
            if cap is not None and ran >= cap:
                capped.append((record["dataset"], record["model"], record["seed"], ran))
        if stopping.get("stopped_early") is False:
            not_stopped.append((record["dataset"], record["model"], record["seed"]))

        if record.get("peak_memory_mb"):
            peak_memory.append((record["peak_memory_mb"], record["dataset"], record["model"]))

    label = "untagged (main matrix)" if tag is None else f"tag {tag!r}"
    print(f"\n=== coverage: {root}/  [{label}] ===")
    header = f"{'Model':<28}" + "".join(f"| {d:<13}" for d in datasets)
    print(header)
    print("-" * len(header))
    for model in models:
        row = f"{MODEL_DISPLAY.get(model, model):<28}"
        for dataset in datasets:
            found = sorted(seeds.get((dataset, model), []))
            cell = ",".join(
                f"{s}!" if ((dataset, model), s) in broken_seeds else str(s)
                for s in found
            )
            row += f"| {(cell or '---'):<13}"
        print(row)

    present = {k for k, v in seeds.items() if v}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    total_runs = sum(len(v) for v in seeds.values())

    print(f"\n  {total_runs} result files; {len(present & expected)} of "
          f"{len(expected)} requested cells have results")
    if missing:
        print(f"  {len(missing)} cells with no results yet:")
        for dataset, model in missing[:20]:
            print(f"    {dataset}/{model}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
    if extra:
        print(f"  {len(extra)} cells outside the requested matrix (check for typos):")
        for dataset, model in extra:
            print(f"    {dataset}/{model}")

    duplicates = sorted(k for k, v in seeds.items() if len(v) != len(set(v)))
    if duplicates:
        print(f"  ! duplicate seed files in {len(duplicates)} cells: {duplicates}")
    if unreadable:
        print(f"  ! {len(unreadable)} unreadable results files:")
        for record in unreadable:
            print(f"    {record['path']}: {record['unreadable']}")
    if no_test_metrics:
        print(f"  ! {len(no_test_metrics)} results files with no test_metrics block")

    print("\n=== health ===")
    if epochs:
        print(f"  epochs run: min {min(epochs)}  median {sorted(epochs)[len(epochs) // 2]}  "
              f"max {max(epochs)}  (n={len(epochs)})")
        rate = 1 - len(capped) / len(epochs)
        print(f"  stopped before the epoch cap: {rate * 100:.0f}% "
              f"({len(epochs) - len(capped)}/{len(epochs)})")
        if capped:
            print(f"  ! {len(capped)} runs hit max_epochs. If this is more than a few, "
                  f"the cap is too low and those runs are undertrained:")
            for dataset, model, seed, ran in capped[:15]:
                print(f"    {dataset}/{model}/seed{seed}  {ran} epochs")
            if len(capped) > 15:
                print(f"    ... and {len(capped) - 15} more")
        print(f"  GPU time in these runs: {gpu_seconds / 3600:.1f} h "
              f"(train + val + test, excludes data loading and startup)")
    else:
        print("  no timing recorded yet")

    if peak_memory:
        peak_memory.sort(reverse=True)
        worst = peak_memory[0]
        print(f"  peak GPU memory: max {worst[0]:.0f} MB "
              f"({worst[1]}/{worst[2]}), median "
              f"{peak_memory[len(peak_memory) // 2][0]:.0f} MB")

    shas = {(r.get("provenance") or {}).get("git_sha") for r in records
            if not r.get("unreadable")}
    shas.discard(None)
    if len(shas) > 1:
        print(f"  ! {len(shas)} distinct git SHAs across these runs. Phase 7's "
              f"provenance audit requires one:")
        for sha in sorted(shas):
            count = sum(1 for r in records
                        if (r.get("provenance") or {}).get("git_sha") == sha)
            print(f"    {sha[:12]}  {count} runs")
    elif shas:
        print(f"  git SHA: {next(iter(shas))[:12]} (all runs)")

    return {
        "root": root, "tag": tag,
        "cells_expected": len(expected), "cells_present": len(present),
        "runs": total_runs,
        "missing": [f"{d}/{m}" for d, m in missing],
        "unexpected": [f"{d}/{m}" for d, m in extra],
        "capped_runs": [f"{d}/{m}/seed{s}" for d, m, s, _ in capped],
        "gpu_hours": round(gpu_seconds / 3600, 2),
        "git_shas": sorted(shas),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate a results tree")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--tag", default=None,
                         help="a tagged variant; default is the untagged main matrix")
    parser.add_argument("--list-tags", action="store_true")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--status", action="store_true",
                         help="coverage and health instead of the tables")
    parser.add_argument("--no-tests", action="store_true",
                         help="skip the paired t-tests")
    parser.add_argument("--json", default=None, help="also write everything here")
    args = parser.parse_args()

    if args.list_tags:
        tags = available_tags(args.results_root)
        print(f"tags under {args.results_root}/: {tags or '(none)'}")
        counts = defaultdict(int)
        for _, _, tag, _, _ in iter_result_files(args.results_root):
            counts[tag] += 1
        for tag, count in sorted(counts.items(), key=lambda kv: (kv[0] or "")):
            print(f"  {(tag or '(untagged)'):<20} {count} runs")
        return

    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else ALL_DATASETS
    models = [m.strip() for m in args.models.split(",")] if args.models else ALL_MODELS
    metrics = [m.strip() for m in args.metrics.split(",")] if args.metrics else ALL_METRICS

    payload = {"results_root": args.results_root, "tag": args.tag}

    if args.status:
        payload["status"] = status_report(args.results_root, args.tag, datasets, models)
    else:
        print_results_table(models=models, datasets=datasets, metrics=metrics,
                             root=args.results_root, tag=args.tag)
        payload["aggregates"] = {
            f"{dataset}/{model}": aggregate_results(dataset, model,
                                                     args.results_root, args.tag)
            for dataset in datasets for model in models
            if aggregate_results(dataset, model, args.results_root, args.tag)
        }

        if not args.no_tests:
            # §7: three seeds gives very little power, so these are
            # reported as suggestive. The header says so here too, because
            # a table of p-values with no such line next to it is exactly
            # how a suggestive result becomes a claimed one.
            print("\n=== paired t-tests (n=3 seeds: suggestive, not confirmatory) ===")
            payload["significance"] = run_all_significance_tests(
                datasets=datasets, root=args.results_root, tag=args.tag)
            print(f"\n  {len(COMPARISONS)} comparisons x {len(datasets)} datasets "
                  f"requested; rows above are those with at least 2 shared seeds.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()