#!/usr/bin/env python3
"""
Preprocessing for the four datasets that ship as DeepCF drop-ins:
ml-1m, lastfm, AMusic, AToy.

These arrive already leave-one-out split (train.rating / test.rating), so
there is nothing to transform. What this script does instead is validate:
confirm the files parse, confirm the stats match §4 exactly, confirm the
one-held-out-item-per-user invariant, and write the checksum manifest that
§6's provenance block reads.

One script rather than four near-identical ones. The gameplan's §7 says
"5 preprocessing scripts"; this is two scripts covering five datasets,
because four of them differ only in a directory name and splitting that
into four copies would mean four places to fix the next time the
validation logic changes.

test.negative is intentionally not read or produced -- full-catalog
evaluation has no use for it (§9).

Usage:
    python -m src.data.preprocess.deepcf_dropin --dataset ml-1m
    python -m src.data.preprocess.deepcf_dropin --all
"""

import argparse
from pathlib import Path

from .common import (
    compute_diagnostics,
    compute_stats,
    read_pairs,
    report,
    verify_stats,
    write_manifest,
)

DROPIN_DATASETS = ["ml-1m", "lastfm", "AMusic", "AToy"]

# Where each dataset's train.rating / test.rating live. All four come from the
# DeepCF release, which names them Data/{name}.train.rating etc.; download
# them into data/{name}/train.rating and data/{name}/test.rating. Column
# counts differ (ml-1m has user, item, rating, timestamp; the others have a
# third column of rating or play count) and lastfm/AToy use CRLF endings --
# read_pairs handles both.
DEFAULT_DATA_ROOT = Path("data")

SOURCE_NOTE = (
    "DeepCF (Deng et al.) benchmark release -- https://github.com/familyld/DeepCF "
    "(Data/{name}.train.rating, Data/{name}.test.rating). Leave-one-out split "
    "upstream. NOT uniformly k-core filtered: ml-1m and lastfm have >= 20 "
    "interactions per user, but AMusic and AToy go down to 1 per user and 1 per "
    "item -- see the manifest's diagnostics block for per-dataset minimums."
)


def process(dataset, data_root=DEFAULT_DATA_ROOT, strict=True):
    data_dir = Path(data_root) / dataset

    train_path = data_dir / "train.rating"
    test_path = data_dir / "test.rating"

    missing = [p for p in (train_path, test_path) if not p.is_file()]
    if missing:
        raise SystemExit(
            f"\n[{dataset}] missing: {', '.join(str(p) for p in missing)}\n"
            f"  Source: {SOURCE_NOTE}\n"
            f"  Expected layout: {data_dir}/train.rating and {data_dir}/test.rating\n"
        )

    train_pairs = read_pairs(train_path)
    test_pairs = read_pairs(test_path)

    stats = compute_stats(train_pairs, test_pairs)
    diagnostics = compute_diagnostics(train_pairs, test_pairs)
    problems = verify_stats(dataset, stats, strict=strict)
    report(dataset, stats, diagnostics, problems)

    write_manifest(
        dataset, data_dir, stats, diagnostics,
        extra={
            "source": SOURCE_NOTE.format(name=dataset),
            "preprocessing": "none (drop-in, validated only)",
        },
    )
    print(f"  wrote {data_dir}/manifest.json")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DROPIN_DATASETS)
    parser.add_argument("--all", action="store_true", help="process every drop-in dataset")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--no-strict", action="store_true",
        help="report §4 mismatches without failing; for inspecting a dataset "
             "that does not match yet, never for a real run",
    )
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("pass --dataset NAME or --all")

    targets = DROPIN_DATASETS if args.all else [args.dataset]
    failed = []
    for dataset in targets:
        try:
            problems = process(dataset, args.data_root, strict=not args.no_strict)
        except SystemExit as exc:
            print(exc)
            failed.append(dataset)
            continue
        # Under --no-strict, verify_stats reports rather than raises, so a
        # mismatch has to be counted here. The previous version printed
        # "All N dataset(s) match §4" whenever nothing raised -- which under
        # --no-strict meant it announced success over visible mismatches.
        if problems:
            failed.append(dataset)

    passed = len(targets) - len(failed)
    print(f"\nPhase 1 gate: {passed} of {len(targets)} dataset(s) passed.")
    if failed:
        print(f"  failed: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()