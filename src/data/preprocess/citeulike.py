#!/usr/bin/env python3
"""
citeulike-a: raw users.dat -> train.rating / test.rating.

Ported from the pre-refactor `src/data/preprocess_citeulike.py`, with two
changes: test.negative is no longer generated (full-catalog evaluation has
no consumer for it, §9), and the output is now checked against §4's stats
before the manifest is written.

Why this one needs real preprocessing while the other four do not:
citeulike-a is distributed only as a raw user-item matrix (`users.dat`,
one line per user, first number = interaction count, remainder = item ids).
No pre-split leave-one-out version circulates the way it does for the
DeepCF benchmark datasets.

Split assumption, stated because it cannot be verified against the
original report's script: users.dat carries no timestamp, so the held-out
test item per user is chosen uniformly at random under a fixed seed rather
than by recency. This is the standard fallback for a dataset with no
temporal signal, and §9 already records "unordered item bag for routing"
as a related deviation, but it does mean the citeulike-a split is not
temporally comparable to a ml-1m-style one.

Seeding note: SEED here fixes the *test* split, which must stay identical
across all runs and all seeds -- it is the upstream split every model is
measured against, and re-drawing it per run would make results
incomparable. This is unrelated to the per-run val split, which
`RecDataset` carves from train at load time using the run seed.

The upstream release is already 5-core filtered (min 10 interactions per
user), so no additional filtering happens here.

Usage:
    python -m src.data.preprocess.citeulike
"""

import argparse
import random
from pathlib import Path

from .common import compute_stats, report, verify_stats, write_pairs, write_manifest

DATASET = "citeulike-a"
SEED = 42
DEFAULT_INPUT = Path("raw/citeulike-a/users.dat")
DEFAULT_OUTPUT = Path("data/citeulike-a")

SOURCE_NOTE = (
    "citeulike-a raw users.dat -- https://github.com/js05212/citeulike-a. "
    "Upstream release is already 5-core filtered."
)


def read_users_dat(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"\nmissing input: {path}\n  Source: {SOURCE_NOTE}\n"
        )

    user_items = []
    with open(path) as f:
        for line_no, line in enumerate(f):
            parts = line.split()
            if not parts:
                continue
            count = int(parts[0])
            items = [int(x) for x in parts[1:]]
            if len(items) != count:
                raise SystemExit(
                    f"\n{path}:{line_no + 1}: declared count {count} but found "
                    f"{len(items)} items\n"
                )
            user_items.append(items)
    return user_items


def split_leave_one_out(user_items, seed=SEED):
    rng = random.Random(seed)
    train_pairs, test_pairs = [], []

    for u, items in enumerate(user_items):
        if not items:
            continue
        shuffled = list(items)
        rng.shuffle(shuffled)
        test_pairs.append((u, shuffled[0]))
        train_pairs.extend((u, i) for i in shuffled[1:])

    return train_pairs, test_pairs


def process(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT, seed=SEED, strict=True):
    user_items = read_users_dat(input_path)
    print(f"Read {len(user_items):,} users from {input_path}")

    train_pairs, test_pairs = split_leave_one_out(user_items, seed=seed)

    output_dir = Path(output_dir)
    write_pairs(output_dir / "train.rating", train_pairs)
    write_pairs(output_dir / "test.rating", test_pairs)

    stats = compute_stats(train_pairs, test_pairs)
    problems = verify_stats(DATASET, stats, strict=strict)
    report(DATASET, stats, problems)

    write_manifest(
        DATASET, output_dir, stats,
        extra={
            "source": SOURCE_NOTE,
            "preprocessing": "leave-one-out, random held-out item (no timestamps)",
            "split_seed": seed,
        },
    )
    print(f"  wrote {output_dir}/manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=SEED,
                         help="fixes the TEST split; changing it makes results "
                              "incomparable to every run that came before")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    process(args.input, args.output_dir, args.seed, strict=not args.no_strict)


if __name__ == "__main__":
    main()