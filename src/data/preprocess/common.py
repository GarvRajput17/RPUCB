"""
Shared preprocessing helpers.

Phase 1's gate is "stats match §4 exactly", so the expected numbers live
here as data and `verify_stats` is what actually enforces them -- a
preprocessing script that silently produces a differently-sized dataset
than the one the writeup claims is the single most expensive kind of error
to discover late, because every downstream number inherits it.

The val split is deliberately NOT baked into files. It is carved at load
time by `RecDataset` from a run seed (data/dataset.py), so the same
train.rating supports all three seeds without three copies on disk, and
the split is reproducible from the seed alone. Preprocessing's job is
therefore just: produce train.rating / test.rating, prove the stats, and
write a manifest the provenance audit can check against.

test.negative is no longer produced. Full-catalog evaluation (§9) ranks
against the entire catalog, so the pre-sampled 99-negative file has no
consumer.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

# §4, verbatim. users / items / interactions / sparsity-percent.
EXPECTED_STATS = {
    "ml-1m":       {"users": 6040, "items": 3706, "interactions": 1000209, "sparsity": 95.53},
    "lastfm":      {"users": 1741, "items": 2665, "interactions": 69149,   "sparsity": 98.51},
    "citeulike-a": {"users": 5551, "items": 16980, "interactions": 204986, "sparsity": 99.78},
    "AMusic":      {"users": 1776, "items": 12926, "interactions": 46087,  "sparsity": 99.80},
    "AToy":        {"users": 3137, "items": 33951, "interactions": 84642,  "sparsity": 99.92},
}

# Interaction counts in §4 are the FULL dataset (train + the one held-out
# test item per user). A leave-one-out split moves exactly one interaction
# per user out of train, so train.rating should hold
# `interactions - users` rows and test.rating exactly `users` rows.
# Checking the reconstructed total rather than the train file alone is what
# makes this robust to a split bug that loses or duplicates rows.


def read_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def write_pairs(path, pairs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for u, i in pairs:
            f.write(f"{u}\t{i}\n")


def compute_stats(train_pairs, test_pairs):
    users = {u for u, _ in train_pairs} | {u for u, _ in test_pairs}
    items = {i for _, i in train_pairs} | {i for _, i in test_pairs}

    # num_users / num_items are max_id + 1, matching how RecDataset sizes
    # its embedding tables -- not len(users), which would silently disagree
    # whenever an id is unused.
    num_users = max(users) + 1 if users else 0
    num_items = max(items) + 1 if items else 0
    total = len(train_pairs) + len(test_pairs)

    density = total / (num_users * num_items) if num_users and num_items else 0.0
    return {
        "users": num_users,
        "items": num_items,
        "interactions": total,
        "sparsity": round(100.0 * (1.0 - density), 2),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "users_with_interactions": len(users),
        "items_with_interactions": len(items),
    }


def verify_stats(dataset, stats, tolerance_sparsity=0.02, strict=True):
    """
    Compare against §4. Returns a list of human-readable mismatches;
    raises when `strict` and anything mismatched.

    Sparsity gets a small tolerance because §4 rounds to two decimals;
    users / items / interactions must match exactly.
    """
    expected = EXPECTED_STATS.get(dataset)
    if expected is None:
        return [f"no expected stats recorded for {dataset!r}"]

    problems = []
    for key in ("users", "items", "interactions"):
        if stats[key] != expected[key]:
            problems.append(
                f"{key}: got {stats[key]:,}, §4 expects {expected[key]:,} "
                f"(difference {stats[key] - expected[key]:+,})"
            )

    if abs(stats["sparsity"] - expected["sparsity"]) > tolerance_sparsity:
        problems.append(
            f"sparsity: got {stats['sparsity']:.2f}%, §4 expects {expected['sparsity']:.2f}%"
        )

    if stats["test_pairs"] != stats["users_with_interactions"]:
        problems.append(
            f"leave-one-out invariant: {stats['test_pairs']:,} test rows for "
            f"{stats['users_with_interactions']:,} users with interactions -- "
            f"expected exactly one held-out item per user"
        )

    if problems and strict:
        raise SystemExit(
            f"\n[{dataset}] Phase 1 gate FAILED:\n  " + "\n  ".join(problems) + "\n"
        )
    return problems


def file_checksum(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(dataset, out_dir, stats, extra=None):
    """
    `manifest.json` beside the rating files: stats plus a checksum per
    file. §6 requires dataset checksums in every results JSON; this is
    what that reads.
    """
    out_dir = Path(out_dir)
    files = {}
    for name in ("train.rating", "test.rating"):
        p = out_dir / name
        if p.is_file():
            files[name] = {"sha256": file_checksum(p), "bytes": p.stat().st_size}

    manifest = {"dataset": dataset, "stats": stats, "files": files}
    if extra:
        manifest.update(extra)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def report(dataset, stats, problems=None):
    expected = EXPECTED_STATS.get(dataset, {})
    print(f"\n[{dataset}]")
    for key in ("users", "items", "interactions", "sparsity"):
        got, want = stats[key], expected.get(key)
        flag = "" if want is None or got == want else "  <-- MISMATCH"
        suffix = "%" if key == "sparsity" else ""
        want_str = f"{want:,}" if isinstance(want, int) else want
        print(f"  {key:<14} {got:>12,}{suffix}   §4: {want_str}{suffix}{flag}"
              if isinstance(got, int)
              else f"  {key:<14} {got:>12}{suffix}   §4: {want_str}{suffix}{flag}")
    print(f"  {'train rows':<14} {stats['train_pairs']:>12,}")
    print(f"  {'test rows':<14} {stats['test_pairs']:>12,}")
    if problems:
        for p in problems:
            print(f"  ! {p}")


def group_by_user(pairs):
    out = defaultdict(set)
    for u, i in pairs:
        out[u].add(i)
    return out