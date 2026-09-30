"""
Shared preprocessing helpers and the Phase 1 gate.

Phase 1's gate is "stats match §4 exactly". The expected numbers live here
as data and `verify_stats` enforces them -- a preprocessing script that
silently produces a differently-sized dataset from the one the writeup
claims is the most expensive kind of error to discover late, because
every downstream number inherits it.

The gate has three tiers:

  §4 stats      users / items / interactions / sparsity must match exactly
                (sparsity to two decimals, since §4 rounds).

  integrity     must hold for the protocol to be valid at all, and fail the
                gate on violation: no duplicate rows, no test item already
                in that user's train set, exactly one test row per user,
                and no unused user or item ids. Unused ids would silently
                add never-trained rows to every full-catalog ranking and
                inflate Coverage@10's denominator.

  diagnostics   properties of the data that cannot fail the gate but decide
                how results must be read, recorded in the manifest. The
                important ones are the COLD counts:

                  train-less users   users whose only interaction is the
                                     test item. Their feature row and
                                     history bag are empty. For MIND the
                                     interest vectors are then exactly zero,
                                     every item scores exactly 0.0, and
                                     under an optimistic tie rule every such
                                     user is a guaranteed rank-1 hit --
                                     which is what inflated MIND's AMusic
                                     test HR@10 in the Phase B first-light
                                     run (43 of its 66 hits).

                  cold test items    test positives with no train
                                     interactions. Every model's item tower
                                     sees an all-zero input and emits one
                                     shared embedding, so all cold items tie
                                     with each other for every user. No
                                     model in this study can rank them.

                On AMusic and AToy these are large (roughly 27% and 31% of
                test positives are cold) and they scale with sparsity, so
                they confound the "RP-UCB helps more as sparsity increases"
                hypothesis unless ranking ties are handled without bias.

The val split is deliberately NOT baked into files. `RecDataset` carves it
at load time from the run seed, so one train.rating serves all seeds and
the split is reproducible from the seed alone. Carving removes one train
item per user, which makes additional items cold; those seed-dependent
counts are RecDataset's to report, not this module's.

test.negative is no longer produced -- full-catalog evaluation ranks
against the entire catalog, so it has no consumer (§9).
"""

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# §4, corrected. The gameplan's table had AMusic at 12,926 items and AToy at
# 33,951. The upstream DeepCF files have 12,929 and 33,953 distinct items,
# both with contiguous ids (no gaps), matching the DeepCF paper -- so the
# table was off, not the data. Sparsity is unchanged at two decimals.
EXPECTED_STATS = {
    "ml-1m":       {"users": 6040, "items": 3706, "interactions": 1000209, "sparsity": 95.53},
    "lastfm":      {"users": 1741, "items": 2665, "interactions": 69149,   "sparsity": 98.51},
    "citeulike-a": {"users": 5551, "items": 16980, "interactions": 204986, "sparsity": 99.78},
    "AMusic":      {"users": 1776, "items": 12929, "interactions": 46087,  "sparsity": 99.80},
    "AToy":        {"users": 3137, "items": 33953, "interactions": 84642,  "sparsity": 99.92},
}

# Integrity fields that must all be zero for the gate to pass.
INTEGRITY_FIELDS = (
    "duplicate_train_rows",
    "duplicate_test_rows",
    "test_item_in_own_train",
    "users_without_test_row",
    "users_with_multiple_test_rows",
    "unused_user_ids",
    "unused_item_ids",
)


def read_pairs(path):
    """(user, item) pairs. Extra columns -- rating, timestamp, play count --
    are ignored, and `strip` handles the CRLF endings some upstream files
    use (lastfm and AToy)."""
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


def group_by_user(pairs):
    out = defaultdict(set)
    for u, i in pairs:
        out[u].add(i)
    return out


def compute_stats(train_pairs, test_pairs):
    """§4 stats plus the integrity fields the gate fails on."""
    users = {u for u, _ in train_pairs} | {u for u, _ in test_pairs}
    items = {i for _, i in train_pairs} | {i for _, i in test_pairs}

    # num_users / num_items are max_id + 1, matching how RecDataset sizes
    # its tables -- which is exactly why unused ids are an integrity failure
    # rather than a curiosity.
    num_users = max(users) + 1 if users else 0
    num_items = max(items) + 1 if items else 0
    total = len(train_pairs) + len(test_pairs)
    density = total / (num_users * num_items) if num_users and num_items else 0.0

    user_train = group_by_user(train_pairs)
    test_rows_per_user = Counter(u for u, _ in test_pairs)

    return {
        "users": num_users,
        "items": num_items,
        "interactions": total,
        "sparsity": round(100.0 * (1.0 - density), 2),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        # integrity
        "duplicate_train_rows": len(train_pairs) - len(set(train_pairs)),
        "duplicate_test_rows": len(test_pairs) - len(set(test_pairs)),
        "test_item_in_own_train": sum(1 for u, i in test_pairs if i in user_train.get(u, ())),
        "users_without_test_row": len(users - set(test_rows_per_user)),
        "users_with_multiple_test_rows": sum(1 for c in test_rows_per_user.values() if c > 1),
        "unused_user_ids": num_users - len(users),
        "unused_item_ids": num_items - len(items),
    }


def compute_diagnostics(train_pairs, test_pairs):
    """Descriptive properties that cannot fail the gate but shape
    interpretation. See the module docstring for why the cold counts
    matter."""
    per_user = Counter(u for u, _ in train_pairs) + Counter(u for u, _ in test_pairs)
    per_item = Counter(i for _, i in train_pairs) + Counter(i for _, i in test_pairs)
    item_train = Counter(i for _, i in train_pairs)
    users_with_train = {u for u, _ in train_pairs}
    all_items = set(per_item)

    def summary(counter):
        values = sorted(counter.values())
        return {
            "min": values[0],
            "median": statistics.median(values),
            "below_5": sum(v < 5 for v in values),
        }

    cold_test = sum(1 for _, i in test_pairs if item_train[i] == 0)
    trainless = sum(1 for u, _ in test_pairs if u not in users_with_train)

    return {
        "interactions_per_user": summary(per_user),
        "interactions_per_item": summary(per_item),
        "trainless_test_users": trainless,
        "trainless_test_users_fraction": round(trainless / len(test_pairs), 4) if test_pairs else 0.0,
        "cold_test_positives": cold_test,
        "cold_test_positives_fraction": round(cold_test / len(test_pairs), 4) if test_pairs else 0.0,
        "cold_catalog_items": sum(1 for i in all_items if item_train[i] == 0),
    }


def verify_stats(dataset, stats, tolerance_sparsity=0.005, strict=True):
    """
    Compare against §4 and check integrity. Returns a list of mismatches;
    raises SystemExit when `strict` and anything failed.
    """
    problems = []

    expected = EXPECTED_STATS.get(dataset)
    if expected is None:
        problems.append(f"no expected stats recorded for {dataset!r}")
    else:
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

    for field in INTEGRITY_FIELDS:
        if stats.get(field, 0):
            problems.append(f"integrity: {field.replace('_', ' ')} = {stats[field]:,}")

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


def write_manifest(dataset, out_dir, stats, diagnostics=None, extra=None):
    """
    `manifest.json` beside the rating files: stats, diagnostics, and a
    sha256 per file. §6 requires dataset checksums in every results JSON,
    and main.py copies this in. Deterministic output (no timestamps), so
    re-running preprocessing on identical data leaves the file
    byte-identical -- which is what makes `git diff` on a committed
    manifest a verification that another machine's data matches.
    """
    out_dir = Path(out_dir)
    files = {}
    for name in ("train.rating", "test.rating"):
        p = out_dir / name
        if p.is_file():
            files[name] = {"sha256": file_checksum(p), "bytes": p.stat().st_size}

    manifest = {"dataset": dataset, "stats": stats, "files": files}
    if diagnostics is not None:
        manifest["diagnostics"] = diagnostics
    if extra:
        manifest.update(extra)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def report(dataset, stats, diagnostics=None, problems=None):
    expected = EXPECTED_STATS.get(dataset, {})
    print(f"\n[{dataset}]")

    for key in ("users", "items", "interactions"):
        got, want = stats[key], expected.get(key)
        flag = "" if want is None or got == want else "   <-- MISMATCH"
        want_str = f"{want:,}" if want is not None else "?"
        print(f"  {key:<14} {got:>12,}   §4: {want_str}{flag}")

    got, want = stats["sparsity"], expected.get("sparsity")
    flag = "" if want is None or abs(got - want) <= 0.005 else "   <-- MISMATCH"
    want_str = f"{want:.2f}%" if want is not None else "?"
    print(f"  {'sparsity':<14} {got:>11.2f}%   §4: {want_str}{flag}")
    print(f"  {'train / test':<14} {stats['train_pairs']:>12,} / {stats['test_pairs']:,}")

    failed = [f for f in INTEGRITY_FIELDS if stats.get(f, 0)]
    print(f"  integrity      {'OK' if not failed else 'FAILED: ' + ', '.join(failed)}")

    if diagnostics:
        pu, pi = diagnostics["interactions_per_user"], diagnostics["interactions_per_item"]
        print(f"  per user       min {pu['min']}, median {pu['median']:g}, {pu['below_5']:,} below 5")
        print(f"  per item       min {pi['min']}, median {pi['median']:g}, {pi['below_5']:,} below 5")
        print(f"  cold           {diagnostics['trainless_test_users']:,} train-less test users "
              f"({100 * diagnostics['trainless_test_users_fraction']:.1f}%), "
              f"{diagnostics['cold_test_positives']:,} cold test positives "
              f"({100 * diagnostics['cold_test_positives_fraction']:.1f}%), "
              f"{diagnostics['cold_catalog_items']:,} cold catalog items")

    for p in problems or ():
        print(f"  ! {p}")