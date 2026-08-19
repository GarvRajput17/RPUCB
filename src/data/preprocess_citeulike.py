#!/usr/bin/env python3
"""
Preprocesses the raw citeulike-a dataset (users.dat, from
https://github.com/js05212/citeulike-a) into the train.rating /
test.rating / test.negative format RecDataset expects.

Why this one needs real preprocessing (unlike ml-1m / AMusic): citeulike-a
is only distributed as a raw user-item rating matrix (`users.dat`: one line
per user, first number = interaction count, rest = item IDs). There's no
pre-split leave-one-out version circulating the way there is for ml-1m and
AMusic (via familyld/DeepCF), and -- importantly -- there's no timestamp
field in this file at all, unlike ml-1m's ratings which carry one.

Assumption this script makes, stated explicitly because it can't be
verified against the original report's exact script: with no timestamp to
define "latest interaction," the held-out test item per user is chosen
uniformly at random (seeded, so it's reproducible) rather than by any
notion of recency. This is a deviation from the strict leave-one-out-by-
time protocol used for ml-1m; it's the standard fallback when a dataset
carries no temporal signal, and is unlikely to change qualitative results,
but it's worth knowing about the results rather than assuming they were
produced identically to a ml-1m-style temporal split.

Verified against the report's own Table I before running this: the raw
users.dat already matches the target scale exactly (5,551 users, 16,980
items, 204,986 interactions) -- no additional 5-core filtering is needed,
the upstream citeulike-a release already applied it (min 10 interactions/
user, confirmed by inspection).
"""

import os
import random

SEED = 42
NUM_NEGATIVES = 99
INPUT_FILE = 'raw/citeulike-a/users.dat'
OUTPUT_DIR = 'data/citeulike'

random.seed(SEED)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE) as f:
        lines = f.read().strip().split('\n')

    num_users = len(lines)
    print(f"Read {num_users} users from {INPUT_FILE}")

    user_items = []  # user_items[u] = list of item ids
    all_items = set()
    for line in lines:
        parts = line.split()
        count = int(parts[0])
        items = [int(x) for x in parts[1:]]
        assert len(items) == count, "declared count doesn't match item list length"
        user_items.append(items)
        all_items.update(items)

    num_items = max(all_items) + 1
    total_interactions = sum(len(items) for items in user_items)
    print(f"num_items (max id + 1): {num_items}")
    print(f"total interactions: {total_interactions}")

    # ── Leave-one-out split (random held-out item per user; see module
    #    docstring for why this differs from a timestamp-based split) ──────
    train_pairs = []
    test_pairs = []
    train_item_sets = []  # per-user set of train items, for negative sampling

    for u, items in enumerate(user_items):
        items = list(items)
        random.shuffle(items)
        held_out = items[0]
        remaining = items[1:]

        test_pairs.append((u, held_out))
        for i in remaining:
            train_pairs.append((u, i))
        train_item_sets.append(set(items))  # full set (train+test) for negative sampling

    print(f"train pairs: {len(train_pairs)}")
    print(f"test pairs: {len(test_pairs)}")

    # ── Write train.rating ──────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, 'train.rating'), 'w') as f:
        for u, i in train_pairs:
            f.write(f"{u}\t{i}\n")

    # ── Write test.rating ───────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, 'test.rating'), 'w') as f:
        for u, i in test_pairs:
            f.write(f"{u}\t{i}\n")

    # ── Write test.negative (99 negatives per test user) ───────────────
    print(f"Sampling {NUM_NEGATIVES} negatives per test user...")
    with open(os.path.join(OUTPUT_DIR, 'test.negative'), 'w') as f:
        for u, pos_i in test_pairs:
            interacted = train_item_sets[u]
            negs = []
            attempts = 0
            while len(negs) < NUM_NEGATIVES and attempts < NUM_NEGATIVES * 200:
                j = random.randint(0, num_items - 1)
                attempts += 1
                if j not in interacted and j not in negs:
                    negs.append(j)
            if len(negs) < NUM_NEGATIVES:
                print(f"  WARNING: user {u} only got {len(negs)} negatives "
                      f"(needed {NUM_NEGATIVES}) -- item pool may be too small "
                      f"relative to this user's interaction count")
            neg_str = '\t'.join(str(x) for x in negs)
            f.write(f"({u},{pos_i})\t{neg_str}\n")

    print(f"\nDone. Wrote to {OUTPUT_DIR}/:")
    print(f"  train.rating:  {len(train_pairs)} interactions")
    print(f"  test.rating:   {len(test_pairs)} held-out test users")
    print(f"  test.negative: {NUM_NEGATIVES} negatives per test user")


if __name__ == '__main__':
    main()