"""
Dataset loading, leave-one-out val/test splits, and negative sampling.

Three changes from the pre-refactor version.

1. A validation split carved from train (§5: "Leave-one-out carved from
   train; upstream test split untouched"). One further item per user is
   held out from train.rating as `val_item`, seeded and reproducible, and
   removed from the dense interaction matrices that back both training
   features and RP-UCB's N_u/N_i counts -- not just from the training
   pair list. Leaving it in those matrices would leak val-item membership
   straight into every model's input features: `interaction_rows` /
   `interaction_cols` literally encode "has this user interacted with
   this item", so failing to remove it would let a model detect its own
   held-out val item at both train and val-eval time, which defeats the
   point of using val HR@10 for early stopping. No timestamp information
   exists in these datasets (§9), so the val item is a seeded uniform
   random pick, not a "most recent" pick.

2. Worker seeding. `get_train_dataloader` previously ran with
   `num_workers=8` and no `worker_init_fn` or `generator=`. PyTorch
   reseeds `torch`'s RNG per worker automatically but not NumPy's or the
   stdlib `random` module's, and `RecTrainDataset.__getitem__` samples
   negatives via `np.random.randint`. Under the default fork start
   method every worker inherited identical NumPy state and therefore
   sampled *the same* negatives -- an 8x collapse in negative diversity,
   silent, for every run to date. Fixed via `reproducibility.seed_worker`
   and `reproducibility.make_generator`.

3. `test.negative` is no longer read. Full-catalog evaluation (§9,
   Krichene & Rendle) ranks the held-out item against the entire catalog
   rather than 99 sampled negatives, so the pre-sampled negative file is
   dead weight -- and was actively misleading to keep, since HR@100 is
   identically 1.0 under a 100-candidate pool.

Also removed: the `torch.manual_seed` / `np.random.seed` / `random.seed`
calls inside the old `RecDataset.__init__`. Seeding is now `main.py`'s
job, via `reproducibility.set_global_seed`, called once before any
dataset or model exists -- seeding as a side effect of object
construction made the effective seed depend on *when* a `RecDataset`
happened to be built relative to everything else in the process.
"""

import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from ..reproducibility import make_generator, seed_worker


class RecTrainDataset(Dataset):
    """
    Negative sampling happens here, per `__getitem__` call, inside
    DataLoader workers -- which is exactly why the worker-seeding fix in
    `get_train_dataloader` matters: this is the code that was silently
    duplicated across all 8 workers before that fix.
    """

    def __init__(self, train_pairs, user_train_items, num_items, num_negatives):
        self.train_pairs = train_pairs
        self.user_train_items = user_train_items
        self.num_items = num_items
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        u, i = self.train_pairs[idx]
        user_interacted = self.user_train_items[u]

        neg_items = []
        while len(neg_items) < self.num_negatives:
            candidates = np.random.randint(0, self.num_items, size=self.num_negatives * 2)
            for neg in candidates:
                neg = int(neg)
                if neg not in user_interacted:
                    neg_items.append(neg)
                    if len(neg_items) == self.num_negatives:
                        break

        return {"user": u, "pos_item": i, "neg_items": neg_items}


def _read_pairs(path):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def _carve_val_split(user_train_items, seed, min_train_remaining=1):
    """
    Seeded leave-one-out val carving, mutating `user_train_items` in
    place (removes the chosen item from each user's set).

    A user needs more than `min_train_remaining` items to donate one to
    val without training on an empty set. Users at or below that
    threshold keep everything in train and get no val item -- they are
    simply absent from val evaluation, and the skip count is returned so
    that is visible rather than silently absorbed.

    Returns: (val_item: {u: item}, n_users_skipped: int)
    """
    rng = random.Random(seed)
    val_item = {}
    n_skipped = 0

    for u in sorted(user_train_items):
        items = user_train_items[u]
        if len(items) <= min_train_remaining:
            n_skipped += 1
            continue
        chosen = rng.choice(sorted(items))
        val_item[u] = chosen
        items.discard(chosen)

    return val_item, n_skipped


class RecDataset:
    def __init__(self, data_path, num_negatives=4, seed=0, min_train_remaining=1):
        self.data_path = data_path
        self.num_negatives = num_negatives
        self.seed = seed

        train_pairs_raw = _read_pairs(os.path.join(data_path, "train.rating"))
        test_pairs = _read_pairs(os.path.join(data_path, "test.rating"))

        user_train_items = defaultdict(set)
        for u, i in train_pairs_raw:
            user_train_items[u].add(i)

        self.test_item = {u: i for u, i in test_pairs}

        self.val_item, n_skipped = _carve_val_split(
            user_train_items, seed=seed, min_train_remaining=min_train_remaining
        )
        if n_skipped:
            print(
                f"[RecDataset] {n_skipped} user(s) had too few train items to "
                f"donate a val item; they train on their full set and are "
                f"absent from val evaluation."
            )

        self.user_train_items = user_train_items
        self.train_pairs = [
            (u, i) for u, items in user_train_items.items() for i in items
        ]

        all_u = [u for u, _ in train_pairs_raw] + [u for u, _ in test_pairs]
        all_i = [i for _, i in train_pairs_raw] + [i for _, i in test_pairs]
        self.num_users = max(all_u) + 1 if all_u else 0
        self.num_items = max(all_i) + 1 if all_i else 0

        # Dense matrices come from the REDUCED train set only -- val_item and
        # test_item must not appear here. See module docstring, point 1.
        train_matrix = sp.lil_matrix((self.num_users, self.num_items), dtype=np.float32)
        for u, i in self.train_pairs:
            train_matrix[u, i] = 1.0
        train_csr = train_matrix.tocsr()

        counts = np.array(train_csr.sum(axis=1)).flatten()
        self.user_interaction_counts = torch.LongTensor(counts)
        item_counts = np.array(train_csr.sum(axis=0)).flatten()
        self.item_interaction_counts = torch.LongTensor(item_counts)

        row_sums = np.maximum(1.0, counts)
        col_sums = np.maximum(1.0, item_counts)

        row_normalized = train_csr.copy()
        for u in range(self.num_users):
            s, e = row_normalized.indptr[u], row_normalized.indptr[u + 1]
            row_normalized.data[s:e] /= row_sums[u]
        self.interaction_rows = torch.from_numpy(row_normalized.toarray()).float()

        col_normalized = train_matrix.tocsc()
        for i in range(self.num_items):
            s, e = col_normalized.indptr[i], col_normalized.indptr[i + 1]
            col_normalized.data[s:e] /= col_sums[i]
        self.interaction_cols = torch.from_numpy(col_normalized.transpose().toarray()).float()

        self.cold_report = self._report_cold_structure()

    # ------------------------------------------------------------------
    def _report_cold_structure(self):
        """
        Count and print the users and items that no model here can rank.

        The preprocessing gate reports this for the raw files; these are the
        post-carve numbers, which are strictly worse and seed-dependent,
        because removing one train item per user turns some singleton items
        cold. Both belong on the record.

        A train-less user is the sharper case. Their feature row is all
        zeros and their history bag is empty, and for the MIND family that
        makes every interest vector exactly zero, so every item in the
        catalog scores exactly 0.0. With an optimistic tie rule that is a
        free rank-1 hit -- 43 of them on AMusic, which was 65% of MIND's
        measured HR@10 in the first-light run. The mid-rank default in
        metrics.py is what neutralises it.
        """
        user_counts = self.user_interaction_counts
        item_counts = self.item_interaction_counts

        trainless_users = int((user_counts == 0).sum())
        cold_items = int((item_counts == 0).sum())
        cold_test = sum(1 for i in self.test_item.values() if item_counts[i] == 0)
        cold_val = sum(1 for i in self.val_item.values() if item_counts[i] == 0)

        report = {
            "trainless_users": trainless_users,
            "cold_catalog_items": cold_items,
            "cold_test_positives": cold_test,
            "cold_val_positives": cold_val,
            "test_users": len(self.test_item),
            "val_users": len(self.val_item),
        }

        if trainless_users or cold_test or cold_val:
            print(
                f"[RecDataset] after val carving (seed {self.seed}): "
                f"{trainless_users:,} train-less users, {cold_items:,} cold catalog items, "
                f"{cold_test:,}/{len(self.test_item):,} cold test positives "
                f"({100 * cold_test / max(len(self.test_item), 1):.1f}%), "
                f"{cold_val:,}/{max(len(self.val_item), 1):,} cold val positives "
                f"({100 * cold_val / max(len(self.val_item), 1):.1f}%). "
                f"These are unrankable by any model here; metrics are also "
                f"reported over the warm subset."
            )

        return report

    # ------------------------------------------------------------------
    def get_train_dataloader(self, batch_size, seed=None, shuffle=True,
                              num_workers=8, pin_memory=None):
        """
        `num_workers` and `pin_memory` come from config rather than being
        hardcoded: 8 workers is wrong on a laptop or inside WSL, where the
        shared-memory limit shows up as workers dying mid-epoch, and
        pinning warns and does nothing without an accelerator. `pin_memory
        =None` means "on iff CUDA/HIP is available".
        """
        dataset = RecTrainDataset(
            self.train_pairs, self.user_train_items, self.num_items, self.num_negatives
        )
        seed = seed if seed is not None else self.seed
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=make_generator(seed),
        )

    def get_val_data(self):
        return list(self.val_item.items())

    def get_test_data(self):
        return list(self.test_item.items())

    def excluded_items(self, u, split):
        """
        Items to mask to -inf when ranking `split`'s target for user u:
        this user's (reduced) train items, plus whichever of
        {val_item, test_item} is NOT the current target -- so val-time
        ranking doesn't see the test item as a spurious candidate, and
        vice versa at test time.
        """
        excl = set(self.user_train_items.get(u, ()))
        if split == "val":
            if u in self.test_item:
                excl.add(self.test_item[u])
        elif split == "test":
            if u in self.val_item:
                excl.add(self.val_item[u])
        else:
            raise ValueError(f"split must be 'val' or 'test', got {split!r}")
        return excl