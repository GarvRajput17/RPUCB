"""
Full-catalog evaluation: val mode (cheap, per-epoch, seeded subsample) and
test mode (expensive, all users, run exactly once on the restored-best
checkpoint) -- §5's eval-cost control.

Replaces the pre-refactor `evaluate_model`, which ranked the held-out item
against 99 sampled negatives. Two consequences of that protocol explain
why the numbers move once this lands (Phase 3's gate): with 100 total
candidates HR@100 was identically 1.0 for every model regardless of
quality, and Coverage@10 was the union of top-10 items drawn from those
100-item pools rather than from the real catalog.

Scoring goes through the encode/score_encoded contract (models/base.py),
not the pairwise `forward()`. The pairwise path consumes raw interaction
profiles -- `num_items + num_users` floats per (user, item) pair -- so
tiling it across a whole catalog materialises 9-39 GB per chunk depending
on dataset. Encoding each side once and combining at d=64 removes that,
and lets the item encoding be computed **once per evaluation** and reused
across every user batch, rather than recomputed per batch.

No per-model branching anywhere in this file: each family supplies its own
encoders and its own `score_encoded`, including the max-over-K reduction
(§5) where it applies.
"""

import torch

from .metrics import FullCatalogMetrics
from .models.base import DEFAULT_ITEM_ENCODE_CHUNK, DEFAULT_PAIR_CHUNK


@torch.no_grad()
def encode_catalog(model, dataset, device, item_cols=None,
                   item_chunk_size=DEFAULT_ITEM_ENCODE_CHUNK):
    """
    Item-side representation for the entire catalog.

    Must be recomputed every time the weights change, so this is called
    once per `evaluate_split`, not cached across epochs.
    """
    cols = item_cols if item_cols is not None else dataset.interaction_cols.to(device)
    item_ids = torch.arange(dataset.num_items, device=device)
    return model.encode_items(cols, item_ids, chunk_size=item_chunk_size)


def _apply_exclusions(scores, batch, dataset, split, device):
    """
    Mask each user's train items -- plus whichever of {val, test} is not
    the current target -- to -inf, in one scatter per batch rather than
    one small CUDA transfer per user.
    """
    rows, cols = [], []
    for row, (user, target) in enumerate(batch):
        excluded = dataset.excluded_items(user, split)
        if not excluded:
            continue
        assert target not in excluded, (
            f"held-out {split} item {target} for user {user} is in its own "
            f"exclusion set; the split carving in data/dataset.py is wrong"
        )
        rows.extend([row] * len(excluded))
        cols.extend(excluded)

    if rows:
        scores[
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(cols, dtype=torch.long, device=device),
        ] = float("-inf")


@torch.no_grad()
def evaluate_split(
    model, dataset, split, device, max_users=None, seed=None,
    item_vectors="auto", user_batch_size=64,
    item_chunk_size=DEFAULT_ITEM_ENCODE_CHUNK,
    pair_chunk=DEFAULT_PAIR_CHUNK,
    interaction_rows_gpu=None, interaction_cols_gpu=None,
):
    """
    Args:
        split: 'val' or 'test'.
        max_users: seeded random subsample of this size, for the cheap
            per-epoch val check (§5's fixed 1,000-user subsample). None
            means every user, used for the single test pass.
        item_vectors: 'auto' uses `dataset.interaction_cols` for ILD --
            metadata-free and identically defined across all five
            datasets, which is what makes ILD comparable between them
            (§9). None skips ILD.
        interaction_rows_gpu / interaction_cols_gpu: tensors the caller
            already has resident. `train.py` preloads both once per run,
            so val evaluation does not re-transfer the full interaction
            matrices on every epoch.
    """
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    model.eval()

    ratings = dataset.get_val_data() if split == "val" else dataset.get_test_data()
    if max_users is not None and len(ratings) > max_users:
        generator = torch.Generator().manual_seed(seed if seed is not None else 0)
        idx = torch.randperm(len(ratings), generator=generator)[:max_users].tolist()
        ratings = [ratings[i] for i in idx]

    rows = (
        interaction_rows_gpu if interaction_rows_gpu is not None
        else dataset.interaction_rows.to(device)
    )
    cols = (
        interaction_cols_gpu if interaction_cols_gpu is not None
        else dataset.interaction_cols.to(device)
    )

    # Once per evaluation, reused by every user batch below.
    item_enc = encode_catalog(model, dataset, device, item_cols=cols,
                              item_chunk_size=item_chunk_size)

    vectors = cols if item_vectors == "auto" else item_vectors
    acc = FullCatalogMetrics(dataset.num_items, item_vectors=vectors)

    for start in range(0, len(ratings), user_batch_size):
        batch = ratings[start:start + user_batch_size]
        user_ids = torch.tensor([u for u, _ in batch], dtype=torch.long, device=device)
        positives = torch.tensor([i for _, i in batch], dtype=torch.long, device=device)

        user_enc = model.encode_users(rows[user_ids], user_ids)
        scores = model.score_encoded(user_enc, item_enc, pair_chunk=pair_chunk)

        _apply_exclusions(scores, batch, dataset, split, device)
        acc.update(scores, positives)

    return acc.compute()