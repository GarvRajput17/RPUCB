"""
Full-catalog evaluation: val mode (cheap, per-epoch, optionally
subsampled) and test mode (all users, once, on the restored-best
checkpoint) -- §5's eval-cost control.

Scoring goes through the encode/score_encoded contract (models/base.py),
not the pairwise `forward()`. The pairwise path consumes raw interaction
profiles -- `num_items + num_users` floats per (user, item) pair -- so
tiling it across a catalog materialises tens of gigabytes per chunk.
Encoding each side once and combining at d=64 removes that, and lets the
item encoding be computed once per evaluation and reused across every user
batch.

Two things this file now supplies that `metrics.py` cannot derive on its
own:

  warm_mask   which users' results reflect model quality at all. A
              positive with no training interactions is one of a large
              group of items that every model maps to the same embedding,
              and a user with no training history has an empty feature row.
              Neither is rankable by any model here, so metrics are
              reported both over all users and over this subset.

  popularity  a reference ranking by train interaction count. It needs no
              model, costs one broadcast, and answers the question every
              result has to survive: is this model beating "recommend the
              most popular items"? First light had DeepCF surfacing only
              780 distinct items on AMusic, which is what popularity looks
              like.

No per-model branching anywhere here: each family supplies its own
encoders and its own `score_encoded`, including the max-over-K reduction.
"""

import torch

from .metrics import DEFAULT_TIE_ATOL, DEFAULT_TIE_POLICY, FullCatalogMetrics
from .models.base import DEFAULT_ITEM_ENCODE_CHUNK, DEFAULT_PAIR_CHUNK


@torch.no_grad()
def encode_catalog(model, dataset, device, item_cols=None,
                   item_chunk_size=DEFAULT_ITEM_ENCODE_CHUNK):
    """
    Item-side representation for the whole catalog. Recomputed on every
    call, since the weights change every epoch.
    """
    cols = item_cols if item_cols is not None else dataset.interaction_cols.to(device)
    item_ids = torch.arange(dataset.num_items, device=device)
    return model.encode_items(cols, item_ids, chunk_size=item_chunk_size)


def _select_users(dataset, split, max_users, seed):
    ratings = dataset.get_val_data() if split == "val" else dataset.get_test_data()
    if max_users is not None and len(ratings) > max_users:
        generator = torch.Generator().manual_seed(seed if seed is not None else 0)
        idx = torch.randperm(len(ratings), generator=generator)[:max_users].tolist()
        ratings = [ratings[i] for i in idx]
    return ratings


def _apply_exclusions(scores, batch, dataset, split, device):
    """
    Mask each user's train items -- plus whichever of {val, test} is not the
    current target -- to -inf, in one scatter per batch rather than a small
    CUDA transfer per user.
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


def _warm_mask(dataset, user_ids, positives, device):
    """
    True where the positive has at least one train interaction and the user
    has some train history. Both tensors are already built from the reduced
    (post-val-carve) train split, so this tracks the data the model
    actually saw.
    """
    item_counts = dataset.item_interaction_counts.to(device)
    user_counts = dataset.user_interaction_counts.to(device)
    return (item_counts[positives] > 0) & (user_counts[user_ids] > 0)


@torch.no_grad()
def _run_eval(score_batch, dataset, split, device, ratings, item_vectors,
              user_batch_size, tie_policy, tie_atol):
    """Shared loop. `score_batch(user_ids) -> [B, num_items]`."""
    acc = FullCatalogMetrics(
        dataset.num_items, item_vectors=item_vectors,
        tie_policy=tie_policy, tie_atol=tie_atol,
    )

    for start in range(0, len(ratings), user_batch_size):
        batch = ratings[start:start + user_batch_size]
        user_ids = torch.tensor([u for u, _ in batch], dtype=torch.long, device=device)
        positives = torch.tensor([i for _, i in batch], dtype=torch.long, device=device)

        scores = score_batch(user_ids)
        _apply_exclusions(scores, batch, dataset, split, device)
        acc.update(scores, positives, warm_mask=_warm_mask(dataset, user_ids, positives, device))

    return acc.compute()


@torch.no_grad()
def evaluate_split(
    model, dataset, split, device, max_users=None, seed=None,
    item_vectors="auto", user_batch_size=64,
    item_chunk_size=DEFAULT_ITEM_ENCODE_CHUNK,
    pair_chunk=DEFAULT_PAIR_CHUNK,
    tie_policy=DEFAULT_TIE_POLICY, tie_atol=DEFAULT_TIE_ATOL,
    interaction_rows_gpu=None, interaction_cols_gpu=None,
):
    """
    Args:
        split: 'val' or 'test'.
        max_users: seeded random subsample of this size for the cheap
            per-epoch val check. None means every user -- used for the
            single test pass, and worth using for val on the smaller
            datasets, where HR@10 over 1,000 users moves in steps of one
            hit and early stopping ends up chasing noise.
        item_vectors: 'auto' uses the co-interaction columns for ILD.
        interaction_rows_gpu / interaction_cols_gpu: tensors the caller
            already has resident, so val evaluation does not re-transfer
            the interaction matrices every epoch.
    """
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    model.eval()
    ratings = _select_users(dataset, split, max_users, seed)

    rows = (interaction_rows_gpu if interaction_rows_gpu is not None
            else dataset.interaction_rows.to(device))
    cols = (interaction_cols_gpu if interaction_cols_gpu is not None
            else dataset.interaction_cols.to(device))

    # Once per evaluation, reused by every user batch below.
    item_enc = encode_catalog(model, dataset, device, item_cols=cols,
                              item_chunk_size=item_chunk_size)

    def score_batch(user_ids):
        user_enc = model.encode_users(rows[user_ids], user_ids)
        return model.score_encoded(user_enc, item_enc, pair_chunk=pair_chunk)

    return _run_eval(
        score_batch, dataset, split, device, ratings,
        cols if item_vectors == "auto" else item_vectors,
        user_batch_size, tie_policy, tie_atol,
    )


@torch.no_grad()
def evaluate_popularity(
    dataset, split, device, max_users=None, seed=None,
    item_vectors="auto", user_batch_size=64,
    tie_policy=DEFAULT_TIE_POLICY, tie_atol=DEFAULT_TIE_ATOL,
    interaction_cols_gpu=None,
):
    """
    Non-personalised reference: every user is scored by train item
    popularity. Model-free, so it costs one broadcast per batch.

    Cold items have a count of zero, so they land at the bottom and tie
    with each other -- the same treatment the tie policy gives them for
    every real model, which is what keeps the comparison fair.
    """
    cols = (interaction_cols_gpu if interaction_cols_gpu is not None
            else dataset.interaction_cols.to(device))
    ratings = _select_users(dataset, split, max_users, seed)
    popularity = dataset.item_interaction_counts.to(device).float()

    def score_batch(user_ids):
        return popularity.unsqueeze(0).expand(user_ids.size(0), -1).clone()

    return _run_eval(
        score_batch, dataset, split, device, ratings,
        cols if item_vectors == "auto" else item_vectors,
        user_batch_size, tie_policy, tie_atol,
    )