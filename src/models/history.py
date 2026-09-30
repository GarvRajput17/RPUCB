"""
Per-user history bags: construction and embedding.

Shared by `pinterest_dcm.py` and `mind_model.py`, which need exactly the
same thing -- a fixed, seeded bag of train items per user, embedded
through the item tower's summarization sub-layer, pre-crossing (Eq. 4's
e_i). It lived in `pinterest_dcm.py` before, which made MIND import from
DCM: the wrong direction, since MIND is the predecessor and the two are
meant to differ only in their routing rule.

Bags are unordered. Our datasets carry no usable timestamps except ml-1m
(§9), and routing operates on a set regardless.
"""

import torch


def build_history_buffers(num_users, user_train_items, max_hist_len, seed=0):
    """
    Fixed per-user history bags: ([num_users, L] long, [num_users, L] bool).

    Users with more than `max_hist_len` train items get a seeded uniform
    sample rather than an arbitrary prefix. The earlier version took
    `list(items)[:max_hist_len]` from a `set` of ints, and CPython iterates
    small-int sets in roughly value order, so truncation systematically
    kept the lowest item ids -- and in these datasets item ids correlate
    with insertion order and popularity, so every user's bag was skewed the
    same way.

    Sampling happens once at construction, so a user's bag is stable across
    epochs and, more importantly, identical between the positive and
    negative forward passes inside one loss term.
    """
    history_item_ids = torch.zeros(num_users, max_hist_len, dtype=torch.long)
    history_mask = torch.zeros(num_users, max_hist_len, dtype=torch.bool)

    if user_train_items is None:
        return history_item_ids, history_mask

    generator = torch.Generator().manual_seed(seed)

    for u, items in user_train_items.items():
        if u >= num_users:
            continue
        items = sorted(items)
        if not items:
            continue
        if len(items) > max_hist_len:
            perm = torch.randperm(len(items), generator=generator)[:max_hist_len]
            chosen = [items[i] for i in perm.tolist()]
        else:
            chosen = items
        n = len(chosen)
        history_item_ids[u, :n] = torch.tensor(chosen, dtype=torch.long)
        history_mask[u, :n] = True

    return history_item_ids, history_mask


def embed_history_bag(summarize, interaction_cols, hist_ids, embed_dim):
    """
    Embed a [B, L] bag of history item ids -> [B, L, d].

    Deduplicates the item ids before touching the tower. The naive version
    gathered [B*L, num_users] raw interaction columns and pushed all of
    them through `summarize`; autograd then retains that whole tensor for
    the first Linear's weight gradient, and training does two forwards per
    batch (positives, then tiled negatives).

    On ml-1m that was 1,024 x 50 = 51,200 rows against a catalog of only
    3,706 items -- the same columns fetched and embedded roughly fourteen
    times over, for 1.24 GB per forward instead of 90 MB. The dense
    datasets are exactly where batch sizes are largest, so this is where
    the ceiling bites; on the sparse ones users share few items and the
    saving is small, which is fine because their totals are small already.

    Exactly equivalent, with one caveat: `summarize` must be row-independent.
    It is -- Linear, GELU, Dropout, Linear -- except that with dropout > 0
    two occurrences of one item would previously draw independent masks and
    now share one. `dropout` is 0.0 throughout the matrix, so nothing
    changes today; revisit if that is ever turned on.

    Args:
        summarize: the item tower's MLP summarization sub-layer.
        interaction_cols: [num_items, num_users], the resident item matrix.
        hist_ids: [B, L] item ids, padding included.
        embed_dim: d.
    """
    B, L = hist_ids.shape

    unique_ids, inverse = torch.unique(hist_ids.reshape(-1), return_inverse=True)
    embedded = summarize(interaction_cols[unique_ids])          # [U, d]

    return embedded[inverse].view(B, L, embed_dim)