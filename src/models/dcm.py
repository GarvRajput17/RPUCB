"""
Differentiable Clustering Module (DCM) -- Pinterest paper Sec 3.2.2, Fig 3.

Implements the two modifications DCM makes over vanilla Capsule-Network
routing (MIND), which the paper identifies as essential to avoid centroid
collapse in production:

  - Validity-Aware Farthest-Point Initialization (VA-FPI), Eq. 5-6
  - Single-Assignment Routing (SAR), Eq. 7
  - the `squash` non-linearity, Eq. 3

Operates on a batch of per-user item-embedding *bags* (Fork 5's locked
decision: unordered, since our datasets carry no timestamps -- sequence
order only matters for Pinterest at realtime-recency scale, not for the
routing math itself, which treats the history as a set either way).

This file is architecture-agnostic: it doesn't know about interaction
matrices, item towers, or user IDs. It only consumes and produces plain
tensors, so it can be reused unchanged by pinterest_dcm.py and
pinterest_dcm_rpucb.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(v, dim=-1, eps=1e-8):
    """
    Pinterest Eq. 3:
        squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)

    Non-linearity that lets the output vector's L2 norm represent the
    "existence" of a cluster centroid.
    """
    norm_sq = (v ** 2).sum(dim=dim, keepdim=True)
    scale = norm_sq / (1.0 + norm_sq)
    return scale * v / torch.sqrt(norm_sq + eps)


class DCM(nn.Module):
    """
    Args:
        embed_dim     : dimensionality of the item embeddings being routed
        K             : number of interest clusters (condition slots) --
                        the gameplan locks this to K=7, matching Pinterest's
                        own reported optimum (their Table 6 ablation)
        routing_iters : number of SAR routing iterations (paper doesn't
                        state an exact count; we use 3, cheap given K=7 and
                        small history lengths)

    Forward:
        item_embeds : [B, L, d] -- e_i for each item in a user's history
                      bag (L = padded history length)
        valid_mask  : [B, L] bool -- True where a slot holds a real
                      interacted item; corresponds to Eq. 6's I_valid,
                      masking out padding slots the same way Pinterest
                      masks out items with missing/invalid features

    Returns:
        centroids : [B, K, d] -- final cluster centroids (the raw,
                    pre-feature-crossing "implicit interest conditions")
        weights   : [B, L, K] -- final routing weights (kept around for a
                    possible later use as retrieval-budget signal, Sec 3.5
                    -- Sum_i b_ij -- not consumed anywhere in Stage A)

    Note on similarity: the paper's Eq. 1/5/7 use raw dot products
    (c_j^T e_i). We L2-normalize both items and centroids before taking
    dot products (i.e. use cosine similarity) for numerical stability at
    our scale -- documented deviation, not a silent one.
    """

    def __init__(self, embed_dim, K, routing_iters=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = K
        self.routing_iters = routing_iters

    # ------------------------------------------------------------------
    def _va_fpi_init(self, item_embeds, valid_mask):
        """
        Eq. 5-6: validity-aware farthest-point initialization.

        Deviation from the paper: Eq. 5 says to "randomly select an item
        as the first cluster centroid." We instead pick a *deterministic*
        first centroid (the first valid item in the padded history) for a
        reason that isn't just simplicity: our training loop scores a
        user's positive and negative items via two separate forward
        passes within the same BPR comparison (train.py). A fresh random
        draw on every forward call would give the same user two
        different, incompatible sets of K condition embeddings for what
        is supposed to be one apples-to-apples comparison -- that would
        silently break BPR's pos-vs-neg semantics. Determinism here is a
        correctness requirement, not a convenience shortcut.
        """
        B, L, d = item_embeds.shape
        device = item_embeds.device

        e = F.normalize(item_embeds, dim=-1)

        neg_inf = torch.finfo(e.dtype).min
        invalid_penalty = torch.where(
            valid_mask, torch.zeros((), device=device, dtype=e.dtype),
            torch.full((), neg_inf, device=device, dtype=e.dtype),
        )

        # First centroid: first valid item per user (deterministic --
        # torch.argmax breaks ties by returning the first occurrence, and
        # every valid slot is tied at score 0 here).
        first_idx = invalid_penalty.argmax(dim=1)                      # [B]
        batch_idx = torch.arange(B, device=device)
        centroids = [e[batch_idx, first_idx]]                          # each [B, d]

        for _ in range(1, self.K):
            stacked = torch.stack(centroids, dim=1)                    # [B, j, d]
            sims = torch.einsum('bld,bjd->blj', e, stacked)            # [B, L, j]
            max_sim = sims.max(dim=2).values                           # [B, L]

            # Eq. 6: arg max_i  min_j  -I_valid(e_i) * c_j^T e_i
            # i.e. among valid items, pick the one farthest (least similar)
            # from every already-chosen centroid.
            score = -max_sim + invalid_penalty
            next_idx = score.argmax(dim=1)                             # [B]
            centroids.append(e[batch_idx, next_idx])

        return torch.stack(centroids, dim=1)                           # [B, K, d]

    # ------------------------------------------------------------------
    def _single_assignment_routing(self, item_embeds, valid_mask, centroids):
        """Eq. 7: single-assignment routing + centroid update (Eq. 2/3)."""
        B, L, d = item_embeds.shape
        e = F.normalize(item_embeds, dim=-1)
        c = F.normalize(centroids, dim=-1)

        sims = torch.einsum('bld,bkd->blk', e, c)                      # [B, L, K]
        sims = sims.masked_fill(~valid_mask.unsqueeze(-1), float('-inf'))

        # Hard-assign each item to its single closest centroid (masks out
        # non-maximum entries in the routing weights, per Eq. 7).
        max_k = sims.argmax(dim=2)                                     # [B, L]
        hard_mask = F.one_hot(max_k, num_classes=self.K).bool()        # [B, L, K]
        hard_mask = hard_mask & valid_mask.unsqueeze(-1)

        masked_sims = sims.masked_fill(~hard_mask, float('-inf'))

        # Softmax-normalize routing weight within each centroid's assigned
        # items only.
        weights = torch.zeros_like(sims)
        for k in range(self.K):
            col = masked_sims[:, :, k]                                 # [B, L]
            finite = torch.isfinite(col)
            col_soft = torch.zeros_like(col)
            if finite.any():
                col_masked = col.masked_fill(~finite, float('-inf'))
                soft = F.softmax(col_masked, dim=1)
                col_soft = torch.where(finite, soft, torch.zeros_like(soft))
            weights[:, :, k] = col_soft

        new_centroids = torch.einsum('blk,bld->bkd', weights, item_embeds)  # [B, K, d]
        new_centroids = squash(new_centroids, dim=-1)
        return new_centroids, weights

    # ------------------------------------------------------------------
    def forward(self, item_embeds, valid_mask):
        centroids = self._va_fpi_init(item_embeds, valid_mask)
        weights = None
        for _ in range(self.routing_iters):
            centroids, weights = self._single_assignment_routing(
                item_embeds, valid_mask, centroids
            )
        return centroids, weights