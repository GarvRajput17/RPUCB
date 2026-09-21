"""
Differentiable Clustering Module (DCM) -- Pinterest Sec 3.2.2, Fig 3.

Implements what DCM changes relative to MIND's capsule routing:
  - Validity-Aware Farthest-Point Initialization (VA-FPI), Eq. 5-6
  - Single-Assignment Routing (SAR), Eq. 7
  - the `squash` non-linearity, Eq. 3 (now shared with mind.py via
    routing_common, so the two backbones cannot drift on it)

Architecture-agnostic: consumes and produces plain tensors, so both
`pinterest_dcm.py` and its masked subclasses reuse it unchanged.

Three fixes in this version.

1. No -inf anywhere. The previous implementation masked invalid slots with
   `float('-inf')` and then softmaxed, which produces NaN for any row that
   is entirely masked. It guarded with `finite.any()` and `torch.where`,
   but `torch.where` propagates NaN gradients through the branch it does
   not select, so the guard protected the forward pass and not the
   backward one. Everything now uses the finite NEG_SCORE sentinel.

2. The per-centroid softmax is vectorised. The old version looped over K
   in Python on every routing iteration of every forward pass.

3. `slot_valid` is returned. VA-FPI always emits K centroids even when a
   user has fewer than K valid history items -- the farthest-point loop
   simply re-picks items it has already chosen, giving duplicate
   centroids that a later max-over-K reduces over as if they were
   distinct interests. After 5-core filtering and carving a validation
   item out of train, users with 3-6 train items exist in all five
   datasets, so this is not hypothetical. The caller now knows which
   slots are real.

Documented deviations from the paper, unchanged: cosine similarity in
place of raw dot products for numerical stability at our scale, and a
deterministic first centroid instead of Eq. 5's random draw. The latter is
a correctness requirement rather than a convenience -- positives and
negatives are scored in separate forward passes, and a fresh random draw
per call would give one user two incompatible sets of conditions inside a
single loss term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing_common import NEG_SCORE, slot_valid_from_counts, squash


class DCM(nn.Module):
    """
    Args:
        embed_dim: width of the item embeddings being routed.
        K: number of interest slots. Locked to 7 (Pinterest Table 6).
        routing_iters: SAR iterations. 3.

    Forward:
        item_embeds: [B, L, d] per-user history bag.
        valid_mask:  [B, L] bool, True where the slot holds a real item.

    Returns:
        centroids:  [B, K, d]
        weights:    [B, L, K] final routing weights
        slot_valid: [B, K] bool, False for slots beyond the user's number
                    of distinct history items
    """

    def __init__(self, embed_dim, K, routing_iters=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = K
        self.routing_iters = routing_iters

    # ------------------------------------------------------------------
    def _va_fpi_init(self, e, valid_mask):
        """Eq. 5-6, on already-normalised item embeddings `e`."""
        B, L, _ = e.shape
        device = e.device

        penalty = torch.where(
            valid_mask,
            torch.zeros((), device=device, dtype=e.dtype),
            torch.full((), NEG_SCORE, device=device, dtype=e.dtype),
        )                                                            # [B, L]

        # First centroid: first valid item. argmax breaks ties by first
        # occurrence and every valid slot is tied at 0.
        batch_idx = torch.arange(B, device=device)
        centroids = [e[batch_idx, penalty.argmax(dim=1)]]

        for _ in range(1, self.K):
            stacked = torch.stack(centroids, dim=1)                  # [B, j, d]
            sims = torch.einsum("bld,bjd->blj", e, stacked)          # [B, L, j]
            max_sim = sims.max(dim=2).values                         # [B, L]
            # Among valid items, the one least similar to every centroid
            # already chosen.
            next_idx = (-max_sim + penalty).argmax(dim=1)            # [B]
            centroids.append(e[batch_idx, next_idx])

        return torch.stack(centroids, dim=1)                         # [B, K, d]

    # ------------------------------------------------------------------
    def _single_assignment_routing(self, item_embeds, e, valid_mask, centroids):
        """Eq. 7 plus the centroid update of Eq. 2/3."""
        c = F.normalize(centroids, dim=-1)
        sims = torch.einsum("bld,bkd->blk", e, c)                    # [B, L, K]

        neg = torch.full_like(sims, NEG_SCORE)
        sims_valid = torch.where(valid_mask.unsqueeze(-1), sims, neg)

        # Hard-assign each item to its single nearest centroid.
        assign = sims_valid.argmax(dim=2)                            # [B, L]
        hard = F.one_hot(assign, num_classes=self.K).bool()          # [B, L, K]
        hard = hard & valid_mask.unsqueeze(-1)

        # Softmax over items within each centroid's assigned set. Done on a
        # finite sentinel so a centroid with no assigned items yields a
        # uniform row rather than NaN; the subsequent multiply by `hard`
        # zeroes it out anyway.
        logits = torch.where(hard, sims, neg)
        weights = F.softmax(logits, dim=1) * hard.to(sims.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        new_centroids = torch.einsum("blk,bld->bkd", weights, item_embeds)
        return squash(new_centroids, dim=-1), weights

    # ------------------------------------------------------------------
    def forward(self, item_embeds, valid_mask):
        e = F.normalize(item_embeds, dim=-1)

        n_valid = valid_mask.sum(dim=1).clamp(min=1)                 # [B]
        slot_valid = slot_valid_from_counts(n_valid, self.K)         # [B, K]

        centroids = self._va_fpi_init(e, valid_mask)
        weights = None
        for _ in range(self.routing_iters):
            centroids, weights = self._single_assignment_routing(
                item_embeds, e, valid_mask, centroids
            )

        return centroids, weights, slot_valid