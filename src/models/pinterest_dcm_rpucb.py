"""
Masked DCM variants -- models 8, 9 and 10, plus the symmetric shared-mask
run §2 wants available without adding a row to the main matrix.

One class, three configurations, following §2's "config flag, not another
model" principle:

    key               K   head_mode   mask_granularity   user mask   capacity
    dcm_rpucb_multi   7   per_slot    per_slot           7 x d       448
    dcm_rpucb_kd      7   concat      -                  K*d         448
    dcm_rpucb_d       1   per_slot    shared             d           64
    (symmetric run)   7   per_slot    shared             d           448

`head_mode` is the architectural axis §2's `7 -> 8` row isolates:

  per_slot  the K routed conditions stay separate, each gated by its own
            mask, each scored against the item, max over K. K masked heads.

  concat    the K conditions are concatenated into one K*d vector, gated
            by a single K*d-dim mask, then projected back to d by a
            learned linear before scoring. One wide masked embedding.

Both hold total user-side capacity at K*d = 448 and both leave the item
tower at d, so `dcm -> dcm_rpucb_multi` isolates the mask and
`dcm_rpucb_multi -> dcm_rpucb_kd` isolates the head architecture with mask
and capacity fixed.

The 448 -> 64 projection is how the wide user embedding meets the item
tower, which stays at d=64 for every model in this family. Widening the
item tower instead would have made models 7 and 9 differ on item-side
capacity too, which is exactly the confound the row exists to avoid.

Item-side masking is present in all three, per the locked decision: one
shared d-dim mask over the item tower output. The item side is
single-embedding in every variant, so there is no per-slot item mask to
choose. N_i comes from train-split item interaction counts.

Interpretation caveat carried forward: with `mask_init=0.5`, a user and an
item both at or above their respective mean counts contribute
representations at 0.5x each, so the scored interaction starts at 0.25x
the unmasked control. This applies to every masked model and is the
leading candidate explanation if the masked rows underperform uniformly
rather than selectively by sparsity.
"""

import torch
import torch.nn as nn

from .base import DEFAULT_ITEM_ENCODE_CHUNK, chunk_bounds
from .pinterest_dcm import PinterestDCM
from .rpucb_mask import RPUCBMask


class PinterestDCMRPUCB(PinterestDCM):
    """
    Args beyond PinterestDCM's:
        head_mode: 'per_slot' or 'concat'.
        mask_granularity: 'per_slot' or 'shared'. Ignored when
            head_mode='concat', which has a single wide mask by
            construction.
        user_interaction_counts / item_interaction_counts: train-split
            counts, LongTensor.
        gamma_init, beta, mask_init: passed to RPUCBMask.
    """

    def __init__(
        self,
        num_users,
        num_items,
        embed_dim=64,
        K=7,
        head_mode="per_slot",
        mask_granularity="per_slot",
        user_train_items=None,
        interaction_cols=None,
        max_hist_len=50,
        summarization_hidden=256,
        n_fields=4,
        n_heads_dhen=2,
        transformer_layers=2,
        attn_heads=2,
        routing_iters=3,
        dropout=0.0,
        history_seed=0,
        user_interaction_counts=None,
        item_interaction_counts=None,
        gamma_init=2.0,
        beta=1.0,
        mask_init=0.5,
    ):
        if head_mode not in ("per_slot", "concat"):
            raise ValueError(f"head_mode must be 'per_slot' or 'concat', got {head_mode!r}")
        if mask_granularity not in ("per_slot", "shared"):
            raise ValueError(
                f"mask_granularity must be 'per_slot' or 'shared', got {mask_granularity!r}"
            )

        super().__init__(
            num_users, num_items, embed_dim=embed_dim, K=K,
            user_train_items=user_train_items, interaction_cols=interaction_cols,
            max_hist_len=max_hist_len, summarization_hidden=summarization_hidden,
            n_fields=n_fields, n_heads_dhen=n_heads_dhen,
            transformer_layers=transformer_layers, attn_heads=attn_heads,
            routing_iters=routing_iters, dropout=dropout,
            history_seed=history_seed,
            _defer_init=True,
        )

        self.head_mode = head_mode
        self.mask_granularity = mask_granularity

        if head_mode == "concat":
            user_mask_dim = K * embed_dim
            user_mask_slots = 1
            # The wide masked user vector meets the d-dim item tower here.
            self.user_proj = nn.Linear(K * embed_dim, embed_dim)
        else:
            user_mask_dim = embed_dim
            user_mask_slots = K if mask_granularity == "per_slot" else 1
            self.user_proj = None

        self.user_mask = RPUCBMask(
            num_users, user_mask_dim, counts=user_interaction_counts,
            n_slots=user_mask_slots, gamma_init=gamma_init, beta=beta,
            mask_init=mask_init,
        )
        self.item_mask = RPUCBMask(
            num_items, embed_dim, counts=item_interaction_counts,
            n_slots=1, gamma_init=gamma_init, beta=beta, mask_init=mask_init,
        )

        self.init_weights()

    # ------------------------------------------------------------------
    def get_condition_masks(self, user_ids):
        """
        User-side masks: [B, K, d] for per_slot granularity, [B, 1, d] for
        shared (broadcasts against the conditions), [B, 1, K*d] for concat.

        N_u is the user's overall train interaction count and is shared
        across slots -- Pinterest defines no per-condition interaction
        count to base an exploration bonus on, so exploration *strength*
        reflects total history while only *where* each slot stays open is
        learned independently.
        """
        return self.user_mask(user_ids)

    def _masked_item_embedding(self, item_col, item_ids):
        q_i = self.item_tower(item_col)
        return q_i * self.item_mask.flat(item_ids)

    # ------------------------------------------------------------------
    def score_multi(self, user_row, item_col, user_ids=None, item_ids=None):
        assert user_ids is not None, "PinterestDCMRPUCB requires user_ids"
        assert item_ids is not None, (
            "PinterestDCMRPUCB masks the item side and requires item_ids"
        )

        conditions, slot_valid = self.get_user_conditions(user_ids)   # [B,K,d]
        q_i = self._masked_item_embedding(item_col, item_ids)         # [B,d]

        if self.head_mode == "concat":
            B = conditions.size(0)
            wide = conditions.reshape(B, self.K * self.embed_dim)     # [B, K*d]
            wide = wide * self.user_mask.flat(user_ids)               # [B, K*d]
            u = self.user_proj(wide).unsqueeze(1)                     # [B, 1, d]
            # One head, always valid: the concatenation already folded any
            # duplicate-centroid padding into the wide vector.
            valid = torch.ones(
                B, 1, dtype=torch.bool, device=conditions.device
            )
            return self._score_conditions(u, q_i, valid), u

        masked = conditions * self.get_condition_masks(user_ids)      # broadcast if shared
        return self._score_conditions(masked, q_i, slot_valid), masked

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        scores, _ = self.score_multi(user_row, item_col, user_ids, item_ids)
        return scores.max(dim=1).values

    def score_with_mask(self, user_row, item_col, user_ids=None, item_ids=None):
        scores = self.score(user_row, item_col, user_ids, item_ids)
        masks = (self.get_condition_masks(user_ids), self.item_mask.flat(item_ids))
        return scores, masks

    # ---- full-catalog scoring ----------------------------------------
    # score_encoded is inherited from PinterestDCM unchanged: concat mode
    # simply produces K=1 conditions with an all-true validity mask.
    def encode_items(self, item_cols, item_ids=None, chunk_size=DEFAULT_ITEM_ENCODE_CHUNK):
        assert item_ids is not None, (
            "PinterestDCMRPUCB masks the item side and requires item_ids"
        )
        out = []
        for start, end in chunk_bounds(item_cols.size(0), chunk_size):
            embedded = self.item_tower(item_cols[start:end])
            out.append(embedded * self.item_mask.flat(item_ids[start:end]))
        return torch.cat(out, dim=0)

    def encode_users(self, user_row, user_ids=None):
        assert user_ids is not None, "PinterestDCMRPUCB requires user_ids"
        conditions, slot_valid = self.get_user_conditions(user_ids)

        if self.head_mode == "concat":
            B = conditions.size(0)
            wide = conditions.reshape(B, self.K * self.embed_dim)
            wide = wide * self.user_mask.flat(user_ids)
            projected = self.user_proj(wide).unsqueeze(1)                 # [B,1,d]
            valid = torch.ones(B, 1, dtype=torch.bool, device=conditions.device)
            return {"conditions": projected, "slot_valid": valid}

        masked = conditions * self.get_condition_masks(user_ids)
        return {"conditions": masked, "slot_valid": slot_valid}