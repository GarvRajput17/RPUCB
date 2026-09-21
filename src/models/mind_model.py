"""
MIND models -- matrix rows 4, 5 and 6. New file.

    key                K            mask                    capacity
    mind               K_u adaptive  -                       <= 448
    mind_rpucb_multi   K_u adaptive  d-dim, shared + item    <= 448
    mind_rpucb         1             d-dim + item            64

`mind -> mind_rpucb_multi` is the pure mask test for this backbone:
routing, interest count and capacity all held fixed, one mask added.
`mind_rpucb_multi -> mind_rpucb` collapses to a single interest with the
mask present throughout.

Mask granularity is shared across slots rather than per-slot, and this is
forced rather than chosen: K_u varies per user, so there is no fixed
(user, k) table to index the way `dcm_rpucb_multi` uses one. §9's
deviations register already carries this asymmetry.

Scoring is MIND-native: inner product between the interest vector and the
item embedding, not the attention-and-fusion scorer the Pinterest models
use. That keeps the family faithful to its own paper and to the sampled
softmax loss §3 assigns it, at the cost of `mind` vs `dcm` differing in
scorer as well as routing. Cross-backbone rows are already flagged as
loss-confounded in §3, and the within-family rows -- which are the ones
carrying the wrapper claim -- are unaffected.

The tower is `PinterestTower`, shared with the DCM family. Reusing it
means the MIND and DCM item sides are identical, so the cross-backbone
difference stays concentrated in routing and scoring rather than leaking
into how items are embedded.

At K=1 the interest machinery is inert -- routing reduces to weighted
pooling plus squash, and label-aware attention has a single slot to attend
over. §2's design note applies: describe `mind_rpucb` as a backbone tower
with one adaptively-masked embedding, not as multi-interest.
"""

import torch
import torch.nn as nn

from .base import DEFAULT_ITEM_ENCODE_CHUNK, DEFAULT_PAIR_CHUNK, BaseCF, chunk_bounds
from .mind import LabelAwareAttention, MINDRouting
from .pinterest_dcm import build_history_buffers
from .pinterest_tower import PinterestTower
from .routing_common import NEG_SCORE
from .rpucb_mask import RPUCBMask


class MIND(BaseCF):
    """Model 4 -- the unmasked MIND control."""

    def __init__(
        self,
        num_users,
        num_items,
        embed_dim=64,
        k_max=7,
        adaptive_k=True,
        user_train_items=None,
        interaction_cols=None,
        max_hist_len=50,
        summarization_hidden=256,
        n_fields=4,
        n_heads_dhen=2,
        transformer_layers=2,
        routing_iters=3,
        dropout=0.0,
        label_aware_power=2.0,
        history_seed=0,
        logit_seed=0,
        _defer_init=False,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.k_max = k_max
        self.max_hist_len = max_hist_len

        history_item_ids, history_mask = build_history_buffers(
            num_users, user_train_items, max_hist_len, seed=history_seed
        )
        self.register_buffer("history_item_ids", history_item_ids, persistent=False)
        self.register_buffer("history_mask", history_mask, persistent=False)

        if interaction_cols is None:
            interaction_cols = torch.zeros(num_items, num_users)
        self.register_buffer("interaction_cols", interaction_cols, persistent=False)

        self.item_tower = PinterestTower(
            input_dim=num_users, embed_dim=embed_dim,
            summarization_hidden=summarization_hidden, n_fields=n_fields,
            n_heads=n_heads_dhen, transformer_layers=transformer_layers,
            dropout=dropout,
        )

        self.routing = MINDRouting(
            embed_dim, k_max=k_max, routing_iters=routing_iters,
            adaptive_k=adaptive_k, max_hist_len=max_hist_len,
            logit_seed=logit_seed,
        )
        self.label_attention = LabelAwareAttention(power=label_aware_power)

        if not _defer_init:
            self.init_weights()

    # ------------------------------------------------------------------
    def _history_item_embeds(self, user_ids):
        hist_ids = self.history_item_ids[user_ids]
        mask = self.history_mask[user_ids]
        B, L = hist_ids.shape
        hist_cols = self.interaction_cols[hist_ids.reshape(-1)]
        e = self.item_tower.summarize(hist_cols).view(B, L, self.embed_dim)
        return e, mask

    def get_user_interests(self, user_ids):
        """
        ([B, K, d], [B, K]). Deduplicates before routing, for the same
        reason PinterestDCM does: one user recurs across every candidate
        item at evaluation and every negative during training.
        """
        unique_ids, inverse = torch.unique(user_ids, return_inverse=True)
        item_embeds, mask = self._history_item_embeds(unique_ids)
        interests, slot_valid = self.routing(item_embeds, mask)
        return interests[inverse], slot_valid[inverse]

    def item_embedding(self, item_col, item_ids=None):
        """Unmasked in the control; overridden by the masked variants."""
        return self.item_tower(item_col)

    def mask_interests(self, interests, user_ids):
        """Identity in the control; overridden by the masked variants."""
        return interests

    # ------------------------------------------------------------------
    def pool_interests(self, interests, slot_valid, target_embed):
        """
        Training-time label-aware pooling: [B, K, d] -> [B, d].

        The training loop calls this once with the *positive* item's
        embedding, then scores both the positive and its negatives against
        the resulting single user vector -- mirroring how the DCM family
        derives j* from the positive and reuses it for negatives.
        """
        return self.label_attention(interests, target_embed, slot_valid)

    def score_multi(self, user_row, item_col, user_ids=None, item_ids=None):
        """Per-interest inner products [B, K], padding slots at NEG_SCORE."""
        assert user_ids is not None, "MIND requires user_ids to look up history"
        interests, slot_valid = self.get_user_interests(user_ids)
        interests = self.mask_interests(interests, user_ids)
        q_i = self.item_embedding(item_col, item_ids)

        scores = torch.einsum("bkd,bd->bk", interests, q_i)
        return self.mask_invalid_slots(scores, slot_valid), interests

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        """Evaluation scoring: max over interests (§5)."""
        scores, _ = self.score_multi(user_row, item_col, user_ids, item_ids)
        return scores.max(dim=1).values

    # ---- full-catalog scoring ----------------------------------------
    # Inherited unchanged by both masked variants: `item_embedding` and
    # `mask_interests` are the two hooks they override, and both encoders
    # below go through those hooks rather than reimplementing them.
    def encode_items(self, item_cols, item_ids=None, chunk_size=DEFAULT_ITEM_ENCODE_CHUNK):
        out = []
        for start, end in chunk_bounds(item_cols.size(0), chunk_size):
            ids = item_ids[start:end] if item_ids is not None else None
            out.append(self.item_embedding(item_cols[start:end], ids))
        return torch.cat(out, dim=0)

    def encode_users(self, user_row, user_ids=None):
        assert user_ids is not None, "MIND requires user_ids to look up history"
        interests, slot_valid = self.get_user_interests(user_ids)
        interests = self.mask_interests(interests, user_ids)
        return {"interests": interests, "slot_valid": slot_valid}

    def score_encoded(self, user_enc, item_enc, pair_chunk=DEFAULT_PAIR_CHUNK):
        """
        Inner product against the whole catalog, then max over interests.

        The cheapest of the three families by a wide margin: MIND scores by
        dot product, so this is one matmul per item chunk with no pair
        tensor at all -- the [B, K, C] result is the only intermediate,
        versus the [B*K*C, d] pair list the attention-based scorers need.
        """
        interests = user_enc["interests"]
        slot_valid = user_enc["slot_valid"]

        B, K, _ = interests.shape
        N = item_enc.size(0)
        out = torch.empty(B, N, device=interests.device, dtype=interests.dtype)

        for start, end in chunk_bounds(N, max(1, pair_chunk // max(B * K, 1))):
            scores = torch.einsum("bkd,nd->bkn", interests, item_enc[start:end])
            scores = scores.masked_fill(~slot_valid.unsqueeze(-1), NEG_SCORE)
            out[:, start:end] = scores.max(dim=1).values

        return out


class MINDRPUCBMulti(MIND):
    """
    Model 5 -- full multi-interest MIND with a shared d-dim user mask and
    a d-dim item mask. The pure mask test for this backbone.
    """

    def __init__(
        self,
        num_users,
        num_items,
        user_interaction_counts=None,
        item_interaction_counts=None,
        gamma_init=2.0,
        beta=1.0,
        mask_init=0.5,
        **kwargs,
    ):
        kwargs["_defer_init"] = True
        super().__init__(num_users, num_items, **kwargs)

        self.user_mask = RPUCBMask(
            num_users, self.embed_dim, counts=user_interaction_counts,
            n_slots=1, gamma_init=gamma_init, beta=beta, mask_init=mask_init,
        )
        self.item_mask = RPUCBMask(
            num_items, self.embed_dim, counts=item_interaction_counts,
            n_slots=1, gamma_init=gamma_init, beta=beta, mask_init=mask_init,
        )

        self.init_weights()

    def mask_interests(self, interests, user_ids):
        # [B, 1, d] broadcast across however many interests this user has.
        return interests * self.user_mask(user_ids)

    def item_embedding(self, item_col, item_ids=None):
        assert item_ids is not None, (
            "MINDRPUCBMulti masks the item side and requires item_ids"
        )
        return self.item_tower(item_col) * self.item_mask.flat(item_ids)

    def score_with_mask(self, user_row, item_col, user_ids=None, item_ids=None):
        scores = self.score(user_row, item_col, user_ids, item_ids)
        return scores, (self.user_mask(user_ids), self.item_mask.flat(item_ids))


class MINDRPUCB(MINDRPUCBMulti):
    """
    Model 6 -- the K=1 collapse. Routing degenerates to weighted pooling
    plus squash and label-aware attention has one slot, so this is the
    MIND tower with a single adaptively-masked embedding rather than a
    multi-interest model. Describe it that way in the writeup.
    """

    def __init__(self, num_users, num_items, **kwargs):
        kwargs["k_max"] = 1
        kwargs["adaptive_k"] = False
        super().__init__(num_users, num_items, **kwargs)