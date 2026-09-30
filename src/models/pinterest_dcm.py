"""
PinterestDCM -- model 7 (`dcm`), the unmasked multi-interest control.

K user embeddings via the Differentiable Clustering Module instead of one.
The item tower stays single-embedding throughout, matching Pinterest
Sec 3.5; that constraint is what makes models 7-10 comparable on user-side
capacity alone.

Pipeline (Fig. 3):
  1. Gather up to `max_hist_len` of the user's train items as an unordered
     bag (no usable timestamps in our datasets; routing treats history as
     a set regardless).
  2. Embed each via the item tower's summarization sub-layer, pre-crossing
     -- Eq. 4's e_i. Reusing the item tower's weights rather than adding a
     parallel embedding pipeline, since our data has none of the separate
     pretrained/categorical item features Pinterest uses at this step.
  3. Route into K centroids (VA-FPI + SAR).
  4. Cross each centroid through a shared LiteDHEN to get the final
     condition embeddings.
  5. Score each condition against the target item, then max over K
     (Pinterest Sec 4.1.1).

Four fixes in this version.

1. `interaction_cols` is no longer a persistent buffer. It is a dense
   [num_items, num_users] float matrix -- 426 MB for AToy, 377 MB for
   citeulike -- and `register_buffer` defaults to persistent=True, so it
   was being written into every checkpoint despite being fully
   reconstructible from the dataset. Across the masked DCM variants x 5
   datasets x 3 seeds that is tens of gigabytes of redundant writes onto
   storage §10 already flags as tight.

2. History sampling is seeded and uniform. The old code did
   `list(items)[:max_hist_len]` on a `set` of ints; CPython iterates
   small-int sets in roughly value order, so truncation systematically
   kept the *lowest item IDs*. Where item IDs correlate with insertion
   order or popularity -- which they do in these datasets -- every user's
   routing bag was skewed the same direction. Now a seeded uniform sample.

3. `init_weights()` runs once, at the end of the leaf constructor, rather
   than once per class in the hierarchy.

4. Invalid interest slots are scored at NEG_SCORE, so max-over-K and the
   training-time argmax both skip them without the caller re-deriving
   validity.

Two levels of deduplication keep this tractable, and both matter.
`get_user_conditions` dedupes user ids before routing, because the same
user recurs across every negative during training and every candidate at
evaluation. `embed_history_bag` then dedupes item ids within the batch's
bags, because at ml-1m's batch size the bags hold 51,200 slots drawn from
a catalog of 3,706 items -- without it the same columns are embedded about
fourteen times over and autograd retains all of them, 1.24 GB per forward
against 90 MB.
"""

import torch
import torch.nn as nn

from .attention import SelfAttentionInteraction
from .base import DEFAULT_ITEM_ENCODE_CHUNK, DEFAULT_PAIR_CHUNK, BaseCF, chunk_bounds
from .dcm import DCM
from .history import build_history_buffers, embed_history_bag
from .pinterest_tower import LiteDHEN, PinterestTower
from .routing_common import NEG_SCORE


class PinterestDCM(BaseCF):
    def __init__(
        self,
        num_users,
        num_items,
        embed_dim=64,
        K=7,
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
        _defer_init=False,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.K = K
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

        self.dcm = DCM(embed_dim, K=K, routing_iters=routing_iters)
        self.condition_cross = LiteDHEN(
            embed_dim, n_fields=n_fields, n_heads=n_heads_dhen,
            transformer_layers=transformer_layers, dropout=dropout,
        )

        self.interaction = SelfAttentionInteraction(
            embed_dim, num_heads=attn_heads, dropout=dropout, output_dim=None
        )
        self.fusion = nn.Linear(2 * embed_dim, 1)

        # Subclasses pass _defer_init=True and call init_weights() once
        # themselves, after their own submodules exist.
        if not _defer_init:
            self.init_weights()

    # ------------------------------------------------------------------
    def _history_item_embeds(self, user_ids):
        """e_i for each user's bag: ([B, L, d], [B, L])."""
        hist_ids = self.history_item_ids[user_ids]
        mask = self.history_mask[user_ids]
        e = embed_history_bag(
            self.item_tower.summarize, self.interaction_cols, hist_ids, self.embed_dim
        )
        return e, mask

    def get_user_conditions(self, user_ids):
        """
        Final condition embeddings and slot validity: ([B, K, d], [B, K]).

        Deduplicates user ids before routing. The same user appears once
        per candidate item at evaluation and once per negative during
        training, and re-running the full routing pipeline for every
        duplicate is enough to exhaust memory outright at eval batch
        scale, not merely waste time.
        """
        unique_ids, inverse = torch.unique(user_ids, return_inverse=True)

        item_embeds, mask = self._history_item_embeds(unique_ids)
        centroids, _, slot_valid = self.dcm(item_embeds, mask)       # [U, K, d]

        U = centroids.size(0)
        flat = centroids.reshape(U * self.K, self.embed_dim)
        crossed = self.condition_cross(flat).view(U, self.K, self.embed_dim)

        return crossed[inverse], slot_valid[inverse]

    def _score_conditions(self, conditions, q_i, slot_valid):
        """Score [B, K, d] conditions against [B, d] items -> [B, K]."""
        B, K, d = conditions.shape
        cond_flat = conditions.reshape(B * K, d)
        q_rep = q_i.unsqueeze(1).expand(-1, K, -1).reshape(B * K, d)

        z = self.interaction(cond_flat, q_rep)
        scores = self.fusion(z).squeeze(-1).view(B, K)
        return self.mask_invalid_slots(scores, slot_valid)

    # ------------------------------------------------------------------
    def score_multi(self, user_row, item_col, user_ids=None, item_ids=None):
        assert user_ids is not None, "PinterestDCM requires user_ids to look up history"
        conditions, slot_valid = self.get_user_conditions(user_ids)
        q_i = self.item_tower(item_col)
        return self._score_conditions(conditions, q_i, slot_valid), conditions

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        # user_row is unused: history comes from a user_ids lookup, not the
        # dense interaction row. Kept in the signature so every model in
        # the registry can be called identically.
        scores, _ = self.score_multi(user_row, item_col, user_ids, item_ids)
        return scores.max(dim=1).values

    # ---- full-catalog scoring ----------------------------------------
    def encode_items(self, item_cols, item_ids=None, chunk_size=DEFAULT_ITEM_ENCODE_CHUNK):
        """[N, d] item tower outputs. item_ids unused (no item mask here)."""
        out = []
        for start, end in chunk_bounds(item_cols.size(0), chunk_size):
            out.append(self.item_tower(item_cols[start:end]))
        return torch.cat(out, dim=0)

    def encode_users(self, user_row, user_ids=None):
        """
        {"conditions": [B, K, d], "slot_valid": [B, K]}. user_row is unused
        -- history comes from a user_ids lookup -- but kept in the
        signature so evaluate.py calls every family identically.
        """
        assert user_ids is not None, "PinterestDCM requires user_ids to look up history"
        conditions, slot_valid = self.get_user_conditions(user_ids)
        return {"conditions": conditions, "slot_valid": slot_valid}

    def score_encoded(self, user_enc, item_enc, pair_chunk=DEFAULT_PAIR_CHUNK):
        """
        [B, N] via max over K. Holds B*K*C pairs in flight rather than B*C,
        so the item chunk is divided by K -- otherwise the seven-condition
        models would use seven times the memory of the DeepCF family at the
        same nominal pair budget.

        Inherited unchanged by PinterestDCMRPUCB: its concat head arrives
        here as K=1 with an all-true validity mask, so the same max-over-K
        reduction is correct without a special case.
        """
        conditions = user_enc["conditions"]
        slot_valid = user_enc["slot_valid"]

        B, K, d = conditions.shape
        N = item_enc.size(0)
        out = torch.empty(B, N, device=conditions.device, dtype=conditions.dtype)

        for start, end in chunk_bounds(N, max(1, pair_chunk // max(B * K, 1))):
            C = end - start
            cond_rep = conditions.unsqueeze(2).expand(B, K, C, d).reshape(-1, d)
            item_rep = item_enc[start:end].view(1, 1, C, d).expand(B, K, C, d).reshape(-1, d)

            z = self.interaction(cond_rep, item_rep)
            scores = self.fusion(z).squeeze(-1).view(B, K, C)
            scores = scores.masked_fill(~slot_valid.unsqueeze(-1), NEG_SCORE)
            out[:, start:end] = scores.max(dim=1).values

        return out