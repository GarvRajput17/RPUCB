"""
PinterestDCM -- Pinterest's implicit-interest arm (Sec 3.2, Fig 3).

K user embeddings via the Differentiable Clustering Module (dcm.py)
instead of a single user embedding. This is the core DCM row in the
gameplan's target table (§6), and the direct competitor to
pinterest_base_rpucb.py at matched total user-side capacity (K * d).
"""

import torch
import torch.nn as nn
from .base import BaseCF
from .pinterest_tower import PinterestTower, LiteDHEN
from .attention import SelfAttentionInteraction
from .dcm import DCM


class PinterestDCM(BaseCF):
    """
    Item tower stays single-embedding throughout -- Pinterest never splits
    the item side (Sec 3.5), and keeping that constraint here is what
    makes this comparable to pinterest_base_rpucb at matched *user-side*
    capacity only.

    Pipeline (mirrors Fig. 3):
      1. For each user, gather up to `max_hist_len` interacted items as an
         *unordered bag* (Fork 5's locked decision -- our data carries no
         timestamps, and DCM's routing math operates on a set regardless).
      2. Embed each history item via the item tower's MLP-summarization
         sub-layer only, pre-crossing (this is Eq. 4's e_i). We reuse the
         item tower's summarization weights for this rather than adding a
         second parallel embedding pipeline -- our datasets don't carry
         the separate pretrained/categorical item features Pinterest uses
         for this step, so `self.item_tower.summarize` is the closest
         faithful stand-in already available.
      3. Route those item embeddings into K cluster centroids via DCM
         (VA-FPI init + Single-Assignment Routing, dcm.py).
      4. Pass each of the K raw centroids through a shared LiteDHEN
         crossing module -- the "Feature Crossing Modules" step in Fig. 3
         -- to get the final K condition embeddings.
      5. Score each condition embedding against the target item via the
         same SelfAttentionInteraction + linear fusion used elsewhere in
         this experiment, then take the max over K -- Pinterest's own
         offline scoring convention (Sec 4.1.1: "we ... take the maximum
         score of each item among all user embeddings"). Because this
         happens inside `score()`/`forward()`, this model plugs into the
         *existing* evaluate.py unmodified for basic HR/NDCG -- the
         max-over-K eval-side change from the gameplan is only needed for
         retrieving a per-condition breakdown (e.g. for coverage metrics
         later), not for the score itself.

    Condition association at training time (Sec 3.2.3's argmax rule --
    which of the K embeddings' gradient gets the positive-item signal) is
    a training-loop-level decision and is intentionally NOT wired in here;
    that belongs to Stage B. `score_multi()` is the hook Stage B's
    training code will call to get per-condition scores for that.

    Practical caps (both documented deviations from an unconstrained
    reproduction, driven by compute/scale, same spirit as Pinterest's own
    "otherwise forgotten due to user history input limit"):
      - `max_hist_len` : caps how many of a user's interacted items are
        used for routing. Pinterest faces the same limit in production.
      - K is fixed at construction time (locked to 7 by the gameplan, not
        swept), matching Pinterest's own reported optimum.
    """

    def __init__(self, num_users, num_items, embed_dim=64, K=7,
                 user_train_items=None, interaction_cols=None,
                 max_hist_len=50, summarization_hidden=256, n_fields=4,
                 n_heads_dhen=2, transformer_layers=2, attn_heads=2,
                 routing_iters=3, dropout=0.0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.K = K
        self.max_hist_len = max_hist_len

        # ── Static per-user history tensors (built once, no gradient) ──
        history_item_ids = torch.zeros(num_users, max_hist_len, dtype=torch.long)
        history_mask = torch.zeros(num_users, max_hist_len, dtype=torch.bool)
        if user_train_items is not None:
            for u, items in user_train_items.items():
                if u >= num_users:
                    continue
                items = list(items)[:max_hist_len]
                if len(items) == 0:
                    continue
                n = len(items)
                history_item_ids[u, :n] = torch.tensor(items, dtype=torch.long)
                history_mask[u, :n] = True
        self.register_buffer('history_item_ids', history_item_ids)
        self.register_buffer('history_mask', history_mask)

        if interaction_cols is None:
            interaction_cols = torch.zeros(num_items, num_users)
        self.register_buffer('interaction_cols', interaction_cols)

        # ── Item tower: shared for both e_i (routing input) and psi(i)
        #    (scoring embedding) -- see docstring point 2 above ─────────
        self.item_tower = PinterestTower(
            input_dim=num_users, embed_dim=embed_dim,
            summarization_hidden=summarization_hidden, n_fields=n_fields,
            n_heads=n_heads_dhen, transformer_layers=transformer_layers,
            dropout=dropout,
        )

        # ── DCM routing + condition crossing ────────────────────────────
        self.dcm = DCM(embed_dim, K=K, routing_iters=routing_iters)
        self.condition_cross = LiteDHEN(
            embed_dim, n_fields=n_fields, n_heads=n_heads_dhen,
            transformer_layers=transformer_layers, dropout=dropout,
        )

        # ── Scoring (shared weights across all K slots) ─────────────────
        self.interaction = SelfAttentionInteraction(
            embed_dim, num_heads=attn_heads, dropout=dropout, output_dim=None
        )
        self.fusion = nn.Linear(2 * embed_dim, 1)

        self.init_weights()

    # ------------------------------------------------------------------
    def _history_item_embeds(self, user_ids):
        """e_i for each user's history bag. Returns ([B, L, d], [B, L])."""
        hist_ids = self.history_item_ids[user_ids]                # [B, L]
        mask = self.history_mask[user_ids]                        # [B, L]
        B, L = hist_ids.shape

        hist_cols = self.interaction_cols[hist_ids.reshape(-1)]   # [B*L, num_users]
        e = self.item_tower.summarize(hist_cols).view(B, L, self.embed_dim)
        return e, mask

    def get_user_conditions(self, user_ids):
        """Final K condition embeddings per user: [B, K, d]."""
        item_embeds, mask = self._history_item_embeds(user_ids)
        centroids, _ = self.dcm(item_embeds, mask)                # [B, K, d]

        B = centroids.size(0)
        flat = centroids.reshape(B * self.K, self.embed_dim)
        crossed = self.condition_cross(flat).view(B, self.K, self.embed_dim)
        return crossed

    def score_multi(self, user_row, item_col, user_ids, item_ids=None):
        """
        Returns per-condition scores [B, K] and the condition embeddings
        [B, K, d]. Stage B's argmax condition-association training loop
        will call this directly; score()/forward() also use it internally
        (taking max over K).
        """
        conditions = self.get_user_conditions(user_ids)           # [B, K, d]
        q_i = self.item_tower(item_col)                           # [B, d]

        B = conditions.size(0)
        cond_flat = conditions.reshape(B * self.K, self.embed_dim)
        q_i_rep = q_i.unsqueeze(1).expand(-1, self.K, -1).reshape(
            B * self.K, self.embed_dim
        )

        z = self.interaction(cond_flat, q_i_rep)                  # [B*K, 2d]
        scores_flat = self.fusion(z).squeeze(-1)                  # [B*K]
        scores = scores_flat.view(B, self.K)                      # [B, K]
        return scores, conditions

    # ------------------------------------------------------------------
    def forward(self, user_row, item_col, user_ids=None, item_ids=None):
        return self.score(user_row, item_col, user_ids, item_ids)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        # user_row (the dense interaction-profile vector) is unused here --
        # history comes from a user_ids lookup instead, not the dense row.
        # Kept in the signature purely for interface consistency with the
        # rest of the codebase (train.py / evaluate.py always pass it).
        assert user_ids is not None, "PinterestDCM requires user_ids to look up history"
        scores, _ = self.score_multi(user_row, item_col, user_ids, item_ids)
        return scores.max(dim=1).values