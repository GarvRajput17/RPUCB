"""
PinterestBaseRPUCB -- Pinterest tower, single embedding, RP-UCB-masked.

This is the row that actually makes the experiment about RP-UCB. Run at
embed_dim=d it's a masking-effect control (isolates masking's contribution
from the tower swap); run at embed_dim=K*d it's the core competitor to
pinterest_dcm.py at matched total user-side capacity -- see gameplan §6.
"""

import torch
import torch.nn as nn
from .base import BaseCF
from .pinterest_tower import PinterestTower
from .attention import SelfAttentionInteraction


class PinterestBaseRPUCB(BaseCF):
    """
    Pinterest tower + user-side RP-UCB masking. Mask math is identical to
    rpucb_attn.py's (mask formula reused unmodified, per the existing
    codebase convention of not re-deriving it per model):

        mask_u = sigmoid(w_u) + beta * sigmoid(gamma) * log(n̄ / N_u)

    Scope is user-side only (matches rpucb_attn.py's convention, and the
    "Scope: U" column in the gameplan's target table) -- the item tower is
    left unmasked. This also happens to be the more architecturally
    faithful choice: Pinterest itself never applies any masking mechanism
    to the item tower either.
    """

    def __init__(self, num_users, num_items, embed_dim=64,
                 summarization_hidden=256, n_fields=4, n_heads_dhen=2,
                 transformer_layers=2, attn_heads=2, dropout=0.0,
                 user_interaction_counts=None, gamma_init=2.0, beta=1.0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.beta = beta

        # ── RP-UCB: user side only ──────────────────────────────────
        self.mask_embeddings = nn.Embedding(num_users, embed_dim)
        self.gamma = nn.Parameter(torch.full((embed_dim,), float(gamma_init)))

        if user_interaction_counts is None:
            user_interaction_counts = torch.ones(num_users, dtype=torch.long)
        self.register_buffer('user_counts', user_interaction_counts)
        self.n_bar = max(1.0, self.user_counts.float().mean().item())

        # ── Towers ───────────────────────────────────────────────────
        self.user_tower = PinterestTower(
            input_dim=num_items, embed_dim=embed_dim,
            summarization_hidden=summarization_hidden, n_fields=n_fields,
            n_heads=n_heads_dhen, transformer_layers=transformer_layers,
            dropout=dropout,
        )
        self.item_tower = PinterestTower(
            input_dim=num_users, embed_dim=embed_dim,
            summarization_hidden=summarization_hidden, n_fields=n_fields,
            n_heads=n_heads_dhen, transformer_layers=transformer_layers,
            dropout=dropout,
        )

        self.interaction = SelfAttentionInteraction(
            embed_dim, num_heads=attn_heads, dropout=dropout, output_dim=None
        )
        self.fusion = nn.Linear(2 * embed_dim, 1)

        self.init_weights()

    # ------------------------------------------------------------------
    def get_mask(self, user_ids):
        w_u = self.mask_embeddings(user_ids)                             # [B, d]
        N_u = self.user_counts[user_ids].float().clamp(min=1.0)          # [B]
        explore_scalar = torch.clamp(
            torch.log(torch.tensor(self.n_bar, device=w_u.device) / N_u),
            min=0.0,
        )                                                                 # [B]
        sigma_gamma = torch.sigmoid(self.gamma)                          # [d]
        explore_vec = sigma_gamma.unsqueeze(0) * explore_scalar.unsqueeze(1)  # [B, d]
        mask = torch.sigmoid(w_u) + self.beta * explore_vec
        return mask.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def forward(self, user_row, item_col, user_ids, item_ids=None):
        return self.score_with_mask(user_row, item_col, user_ids, item_ids)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        scores, _ = self.score_with_mask(user_row, item_col, user_ids, item_ids)
        return scores

    def score_with_mask(self, user_row, item_col, user_ids, item_ids=None):
        mask = self.get_mask(user_ids)                # [B, d]

        p_u = self.user_tower(user_row)                # [B, d]
        q_i = self.item_tower(item_col)                # [B, d]
        p_u_masked = p_u * mask

        z = self.interaction(p_u_masked, q_i)          # [B, 2d]
        scores = self.fusion(z).squeeze(-1)
        return scores, mask