import torch
import torch.nn as nn
from .base import BaseCF, build_mlp
from .attention import SelfAttentionInteraction


class RPUCBAttn(BaseCF):
    """
    RP-UCB + Attention (user-side masking only).

    Architecture:
      • CFNet-rl branch  : user embeddings are masked via RP-UCB; item branch unchanged.
      • CFNet-ml branch  : user embeddings are masked via RP-UCB, then a
                           SelfAttentionInteraction module fuses masked user
                           and *standard* item embeddings.
      • Fusion           : [z_rl || z_ml] → linear → scalar score.

    The RP-UCB mask for users:
        mask_u = sigmoid(w_u) + beta * sigmoid(gamma) * log(n̄ / N_u)
    Item embeddings remain standard (no exploration bonus).

    Args:
        num_users               : number of users
        num_items               : number of items
        embed_dim               : latent dimensionality
        rl_layers               : hidden sizes for CFNet-rl MLPs
        ml_layers               : optional hidden sizes applied AFTER attention
                                  (None → identity)
        user_interaction_counts : LongTensor [num_users]
        attn_heads              : number of multi-head attention heads
        dropout                 : dropout probability
        gamma_init              : initial value for the learnable gamma parameter
        beta                    : scale factor for the exploration bonus
    """

    def __init__(
        self,
        num_users,
        num_items,
        embed_dim=64,
        rl_layers=None,
        ml_layers=None,
        user_interaction_counts=None,
        attn_heads=2,
        dropout=0.0,
        gamma_init=2.0,
        beta=1.0,
    ):
        super().__init__()
        if rl_layers is None:
            rl_layers = [512, 256, 128, 64]

        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.beta = beta

        # ── RP-UCB (user side only) ──────────────────────────────────────────
        self.mask_embeddings = nn.Embedding(num_users, embed_dim)
        self.gamma = nn.Parameter(torch.full((embed_dim,), float(gamma_init)))

        if user_interaction_counts is None:
            user_interaction_counts = torch.ones(num_users, dtype=torch.long)
        self.register_buffer('user_counts', user_interaction_counts)
        mean_cnt = self.user_counts.float().mean().item()
        self.n_bar = max(1.0, mean_cnt)

        # ── CFNet-rl branch ──────────────────────────────────────────────────
        self.f_rl_user = build_mlp([num_items] + rl_layers, dropout=dropout)
        self.f_rl_item = build_mlp([num_users] + rl_layers, dropout=dropout)

        # ── CFNet-ml branch : Attention + optional MLP ───────────────────────
        self.user_embedding = nn.Linear(num_items, embed_dim, bias=False)   # P^T
        self.item_embedding = nn.Linear(num_users, embed_dim, bias=False)   # Q^T

        # Attention fuses masked-user and *standard* item embeddings
        self.z_attn_module = SelfAttentionInteraction(
            embed_dim, num_heads=attn_heads, dropout=dropout, output_dim=None
        )

        if ml_layers:
            self.f_ml = build_mlp([2 * embed_dim] + ml_layers, dropout=dropout)
        else:
            self.f_ml = nn.Identity()

        # ── Fusion ───────────────────────────────────────────────────────────
        self.fusion = nn.Linear(2 * embed_dim, 1)

        self.init_weights()

    # ------------------------------------------------------------------
    def get_mask(self, user_ids):
        """RP-UCB mask for users."""
        w_u = self.mask_embeddings(user_ids)                        # [B, d]
        N_u = self.user_counts[user_ids].float().clamp(min=1.0)    # [B]

        explore_scalar = torch.clamp(
            torch.log(torch.tensor(self.n_bar, device=w_u.device) / N_u),
            min=0.0,
        )                                                           # [B]

        sigma_gamma = torch.sigmoid(self.gamma)                     # [d]
        explore_vec = sigma_gamma.unsqueeze(0) * explore_scalar.unsqueeze(1)  # [B, d]

        mask = torch.sigmoid(w_u) + self.beta * explore_vec
        return mask.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def forward(self, user_row, item_col, user_ids, item_ids=None):
        return self.score_with_mask(user_row, item_col, user_ids)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        scores, _ = self.score_with_mask(user_row, item_col, user_ids)
        return scores

    def score_with_mask(self, user_row, item_col, user_ids, item_ids=None):
        mask = self.get_mask(user_ids)

        # ── CFNet-rl ─────────────────────────────────────────────────────
        p_u_rl = self.f_rl_user(user_row)
        q_i_rl = self.f_rl_item(item_col)

        p_u_rl_masked = p_u_rl * mask
        z_rl = p_u_rl_masked * q_i_rl          # item side: unmasked

        # ── CFNet-ml + Attention ─────────────────────────────────────────
        p_u_ml = self.user_embedding(user_row)
        q_i_ml = self.item_embedding(item_col)  # item side: unmasked

        p_u_ml_masked = p_u_ml * mask
        z_attn = self.z_attn_module(p_u_ml_masked, q_i_ml)
        z_ml = self.f_ml(z_attn)

        # ── Fusion ───────────────────────────────────────────────────────
        z = torch.cat([z_rl, z_ml], dim=1)
        scores = self.fusion(z).squeeze(-1)
        return scores, mask
