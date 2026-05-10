import torch
import torch.nn as nn
from .base import BaseCF, build_mlp
from .attention import SelfAttentionInteraction


class RPUCBAttnFull(BaseCF):
    """
    RP-UCB + Attention with masking on *both* user and item sides.

    Architecture:
      • CFNet-rl branch  : user AND item embeddings are masked via their
                           respective RP-UCB masks before element-wise product.
      • CFNet-ml branch  : user AND item embeddings are masked, then fused
                           via SelfAttentionInteraction.
      • Fusion           : [z_rl || z_ml] → linear → scalar score.

    User RP-UCB mask:
        mask_u = sigmoid(w_u) + beta * sigmoid(gamma_u) * log(n̄_u / N_u)
    Item RP-UCB mask:
        mask_i = sigmoid(w_i) + beta * sigmoid(gamma_i) * log(n̄_i / N_i)

    Args:
        num_users               : number of users
        num_items               : number of items
        embed_dim               : latent dimensionality
        rl_layers               : hidden sizes for CFNet-rl MLPs
        ml_layers               : optional hidden sizes applied AFTER attention
        user_interaction_counts : LongTensor [num_users]
        item_interaction_counts : LongTensor [num_items]
        attn_heads              : number of multi-head attention heads
        dropout                 : dropout probability
        gamma_init              : initial value of learnable gamma parameters
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
        item_interaction_counts=None,
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

        # ── RP-UCB: user side ────────────────────────────────────────────────
        self.user_mask_embed = nn.Embedding(num_users, embed_dim)
        self.user_gamma = nn.Parameter(torch.full((embed_dim,), float(gamma_init)))

        if user_interaction_counts is None:
            user_interaction_counts = torch.ones(num_users, dtype=torch.long)
        self.register_buffer('user_counts', user_interaction_counts)
        self.n_bar_user = max(1.0, self.user_counts.float().mean().item())

        # ── RP-UCB: item side ────────────────────────────────────────────────
        self.item_mask_embed = nn.Embedding(num_items, embed_dim)
        self.item_gamma = nn.Parameter(torch.full((embed_dim,), float(gamma_init)))

        if item_interaction_counts is None:
            item_interaction_counts = torch.ones(num_items, dtype=torch.long)
        self.register_buffer('item_counts', item_interaction_counts)
        self.n_bar_item = max(1.0, self.item_counts.float().mean().item())

        # ── CFNet-rl branch ──────────────────────────────────────────────────
        self.f_rl_user = build_mlp([num_items] + rl_layers, dropout=dropout)
        self.f_rl_item = build_mlp([num_users] + rl_layers, dropout=dropout)

        # ── CFNet-ml branch : Attention + optional MLP ───────────────────────
        self.user_embedding = nn.Linear(num_items, embed_dim, bias=False)   # P^T
        self.item_embedding = nn.Linear(num_users, embed_dim, bias=False)   # Q^T

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
    def _rpucb_mask(self, ids, embed_layer, counts_buf, n_bar, gamma_param):
        """Generic RP-UCB mask computation."""
        w = embed_layer(ids)                                       # [B, d]
        N = counts_buf[ids].float().clamp(min=1.0)                # [B]
        explore_scalar = torch.clamp(
            torch.log(torch.tensor(n_bar, device=w.device) / N),
            min=0.0,
        )                                                          # [B]
        sigma_gamma = torch.sigmoid(gamma_param)                  # [d]
        explore_vec = sigma_gamma.unsqueeze(0) * explore_scalar.unsqueeze(1)  # [B, d]
        return (torch.sigmoid(w) + self.beta * explore_vec).clamp(0.0, 1.0)

    def get_user_mask(self, user_ids):
        return self._rpucb_mask(
            user_ids, self.user_mask_embed,
            self.user_counts, self.n_bar_user, self.user_gamma
        )

    def get_item_mask(self, item_ids):
        return self._rpucb_mask(
            item_ids, self.item_mask_embed,
            self.item_counts, self.n_bar_item, self.item_gamma
        )

    # ------------------------------------------------------------------
    def forward(self, user_row, item_col, user_ids, item_ids=None):
        return self.score_with_mask(user_row, item_col, user_ids, item_ids)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        scores, _ = self.score_with_mask(user_row, item_col, user_ids, item_ids)
        return scores

    def score_with_mask(self, user_row, item_col, user_ids, item_ids=None):
        user_mask = self.get_user_mask(user_ids)                   # [B, d]

        if item_ids is not None:
            item_mask = self.get_item_mask(item_ids)               # [B, d]
        else:
            item_mask = torch.ones_like(user_mask)

        # ── CFNet-rl ─────────────────────────────────────────────────────
        p_u_rl = self.f_rl_user(user_row)
        q_i_rl = self.f_rl_item(item_col)

        p_u_rl_masked = p_u_rl * user_mask
        q_i_rl_masked = q_i_rl * item_mask
        z_rl = p_u_rl_masked * q_i_rl_masked

        # ── CFNet-ml + Attention ─────────────────────────────────────────
        p_u_ml = self.user_embedding(user_row)
        q_i_ml = self.item_embedding(item_col)

        p_u_ml_masked = p_u_ml * user_mask
        q_i_ml_masked = q_i_ml * item_mask
        z_attn = self.z_attn_module(p_u_ml_masked, q_i_ml_masked)
        z_ml = self.f_ml(z_attn)

        # ── Fusion ───────────────────────────────────────────────────────
        z = torch.cat([z_rl, z_ml], dim=1)
        scores = self.fusion(z).squeeze(-1)
        return scores, (user_mask, item_mask)
