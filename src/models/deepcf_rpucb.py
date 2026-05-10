import torch
import torch.nn as nn
from .base import BaseCF, build_mlp


class DeepCFRPUCB(BaseCF):
    """
    DeepCF with RP-UCB adaptive masking applied to *both* user and item
    embedding branches.

    The RP-UCB mask modulates each latent dimension according to how much
    exploration is warranted given the user's interaction count N_u:

        mask_u(u) = sigmoid(w_u  + gamma * log(n̄ / N_u))   clamped to [0,1]
        mask_i(i) = sigmoid(w_i  + gamma * log(n̄ / N_i))   clamped to [0,1]

    Both masks are applied in:
      • the CFNet-rl branch  (element-wise multiplication before dot-product)
      • the CFNet-ml branch  (element-wise multiplication before concatenation)

    Args:
        num_users               : number of users
        num_items               : number of items
        embed_dim               : latent dimensionality
        rl_layers               : hidden layers for the RL (CFNet-rl) MLPs
        ml_layers               : hidden layers for the ML (CFNet-ml) MLP
        user_interaction_counts : LongTensor [num_users] — #interactions per user
        item_interaction_counts : LongTensor [num_items] — #interactions per item
        dropout                 : dropout probability
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
        dropout=0.0,
    ):
        super().__init__()
        if rl_layers is None:
            rl_layers = [512, 256, 128, 64]
        if ml_layers is None:
            ml_layers = [512, 256, 128, 64]

        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        # ── RP-UCB: user side ────────────────────────────────────────────────
        self.user_mask_embeddings = nn.Embedding(num_users, embed_dim)
        self.user_gamma = nn.Parameter(torch.zeros(embed_dim))

        if user_interaction_counts is None:
            user_interaction_counts = torch.ones(num_users, dtype=torch.long)
        self.register_buffer('user_counts', user_interaction_counts)
        mean_user_cnt = self.user_counts.float().mean().item()
        self.n_bar_user = max(1.0, mean_user_cnt)

        # ── RP-UCB: item side ────────────────────────────────────────────────
        self.item_mask_embeddings = nn.Embedding(num_items, embed_dim)
        self.item_gamma = nn.Parameter(torch.zeros(embed_dim))

        if item_interaction_counts is None:
            item_interaction_counts = torch.ones(num_items, dtype=torch.long)
        self.register_buffer('item_counts', item_interaction_counts)
        mean_item_cnt = self.item_counts.float().mean().item()
        self.n_bar_item = max(1.0, mean_item_cnt)

        # ── CFNet-rl branch ──────────────────────────────────────────────────
        self.f_rl_user = build_mlp([num_items] + rl_layers, dropout=dropout)
        self.f_rl_item = build_mlp([num_users] + rl_layers, dropout=dropout)

        # ── CFNet-ml branch ──────────────────────────────────────────────────
        self.user_embedding = nn.Linear(num_items, embed_dim, bias=False)  # P^T
        self.item_embedding = nn.Linear(num_users, embed_dim, bias=False)  # Q^T
        self.f_ml = build_mlp([2 * embed_dim] + ml_layers, dropout=dropout)

        # ── Fusion ───────────────────────────────────────────────────────────
        self.fusion = nn.Linear(2 * embed_dim, 1)

        self.init_weights()

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------
    def get_user_mask(self, user_ids):
        w_u = self.user_mask_embeddings(user_ids)           # [B, d]
        N_u = self.user_counts[user_ids].float().clamp(min=1.0)  # [B]
        explore_scalar = torch.clamp(
            torch.log(torch.tensor(self.n_bar_user, device=w_u.device) / N_u),
            min=0.0,
        )                                                    # [B]
        explore_vec = self.user_gamma.unsqueeze(0) * explore_scalar.unsqueeze(1)  # [B, d]
        return torch.sigmoid(w_u + explore_vec)             # [B, d]

    def get_item_mask(self, item_ids):
        w_i = self.item_mask_embeddings(item_ids)           # [B, d]
        N_i = self.item_counts[item_ids].float().clamp(min=1.0)  # [B]
        explore_scalar = torch.clamp(
            torch.log(torch.tensor(self.n_bar_item, device=w_i.device) / N_i),
            min=0.0,
        )                                                    # [B]
        explore_vec = self.item_gamma.unsqueeze(0) * explore_scalar.unsqueeze(1)  # [B, d]
        return torch.sigmoid(w_i + explore_vec)             # [B, d]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, user_row, item_col, user_ids, item_ids=None):
        return self.score_with_mask(user_row, item_col, user_ids, item_ids)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        scores, _ = self.score_with_mask(user_row, item_col, user_ids, item_ids)
        return scores

    def score_with_mask(self, user_row, item_col, user_ids, item_ids=None):
        user_mask = self.get_user_mask(user_ids)            # [B, d]

        # Derive item_ids from item_col if not supplied
        # (item_col is the normalised column vector; item IDs were stored
        #  externally in the batch — passed as item_ids when available)
        if item_ids is not None:
            item_mask = self.get_item_mask(item_ids)        # [B, d]
        else:
            # Fallback: no item mask (uniform ones)
            item_mask = torch.ones_like(user_mask)

        # ── CFNet-rl ─────────────────────────────────────────────────────
        p_u_rl = self.f_rl_user(user_row)                  # [B, d_rl]
        q_i_rl = self.f_rl_item(item_col)                  # [B, d_rl]

        p_u_rl_masked = p_u_rl * user_mask
        q_i_rl_masked = q_i_rl * item_mask
        z_rl = p_u_rl_masked * q_i_rl_masked

        # ── CFNet-ml ─────────────────────────────────────────────────────
        p_u_ml = self.user_embedding(user_row)              # [B, d]
        q_i_ml = self.item_embedding(item_col)              # [B, d]

        p_u_ml_masked = p_u_ml * user_mask
        q_i_ml_masked = q_i_ml * item_mask
        concat_ml = torch.cat([p_u_ml_masked, q_i_ml_masked], dim=1)
        z_ml = self.f_ml(concat_ml)

        # ── Fusion ───────────────────────────────────────────────────────
        z = torch.cat([z_rl, z_ml], dim=1)
        scores = self.fusion(z).squeeze(-1)
        return scores, (user_mask, item_mask)
