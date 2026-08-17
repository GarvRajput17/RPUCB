"""
PinterestDCMRPUCB -- stretch model (gameplan §4/§6, lowest execution
priority -- cut first if the single training pass runs short on
compute/time).

DCM's K condition embeddings, each independently gated by its own RP-UCB
mask (Fork 4's locked decision: per-(user, k) masks, not one mask shared
identically across all K slots). Answers a different question from the
core K-vs-1 test: does masking help *even after* multiplicity has already
been added, i.e. does the effect stack.
"""

import torch
import torch.nn as nn
from .pinterest_dcm import PinterestDCM


class PinterestDCMRPUCB(PinterestDCM):
    """
    mask_{u,k} = sigmoid(w_{u,k}) + beta * sigmoid(gamma) * log(n̄_u / N_u)

    N_u (the user's overall interaction count) is shared across all K
    slots for a given user -- only the learned exploitation term w_{u,k}
    (and the resulting mask) differs per slot. Pinterest doesn't define a
    per-condition interaction count to base the exploration bonus on
    instead, so the exploration *strength* still reflects how much history
    that user has overall; only *where* in the latent space each slot
    chooses to stay open is independent per-slot.

    Subclasses PinterestDCM and reuses its towers, DCM routing, condition
    crossing, attention scorer, and fusion layer unchanged -- only the
    mask and the score_multi() call site that applies it are new.
    """

    def __init__(self, num_users, num_items, embed_dim=64, K=7,
                 user_train_items=None, interaction_cols=None,
                 max_hist_len=50, summarization_hidden=256, n_fields=4,
                 n_heads_dhen=2, transformer_layers=2, attn_heads=2,
                 routing_iters=3, dropout=0.0,
                 user_interaction_counts=None, gamma_init=2.0, beta=1.0):
        super().__init__(
            num_users, num_items, embed_dim=embed_dim, K=K,
            user_train_items=user_train_items, interaction_cols=interaction_cols,
            max_hist_len=max_hist_len, summarization_hidden=summarization_hidden,
            n_fields=n_fields, n_heads_dhen=n_heads_dhen,
            transformer_layers=transformer_layers, attn_heads=attn_heads,
            routing_iters=routing_iters, dropout=dropout,
        )
        self.beta = beta

        # One mask-embedding row per (user, k) slot, flattened as
        # user_id * K + k.
        self.mask_embeddings = nn.Embedding(num_users * K, embed_dim)
        self.gamma = nn.Parameter(torch.full((embed_dim,), float(gamma_init)))

        if user_interaction_counts is None:
            user_interaction_counts = torch.ones(num_users, dtype=torch.long)
        self.register_buffer('user_counts', user_interaction_counts)
        self.n_bar = max(1.0, self.user_counts.float().mean().item())

        self.init_weights()

    # ------------------------------------------------------------------
    def get_condition_masks(self, user_ids):
        """Per-(user, k) masks: [B, K, d]."""
        B = user_ids.size(0)
        k_offsets = torch.arange(self.K, device=user_ids.device)              # [K]
        flat_ids = user_ids.unsqueeze(1) * self.K + k_offsets.unsqueeze(0)    # [B, K]

        w = self.mask_embeddings(flat_ids)                                    # [B, K, d]
        N_u = self.user_counts[user_ids].float().clamp(min=1.0)               # [B]
        explore_scalar = torch.clamp(
            torch.log(torch.tensor(self.n_bar, device=w.device) / N_u),
            min=0.0,
        ).view(B, 1, 1)                                                        # [B,1,1]
        sigma_gamma = torch.sigmoid(self.gamma).view(1, 1, -1)                # [1,1,d]
        explore_vec = sigma_gamma * explore_scalar                            # -> [B,K,d]

        mask = torch.sigmoid(w) + self.beta * explore_vec
        return mask.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def score_multi(self, user_row, item_col, user_ids, item_ids=None):
        conditions = self.get_user_conditions(user_ids)       # [B, K, d]
        masks = self.get_condition_masks(user_ids)             # [B, K, d]
        conditions = conditions * masks

        q_i = self.item_tower(item_col)                        # [B, d]

        B = conditions.size(0)
        cond_flat = conditions.reshape(B * self.K, self.embed_dim)
        q_i_rep = q_i.unsqueeze(1).expand(-1, self.K, -1).reshape(
            B * self.K, self.embed_dim
        )

        z = self.interaction(cond_flat, q_i_rep)
        scores_flat = self.fusion(z).squeeze(-1)
        scores = scores_flat.view(B, self.K)
        return scores, conditions

    def forward(self, user_row, item_col, user_ids=None, item_ids=None):
        return self.score(user_row, item_col, user_ids, item_ids)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        assert user_ids is not None, "PinterestDCMRPUCB requires user_ids to look up history"
        scores, _ = self.score_multi(user_row, item_col, user_ids, item_ids)
        return scores.max(dim=1).values