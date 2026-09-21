"""
DeepCF + RP-UCB + attention -- model 3, recovered from the pre-refactor
`rpucb_attn_full.py` and renamed.

Identical to model 2 in every respect except the CFNet-ml fusion: a
`SelfAttentionInteraction` over the two masked embeddings replaces
concatenate-then-MLP. Same masks, same RL branch, same capacity. That
makes the DeepCF backbone a three-rung ladder:

    deepcf            -> deepcf_rpucb        pure mask effect
    deepcf_rpucb      -> deepcf_rpucb_attn   pure attention effect
    deepcf            -> deepcf_rpucb_attn   combined (the mid-sem headline)

which mirrors the structure §1 built for the DCM backbone, and lets the
mid-semester finding -- that RP-UCB alone lost to DeepCF while RP-UCB with
attention beat it -- be explained rather than just reported. If the mask
only pays when the fusion is expressive enough to exploit it, that
predicts it helps on MIND and DCM, which have attention in the tower, and
not on bare DeepCF. The middle rung is what tests that.

This model was already using the canonical mask formula, so the only
substantive change is routing both masks through the shared `RPUCBMask`
and requiring `item_ids` rather than silently falling back to a
user-only mask.
"""

import torch
import torch.nn as nn

from .attention import SelfAttentionInteraction
from .base import DEFAULT_ITEM_ENCODE_CHUNK, DEFAULT_PAIR_CHUNK, BaseCF, build_mlp, chunk_bounds
from .rpucb_mask import RPUCBMask


class DeepCFRPUCBAttn(BaseCF):
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
        mask_init=0.5,
    ):
        super().__init__()
        if rl_layers is None:
            rl_layers = [512, 256, 128, 64]

        assert rl_layers[-1] == embed_dim, (
            f"rl_layers[-1] ({rl_layers[-1]}) must equal embed_dim ({embed_dim}); "
            f"see deepcf_rpucb.py's docstring."
        )
        assert embed_dim % attn_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by attn_heads ({attn_heads})"
        )

        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        self.user_mask = RPUCBMask(
            num_users, embed_dim, counts=user_interaction_counts,
            n_slots=1, gamma_init=gamma_init, beta=beta, mask_init=mask_init,
        )
        self.item_mask = RPUCBMask(
            num_items, embed_dim, counts=item_interaction_counts,
            n_slots=1, gamma_init=gamma_init, beta=beta, mask_init=mask_init,
        )

        self.f_rl_user = build_mlp([num_items] + rl_layers, dropout=dropout)
        self.f_rl_item = build_mlp([num_users] + rl_layers, dropout=dropout)

        self.user_embedding = nn.Linear(num_items, embed_dim, bias=False)
        self.item_embedding = nn.Linear(num_users, embed_dim, bias=False)

        # The one architectural difference from model 2.
        self.z_attn_module = SelfAttentionInteraction(
            embed_dim, num_heads=attn_heads, dropout=dropout, output_dim=None
        )

        if ml_layers:
            self.f_ml = build_mlp([2 * embed_dim] + ml_layers, dropout=dropout)
            ml_out = ml_layers[-1]
        else:
            self.f_ml = nn.Identity()
            ml_out = 2 * embed_dim

        self.fusion = nn.Linear(rl_layers[-1] + ml_out, 1)

        self.init_weights()

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        scores, _ = self.score_with_mask(user_row, item_col, user_ids, item_ids)
        return scores

    def score_with_mask(self, user_row, item_col, user_ids=None, item_ids=None):
        assert user_ids is not None, "DeepCFRPUCBAttn requires user_ids"
        assert item_ids is not None, "DeepCFRPUCBAttn masks the item side and requires item_ids"

        user_mask = self.user_mask.flat(user_ids)
        item_mask = self.item_mask.flat(item_ids)

        p_u_rl = self.f_rl_user(user_row) * user_mask
        q_i_rl = self.f_rl_item(item_col) * item_mask
        z_rl = p_u_rl * q_i_rl

        p_u_ml = self.user_embedding(user_row) * user_mask
        q_i_ml = self.item_embedding(item_col) * item_mask
        z_ml = self.f_ml(self.z_attn_module(p_u_ml, q_i_ml))

        scores = self.fusion(torch.cat([z_rl, z_ml], dim=1)).squeeze(-1)
        return scores, (user_mask, item_mask)

    # ---- full-catalog scoring ----------------------------------------
    def encode_items(self, item_cols, item_ids=None, chunk_size=DEFAULT_ITEM_ENCODE_CHUNK):
        assert item_ids is not None, "DeepCFRPUCBAttn masks the item side and requires item_ids"
        rl, ml = [], []
        for start, end in chunk_bounds(item_cols.size(0), chunk_size):
            chunk = item_cols[start:end]
            mask = self.item_mask.flat(item_ids[start:end])
            rl.append(self.f_rl_item(chunk) * mask)
            ml.append(self.item_embedding(chunk) * mask)
        return {"rl": torch.cat(rl, dim=0), "ml": torch.cat(ml, dim=0)}

    def encode_users(self, user_row, user_ids=None):
        assert user_ids is not None, "DeepCFRPUCBAttn requires user_ids"
        mask = self.user_mask.flat(user_ids)
        return {
            "rl": self.f_rl_user(user_row) * mask,
            "ml": self.user_embedding(user_row) * mask,
        }

    def score_encoded(self, user_enc, item_enc, pair_chunk=DEFAULT_PAIR_CHUNK):
        """
        Cannot share `deepcf_score_encoded`: SelfAttentionInteraction takes
        a flat [X, d] pair list rather than broadcasting over [B, C, d], so
        the ML branch has to be reshaped to B*C rows and back. The RL
        branch is identical to the concat models.
        """
        p_rl, p_ml = user_enc["rl"], user_enc["ml"]
        q_rl, q_ml = item_enc["rl"], item_enc["ml"]

        B, N = p_rl.size(0), q_rl.size(0)
        d = p_ml.size(-1)
        out = torch.empty(B, N, device=p_rl.device, dtype=p_rl.dtype)

        for start, end in chunk_bounds(N, max(1, pair_chunk // max(B, 1))):
            C = end - start
            z_rl = p_rl.unsqueeze(1) * q_rl[start:end].unsqueeze(0)       # [B,C,d]

            pm = p_ml.unsqueeze(1).expand(B, C, d).reshape(B * C, d)
            qm = q_ml[start:end].unsqueeze(0).expand(B, C, d).reshape(B * C, d)
            z_ml = self.f_ml(self.z_attn_module(pm, qm)).view(B, C, -1)

            out[:, start:end] = self.fusion(
                torch.cat([z_rl, z_ml], dim=-1)
            ).squeeze(-1)

        return out