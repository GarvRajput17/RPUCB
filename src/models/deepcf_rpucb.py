"""
DeepCF + RP-UCB -- model 2. The pure mask test for the DeepCF backbone:
identical architecture and capacity to model 1, differing only in the two
masks.

Fixed here: the mask formula. This model previously computed
`sigmoid(w + gamma * explore)` with gamma initialised to zeros and no beta
parameter, while every other masked model in the matrix used
`clamp(sigmoid(w) + beta * sigmoid(gamma) * explore, 0, 1)` with
gamma_init=2.0. Two consequences of the old form, both bad for the
comparison it is supposed to support: the exploration bonus started at
exactly zero and had to learn its way out, and there was no beta, so §3's
"beta must be re-tuned on validation per family" was unimplementable for
this family. Both masks now come from the shared `RPUCBMask`.

Mask scope is user *and* item, per the locked decision. Recorded as an
interpretation caveat: with `mask_init=0.5` the two masks multiply, so at
initialisation a dense user's score is built from representations at 0.25x
the magnitude the unmasked baseline sees. If this row comes out negative,
rerun with `mask_init` near 1.0 before concluding anything about RP-UCB
itself.

Known wart, preserved deliberately. The user mask is d-dimensional and is
applied both to the CFNet-ml embeddings (genuinely d-dimensional) and to
the CFNet-rl branch output, whose width is `rl_layers[-1]`. Those are
different representation spaces that happen to share a width because the
default rl_layers ends at 64 = embed_dim. Masking the RL branch is
defensible -- it is a learned representation and the README describes
RP-UCB as gating representations before scoring -- but reusing the same
mask vector across both spaces is a modelling choice that only typechecks
by coincidence. Changing it would change the model relative to the
mid-semester results, so it stays, with an assertion that makes the
coincidence explicit instead of silent.
"""

import torch
import torch.nn as nn

from .base import DEFAULT_ITEM_ENCODE_CHUNK, DEFAULT_PAIR_CHUNK, BaseCF, build_mlp, chunk_bounds
from .deepcf import deepcf_score_encoded
from .rpucb_mask import RPUCBMask


class DeepCFRPUCB(BaseCF):
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
        gamma_init=2.0,
        beta=1.0,
        mask_init=0.5,
    ):
        super().__init__()
        if rl_layers is None:
            rl_layers = [512, 256, 128, 64]
        if ml_layers is None:
            ml_layers = [512, 256, 128, 64]

        assert rl_layers[-1] == embed_dim, (
            f"the d-dim RP-UCB mask is applied to the CFNet-rl branch output, so "
            f"rl_layers[-1] ({rl_layers[-1]}) must equal embed_dim ({embed_dim}). "
            f"See the module docstring before changing either."
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
        self.f_ml = build_mlp([2 * embed_dim] + ml_layers, dropout=dropout)

        self.fusion = nn.Linear(rl_layers[-1] + ml_layers[-1], 1)

        self.init_weights()

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        scores, _ = self.score_with_mask(user_row, item_col, user_ids, item_ids)
        return scores

    def score_with_mask(self, user_row, item_col, user_ids=None, item_ids=None):
        # Required, not optional. The old code fell back to an all-ones item
        # mask when item_ids was absent, which silently turned a both-sides
        # model into a user-only one with no error anywhere.
        assert user_ids is not None, "DeepCFRPUCB requires user_ids"
        assert item_ids is not None, "DeepCFRPUCB masks the item side and requires item_ids"

        user_mask = self.user_mask.flat(user_ids)      # [B, d]
        item_mask = self.item_mask.flat(item_ids)      # [B, d]

        p_u_rl = self.f_rl_user(user_row) * user_mask
        q_i_rl = self.f_rl_item(item_col) * item_mask
        z_rl = p_u_rl * q_i_rl

        p_u_ml = self.user_embedding(user_row) * user_mask
        q_i_ml = self.item_embedding(item_col) * item_mask
        z_ml = self.f_ml(torch.cat([p_u_ml, q_i_ml], dim=1))

        scores = self.fusion(torch.cat([z_rl, z_ml], dim=1)).squeeze(-1)
        return scores, (user_mask, item_mask)

    # ---- full-catalog scoring ----------------------------------------
    def encode_items(self, item_cols, item_ids=None, chunk_size=DEFAULT_ITEM_ENCODE_CHUNK):
        assert item_ids is not None, "DeepCFRPUCB masks the item side and requires item_ids"
        rl, ml = [], []
        for start, end in chunk_bounds(item_cols.size(0), chunk_size):
            chunk = item_cols[start:end]
            mask = self.item_mask.flat(item_ids[start:end])
            rl.append(self.f_rl_item(chunk) * mask)
            ml.append(self.item_embedding(chunk) * mask)
        return {"rl": torch.cat(rl, dim=0), "ml": torch.cat(ml, dim=0)}

    def encode_users(self, user_row, user_ids=None):
        assert user_ids is not None, "DeepCFRPUCB requires user_ids"
        mask = self.user_mask.flat(user_ids)
        return {
            "rl": self.f_rl_user(user_row) * mask,
            "ml": self.user_embedding(user_row) * mask,
        }

    def score_encoded(self, user_enc, item_enc, pair_chunk=DEFAULT_PAIR_CHUNK):
        return deepcf_score_encoded(self.f_ml, self.fusion, user_enc, item_enc, pair_chunk)