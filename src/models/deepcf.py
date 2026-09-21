"""
DeepCF (Deng et al.) -- model 1, the unmasked control for the DeepCF
backbone.

Unchanged from the pre-refactor version apart from the `item_ids` argument
on `score`, which every model now accepts so the registry can call them
identically. DeepCF ignores it.
"""

import torch
import torch.nn as nn

from .base import DEFAULT_ITEM_ENCODE_CHUNK, DEFAULT_PAIR_CHUNK, BaseCF, build_mlp, chunk_bounds


class DeepCF(BaseCF):
    def __init__(
        self,
        num_users,
        num_items,
        embed_dim=64,
        rl_layers=None,
        ml_layers=None,
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

        # CFNet-rl: interaction profiles through separate MLPs, then
        # element-wise product (representation learning branch).
        self.f_rl_user = build_mlp([num_items] + rl_layers, dropout=dropout)
        self.f_rl_item = build_mlp([num_users] + rl_layers, dropout=dropout)

        # CFNet-ml: linear embeddings, concatenated, then an MLP
        # (matching-function learning branch).
        self.user_embedding = nn.Linear(num_items, embed_dim, bias=False)
        self.item_embedding = nn.Linear(num_users, embed_dim, bias=False)
        self.f_ml = build_mlp([2 * embed_dim] + ml_layers, dropout=dropout)

        self.fusion = nn.Linear(rl_layers[-1] + ml_layers[-1], 1)

        self.init_weights()

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        p_u_rl = self.f_rl_user(user_row)
        q_i_rl = self.f_rl_item(item_col)
        z_rl = p_u_rl * q_i_rl

        p_u_ml = self.user_embedding(user_row)
        q_i_ml = self.item_embedding(item_col)
        z_ml = self.f_ml(torch.cat([p_u_ml, q_i_ml], dim=1))

        return self.fusion(torch.cat([z_rl, z_ml], dim=1)).squeeze(-1)

    # ---- full-catalog scoring ----------------------------------------
    def encode_items(self, item_cols, item_ids=None, chunk_size=DEFAULT_ITEM_ENCODE_CHUNK):
        rl, ml = [], []
        for start, end in chunk_bounds(item_cols.size(0), chunk_size):
            chunk = item_cols[start:end]
            rl.append(self.f_rl_item(chunk))
            ml.append(self.item_embedding(chunk))
        return {"rl": torch.cat(rl, dim=0), "ml": torch.cat(ml, dim=0)}

    def encode_users(self, user_row, user_ids=None):
        return {"rl": self.f_rl_user(user_row), "ml": self.user_embedding(user_row)}

    def score_encoded(self, user_enc, item_enc, pair_chunk=DEFAULT_PAIR_CHUNK):
        return deepcf_score_encoded(self.f_ml, self.fusion, user_enc, item_enc, pair_chunk)


def deepcf_score_encoded(f_ml, fusion, user_enc, item_enc, pair_chunk=DEFAULT_PAIR_CHUNK):
    """
    Shared [B, N] scorer for the concat-fusion DeepCF models (models 1 and
    2). Module-level rather than a method so `DeepCFRPUCB` can reuse it
    without inheriting from `DeepCF` -- the two are siblings under BaseCF,
    and making one the parent of the other would imply a specialisation
    relationship that does not hold (model 2 is not a DeepCF with extra
    behaviour; it is a separate model that happens to share a fusion).

    user_enc / item_enc: {"rl": [B|N, d], "ml": [B|N, d]}, already masked
    by the caller's encoder if the model masks at all. Chunked over items
    so the [B, C, d] outer products stay bounded.
    """
    p_rl, p_ml = user_enc["rl"], user_enc["ml"]
    q_rl, q_ml = item_enc["rl"], item_enc["ml"]

    B, N = p_rl.size(0), q_rl.size(0)
    out = torch.empty(B, N, device=p_rl.device, dtype=p_rl.dtype)

    for start, end in chunk_bounds(N, max(1, pair_chunk // max(B, 1))):
        C = end - start
        z_rl = p_rl.unsqueeze(1) * q_rl[start:end].unsqueeze(0)          # [B,C,d]
        pm = p_ml.unsqueeze(1).expand(B, C, -1)
        qm = q_ml[start:end].unsqueeze(0).expand(B, C, -1)
        z_ml = f_ml(torch.cat([pm, qm], dim=-1))                          # [B,C,d_ml]
        out[:, start:end] = fusion(torch.cat([z_rl, z_ml], dim=-1)).squeeze(-1)

    return out