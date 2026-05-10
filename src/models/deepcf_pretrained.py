import torch
import torch.nn as nn

from .base import BaseCF, build_mlp


class DeepCFRLPretrain(BaseCF):
    """CFNet-rl pretraining module used only by the pretrained DeepCF run."""

    def __init__(self, num_users, num_items, embed_dim=64,
                 rl_layers=None, dropout=0.0):
        super().__init__()
        if rl_layers is None:
            rl_layers = [512, 256, 128, 64]

        self.f_rl_user = build_mlp([num_items] + rl_layers, dropout=dropout)
        self.f_rl_item = build_mlp([num_users] + rl_layers, dropout=dropout)
        self.output = nn.Linear(rl_layers[-1], 1)
        self.init_weights()

    def forward(self, user_row, item_col, user_ids=None, item_ids=None):
        return self.score(user_row, item_col)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        p_u = self.f_rl_user(user_row)
        q_i = self.f_rl_item(item_col)
        return self.output(p_u * q_i).squeeze(-1)


class DeepCFMLPretrain(BaseCF):
    """CFNet-ml pretraining module used only by the pretrained DeepCF run."""

    def __init__(self, num_users, num_items, embed_dim=64,
                 ml_layers=None, dropout=0.0):
        super().__init__()
        if ml_layers is None:
            ml_layers = [512, 256, 128, 64]

        self.user_embedding = nn.Linear(num_items, embed_dim, bias=False)
        self.item_embedding = nn.Linear(num_users, embed_dim, bias=False)
        self.f_ml = build_mlp([2 * embed_dim] + ml_layers, dropout=dropout)
        self.output = nn.Linear(ml_layers[-1], 1)
        self.init_weights()

    def forward(self, user_row, item_col, user_ids=None, item_ids=None):
        return self.score(user_row, item_col)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        p_u = self.user_embedding(user_row)
        q_i = self.item_embedding(item_col)
        z_ml = self.f_ml(torch.cat([p_u, q_i], dim=1))
        return self.output(z_ml).squeeze(-1)


class DeepCFPretrained(BaseCF):
    """
    DeepCF architecture used by the pretrained experiment path.

    This intentionally lives outside deepcf.py so the non-pretrained baseline
    remains a separate implementation and result path.
    """

    def __init__(self, num_users, num_items, embed_dim=64,
                 rl_layers=None, ml_layers=None, dropout=0.0):
        super().__init__()
        if rl_layers is None:
            rl_layers = [512, 256, 128, 64]
        if ml_layers is None:
            ml_layers = [512, 256, 128, 64]

        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.rl_out_dim = rl_layers[-1]
        self.ml_out_dim = ml_layers[-1]

        self.f_rl_user = build_mlp([num_items] + rl_layers, dropout=dropout)
        self.f_rl_item = build_mlp([num_users] + rl_layers, dropout=dropout)

        self.user_embedding = nn.Linear(num_items, embed_dim, bias=False)
        self.item_embedding = nn.Linear(num_users, embed_dim, bias=False)
        self.f_ml = build_mlp([2 * embed_dim] + ml_layers, dropout=dropout)

        self.fusion = nn.Linear(self.rl_out_dim + self.ml_out_dim, 1)
        self.init_weights()

    def init_from_pretrain(self, rl_model, ml_model):
        self.f_rl_user.load_state_dict(rl_model.f_rl_user.state_dict())
        self.f_rl_item.load_state_dict(rl_model.f_rl_item.state_dict())
        self.user_embedding.load_state_dict(ml_model.user_embedding.state_dict())
        self.item_embedding.load_state_dict(ml_model.item_embedding.state_dict())
        self.f_ml.load_state_dict(ml_model.f_ml.state_dict())

        with torch.no_grad():
            self.fusion.weight[:, :self.rl_out_dim].copy_(0.5 * rl_model.output.weight)
            self.fusion.weight[:, self.rl_out_dim:].copy_(0.5 * ml_model.output.weight)
            self.fusion.bias.copy_(0.5 * (rl_model.output.bias + ml_model.output.bias))

    def forward(self, user_row, item_col, user_ids=None, item_ids=None):
        return self.score(user_row, item_col)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        p_u_rl = self.f_rl_user(user_row)
        q_i_rl = self.f_rl_item(item_col)
        z_rl = p_u_rl * q_i_rl

        p_u_ml = self.user_embedding(user_row)
        q_i_ml = self.item_embedding(item_col)
        z_ml = self.f_ml(torch.cat([p_u_ml, q_i_ml], dim=1))

        z = torch.cat([z_rl, z_ml], dim=1)
        return self.fusion(z).squeeze(-1)
