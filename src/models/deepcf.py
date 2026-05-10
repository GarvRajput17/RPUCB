import torch
import torch.nn as nn
from .base import BaseCF, build_mlp

class DeepCF(BaseCF):
    def __init__(self, num_users, num_items, embed_dim=64, rl_layers=[512, 256, 128, 64], ml_layers=[512, 256, 128, 64], dropout=0.0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        
        # CFNet-rl branch
        self.f_rl_user = build_mlp([num_items] + rl_layers, dropout=dropout)
        self.f_rl_item = build_mlp([num_users] + rl_layers, dropout=dropout)
        
        # CFNet-ml branch
        self.user_embedding = nn.Linear(num_items, embed_dim, bias=False)  # P^T
        self.item_embedding = nn.Linear(num_users, embed_dim, bias=False)  # Q^T
        self.f_ml = build_mlp([2 * embed_dim] + ml_layers, dropout=dropout)
        
        # Fusion
        self.fusion = nn.Linear(2 * embed_dim, 1)
        
        self.init_weights()

    def forward(self, user_row, item_col, user_ids=None, item_ids=None):
        return self.score(user_row, item_col, user_ids)
        
    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        # CFNet-rl
        p_u_rl = self.f_rl_user(user_row)
        q_i_rl = self.f_rl_item(item_col)
        z_rl = p_u_rl * q_i_rl
        
        # CFNet-ml
        p_u_ml = self.user_embedding(user_row)
        q_i_ml = self.item_embedding(item_col)
        concat_ml = torch.cat([p_u_ml, q_i_ml], dim=1)
        z_ml = self.f_ml(concat_ml)
        
        # Fusion
        z = torch.cat([z_rl, z_ml], dim=1)
        scores = self.fusion(z).squeeze(-1)
        return scores
