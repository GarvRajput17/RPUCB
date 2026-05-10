import math

import torch
import torch.nn as nn

from .base import BaseCF, build_mlp


class RPUCBUserItemPretrained(BaseCF):
    """
    User+item RP-UCB model for the pretrained experiment path.

    Pretraining uses the same backbone with confidence modulation disabled.
    Fine-tuning enables the user and item masks from an identity prior.
    """

    def __init__(
        self,
        num_users,
        num_items,
        embed_dim=64,
        hidden_layers=None,
        user_interaction_counts=None,
        item_interaction_counts=None,
        dropout=0.0,
        gamma_init=0.1,
        beta=1.0,
        identity_mask_value=0.95,
        pretrain_mode=False,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 256, 128, 64]

        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.beta = beta
        self.pretrain_mode = pretrain_mode
        self.identity_mask_value = identity_mask_value

        self.user_embedding = nn.Linear(num_items, embed_dim, bias=False)
        self.item_embedding = nn.Linear(num_users, embed_dim, bias=False)
        self.matching = build_mlp([2 * embed_dim] + hidden_layers, dropout=dropout)
        self.output = nn.Linear(hidden_layers[-1], 1)

        self.user_mask_embed = nn.Embedding(num_users, embed_dim)
        self.item_mask_embed = nn.Embedding(num_items, embed_dim)
        self.user_gamma = nn.Parameter(torch.full((embed_dim,), float(gamma_init)))
        self.item_gamma = nn.Parameter(torch.full((embed_dim,), float(gamma_init)))

        if user_interaction_counts is None:
            user_interaction_counts = torch.ones(num_users, dtype=torch.long)
        if item_interaction_counts is None:
            item_interaction_counts = torch.ones(num_items, dtype=torch.long)

        self.register_buffer('user_counts', user_interaction_counts)
        self.register_buffer('item_counts', item_interaction_counts)
        self.n_bar_user = max(1.0, self.user_counts.float().mean().item())
        self.n_bar_item = max(1.0, self.item_counts.float().mean().item())

        self.init_weights()
        self.init_identity_masks(identity_mask_value, gamma_init)

    def init_identity_masks(self, value=0.95, gamma_init=0.1):
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        logit = math.log(value / (1.0 - value))
        nn.init.constant_(self.user_mask_embed.weight, logit)
        nn.init.constant_(self.item_mask_embed.weight, logit)
        nn.init.constant_(self.user_gamma, float(gamma_init))
        nn.init.constant_(self.item_gamma, float(gamma_init))

    def set_pretrain_mode(self, enabled):
        self.pretrain_mode = bool(enabled)

    def freeze_backbone(self):
        for module in (self.user_embedding, self.item_embedding, self.matching):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def mask_parameters(self):
        yield from self.user_mask_embed.parameters()
        yield from self.item_mask_embed.parameters()
        yield self.user_gamma
        yield self.item_gamma
        yield from self.output.parameters()

    def _rpucb_mask(self, ids, embed_layer, counts_buf, n_bar, gamma_param):
        if self.pretrain_mode:
            return torch.ones(ids.size(0), self.embed_dim, device=ids.device)

        w = embed_layer(ids)
        counts = counts_buf[ids].float().clamp(min=1.0)
        explore_scalar = torch.clamp(
            torch.log(torch.tensor(n_bar, device=w.device) / counts),
            min=0.0,
        )
        explore_vec = torch.sigmoid(gamma_param).unsqueeze(0) * explore_scalar.unsqueeze(1)
        return (torch.sigmoid(w) + self.beta * explore_vec).clamp(0.0, 1.0)

    def get_user_mask(self, user_ids):
        return self._rpucb_mask(
            user_ids, self.user_mask_embed,
            self.user_counts, self.n_bar_user, self.user_gamma,
        )

    def get_item_mask(self, item_ids):
        return self._rpucb_mask(
            item_ids, self.item_mask_embed,
            self.item_counts, self.n_bar_item, self.item_gamma,
        )

    def forward(self, user_row, item_col, user_ids, item_ids=None):
        return self.score_with_mask(user_row, item_col, user_ids, item_ids)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        scores, _ = self.score_with_mask(user_row, item_col, user_ids, item_ids)
        return scores

    def score_with_mask(self, user_row, item_col, user_ids, item_ids=None):
        user_vec = self.user_embedding(user_row)
        item_vec = self.item_embedding(item_col)

        if item_ids is None:
            item_mask = torch.ones_like(item_vec)
        else:
            item_mask = self.get_item_mask(item_ids)
        user_mask = self.get_user_mask(user_ids)

        user_vec = user_vec * user_mask
        item_vec = item_vec * item_mask
        z = self.matching(torch.cat([user_vec, item_vec], dim=1))
        scores = self.output(z).squeeze(-1)
        return scores, (user_mask, item_mask)
