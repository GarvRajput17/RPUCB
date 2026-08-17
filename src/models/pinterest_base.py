"""
PinterestBase -- Pinterest-style tower, single embedding, unmasked.

This is the *architecture control* in the gameplan's target table (§6):
it isolates "does the new tower help" from "does RP-UCB masking help"
(pinterest_base_rpucb) and "does multiplicity help" (pinterest_dcm).
Run at embed_dim=d only -- the K*d unmasked variant was cut from the
gameplan as redundant once the masked K*d row exists.
"""

import torch.nn as nn
from .base import BaseCF
from .pinterest_tower import PinterestTower
from .attention import SelfAttentionInteraction


class PinterestBase(BaseCF):
    """
    Single user embedding, single item embedding, both produced by the
    Pinterest tower (MLP summarization + lite DHEN feature crossing --
    see pinterest_tower.py) instead of DeepCF's CFNet-rl/CFNet-ml dual
    branch fusion (Fork 1's locked decision: full rebuild, not a bolt-on).

    User and item embeddings are combined via the existing
    SelfAttentionInteraction module (same one rpucb_attn_full.py already
    uses), so the scoring function is held constant across every new
    model built for this experiment (Fork 3's locked decision) -- any
    performance difference between models in this experiment should come
    from the tower/masking/multiplicity design, not from a weaker or
    stronger scorer being used unevenly.
    """

    def __init__(self, num_users, num_items, embed_dim=64,
                 summarization_hidden=256, n_fields=4, n_heads_dhen=2,
                 transformer_layers=2, attn_heads=2, dropout=0.0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        self.user_tower = PinterestTower(
            input_dim=num_items, embed_dim=embed_dim,
            summarization_hidden=summarization_hidden, n_fields=n_fields,
            n_heads=n_heads_dhen, transformer_layers=transformer_layers,
            dropout=dropout,
        )
        self.item_tower = PinterestTower(
            input_dim=num_users, embed_dim=embed_dim,
            summarization_hidden=summarization_hidden, n_fields=n_fields,
            n_heads=n_heads_dhen, transformer_layers=transformer_layers,
            dropout=dropout,
        )

        self.interaction = SelfAttentionInteraction(
            embed_dim, num_heads=attn_heads, dropout=dropout, output_dim=None
        )
        self.fusion = nn.Linear(2 * embed_dim, 1)

        self.init_weights()

    def forward(self, user_row, item_col, user_ids=None, item_ids=None):
        return self.score(user_row, item_col, user_ids, item_ids)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        p_u = self.user_tower(user_row)          # [B, d]
        q_i = self.item_tower(item_col)           # [B, d]
        z = self.interaction(p_u, q_i)            # [B, 2d]
        return self.fusion(z).squeeze(-1)