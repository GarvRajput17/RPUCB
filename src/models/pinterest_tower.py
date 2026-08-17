"""
Pinterest-style tower components (Stage A, new files only).

Implements the pieces of the Pinterest paper's architecture that the
gameplan's Fork 1 locked in: a real rebuild of the tower (MLP
summarization + lite DHEN feature crossing), rather than reusing DeepCF's
CFNet-rl / CFNet-ml dual-branch fusion.

Reference points in the paper:
  - Eq. 4                : MLP summarization layer
  - Appendix A            : DHEN feature-crossing recipe (2 hierarchies,
                             each a sum of parallel sub-modules)
  - Sec 3.4               : "Both implicit and explicit interest modeling
                             share the same architecture in each tower" --
                             this is why PinterestTower is used identically
                             to build both the user tower and the item
                             tower, with separate parameter instances.

Simplifications relative to the paper, documented inline where they occur:
  - DHEN's Appendix A uses a 256-dim / 4-head transformer and a 4-block
    MaskNet; we use much smaller sizes appropriate to our embedding scale
    (embed_dim=64, or 448 for the matched-capacity runs) and a single
    MaskNet block instead of four.
  - "Feature fields" for the within-tower transformer are constructed by
    splitting the summarized embedding into `n_fields` equal chunks, since
    our datasets don't carry Pinterest's separate named feature fields
    (pretrained embeddings, categorical inputs, etc.) -- we only have one
    interaction-profile vector per user/item to begin with.
"""

import torch
import torch.nn as nn


class MLPSummarization(nn.Module):
    """
    Pinterest Eq. 4:
        e_i = W2^T( GELU( W1^T · concat(f_i1, ..., f_iN) ) )

    A 2-layer GELU MLP mapping a raw input feature vector into a
    fixed-size embedding. In our setup the "input features" are a user's
    or item's normalized interaction-profile vector (same convention
    DeepCF uses for `user_row` / `item_col`), not Pinterest's pretrained +
    categorical Pin features.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class LiteMaskNet(nn.Module):
    """
    Lightweight stand-in for Wang et al.'s MaskNet, used inside DHEN's
    second hierarchy: an instance-guided mask gates the input feature-wise
    before a linear projection, with a residual connection + LayerNorm.

    Simplification: the paper's DHEN uses 4 parallel MaskNet blocks; this
    is a single-block version, appropriate given our much smaller
    embedding dimensionality.
    """

    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.mask_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
        )
        self.feat_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        mask = torch.sigmoid(self.mask_proj(x))
        gated = x * mask
        out = self.feat_proj(gated)
        return self.norm(out + x)


class LiteDHEN(nn.Module):
    """
    Lightweight DHEN (Zhang et al. 2022) feature-crossing block, following
    the recipe Pinterest describes in Appendix A but scaled down:

        Hierarchy 1: small Transformer encoder over `n_fields` chunks of
                     the embedding, summed with a parallel 2-layer MLP.
        Hierarchy 2: LiteMaskNet block, summed with a parallel 2-layer MLP.

    This module performs *within-tower* feature crossing -- i.e. it is
    applied identically to the user tower's own embedding and the item
    tower's own embedding, separately. It is NOT the cross-tower
    user/item interaction (that stays as SelfAttentionInteraction, reused
    unchanged from the existing codebase, per Fork 3).

    `dim` must be divisible by `n_fields`.
    """

    def __init__(self, dim, n_fields=4, n_heads=2, transformer_layers=2,
                 dropout=0.0):
        super().__init__()
        assert dim % n_fields == 0, (
            f"embed dim ({dim}) must be divisible by n_fields ({n_fields})"
        )
        self.n_fields = n_fields
        self.field_dim = dim // n_fields

        # ── Hierarchy 1: field-wise transformer + parallel MLP ─────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.field_dim,
            nhead=n_heads,
            dim_feedforward=max(4 * self.field_dim, 8),
            dropout=dropout,
            batch_first=True,
        )
        self.field_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.h1_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(dim, dim),
        )
        self.h1_norm = nn.LayerNorm(dim)

        # ── Hierarchy 2: MaskNet + parallel MLP ─────────────────────────
        self.masknet = LiteMaskNet(dim, dropout=dropout)
        self.h2_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(dim, dim),
        )
        self.h2_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, dim]
        B = x.size(0)

        fields = x.view(B, self.n_fields, self.field_dim)
        cross = self.field_transformer(fields).reshape(B, -1)
        h1 = self.h1_norm(cross + self.h1_mlp(x))

        h2 = self.h2_norm(self.masknet(h1) + self.h2_mlp(h1))
        return h2


class PinterestTower(nn.Module):
    """
    Full Pinterest-style tower: MLP summarization (Eq. 4) -> LiteDHEN
    feature crossing (Appendix A). Used identically for the user side and
    the item side (Sec 3.4) -- instantiate two separate copies, one per
    side, as pinterest_base.py / pinterest_dcm.py do.

    input_dim  : raw feature width (num_items for the user tower,
                 num_users for the item tower -- same convention DeepCF
                 uses for f_rl_user / f_rl_item)
    embed_dim  : output embedding width. Must be divisible by n_fields.
    """

    def __init__(self, input_dim, embed_dim=64, summarization_hidden=256,
                 n_fields=4, n_heads=2, transformer_layers=2, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.summarize = MLPSummarization(
            input_dim, summarization_hidden, embed_dim, dropout=dropout
        )
        self.cross = LiteDHEN(
            embed_dim, n_fields=n_fields, n_heads=n_heads,
            transformer_layers=transformer_layers, dropout=dropout,
        )

    def forward(self, x):
        e = self.summarize(x)
        return self.cross(e)