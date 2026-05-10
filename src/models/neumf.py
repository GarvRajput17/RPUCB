import torch
import torch.nn as nn
from .base import BaseCF, build_mlp


class NeuMF(BaseCF):
    """
    Neural Matrix Factorisation (He et al., 2017).

    Combines:
      - GMF branch:  element-wise product of user / item latent embeddings
      - MLP branch:  concatenation of separate user / item embeddings fed
                     through a deep MLP
    Both branches are fused with a final linear layer → scalar score.

    Args:
        num_users    : total number of users
        num_items    : total number of items
        embed_dim    : latent dimensionality used for *both* GMF and MLP branches
        mlp_layers   : hidden layer sizes for the MLP branch
                       (input is 2*embed_dim, output of last layer feeds fusion)
        dropout      : dropout probability applied after each MLP hidden layer
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        mlp_layers: list = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 128, 64]

        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        # ── GMF branch ──────────────────────────────────────────────────────
        self.gmf_user_embed = nn.Embedding(num_users, embed_dim)
        self.gmf_item_embed = nn.Embedding(num_items, embed_dim)

        # ── MLP branch ──────────────────────────────────────────────────────
        self.mlp_user_embed = nn.Embedding(num_users, embed_dim)
        self.mlp_item_embed = nn.Embedding(num_items, embed_dim)
        self.mlp = build_mlp([2 * embed_dim] + mlp_layers, dropout=dropout)

        # ── Fusion ──────────────────────────────────────────────────────────
        # GMF output dim  = embed_dim
        # MLP output dim  = mlp_layers[-1]
        self.fusion = nn.Linear(embed_dim + mlp_layers[-1], 1)

        self.init_weights()

    # ------------------------------------------------------------------
    # forward / score
    # ------------------------------------------------------------------
    def forward(self, user_row, item_col, user_ids):
        """
        NeuMF does not use the raw interaction rows / columns;
        it uses the explicit user_ids to look up learned embeddings.

        user_row  : [B, num_items]  – passed for API compatibility, unused
        item_col  : [B, num_users]  – passed for API compatibility, unused
        user_ids  : [B]             – integer user indices
        """
        return self.score(user_row, item_col, user_ids)

    def score(self, user_row, item_col, user_ids=None):
        # We need item_ids to look up item embeddings.
        # Recover them from item_col (the one-hot / normalised column vector):
        # item_col is already the column of the interaction matrix for each
        # item, but NeuMF needs the actual item index.
        # We store item_ids as a side-channel in score(); the caller (train /
        # evaluate) always passes user_ids; for item IDs we derive them from
        # the batch structure used elsewhere.  To keep the API contract, we
        # expose a second entry-point `score_neumf`.
        raise RuntimeError(
            "Call score_neumf(user_ids, item_ids) directly for NeuMF."
        )

    def score_neumf(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Primary scoring method for NeuMF.

        Args:
            user_ids : [B] – LongTensor of user indices
            item_ids : [B] – LongTensor of item indices

        Returns:
            scores   : [B] – scalar relevance scores
        """
        # GMF
        p_gmf = self.gmf_user_embed(user_ids)   # [B, d]
        q_gmf = self.gmf_item_embed(item_ids)   # [B, d]
        z_gmf = p_gmf * q_gmf                   # [B, d]

        # MLP
        p_mlp = self.mlp_user_embed(user_ids)   # [B, d]
        q_mlp = self.mlp_item_embed(item_ids)   # [B, d]
        z_mlp = self.mlp(torch.cat([p_mlp, q_mlp], dim=1))  # [B, mlp_layers[-1]]

        # Fusion
        z = torch.cat([z_gmf, z_mlp], dim=1)   # [B, d + mlp_layers[-1]]
        return self.fusion(z).squeeze(-1)        # [B]
