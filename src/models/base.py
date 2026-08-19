import torch
import torch.nn as nn

def build_mlp(layer_sizes, dropout=0.0, activation=nn.ReLU):
    """
    Builds a sequential MLP.
    layer_sizes: list of ints, e.g. [input_dim, 256, 128, output_dim]
    Returns: nn.Sequential
    Each hidden layer: Linear → activation
    Final layer: Linear only (no activation)
    Dropout applied after each hidden activation if dropout > 0
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i+1]
        layers.append(nn.Linear(in_dim, out_dim))
        
        # Add activation and dropout for all but the last layer
        if i < len(layer_sizes) - 2:
            if activation is not None:
                layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
                
    return nn.Sequential(*layers)

class BaseCF(nn.Module):
    def score(self, user_row, item_col, user_ids=None):
        raise NotImplementedError

    def score_with_mask(self, user_row, item_col, user_ids):
        return self.score(user_row, item_col, user_ids), None

    def score_multi(self, user_row, item_col, user_ids, item_ids=None):
        """For K-embedding models: returns (scores_per_k [B, K], conditions).
        Default: wraps score() as a single-column K=1 tensor."""
        return self.score(user_row, item_col, user_ids).unsqueeze(1), None
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)