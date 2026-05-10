import torch
import torch.nn as nn

class SelfAttentionInteraction(nn.Module):
    def __init__(self, embed_dim, num_heads=2, dropout=0.0, output_dim=None):
        super().__init__()
            
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        if output_dim is not None:
            self.output_proj = nn.Linear(2 * embed_dim, output_dim)
        else:
            self.output_proj = None
        
    def forward(self, p_u_masked, q_i):
        # Stack as token sequence: shape [batch, 2, embed_dim]
        tokens = torch.stack([p_u_masked, q_i], dim=1)
        
        # Self-attention: each token attends to both tokens
        attn_out, _ = self.attn(tokens, tokens, tokens)   # [batch, 2, embed_dim]
        
        # Residual + LayerNorm
        attn_out = self.norm(attn_out + tokens)
        
        # Flatten: [batch, 2 * embed_dim]
        attn_out = attn_out.view(attn_out.size(0), -1)
        
        if self.output_proj is not None:
            return self.output_proj(attn_out)    # [batch, output_dim]
        else:
            return attn_out
