# import torch
import torch.nn as nn
# from functools import partial
from torch.nn import LayerNorm
from torch import Tensor
from ..mlps.simpleMlp import FeedForward

class Encoder(nn.Module):
    """
    One ViT block:
       x → LN → MHA → +residual  → FeedForward → +residual
    """
    def __init__(self, 
                 dim: int, 
                 num_heads: int, 
                 ff_hidden_dims: list[int] | list[Tensor], 
                 drop: float =0.1
                ):
        '''
            dim: input embedding dimention (D)
            num_heads: number of heads in multiheaded cross attention
            ff_hiden_dims: feed forward model's hiden dims.
            drop: dropout
        '''
        super().__init__()
        self.norm_input = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=drop, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(dim)
        self.ff    = FeedForward(dim, ff_hidden_dims, dropout=drop)

    def forward(self, x):
        # Self‐attention block
        norm_x = self.norm_input(x)
        y, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + y # add residual connection
        # Feed-forward block
        x = x + self.ff(self.norm_attn(x))
        return x