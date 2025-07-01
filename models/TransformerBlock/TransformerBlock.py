import torch
import torch.nn as nn
from functools import partial
from torch.nn import LayerNorm



class FeedForward(nn.Module):
    """
    A customizable MLP: you pass a list of hidden dims, and it'll
    chain [Linear → Activation → Dropout] for each, ending by projecting
    back to `dim`.
    z0  = [xclass, x1, x2, · · · , xN] ; xi ∈ R^(D), z0  ∈ R^((N+1)xD) , where N is the number of voxels in scene.
    zl' = MSA(LN(zl-1)) + zl-1,  l= 1 . . . L (2)
    zl  = MLP(LN(zl')) + zl-1 ,   l= 1 . . . L (3)
    y = LN(zL)
    """
    def __init__(self, input_dim, hidden_dims, activation=nn.GELU, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        self.LN = LayerNorm(normalized_shape=input_dim) # normalize just the 'D' part of the inputs...
        # Layer Norm!!

        layers += self.LN # add an initial norm 

        for h in hidden_dims: # MLP
            layers += [
                nn.Linear(in_dim, h),
                activation(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        # final projection back to `dim`
        layers += [
            nn.Linear(in_dim, input_dim),
            nn.Dropout(dropout),
        ]
        self.feedForward = nn.Sequential(*layers)

    def forward(self, x):
        '''
        x:  (batch, (N+1),D)
        '''
        return self.feedForward(x) + x # add prev to preserve context


class TransformerBlock(nn.Module):
    """
    One ViT block:
       x → LN → MHA → +residual → LN → FeedForward → +residual
    """
    def __init__(self, dim, num_heads, ff_hidden_dims, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = FeedForward(dim, ff_hidden_dims, drop=drop)

    def forward(self, x):
        # Self‐attention block
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + y
        # Feed-forward block
        x = x + self.ff(self.norm2(x))
        return x