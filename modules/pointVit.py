import torch
import torch.nn as nn
from functools import partial
from torch.nn import LayerNorm

from .encoders.pointVitEncoder import Encoder
from .voxelization.cloudVoxelizer import voxelize
from .mlps.simpleMlp import FeedForward

class PointViT(nn.Module):
    def __init__(
        self,
        *,
        voxelFeatureDims=224,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth_enc=12,
        num_heads_enc=12,
        ff_hidden_dims=[256, 512, 512 , 256],
        drop_rate=0.1
    ):
        super().__init__()


        # 1) Class token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout   = nn.Dropout(drop_rate)
        # self.positionEmbedder = 

        # 2) Stacked transformer blocks
        self.pipeline = []
        for _ in range(depth_enc):
            self.pipeline += [
                Encoder(
                    dim=embed_dim,
                    num_heads=num_heads_enc,
                    ff_hidden_dims=ff_hidden_dims,
                    drop=drop_rate
                )
            ]
                
        # self.pipeline += [FeedForward(input_dim=voxel, hidden_dims=ff_hidden_dims, activation=nn.GELU, dropout=0.1)]

        # 3) Final norm & head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # init (you can customize your own init scheme)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)                          # [B, N, E]
        cls_tokens = self.cls_token.expand(B, -1, -1)    # [B, 1, E]
        x = torch.cat([cls_tokens, x], dim=1)            # [B, 1+N, E]
        x = x + self.pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                                # take class token
        return self.head(cls_out)
