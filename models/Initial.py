import torch
import torch.nn as nn
from functools import partial
from torch.nn import LayerNorm




class PatchEmbedding(nn.Module):
    """
    Turn an image into a sequence of patch‐tokens.
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x: [B, C, H, W] → [B, E, H/ps, W/ps]
        x = self.proj(x)
        # flatten to [B, num_patches, E]
        return x.flatten(2).transpose(1, 2)


class PointViT(nn.Module):
    def __init__(
        self,
        *,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ff_hidden_dims=[3072],    # if you want a single 2-layer MLP like ViT
                                   # you can set [4*embed_dim]; for more layers,
                                   # pass e.g. [4*embed_dim, 4*embed_dim//2]
        drop_rate=0.1
    ):
        super().__init__()

        # 1) Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_chans, embed_dim
        )
        # 2) Class token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )
        self.dropout   = nn.Dropout(drop_rate)

        # 3) Stacked transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dims, drop_rate)
            for _ in range(depth)
        ])

        # 4) Final norm & head
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
