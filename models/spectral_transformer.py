"""models/spectral_transformer.py — Patch-based Transformer for 1D XRD."""
import torch
import torch.nn as nn
from models.base import BaseModel


class PatchEmbedding(nn.Module):
    def __init__(self, seq_len: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = seq_len // patch_size
        self.projection = nn.Linear(patch_size, embed_dim)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(
            torch.randn(1, self.n_patches + 1, embed_dim) * 0.02
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.squeeze(1).reshape(B, self.n_patches, self.patch_size)
        x = self.projection(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int,
                 ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, n_heads,
                                            dropout=dropout,
                                            batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x, return_attn: bool = False):
        normed   = self.norm1(x)
        out, attn_w = self.attn(normed, normed, normed,
                                 need_weights=return_attn,
                                 average_attn_weights=True)
        x = x + out + self.ff(self.norm2(x + out))
        return x, attn_w


class SpectralTransformer(BaseModel):
    def __init__(self, n_classes: int, seq_len: int = 1000,
                 patch_size: int = 20, embed_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 4,
                 ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(seq_len, patch_size, embed_dim)
        self.blocks      = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(embed_dim, n_classes)

    @property
    def supports_attention(self) -> bool:
        return True

    def forward(self, x, return_attn: bool = False):
        x        = self.patch_embed(x)
        attn_all = []
        for block in self.blocks:
            x, w = block(x, return_attn=return_attn)
            if return_attn and w is not None:
                attn_all.append(w)
        cls    = self.dropout(self.norm(x)[:, 0])
        logits = self.head(cls)
        return (logits, attn_all) if return_attn else logits
