"""models/conformer.py — CNN + Transformer hybrid for 1D XRD patterns."""
import torch.nn as nn
from models.base import BaseModel


class ConformerBlock(nn.Module):
    """
    Local (CNN) and global (attention) modeling within one block.
    Suited to XRD: individual peaks are local features, while correlations
    between peaks are global.
    """
    def __init__(self, channels: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # CNN branch: local peak features
        self.cnn = nn.Sequential(
            nn.Conv1d(channels, channels,
                      kernel_size=31, padding=15, groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        # Attention branch: global relationships
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads,
                                           dropout=dropout,
                                           batch_first=True)
        self.ff   = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn: bool = False):
        # x: (B, L, C)
        cnn_out = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop(cnn_out)

        normed = self.norm(x)
        attn_out, attn_w = self.attn(normed, normed, normed,
                                      need_weights=return_attn,
                                      average_attn_weights=True)
        x = x + self.drop(attn_out)
        x = x + self.ff(x)
        return x, attn_w


class Conformer(BaseModel):
    def __init__(self, n_classes: int, channels: int = 128,
                 n_heads: int = 4, n_layers: int = 4,
                 dropout: float = 0.1, seq_len: int = 1000):
        super().__init__()
        # CNN stem: seq_len -> seq_len/4
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels//2, kernel_size=11, padding=5),
            nn.BatchNorm1d(channels//2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(channels//2, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels), nn.GELU(), nn.MaxPool1d(2),
        )
        self.blocks = nn.ModuleList([
            ConformerBlock(channels, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm       = nn.LayerNorm(channels)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, n_classes),
        )

    @property
    def supports_attention(self) -> bool:
        return True

    def forward(self, x, return_attn: bool = False):
        x        = self.stem(x).transpose(1, 2)   # (B, seq_len/4, C)
        attn_all = []
        for block in self.blocks:
            x, w = block(x, return_attn=return_attn)
            if return_attn and w is not None:
                attn_all.append(w)
        x      = self.norm(x).mean(dim=1)          # global average pool
        x      = self.dropout(x)
        logits = self.classifier(x)
        return (logits, attn_all) if return_attn else logits
