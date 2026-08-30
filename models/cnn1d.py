"""models/cnn1d.py — 1D CNN baseline."""
import torch.nn as nn
from models.base import BaseModel


class CNN1D(BaseModel):
    def __init__(self, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1,   32, 7, padding=3), nn.BatchNorm1d(32),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,  64, 5, padding=2), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128,256, 3, padding=1), nn.BatchNorm1d(256),
            nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*8, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x, return_attn: bool = False):
        logits = self.classifier(self.features(x))
        return (logits, None) if return_attn else logits
