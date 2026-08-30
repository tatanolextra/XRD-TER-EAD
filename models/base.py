"""models/base.py — Abstract base class for all models."""
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Every model must inherit from this class and implement forward().
    If a model supports attention visualization:
        override supports_attention = True
    """

    @abstractmethod
    def forward(self, x, return_attn: bool = False):
        """
        x: (batch, 1, seq_len)
        return_attn=True  -> returns (logits, attn_list)
        return_attn=False -> returns logits
        """
        pass

    @property
    def supports_attention(self) -> bool:
        return False
