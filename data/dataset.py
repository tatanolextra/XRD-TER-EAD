"""
data/dataset.py
────────────────────────────────────────────────────
PyTorch Dataset wrapper — independent of crystal family.
Preloads onto GPU when available, for faster training.
"""
import torch
from torch.utils.data import Dataset
from config import DEVICE


class XRDDataset(Dataset):
    """
    PyTorch Dataset for XRD patterns.
    If preload_gpu=True and CUDA is available:
        -> the entire tensor is moved to GPU once
        -> no per-batch host-to-device transfer -> ~3x faster
    """
    def __init__(self, X, y, preload_gpu: bool = True):
        X_tensor = torch.FloatTensor(X).unsqueeze(1)  # (N, 1, seq_len)
        y_tensor = torch.LongTensor(y)

        if preload_gpu and DEVICE.type == "cuda":
            self.X      = X_tensor.to(DEVICE)
            self.y      = y_tensor.to(DEVICE)
            self.on_gpu = True
        else:
            self.X      = X_tensor
            self.y      = y_tensor
            self.on_gpu = False

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
