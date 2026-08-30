"""
training/ensemble.py
────────────────────────────────────────────────────
Deep Ensemble: train N models independently,
inference = mean +/- std over the N predictions.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.trainer import train_one_model
from data.dataset import XRDDataset
from config import DEVICE, BATCH_SIZE, N_ENSEMBLE


def train_ensemble(train_X, train_y, n_classes: int):
    """Train N_ENSEMBLE models with different seeds."""
    models = []
    for i in range(N_ENSEMBLE):
        print(f"    Training model {i+1}/{N_ENSEMBLE}...", flush=True)
        m = train_one_model(train_X, train_y,
                             n_classes, seed=i * 7 + 42)
        models.append(m)
    return models


def ensemble_predict(models: list, X: np.ndarray):
    """
    Deep Ensemble inference.
    Returns:
        mean_probs : (N, n_classes) — mean probability
        entropy    : (N,)           — predictive entropy
        pred_class : (N,)           — predicted class index
    """
    dummy_y = np.zeros(len(X), dtype=int)
    dataset = XRDDataset(X, dummy_y, preload_gpu=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                          num_workers=0)
    all_probs = []

    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                if not dataset.on_gpu:
                    xb = xb.to(DEVICE)
                probs = torch.softmax(model(xb), dim=1)
                preds.append(probs.cpu().numpy())
        all_probs.append(np.concatenate(preds, axis=0))

    all_probs  = np.array(all_probs)           # (N_ens, N, n_cls)
    mean_probs = all_probs.mean(axis=0)        # (N, n_cls)
    entropy    = -np.sum(
        mean_probs * np.log(mean_probs + 1e-10), axis=1
    )                                           # (N,)
    pred_class = mean_probs.argmax(axis=1)     # (N,)

    return mean_probs, entropy, pred_class
