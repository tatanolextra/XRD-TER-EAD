"""evaluation/metrics.py — Accuracy, F1, and entropy statistics."""
import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    classes) -> dict:
    acc = float((y_pred == y_true).mean() * 100)
    f1  = float(f1_score(y_true, y_pred,
                          average="macro", zero_division=0))
    return {"acc": acc, "f1": f1}


def compute_per_class_f1(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          classes) -> dict:
    """Per-class F1 score."""
    scores = f1_score(y_true, y_pred,
                       labels=list(range(len(classes))),
                       average=None, zero_division=0)
    return {cls: float(s) for cls, s in zip(classes, scores)}


def compute_entropy_stats(entropy: np.ndarray,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           classes) -> dict:
    """Mean predictive entropy per class, split by correct vs incorrect."""
    stats = {}
    for i, cls in enumerate(classes):
        mask = y_true == i
        if mask.sum() > 0:
            stats[cls] = {
                "mean_entropy": float(entropy[mask].mean()),
                "acc":          float((y_pred[mask] == i).mean()),
                "n_samples":    int(mask.sum()),
            }
    return stats
