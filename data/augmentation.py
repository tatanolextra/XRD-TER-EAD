"""
data/augmentation.py
────────────────────────────────────────────────────
XRD signal augmentation — independent of crystal family.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from config import N_COPIES


def augment_xrd(pattern: np.ndarray, n_copies: int = N_COPIES) -> np.ndarray:
    """
    Generate n_copies physically-motivated variants of one XRD pattern.
    1. Gaussian noise     -> detector noise
    2. Peak broadening    -> Scherrer broadening (50% probability)
    3. Peak shift         -> angular calibration error (+/-2 grid points)
    4. Background slope   -> fluorescence background
    5. Intensity scaling  -> different acquisition times
    """
    augmented = []
    for _ in range(n_copies):
        p = pattern.copy()

        # 1. Gaussian noise
        noise = np.random.uniform(0.01, 0.03) * p.max()
        p += np.random.normal(0, noise, len(p))

        # 2. Peak broadening
        if np.random.random() > 0.5:
            p = gaussian_filter1d(p, sigma=np.random.uniform(0.5, 2.0))

        # 3. Peak shift
        shift = np.random.randint(-2, 3)
        if shift:
            p = np.roll(p, shift)

        # 4. Background slope
        p += np.linspace(0, np.random.uniform(0, 3), len(p))

        # 5. Intensity scaling
        p *= np.random.uniform(0.90, 1.10)

        # Normalize
        p = np.clip(p, 0, None)
        if p.max() > 0:
            p = p / p.max() * 100

        augmented.append(p)
    return np.array(augmented)


def make_augmented_dataset(X: np.ndarray,
                            y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Augment the full set, returns (X_aug, y_aug)."""
    X_aug, y_aug = [], []
    for xi, yi in zip(X, y):
        X_aug.append(xi); y_aug.append(yi)
        for aug in augment_xrd(xi):
            X_aug.append(aug); y_aug.append(yi)
    return np.array(X_aug), np.array(y_aug)
