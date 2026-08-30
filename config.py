"""
config.py
────────────────────────────────────────────────────
All hyperparameters and settings.

To switch to a different crystal family:
    ACTIVE_FAMILY = "garnet"   # "spinel" | "garnet" | "scheelite"

No other file needs to be modified.
"""
import os
import torch

# ── SELECT FAMILY HERE ────────────────────────────────────────────
ACTIVE_FAMILY = "spinel"
# ACTIVE_FAMILY = "garnet"
# ACTIVE_FAMILY = "scheelite"

# ── SELECT MODEL HERE ─────────────────────────────────────────────
MODEL_NAME = "conformer"
# MODEL_NAME = "cnn1d"
# MODEL_NAME = "spectral_transformer"

# ── Paths (subfolders are created automatically per family) ───────
BASE_DIR      = "output"
DATA_DIR      = os.path.join(BASE_DIR, ACTIVE_FAMILY, "dataset")
RESULTS_DIR   = os.path.join(BASE_DIR, ACTIVE_FAMILY, "results")
XRD_NPY       = os.path.join(DATA_DIR, "dataset_xrd.npy")
METADATA_JSON = os.path.join(DATA_DIR, "metadata.json")

for _dir in [DATA_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Signal ──────────────────────────────────────────────────────
SEQ_LEN       = 1000
TWO_THETA_MIN = 10
TWO_THETA_MAX = 90

# ── Augmentation ────────────────────────────────────────────────
N_COPIES      = 20

# ── Training ────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 64
EPOCHS        = 80
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
N_FOLDS       = 5
N_ENSEMBLE    = 3
RANDOM_SEED   = 42

# ── Model hyperparameters ──────────────────────────────────────
MODEL_CONFIGS = {
    "spectral_transformer": {
        "patch_size": 20,
        "embed_dim":  128,
        "n_heads":    4,
        "n_layers":   4,
        "ff_dim":     256,
        "dropout":    0.1,
    },
    "cnn1d": {
        "dropout": 0.5,
    },
    "conformer": {
        "channels":  128,
        "n_heads":   4,
        "n_layers":  4,
        "dropout":   0.1,
    },
}
