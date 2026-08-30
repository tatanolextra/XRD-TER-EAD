# models package
import config
from config import MODEL_CONFIGS
from models.cnn1d import CNN1D
from models.spectral_transformer import SpectralTransformer
from models.conformer import Conformer

_REGISTRY = {
    "cnn1d":                CNN1D,
    "spectral_transformer": SpectralTransformer,
    "conformer":            Conformer,
}


def get_model(n_classes: int, model_name: str = None):
    """
    Factory — switch models by editing MODEL_NAME in config.py.
    model_name=None -> reads config.MODEL_NAME AT CALL TIME
    (not a static default argument, to avoid the value getting
    "frozen" at module import time).
    """
    if model_name is None:
        model_name = config.MODEL_NAME

    if model_name not in _REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found.\n"
            f"Available: {list(_REGISTRY.keys())}"
        )
    cfg   = MODEL_CONFIGS.get(model_name, {})
    model = _REGISTRY[model_name](n_classes=n_classes, **cfg)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Params: {total:,}")
    return model
