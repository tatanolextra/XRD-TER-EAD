"""
training/trainer.py
────────────────────────────────────────────────────
Train a single model from scratch.
Agnostic to family or model type — both are pulled from config.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import get_model
from data.dataset import XRDDataset
from config import DEVICE, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY


def train_one_model(train_X, train_y, n_classes: int,
                    seed: int = 42):
    """
    Train one model with a fixed seed.
    Uses early stopping when the training loss plateaus.
    """
    torch.manual_seed(seed)

    model     = get_model(n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    dataset = XRDDataset(train_X, train_y, preload_gpu=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)

    best_loss  = float("inf")
    no_improve = 0
    patience   = 20

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            if not dataset.on_gpu:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"      early stop epoch {epoch+1} "
                      f"loss={avg_loss:.4f}", flush=True)
                break

        if (epoch + 1) % 30 == 0:
            print(f"      epoch {epoch+1}/{EPOCHS} "
                  f"loss={avg_loss:.4f}", flush=True)

    return model
