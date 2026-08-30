"""
evaluation/cross_validation.py
────────────────────────────────────────────────────
5-Fold CV with Deep Ensemble.
Experiment A: Real only  |  Experiment B: Real + TER
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from training.ensemble import train_ensemble, ensemble_predict
from evaluation.metrics import compute_metrics
from data.augmentation import make_augmented_dataset
from config import N_FOLDS, RANDOM_SEED


def run_cv(data: dict) -> dict:
    patterns   = data["patterns"]
    labels     = data["labels"]
    is_virtual = np.array(data["is_virtual"], dtype=bool)
    origins    = data.get("origins", [None] * len(labels))
    classes    = data["classes"]
    n_classes  = data["n_classes"]
    family_cfg = data.get("family_cfg", {})

    n_folds = family_cfg.get("n_folds", N_FOLDS)

    real_mask = ~is_virtual
    real_X    = patterns[real_mask]
    real_y    = labels[real_mask]
    real_idx  = np.where(real_mask)[0]   # indices into the full array

    # Safety check
    min_class_count = int(np.bincount(real_y).min())
    n_folds = min(n_folds, min_class_count)
    if n_folds < 2:
        print(f"Warning: only {min_class_count} sample(s)/class "
              f"-> cannot run CV.")
        return {"A": [], "B": [], "entropy_records": []}

    print(f"Running {n_folds}-Fold CV "
          f"(min class count={min_class_count})")

    skf     = StratifiedKFold(n_splits=n_folds,
                               shuffle=True,
                               random_state=RANDOM_SEED)
    results = {"A": [], "B": [], "entropy_records": []}

    for fold, (tr_idx, te_idx) in enumerate(
            skf.split(real_X, real_y)):

        print(f"\n── Fold {fold+1}/{n_folds} ──")

        te_X = real_X[te_idx]
        te_y = real_y[te_idx]

        # ── Experiment A: Real only ──────────────────────────────
        tr_X_A, tr_y_A = make_augmented_dataset(
            real_X[tr_idx], real_y[tr_idx]
        )
        models_A = train_ensemble(tr_X_A, tr_y_A, n_classes)
        _, _, pred_A = ensemble_predict(models_A, te_X)
        m_A = compute_metrics(te_y, pred_A, classes)
        m_A["y_true"] = te_y.tolist()
        m_A["y_pred"] = pred_A.tolist()
        m_A["test_idx"] = te_idx.tolist()
        results["A"].append(m_A)
        print(f"  A (Real only) : acc={m_A['acc']:.1f}% "
              f"| F1={m_A['f1']:.3f}")

        # ── Experiment B: Real + TER (data-leakage-safe) ─────────
        # Only include virtual structures whose origin is in the TRAIN split
        train_formulas = set(
            origins[real_idx[i]] for i in tr_idx
        )

        safe_virt_mask = np.array([
            is_v and (orig in train_formulas)
            for is_v, orig in zip(is_virtual, origins)
        ], dtype=bool)

        n_safe_virt = safe_virt_mask.sum()

        tr_X_B = np.concatenate([
            real_X[tr_idx],
            patterns[safe_virt_mask]
        ])
        tr_y_B = np.concatenate([
            real_y[tr_idx],
            labels[safe_virt_mask]
        ])
        tr_X_B, tr_y_B = make_augmented_dataset(tr_X_B, tr_y_B)

        models_B = train_ensemble(tr_X_B, tr_y_B, n_classes)
        _, ent_B, pred_B = ensemble_predict(models_B, te_X)
        m_B = compute_metrics(te_y, pred_B, classes)
        m_B["y_true"] = te_y.tolist()
        m_B["y_pred"] = pred_B.tolist()
        m_B["test_idx"] = te_idx.tolist()
        results["B"].append(m_B)
        print(f"  B (Real+TER)  : acc={m_B['acc']:.1f}% "
              f"| F1={m_B['f1']:.3f} "
              f"[{n_safe_virt} virtual samples used]")

        # Entropy records
        for ent, pred, true in zip(ent_B, pred_B, te_y):
            results["entropy_records"].append({
                "entropy":    float(ent),
                "correct":    int(pred == true),
                "true_class": str(classes[true]),
                "pred_class": str(classes[pred]),
            })

        if fold == n_folds - 1:
            results["last_models"] = models_B

    return results
