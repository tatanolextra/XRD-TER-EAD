"""
main.py
────────────────────────────────────────────────────

Entry point. Runs the pipeline for one crystal family.

Usage:
    python main.py                          # uses ACTIVE_FAMILY from config.py
    python main.py --family garnet          # override family
    python main.py --family scheelite --key YOUR_API_KEY
    python main.py --build-only             # build the dataset only, no training

The Materials Project API key can also be set via the MP_API_KEY
environment variable instead of --key, to avoid leaving it in shell history:
    export MP_API_KEY="your-key-here"
    python main.py --family garnet
"""
import argparse
import os
import sys
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="XRD Crystal Phase Classification"
    )
    parser.add_argument(
        "--family", type=str, default=None,
        help="Crystal family: spinel | garnet | scheelite"
    )
    parser.add_argument(
        "--key", type=str, default=None,
        help="Materials Project API key (only needed to build the dataset "
             "the first time; falls back to the MP_API_KEY env var)"
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="Only build the dataset, skip training"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Override ACTIVE_FAMILY if --family was passed ────────────
    if args.family:
        import config
        config.ACTIVE_FAMILY = args.family
        # Re-derive paths for the new family
        config.DATA_DIR      = os.path.join(config.BASE_DIR, args.family, "dataset")
        config.RESULTS_DIR   = os.path.join(config.BASE_DIR, args.family, "results")
        config.XRD_NPY       = os.path.join(config.DATA_DIR, "dataset_xrd.npy")
        config.METADATA_JSON = os.path.join(config.DATA_DIR, "metadata.json")
        os.makedirs(config.DATA_DIR,    exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

    import config as cfg
    from data.crystal_families import get_family

    family = get_family(cfg.ACTIVE_FAMILY)
    api_key = args.key or os.environ.get("MP_API_KEY")

    print("=" * 55)
    print(f"XRD Phase Classification")
    print(f"Family : {family['name']} ({family['formula']})")
    print(f"Model  : {cfg.MODEL_NAME}")
    print(f"Device : {cfg.DEVICE}")
    print(f"Output : {cfg.BASE_DIR}/{cfg.ACTIVE_FAMILY}/")
    print("=" * 55)

    # ── 1. Load / build dataset ───────────────────────────────────
    from data.loader import load_dataset
    data = load_dataset(api_key=api_key)

    if args.build_only:
        print("\nDataset built successfully. Exiting (--build-only).")
        return

    # ── 2. Train + Evaluate ────────────────────────────────────────
    from evaluation.cross_validation import run_cv
    results = run_cv(data)

    # ── 3. Summary ────────────────────────────────────────────────
    acc_A = np.mean([r["acc"] for r in results["A"]])
    acc_B = np.mean([r["acc"] for r in results["B"]])
    f1_A  = np.mean([r["f1"]  for r in results["A"]])
    f1_B  = np.mean([r["f1"]  for r in results["B"]])
    std_A = np.std([r["acc"] for r in results["A"]])
    std_B = np.std([r["acc"] for r in results["B"]])

    print(f"""
╔══════════════════════════════════════════════╗
║  {family['name']:^42} ║
╠══════════════════════════════════════════════╣
║  Real only  : {acc_A:5.1f}% ± {std_A:.1f}% | F1={f1_A:.3f}     ║
║  Real + TER : {acc_B:5.1f}% ± {std_B:.1f}% | F1={f1_B:.3f}     ║
║  Δ Accuracy : {acc_B-acc_A:+.1f}%                          ║
╚══════════════════════════════════════════════╝
""")

    # Save summary JSON, used for cross-family comparison
    import json
    summary = {
        "family":    cfg.ACTIVE_FAMILY,
        "model":     cfg.MODEL_NAME,
        "n_classes": data["n_classes"],
        "n_samples": len(data["labels"]),
        "real_only": {"acc": acc_A, "std": std_A, "f1": f1_A},
        "ter":       {"acc": acc_B, "std": std_B, "f1": f1_B},
        "delta_acc": acc_B - acc_A,
        "fold_results": {
            "A": results["A"],
            "B": results["B"],
        }
    }
    summary_path = os.path.join(cfg.RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # ── 3b. Per-class F1 (Fig 5, Fig 6) ──────────────────────────
    from sklearn.metrics import f1_score
    per_class_f1_A = [
        f1_score(r["y_true"], r["y_pred"], average=None,
                  labels=list(range(data["n_classes"]))).tolist()
        for r in results["A"]
    ]
    per_class_f1_B = [
        f1_score(r["y_true"], r["y_pred"], average=None,
                  labels=list(range(data["n_classes"]))).tolist()
        for r in results["B"]
    ]
    per_class_path = os.path.join(cfg.RESULTS_DIR, "per_class_f1.json")
    with open(per_class_path, "w") as f:
        json.dump({
            "per_class_f1_A": per_class_f1_A,
            "per_class_f1_B": per_class_f1_B,
            "mean_f1_A": np.mean(per_class_f1_A, axis=0).tolist(),
            "mean_f1_B": np.mean(per_class_f1_B, axis=0).tolist(),
            "classes": list(data["classes"]),
        }, f, indent=2)
    print(f"Per-class F1 saved: {per_class_path}")

    # ── 3c. Save ensemble weights (Fig 7) ────────────────────────
    if "last_models" in results:
        import torch
        model_dir = os.path.join(cfg.RESULTS_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        for i, m in enumerate(results["last_models"]):
            torch.save(m.state_dict(),
                       os.path.join(model_dir, f"ensemble_{i}.pt"))
        with open(os.path.join(model_dir, "model_meta.json"), "w") as f:
            json.dump({
                "model_name": cfg.MODEL_NAME,
                "n_classes":  data["n_classes"],
                "config":     cfg.MODEL_CONFIGS.get(cfg.MODEL_NAME, {}),
            }, f, indent=2)
        print(f"Ensemble weights saved: {model_dir}")

    # ── 3d. Save entropy records (Fig 6) ─────────────────────────
    entropy_path = os.path.join(cfg.RESULTS_DIR, "entropy_records.json")
    with open(entropy_path, "w") as f:
        json.dump(results["entropy_records"], f, indent=2)
    print(f"Entropy records saved: {entropy_path}")

    # ── 4. Figures ────────────────────────────────────────────────
    from visualization.figures import plot_accuracy, plot_entropy
    plot_accuracy(results, data)
    plot_entropy(results, data)

    if "last_models" in results:
        from visualization.attention import plot_attention
        sample_classes = list(data["classes"][:3])
        plot_attention(results["last_models"], data,
                       sample_classes=sample_classes)

    print(f"\nAll figures saved to {cfg.BASE_DIR}/{cfg.ACTIVE_FAMILY}/results/")


if __name__ == "__main__":
    main()
