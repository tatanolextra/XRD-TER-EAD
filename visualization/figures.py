"""
visualization/figures.py
────────────────────────────────────────────────────
All figures for a single crystal family.
Reads a `data` dict to pull the family name, ionic radii, etc.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import RESULTS_DIR


def plot_accuracy(results: dict, data: dict):
    """Figure 1: Accuracy & F1 comparison (A vs B)."""
    family_name = data.get("family_name", "unknown")
    family_cfg  = data.get("family_cfg", {})

    acc_A = [r["acc"] for r in results["A"]]
    acc_B = [r["acc"] for r in results["B"]]
    f1_A  = [r["f1"]  for r in results["A"]]
    f1_B  = [r["f1"]  for r in results["B"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{family_cfg.get('name','')}"
        f" ({family_cfg.get('formula','')}) — "
        "Effect of TER Augmentation",
        fontsize=13, fontweight="bold"
    )

    for ax, vals_A, vals_B, ylabel, ylim in zip(
        axes,
        [acc_A, f1_A], [acc_B, f1_B],
        ["5-Fold CV Accuracy (%)", "Macro-F1"],
        [100, 1.0],
    ):
        labels = ["Real only", "Real + TER"]
        means  = [np.mean(vals_A), np.mean(vals_B)]
        stds   = [np.std(vals_A),  np.std(vals_B)]
        bars   = ax.bar(labels, means, yerr=stds,
                         color=["#534AB7", "#D85A30"],
                         width=0.5, capsize=8, edgecolor="white")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ylim)
        ax.grid(True, alpha=0.3, axis="y")
        for i, (m, s) in enumerate(zip(means, stds)):
            label = f"{m:.1f}±{s:.1f}%" if "Acc" in ylabel \
                    else f"{m:.3f}"
            ax.text(i, m + s + ylim * 0.02, label,
                     ha="center", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_accuracy.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out}")


def plot_entropy(results: dict, data: dict):
    """Figure 2: Entropy violin plot + entropy-vs-ionic-radius scatter."""
    family_cfg  = data.get("family_cfg", {})
    ionic_radius = family_cfg.get("ionic_radius", {})
    classes      = data["classes"]

    records    = results["entropy_records"]
    ent_ok     = [r["entropy"] for r in records if r["correct"]]
    ent_wrong  = [r["entropy"] for r in records if not r["correct"]]

    class_ent = {}
    for r in records:
        class_ent.setdefault(r["true_class"], []).append(r["entropy"])
    mean_ent = {c: np.mean(v) for c, v in class_ent.items()}

    valid = [c for c in classes
             if c in mean_ent and c in ionic_radius]
    x_r   = [ionic_radius[c] for c in valid]
    y_e   = [mean_ent[c]     for c in valid]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Uncertainty Analysis — Predictive Entropy",
                  fontsize=13, fontweight="bold")

    # Violin plot
    if ent_ok and ent_wrong:
        axes[0].violinplot([ent_ok, ent_wrong],
                            positions=[0, 1], showmedians=True)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Correct", "Incorrect"])
    axes[0].set_ylabel("Predictive Entropy")
    axes[0].set_title("Entropy: Correct vs Incorrect")
    axes[0].grid(True, alpha=0.3)

    # Scatter plot
    if x_r:
        axes[1].scatter(x_r, y_e, s=120, color="#534AB7",
                         zorder=5, edgecolors="white")
        for c, x, y in zip(valid, x_r, y_e):
            axes[1].annotate(c, (x, y), xytext=(5, 4),
                              textcoords="offset points", fontsize=9)
        if len(x_r) >= 2:
            z  = np.polyfit(x_r, y_e, 1)
            xr = np.linspace(min(x_r)-.05, max(x_r)+.05, 100)
            axes[1].plot(xr, np.poly1d(z)(xr), "--",
                          color="#D85A30", alpha=0.7)
    axes[1].set_xlabel("B-site Ionic Radius (Å)")
    axes[1].set_ylabel("Mean Predictive Entropy")
    axes[1].set_title("Entropy vs Ionic Radius")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_entropy.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out}")
