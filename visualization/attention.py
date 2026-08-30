"""
visualization/attention.py
────────────────────────────────────────────────────
Attention map visualization — auto-detects the model type
to handle Conformer vs. SpectralTransformer correctly.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import DEVICE, RESULTS_DIR, SEQ_LEN, TWO_THETA_MIN, TWO_THETA_MAX


def plot_attention(models: list, data: dict,
                   sample_classes: list = None):
    """Plot attention maps for a few representative classes."""
    model = models[0]
    if not model.supports_attention:
        print("Model does not support attention visualization.")
        return

    patterns    = data["patterns"]
    labels      = data["labels"]
    classes     = data["classes"]
    two_theta   = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, SEQ_LEN)
    model_type  = model.__class__.__name__

    if sample_classes is None:
        sample_classes = list(classes[:3])

    fig, axes = plt.subplots(
        len(sample_classes), 1,
        figsize=(12, 4 * len(sample_classes))
    )
    if len(sample_classes) == 1:
        axes = [axes]

    fig.suptitle(
        f"Attention Maps — {model_type}\n"
        f"(Which 2θ regions does the model attend to?)",
        fontsize=13, fontweight="bold"
    )

    colors = ["#1D9E75", "#534AB7", "#D85A30",
              "#BA7517", "#185FA5"]
    model.eval()

    for ax, cls_name, color in zip(axes, sample_classes, colors):
        cls_idx = data["le"].transform([cls_name])[0]
        idxs    = np.where(labels == cls_idx)[0]
        if len(idxs) == 0:
            ax.set_title(f"{cls_name}: no samples"); continue

        sample = patterns[idxs[0]]
        x_t    = torch.FloatTensor(sample).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, attn_list = model(x_t, return_attn=True)

        if not attn_list:
            continue

        attn = attn_list[-1][0].cpu().numpy()   # (seq, seq)

        if model_type == "SpectralTransformer":
            # CLS token is at position 0, patches start at 1
            cls_a         = attn[0, 1:]          # (n_patches,)
            repeat_factor = SEQ_LEN // len(cls_a)
        else:
            # Conformer: no CLS token, average over positions
            cls_a         = attn.mean(axis=0)    # (seq_len/4,)
            repeat_factor = SEQ_LEN // len(cls_a)

        cls_a     = cls_a / (cls_a.max() + 1e-8)
        attn_full = np.repeat(cls_a, repeat_factor)
        # Pad if the repeated array is short due to integer division
        if len(attn_full) < SEQ_LEN:
            attn_full = np.pad(attn_full,
                                (0, SEQ_LEN - len(attn_full)),
                                mode="edge")

        ax2 = ax.twinx()
        ax.plot(two_theta, sample, color=color,
                 linewidth=1.2, alpha=0.85)
        ax2.fill_between(two_theta, attn_full,
                          color=color, alpha=0.25)
        ax2.set_ylim(0, 2.5)
        ax2.set_ylabel("Attention", color=color, fontsize=9)
        ax.set_ylabel("Intensity")
        ax.set_xlim(TWO_THETA_MIN, TWO_THETA_MAX)
        ax.set_title(f"{cls_name}-{data['family_name']} | "
                      f"{model_type}", fontsize=11)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("2θ (degrees)")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_attention.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out}")
