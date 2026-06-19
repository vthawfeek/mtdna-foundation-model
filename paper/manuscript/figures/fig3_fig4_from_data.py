"""
Generate Figures 3 and 4 at 300 DPI from saved JSON/NPZ data.
No model loading required.

Run: uv run python paper/manuscript/figures/fig3_fig4_from_data.py
Outputs: fig3.pdf, fig3.png, fig4.pdf, fig4.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.manifold import TSNE

ROOT = Path(__file__).parent.parent.parent.parent
FIG_DIR = Path(__file__).parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(str(FIG_DIR / f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ============================================================
# Figure 3: Haplogroup embeddings — t-SNE + bar chart
# ============================================================

def figure3():
    npz = np.load(str(ROOT / "app_reference.npz"), allow_pickle=True)
    embeddings = npz["embeddings"]   # (100, 256)
    labels = npz["labels"]           # (100,) major haplogroup
    umap_coords = npz["umap_2d"]     # (100, 2) pre-computed

    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20", len(unique_labels))
    color_map = {hg: cmap(i) for i, hg in enumerate(unique_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel A: UMAP coloured by haplogroup
    ax = axes[0]
    for hg in unique_labels:
        mask = np.array([l == hg for l in labels])
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=[color_map[hg]], s=55, alpha=0.85,
                   edgecolors="white", linewidths=0.4, label=hg, zorder=4)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("(a)  mtDNA-FM embeddings in UMAP space\n"
                 "(100 sequences, zero-shot, no fine-tuning)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.8,
              markerscale=1.4, handlelength=1.0)

    # Panel B: Bar chart — method comparison
    ax = axes[1]
    methods = ["Random\nbaseline\n(8-class)", "mtDNA-FM\nzero-shot k-NN\n(8-class)"]
    accs = [12.5, 50.0]
    colors = ["#aaaaaa", "#d6604d"]
    bars = ax.bar(methods, accs, color=colors, edgecolor="gray",
                  linewidth=0.8, width=0.45)
    ax.set_ylabel("8-class haplogroup accuracy (%)")
    ax.set_title("(b)  Zero-shot haplogroup verification\n"
                 "(8-class panel, zero-shot 3-NN)", fontweight="bold")
    ax.set_ylim(0, 65)
    ax.axhline(12.5, color="#cc0000", linestyle="--", linewidth=1.0,
               label="Random baseline (12.5%)")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=9.5)
    ax.text(1.0, 53.0, "4× above\nrandom", ha="center", va="bottom",
            fontsize=8.5, color="#d6604d", style="italic")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Figure 3: Haplogroup embedding structure and zero-shot classification accuracy",
                 fontweight="bold", fontsize=11, y=1.02)
    _save(fig, "fig3")


# ============================================================
# Figure 4: Pathogenicity — ROC + PR curves + per-type bar
# ============================================================

def figure4():
    with open(str(ROOT / "reports" / "zeroshot_pathogenicity_knn.json")) as f:
        data = json.load(f)

    auroc = data["auroc"]
    ci_lo = data["auroc_ci_95_lo"]
    ci_hi = data["auroc_ci_95_hi"]
    auprc = data["auprc"]
    random_auprc = data["random_auprc"]

    fpr = [0.0] + data["roc_curve"]["fpr"]
    tpr = [0.0] + data["roc_curve"]["tpr"]
    prec = data["pr_curve"]["precision"]
    rec = [0.0] + data["pr_curve"]["recall"]
    # Align precision with recall (precision doesn't have a 0 start point)
    prec_plot = [prec[0]] + prec  # hold first value back to 0 recall

    per_type = data["per_type"]
    type_labels = ["Missense\n(n=56)", "tRNA\n(n=44)", "rRNA\n(n=5)"]
    type_keys = ["missense", "tRNA", "rRNA"]
    aurocs_type = [per_type[k]["auroc"] for k in type_keys]
    auprcs_type = [per_type[k]["auprc"] for k in type_keys]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: ROC curve
    ax = axes[0]
    ax.plot(fpr, tpr, color="#2166ac", lw=2.2,
            label=f"mtDNA-FM zero-shot\nAUROC {auroc:.3f} "
                  f"[{ci_lo:.3f}–{ci_hi:.3f}]")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#2166ac")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Random (AUROC 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("(a)  ROC curve\n(118 pathogenic vs 419 benign SNPs)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    # Annotate operating point
    fpr_op, tpr_op = data["roc_curve"]["fpr"][3], data["roc_curve"]["tpr"][3]
    ax.annotate(f"FPR={fpr_op:.2f}\nTPR={tpr_op:.2f}",
                xy=(fpr_op, tpr_op), xytext=(fpr_op + 0.12, tpr_op - 0.12),
                fontsize=8, color="#2166ac",
                arrowprops=dict(arrowstyle="->", color="#2166ac", lw=1.0))
    ax.scatter([fpr_op], [tpr_op], c="#2166ac", s=50, zorder=5)

    # Panel B: PR curve
    ax = axes[1]
    ax.plot(rec, prec_plot, color="#d6604d", lw=2.2,
            label=f"mtDNA-FM zero-shot\nAUPRC {auprc:.3f}")
    ax.axhline(random_auprc, color="gray", linestyle="--", lw=1.0,
               label=f"Random baseline ({random_auprc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(b)  Precision-recall curve", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)

    # Panel C: Per-variant-type AUROC and AUPRC
    ax = axes[2]
    x = np.arange(len(type_labels))
    w = 0.38
    bars1 = ax.bar(x - w / 2, aurocs_type, w, label="AUROC",
                   color="#4393c3", edgecolor="gray", linewidth=0.7)
    bars2 = ax.bar(x + w / 2, auprcs_type, w, label="AUPRC",
                   color="#d6604d", edgecolor="gray", linewidth=0.7)
    ax.axhline(0.5, color="#888888", linestyle="--", lw=1.0, label="Random AUROC")
    ax.set_xticks(x); ax.set_xticklabels(type_labels)
    ax.set_ylabel("Score")
    ax.set_title("(c)  Performance by variant class\n(zero-shot k-NN)", fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=8.5)
    for bar, val in zip(list(bars1) + list(bars2), aurocs_type + auprcs_type):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.3f}", ha="center", fontsize=7.5)

    fig.suptitle(
        "Figure 4: Zero-shot pathogenic variant discrimination\n"
        "(no pathogenicity labels in pre-training)",
        fontweight="bold", fontsize=11, y=1.03)
    _save(fig, "fig4")


if __name__ == "__main__":
    print("Generating Figure 3 (haplogroup UMAP + bar chart)...")
    figure3()
    print("Generating Figure 4 (pathogenicity ROC + PR + per-type)...")
    figure4()
    print(f"\nDone. Saved to {FIG_DIR}")
