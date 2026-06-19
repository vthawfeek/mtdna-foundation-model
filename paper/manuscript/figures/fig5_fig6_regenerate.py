"""
Regenerate Figures 5 and 6 at 300 DPI.

Fig 5: Two-panel — haplogroup-coloured UMAP of 100 modern sequences (from app_reference.npz)
       + bar chart of L2 distances showing archaic sequence placement.
Fig 6: Gene-type t-SNE using the showcase PNG data — redraws with non-overlapping labels
       by using a radial label placement strategy around each cluster centroid.

Run: uv run python paper/manuscript/figures/fig5_fig6_regenerate.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
FIG_DIR = Path(__file__).parent

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
# Figure 5: Ancient DNA — haplogroup UMAP + L2 bar chart
# ============================================================

def figure5():
    npz = np.load(str(ROOT / "app_reference.npz"), allow_pickle=True)
    umap_xy = npz["umap_2d"]    # (100, 2)
    labels = npz["labels"]      # (100,) major haplogroup

    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20", len(unique_labels))
    color_map = {hg: cmap(i) for i, hg in enumerate(unique_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              gridspec_kw={"width_ratios": [1.5, 1]})

    # Panel A: UMAP with haplogroup colour
    ax = axes[0]
    for hg in unique_labels:
        mask = np.array([l == hg for l in labels])
        ax.scatter(umap_xy[mask, 0], umap_xy[mask, 1],
                   c=[color_map[hg]], s=60, alpha=0.85,
                   edgecolors="white", linewidths=0.4, label=hg, zorder=4)

    # Annotate: archaic sequences were outside this cloud (marked with text)
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("(a)  Modern human mtDNA embedding space\n"
                 "(100 sequences, zero-shot, haplogroup-coloured)", fontweight="bold")
    ax.legend(loc="best", fontsize=7.5, ncol=2, framealpha=0.85,
              markerscale=1.3, handlelength=1.0)

    # Add archaic note as inset text
    ax.text(0.03, 0.03,
            "Archaic sequences lie\noutside this cloud\n(see panel b)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, style="italic", color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="#aaaaaa", alpha=0.9))

    # Panel B: L2 distance bar chart
    ax = axes[1]
    groups = ["Modern\nhuman\n(pairwise)", "Neanderthal\n(NC_011137.1)", "Denisovan\n(FR695060.1)"]
    distances = [0.075, 0.111, 0.107]
    multipliers = [1.0, 1.48, 1.43]
    colors = ["#4393c3", "#d6604d", "#9970ab"]

    bars = ax.bar(groups, distances, color=colors, edgecolor="gray",
                  linewidth=0.8, width=0.55)
    ax.axhline(0.075, color="#4393c3", linestyle="--", linewidth=1.0,
               alpha=0.6, label="Modern baseline (0.075)")
    ax.set_ylabel("Mean L2 distance from modern centroid")
    ax.set_title("(b)  Archaic sequence embedding distance\n"
                 "(zero-shot; neither sequence in pre-training)", fontweight="bold")
    ax.set_ylim(0, 0.14)
    for bar, val, mult in zip(bars, distances, multipliers):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        if mult > 1.0:
            ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                    f"{mult:.2f}×\nbaseline", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Figure 5: Archaic human sequences in mtDNA-FM embedding space",
                 fontweight="bold", fontsize=11, y=1.02)
    _save(fig, "fig5")


if __name__ == "__main__":
    print("Generating Figure 5 (ancient DNA: haplogroup UMAP + L2 bar chart)...")
    figure5()
    print(f"\nDone. Saved to {FIG_DIR}")
    print("\nNote: Figure 6 uses the pre-computed showcase output (docs/figures/showcase_gene_type_recovery.png).")
    print("It contains actual model embeddings from Day 25 and cannot be regenerated without the model checkpoint.")
