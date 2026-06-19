"""
Standalone generator for Figures 1 and 2 — no model required.
Run: uv run python paper/manuscript/figures/fig1_fig2_standalone.py
Outputs: paper/manuscript/figures/fig1.pdf, fig1.png, fig2.pdf, fig2.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIG_DIR = Path(__file__).parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(str(FIG_DIR / f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ============================================================
# Figure 1: Architecture + Circular PE
# ============================================================

def figure1():
    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.2, 1.4], wspace=0.35)

    # ── Panel A: Circular genome diagram ──────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_aspect("equal")
    ax_a.axis("off")
    ax_a.set_title("(a)  mtDNA circular genome", fontweight="bold", pad=8)

    theta = np.linspace(0, 2 * np.pi, 1000)
    ax_a.plot(np.cos(theta), np.sin(theta), color="#444444", lw=1.5)

    L = 16569
    # Clockwise from top: angle = pi/2 - 2*pi*pos/L
    # Labels well-spaced around the circle; D-loop at top (12 o'clock)
    gene_annotations = [
        (0,     "D-loop\n(pos 0 / 16,569)",  "#c0392b", "bold",   1.52),
        (3307,  "ND1",                        "#8e44ad", "normal", 1.45),
        (7445,  "COX1",                       "#8e44ad", "normal", 1.45),
        (11369, "ND4",                        "#8e44ad", "normal", 1.45),
        (14747, "CYB",                        "#8e44ad", "normal", 1.45),
    ]
    for pos, lbl, color, fw, radius in gene_annotations:
        angle = np.pi / 2 - 2 * np.pi * pos / L  # clockwise, pos 0 at top
        xp, yp = np.cos(angle), np.sin(angle)
        ax_a.scatter([xp], [yp], c=color, s=55, zorder=5,
                     edgecolors="white", linewidths=0.5)
        xoff = xp * radius
        yoff = yp * radius
        ha = "center" if abs(xp) < 0.3 else ("left" if xp > 0 else "right")
        va = "bottom" if yp > 0.3 else ("top" if yp < -0.3 else "center")
        ax_a.text(xoff, yoff, lbl, ha=ha, va=va,
                  fontsize=8, color=color, fontweight=fw,
                  bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                            edgecolor="none", alpha=0.75))

    # Clockwise direction arrows
    for angle in [np.pi / 2 - 0.6, np.pi / 2 - 2.2, np.pi / 2 - 3.8]:
        ax_a.annotate("", xy=(np.cos(angle - 0.22), np.sin(angle - 0.22)),
                      xytext=(np.cos(angle), np.sin(angle)),
                      arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.8))

    ax_a.text(0, 0, "16,569 bp\n37 genes\n(circular)", ha="center", va="center",
              fontsize=9, style="italic", color="#555555")
    ax_a.set_xlim(-1.85, 1.85)
    ax_a.set_ylim(-1.65, 1.85)

    # ── Panel B: Positional encoding comparison ───────────────────────────
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_title("(b)  Positional encoding at\ngenome junction", fontweight="bold", pad=8)

    pos = np.linspace(0, L, 300)
    # Distance of PE(p) from PE(0) — first dimension
    std_dist = 1 - np.cos(pos / L * np.pi)           # goes 0→2→0 but we want 0→max for linear
    std_dist = np.abs(pos - pos.max() / 2) / (pos.max() / 2)  # linear ramp then ramp down
    # Simpler: use actual first PE component
    circ_sim = (1 + np.cos(2 * np.pi * pos / L)) / 2  # similarity to pos=0

    ax_b.plot(pos, 1 - circ_sim, color="#2166ac", lw=2.0,
              label="Circular PE\n(mtDNA-FM)")
    lin_sim_to_zero = np.abs(pos) / L           # distance 0→1 linearly
    ax_b.plot(pos, lin_sim_to_zero, color="#d73027", lw=2.0, ls="--",
              label="Standard linear PE")

    ax_b.axvspan(0, 200, alpha=0.12, color="#c0392b", label="D-loop (junction)")
    ax_b.axvspan(L - 200, L, alpha=0.12, color="#c0392b")

    ax_b.set_xlabel("Genomic position (bp)")
    ax_b.set_ylabel("Distance from position 0")
    ax_b.set_xlim(0, L)
    ax_b.set_ylim(-0.05, 1.1)
    ax_b.set_xticks([0, 4000, 8000, 12000, 16569])
    ax_b.set_xticklabels(["0", "4k", "8k", "12k", "16,569"])

    # Annotation: at pos=16569, circular PE returns to 0
    ax_b.annotate("PE(0) = PE(16,569)",
                  xy=(L, 0.02), xytext=(L - 5200, 0.18),
                  arrowprops=dict(arrowstyle="->", color="#2166ac", lw=1.2),
                  color="#2166ac", fontsize=8, fontstyle="italic")
    ax_b.legend(fontsize=8, loc="center left")
    ax_b.grid(True, alpha=0.2)

    # ── Panel C: Architecture block diagram ──────────────────────────────
    ax_c = fig.add_subplot(gs[2])
    ax_c.axis("off")
    ax_c.set_title("(c)  mtDNA-FM architecture\n(~5.8M parameters)", fontweight="bold", pad=8)

    blocks = [
        # (y_center, text, face_color, text_color)
        (0.88, "Input: 6-mer tokens  +  het values h∈[0,1]",       "#eaf4fb", "#333"),
        (0.74, "K-mer embedding  (vocab=4,102, d=256)",             "#d0e8f5", "#333"),
        (0.60, "+ Circular PE  (2π·p / 16,569)",                    "#a8d1ed", "#333"),
        (0.46, "+ Het projection  Linear(1,256) + LayerNorm",       "#80bade", "#333"),
        (0.31, "6 × Transformer block\n(8-head, FFN=1024, pre-LN)", "#4393c3", "white"),
        (0.14, "Task heads:\nHaplogroup (26) | Pathogenicity | Het-reg", "#2166ac", "white"),
    ]
    for y, text, fc, tc in blocks:
        ax_c.add_patch(mpatches.FancyBboxPatch(
            (0.05, y - 0.065), 0.90, 0.11,
            boxstyle="round,pad=0.025", facecolor=fc,
            edgecolor="#888888", linewidth=0.8
        ))
        ax_c.text(0.50, y - 0.005, text, ha="center", va="center",
                  fontsize=7.8, color=tc, transform=ax_c.transAxes)

    # Downward arrows between blocks
    for y_top in [0.88, 0.74, 0.60, 0.46]:
        ax_c.annotate("", xy=(0.50, y_top - 0.065 - 0.01),
                      xytext=(0.50, y_top - 0.065 - 0.055),
                      xycoords="axes fraction", textcoords="axes fraction",
                      arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.0))

    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)

    fig.suptitle("Figure 1: mtDNA-FM — Circular Genome Representation and Architecture",
                 fontsize=12, fontweight="bold", y=1.02)
    _save(fig, "fig1")


# ============================================================
# Figure 2: Circular PE vs Linear PE — similarity matrices
# ============================================================

def figure2():
    N = 300   # downsample 16,569 → 300 grid points for display
    L = 16569
    pos = np.linspace(0, L - 1, N)

    # Compute pairwise positional similarity for first PE dimension only
    # (generalises: use cosine similarity of full PE vector)
    def linear_sim_matrix(pos):
        diff = np.abs(pos[:, None] - pos[None, :])
        max_diff = L
        return 1 - diff / max_diff

    def circular_sim_matrix(pos):
        theta = 2 * np.pi * pos / L
        return (1 + np.cos(theta[:, None] - theta[None, :])) / 2

    lin_sim = linear_sim_matrix(pos)
    circ_sim = circular_sim_matrix(pos)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2),
                              gridspec_kw={"width_ratios": [1, 1, 0.05]})
    ax1, ax2, ax_cb = axes

    ticks = np.linspace(0, N - 1, 5).astype(int)
    tick_labels = [f"{int(pos[t] / 1000):.0f}k" for t in ticks]
    tick_labels[0] = "0"
    tick_labels[-1] = "16,569"

    vmin, vmax = 0.0, 1.0
    cmap = "RdBu_r"

    for ax, sim, title, subtitle in [
        (ax1, lin_sim,
         "(a)  Standard linear PE",
         "positions 0 and 16,569 maximally\ndissimilar — D-loop region SPLIT"),
        (ax2, circ_sim,
         "(b)  Circular PE (mtDNA-FM)",
         "positions 0 and 16,569 identical —\nD-loop junction correctly closed"),
    ]:
        im = ax.imshow(sim, aspect="auto", origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_yticks(ticks); ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Genomic position (bp)")
        ax.set_ylabel("Genomic position (bp)")
        ax.set_title(title, fontweight="bold", pad=6)

        # Annotate D-loop region
        dloop_px_start = 0
        dloop_px_end = int(576 / L * N)
        for corner_px in [dloop_px_start, int((L - 576) / L * N)]:
            ax.axvline(corner_px, color="#c0392b", lw=0.8, ls=":", alpha=0.7)
            ax.axhline(corner_px, color="#c0392b", lw=0.8, ls=":", alpha=0.7)

        ax.text(0.98, 0.02, subtitle, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="black",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                          edgecolor="#aaaaaa", alpha=0.85))

    # Highlight the junction corner for each plot
    corner_px = int((L - 1) / L * N)
    ax1.scatter([0, corner_px], [corner_px, 0], c="#c0392b", s=50, zorder=5,
                marker="x", linewidths=1.5, label="D-loop corners (similarity≈0)")
    ax1.legend(fontsize=7.5, loc="upper right")

    ax2.scatter([0, corner_px], [corner_px, 0], c="#2166ac", s=50, zorder=5,
                marker="o", linewidths=1.5, label="D-loop corners (similarity=1)")
    ax2.legend(fontsize=7.5, loc="upper right")

    plt.colorbar(im, cax=ax_cb, label="Positional similarity")

    fig.suptitle(
        "Figure 2: Positional similarity matrix — standard vs circular positional encoding\n"
        "Red dashed lines mark the D-loop region (positions 0 and ~16,024–16,569)",
        fontsize=11, fontweight="bold", y=1.03
    )
    _save(fig, "fig2")


if __name__ == "__main__":
    print("Generating Figure 1 (architecture + circular PE)...")
    figure1()
    print("Generating Figure 2 (positional similarity matrices)...")
    figure2()
    print(f"\nDone. Files saved to: {FIG_DIR}")
    print("  fig1.pdf  fig1.png")
    print("  fig2.pdf  fig2.png")
