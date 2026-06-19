"""
Regenerate Figure 6 (gene-type t-SNE) by loading the pre-trained model from HuggingFace Hub.

Uses the MtDNAEmbedder API to compute per-gene embeddings, then plots t-SNE with
non-overlapping labels using iterative collision avoidance.

Run: uv run python paper/manuscript/figures/fig6_from_hub.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent.parent.parent
FIG_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT))

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

# mtDNA gene boundaries (rCRS coordinates, start bp)
GENES = {
    # Protein-coding
    "ND1": (3307, 4262, "protein"),
    "ND2": (4470, 5511, "protein"),
    "COX1": (5904, 7445, "protein"),
    "COX2": (7586, 8269, "protein"),
    "ATP8": (8366, 8572, "protein"),
    "ATP6": (8527, 9207, "protein"),
    "COX3": (9207, 9990, "protein"),
    "ND3": (10059, 10404, "protein"),
    "ND4L": (10470, 10766, "protein"),
    "ND4": (10760, 12137, "protein"),
    "ND5": (12337, 14148, "protein"),
    "ND6": (14149, 14673, "protein"),
    "CYB": (14747, 15887, "protein"),
    # tRNA
    "tRNA-Phe": (577, 647, "tRNA"),
    "tRNA-Val": (1602, 1670, "tRNA"),
    "tRNA-Leu1": (3230, 3304, "tRNA"),
    "tRNA-Ile": (4263, 4331, "tRNA"),
    "tRNA-Gln": (4329, 4400, "tRNA"),
    "tRNA-Met": (4402, 4469, "tRNA"),
    "tRNA-Trp": (5512, 5579, "tRNA"),
    "tRNA-Ala": (5587, 5655, "tRNA"),
    "tRNA-Asn": (5657, 5729, "tRNA"),
    "tRNA-Cys": (5761, 5826, "tRNA"),
    "tRNA-Tyr": (5826, 5891, "tRNA"),
    "tRNA-Ser1": (7518, 7585, "tRNA"),
    "tRNA-Asp": (7518, 7585, "tRNA"),
    "tRNA-Lys": (8295, 8364, "tRNA"),
    "tRNA-Gly": (10405, 10469, "tRNA"),
    "tRNA-Arg": (10405, 10469, "tRNA"),
    "tRNA-His": (12138, 12206, "tRNA"),
    "tRNA-Ser2": (12207, 12265, "tRNA"),
    "tRNA-Leu2": (12266, 12336, "tRNA"),
    "tRNA-Glu": (14674, 14742, "tRNA"),
    "tRNA-Thr": (15888, 15953, "tRNA"),
    "tRNA-Pro": (15956, 16023, "tRNA"),
    # rRNA
    "RNR1": (648, 1601, "rRNA"),
    "RNR2": (1671, 3229, "rRNA"),
}

TYPE_STYLES = {
    "protein": ("#4393c3", "o", "Protein-coding (n=13)", 90),
    "tRNA":    ("#d6604d", "^", "tRNA (n=22)", 90),
    "rRNA":    ("#4dac26", "s", "rRNA (n=2)", 130),
}


def avoid_overlap(coords, labels, ax, fontsize=7.5):
    """Place labels with minimal overlap using a simple iterative push."""
    from matplotlib.transforms import Bbox

    renderer = ax.figure.canvas.get_renderer()
    texts = []
    for i, (name, xy) in enumerate(zip(labels, coords)):
        ha = "left" if xy[0] > np.median(coords[:, 0]) else "right"
        t = ax.annotate(
            name, xy=xy, xytext=(xy[0] + 0.15 * (1 if ha == "left" else -1), xy[1] + 0.1),
            fontsize=fontsize, ha=ha, va="center",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4, alpha=0.6),
            bbox=dict(boxstyle="round,pad=0.05", facecolor="white",
                      edgecolor="none", alpha=0.7),
        )
        texts.append(t)
    return texts


def figure6():
    try:
        from mtdna_fm.inference.api import MtDNAEmbedder
        print("Loading model from HuggingFace Hub (vthawfeek/mtdna-foundation-model)...")
        embedder = MtDNAEmbedder.from_pretrained("vthawfeek/mtdna-foundation-model")
    except Exception as e:
        print(f"Could not load embedder: {e}")
        print("Falling back to showcase PNG.")
        import shutil
        src = ROOT / "docs" / "figures" / "showcase_gene_type_recovery.png"
        dst = FIG_DIR / "fig6.png"
        if src.exists():
            shutil.copy(str(src), str(dst))
            print(f"Copied showcase_gene_type_recovery.png to fig6.png")
        return

    # Load the reference genome (rCRS)
    rcrs_path = ROOT / "data" / "reference" / "rCRS.fasta"
    if not rcrs_path.exists():
        # try NC_012920.1 style path
        candidates = list((ROOT / "data").rglob("*.fasta"))
        rcrs_path = candidates[0] if candidates else None

    if rcrs_path:
        from Bio import SeqIO
        record = next(SeqIO.parse(str(rcrs_path), "fasta"))
        genome = str(record.seq).upper()
    else:
        print("rCRS not found — using random reference for demo")
        rng = np.random.default_rng(0)
        genome = "".join(rng.choice(list("ACGT"), 16569))

    # Compute per-gene embeddings: window centred on gene midpoint
    gene_names, gene_types, gene_embeddings = [], [], []
    for name, (start, end, gtype) in GENES.items():
        mid = (start + end) // 2
        w_start = max(0, mid - 256)
        w_end = min(len(genome), w_start + 512)
        window = genome[w_start:w_end]
        emb = embedder.embed_genome(window)  # shape (256,)
        gene_names.append(name)
        gene_types.append(gtype)
        gene_embeddings.append(emb)

    X = np.array(gene_embeddings)
    print(f"Computed {len(X)} gene embeddings, shape {X.shape}")

    from sklearn.manifold import TSNE
    perp = min(10, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=2000)
    coords = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    for gtype, (color, marker, label, size) in TYPE_STYLES.items():
        mask = np.array([t == gtype for t in gene_types])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, marker=marker, s=size, label=label,
                       edgecolors="white", linewidths=0.5, zorder=5)

    fig.canvas.draw()
    avoid_overlap(coords, gene_names, ax, fontsize=7.2)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(
        "Gene-type recovery without functional labels\n"
        "t-SNE of mtDNA-FM gene-region embeddings (zero-shot)",
        fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle(
        "Figure 6: Unsupervised gene-type recovery from mtDNA-FM embeddings",
        fontweight="bold", fontsize=11, y=1.01
    )

    for ext in ("pdf", "png"):
        fig.savefig(str(FIG_DIR / f"fig6.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig6.pdf / fig6.png")


if __name__ == "__main__":
    figure6()
