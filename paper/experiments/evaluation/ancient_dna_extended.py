"""
G7: Extended ancient DNA zero-shot phylogenetic placement.

Uses MtDNAEmbedder.from_pretrained() to embed 10+ ancient hominin mtDNA sequences
and evaluates whether they place correctly relative to modern human haplogroups.

Usage:
    uv run python paper/experiments/evaluation/ancient_dna_extended.py

Outputs:
    paper/experiments/evaluation/ancient_dna_results.json
    paper/experiments/evaluation/ancient_dna_umap.png  (if umap-learn installed)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path("paper/experiments/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("data/raw/ancient")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ancient samples with known phylogenetic expectations
ANCIENT_SAMPLES = [
    {"accession": "NC_011137.1", "label": "Neanderthal (Vindija)", "type": "neanderthal", "age_kya": 38},
    {"accession": "FM865408.1",  "label": "Neanderthal (Feldhofer 1)", "type": "neanderthal", "age_kya": 40},
    {"accession": "AY008144.1",  "label": "Neanderthal (Mezmaiskaya)", "type": "neanderthal", "age_kya": 60},
    {"accession": "KX198085.1",  "label": "Neanderthal (El Sidrón)", "type": "neanderthal", "age_kya": 49},
    {"accession": "FR695060.1",  "label": "Denisovan (Altai)", "type": "denisovan", "age_kya": 50},
    {"accession": "KU131959.1",  "label": "Early modern (Kostenki, 37 kya)", "type": "early_modern", "age_kya": 37},
    {"accession": "HQ153842.1",  "label": "Early modern (Tianyuan, 40 kya)", "type": "early_modern", "age_kya": 40},
    {"accession": "AF347017.1",  "label": "Modern human L0 (San, deep root)", "type": "modern_african", "age_kya": 0},
]


def fetch_sequence(accession: str) -> str | None:
    """Fetch mtDNA from NCBI with caching."""
    cache = CACHE_DIR / f"{accession}.fasta"
    if cache.exists():
        seq = "".join(
            line.strip() for line in open(cache) if not line.startswith(">")
        ).upper()
        if seq:
            return seq

    logger.info(f"  Fetching {accession} from NCBI...")
    try:
        from Bio import Entrez, SeqIO
        Entrez.email = "mtdnafm@paper.local"
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        seq = str(record.seq).upper()
        with open(cache, "w") as f:
            f.write(f">{accession}\n{seq}\n")
        logger.info(f"    {len(seq)} bp downloaded and cached")
        return seq
    except Exception as e:
        logger.warning(f"  Failed to fetch {accession}: {e}")
        return None


def embed_sequences(sequences: list[str]) -> np.ndarray:
    """Embed sequences using MtDNAEmbedder (CLS pooling via sliding windows)."""
    from mtdna_fm.inference.api import MtDNAEmbedder

    model_path = "models/phase2_v1" if Path("models/phase2_v1/config.json").exists() else "models/phase1_v1"
    logger.info(f"Using model: {model_path}")
    embedder = MtDNAEmbedder.from_pretrained(model_path)

    embeddings = []
    for i, seq in enumerate(sequences):
        emb = embedder.embed_genome(seq)
        embeddings.append(emb)
        if (i + 1) % 10 == 0:
            logger.info(f"  Embedded {i+1}/{len(sequences)}")
    return np.array(embeddings)


def pairwise_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute pairwise L2 distances between rows of A and B."""
    return np.sqrt(np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1))


def main() -> None:
    # Fetch ancient sequences
    logger.info("=== Fetching ancient DNA sequences ===")
    ancient_records = []
    for s in ANCIENT_SAMPLES:
        seq = fetch_sequence(s["accession"])
        if seq:
            ancient_records.append({**s, "sequence": seq})

    if not ancient_records:
        logger.error("No ancient sequences fetched. Check network/NCBI access.")
        return

    ancient_df = pd.DataFrame(ancient_records)
    logger.info(f"Fetched {len(ancient_df)} ancient sequences")

    # Load modern human sequences for reference
    held_out = Path("paper/experiments/evaluation/held_out_test.parquet")
    if held_out.exists():
        modern_df = pd.read_parquet(held_out)
        # Sample up to 50 modern human sequences
        modern_sample = modern_df.sample(min(50, len(modern_df)), random_state=42)
        logger.info(f"Loaded {len(modern_sample)} modern human sequences for comparison")
    else:
        modern_sample = pd.read_parquet("data/processed/test.parquet").sample(
            min(50, 1263), random_state=42
        )
        logger.info(f"Using test.parquet for modern comparison: {len(modern_sample)} sequences")

    # Embed all sequences
    logger.info("=== Embedding sequences ===")
    all_seqs = ancient_df["sequence"].tolist() + modern_sample["sequence"].tolist()
    all_labels = ancient_df["label"].tolist() + modern_sample.get("major_haplogroup", modern_sample.get("haplogroup", pd.Series(["H"] * len(modern_sample)))).tolist()
    all_types = ancient_df["type"].tolist() + ["modern_human"] * len(modern_sample)

    embeddings = embed_sequences(all_seqs)
    n_ancient = len(ancient_df)
    ancient_embs = embeddings[:n_ancient]
    modern_embs = embeddings[n_ancient:]

    # Pairwise distances
    ancient_to_modern = pairwise_l2(ancient_embs, modern_embs)  # (n_ancient, n_modern)
    modern_to_modern = pairwise_l2(modern_embs, modern_embs)    # (n_modern, n_modern)
    np.fill_diagonal(modern_to_modern, np.inf)
    modern_baseline = float(np.mean(modern_to_modern[modern_to_modern < np.inf]))

    per_sample = []
    for i, row in ancient_df.iterrows():
        a2m = float(np.mean(ancient_to_modern[i]))
        per_sample.append({
            "accession": row["accession"],
            "label": row["label"],
            "type": row["type"],
            "age_kya": row.get("age_kya"),
            "mean_dist_to_modern": a2m,
            "ratio_vs_modern_baseline": a2m / modern_baseline,
            "correctly_placed": a2m > modern_baseline * (1.2 if row["type"] == "early_modern" else 1.4),
        })
        logger.info(
            f"  {row['label']}: dist={a2m:.4f}, "
            f"ratio={a2m/modern_baseline:.3f}x, "
            f"correct={'✓' if per_sample[-1]['correctly_placed'] else '✗'}"
        )

    concordance = {}
    for t in ["neanderthal", "denisovan", "early_modern"]:
        subset = [s for s in per_sample if s["type"] == t]
        if subset:
            n_correct = sum(1 for s in subset if s["correctly_placed"])
            concordance[t] = {"n": len(subset), "n_correct": n_correct, "fraction": n_correct / len(subset)}

    results = {
        "status": "completed",
        "model": "phase1_v1" if not Path("models/phase2_v1/config.json").exists() else "phase2_v1",
        "n_ancient": n_ancient,
        "n_modern": len(modern_embs),
        "modern_baseline_l2": modern_baseline,
        "per_sample": per_sample,
        "concordance_by_type": concordance,
    }

    out = EVAL_DIR / "ancient_dna_results.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info(f"Results → {out}")

    # UMAP visualization
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
        coords = reducer.fit_transform(embeddings)

        type_colors = {
            "neanderthal": "#d73027", "denisovan": "#fc8d59",
            "early_modern": "#4dac26", "modern_african": "#762a83",
            "modern_human": "#abd9e9",
        }
        fig, ax = plt.subplots(figsize=(10, 8))
        for t, color in type_colors.items():
            idx = [i for i, tp in enumerate(all_types) if tp == t]
            if not idx:
                continue
            size = 120 if t != "modern_human" else 15
            marker = "*" if t in ("neanderthal", "denisovan") else "o"
            ax.scatter(coords[idx, 0], coords[idx, 1], c=color, s=size, marker=marker,
                       alpha=0.8, label=t.replace("_", " ").title(), zorder=5 if size > 20 else 2)
            if t != "modern_human":
                for i in idx:
                    ax.annotate(all_labels[i].split("(")[0].strip(),
                                (coords[i, 0] + 0.1, coords[i, 1] + 0.1), fontsize=7)

        ax.legend(fontsize=9)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("mtDNA-FM: Ancient and Modern Human Embeddings (zero-shot)", fontsize=11)
        fig.tight_layout()
        umap_path = EVAL_DIR / "ancient_dna_umap.png"
        fig.savefig(str(umap_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"UMAP → {umap_path}")
    except ImportError:
        logger.info("umap-learn not installed — skipping UMAP visualization")

    print(f"\n=== Ancient DNA Results ===")
    print(f"Modern human baseline L2: {modern_baseline:.4f}")
    print(f"\n{'Sample':<45} {'Ratio':>8} {'Correct'}")
    print("-" * 60)
    for s in sorted(per_sample, key=lambda x: -x["ratio_vs_modern_baseline"]):
        label = s['label'][:43]
        print(f"{label:<45} {s['ratio_vs_modern_baseline']:>8.3f}x {'✓' if s['correctly_placed'] else '✗'}")
    print(f"\nConcordance: {concordance}")


if __name__ == "__main__":
    main()
