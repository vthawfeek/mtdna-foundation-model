#!/usr/bin/env python3
"""
Zero-shot k-NN pathogenicity evaluation.

Tests whether the pre-trained mtDNA-FM encoder's variant-position hidden
states separate pathogenic from benign variants without any supervised
fine-tuning on pathogenicity labels.

Pipeline:
  1. Download ClinVar chrM VCF + gnomAD chrM VCF (idempotent)
  2. Parse into labeled DataFrame (pathogenic=1, benign=0)
  3. Load rCRS reference (NC_012920.1)
  4. For each variant: apply alt to rCRS, embed with pre-trained encoder
  5. 5-fold stratified k-NN (k=5, cosine distance)
  6. Compute AUROC, AUPRC, per-variant-type breakdown
  7. Save to reports/zeroshot_pathogenicity_knn.json

Usage:
  uv run python scripts/zeroshot_patho_eval.py
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mtdna_fm.data.variant_downloader import (
    GNOMAD_BGZ_FILENAME,
    GNOMAD_BGZ_URL,
    GNOMAD_VCF_FILENAME,
    download_clinvar_chrm,
)
from mtdna_fm.data.variant_processor import (
    add_benign_proxies,
    parse_clinvar_chrm_vcf,
    parse_gnomad_chrm_vcf,
)
from mtdna_fm.evaluation.variant_eval import compute_metrics
from mtdna_fm.inference.api import MtDNAEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLINVAR_DIR = RAW_DIR / "clinvar"
GNOMAD_DIR = RAW_DIR / "gnomad"
NCBI_FASTA = RAW_DIR / "ncbi" / "vertebrate_mtdna.fasta"
_local_model = PROJECT_ROOT / "models" / "phase1_v1"
MODEL_PATH = _local_model if (_local_model / "config.json").exists() else "vthawfeek/mtdna-foundation-model"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_JSON = REPORTS_DIR / "zeroshot_pathogenicity_knn.json"
EMBEDDINGS_NPZ = REPORTS_DIR / "zeroshot_patho_embeddings.npz"


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------


def _stream_download(url: str, dest: Path, desc: str = "") -> None:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with (
        open(dest, "wb") as fh,
        tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
            bar.update(len(chunk))


def download_gnomad_no_tabix(output_dir: Path) -> Path:
    """Download gnomAD chrM .bgz and decompress to plain-text VCF (no tabix needed)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vcf_path = output_dir / GNOMAD_VCF_FILENAME
    if vcf_path.exists():
        log.info("gnomAD chrM VCF already exists: %s", vcf_path)
        return vcf_path

    bgz_path = output_dir / GNOMAD_BGZ_FILENAME
    if not bgz_path.exists():
        log.info("Downloading gnomAD chrM .bgz …")
        _stream_download(GNOMAD_BGZ_URL, bgz_path, desc="gnomAD chrM")

    log.info("Decompressing gnomAD .bgz → plain VCF …")
    with gzip.open(bgz_path, "rt", errors="replace") as fin, open(vcf_path, "w") as fout:
        for line in fin:
            fout.write(line)
    log.info("gnomAD chrM VCF written: %s  (%d bytes)", vcf_path, vcf_path.stat().st_size)
    return vcf_path


# ---------------------------------------------------------------------------
# Reference sequence
# ---------------------------------------------------------------------------


def load_rcrs(fasta_path: Path) -> str:
    """Extract NC_012920.1 (rCRS) from the NCBI vertebrate mtDNA fasta, or fetch from Entrez."""
    from Bio import SeqIO

    if fasta_path.exists():
        for rec in SeqIO.parse(str(fasta_path), "fasta"):
            if "NC_012920" in rec.id:
                seq = str(rec.seq).upper()
                log.info("Loaded rCRS %s: %d bp", rec.id, len(seq))
                return seq

    # Fallback: download NC_012920.1 directly from NCBI Entrez
    log.info("Vertebrate FASTA not found; fetching NC_012920.1 from NCBI Entrez ...")
    from Bio import Entrez
    Entrez.email = "vthawfeek@gmail.com"
    handle = Entrez.efetch(db="nucleotide", id="NC_012920.1", rettype="fasta", retmode="text")
    rec = SeqIO.read(handle, "fasta")
    handle.close()
    seq = str(rec.seq).upper()
    log.info("Fetched rCRS %s from Entrez: %d bp", rec.id, len(seq))
    # Cache it locally so subsequent runs don't need to re-download
    rcrs_cache = fasta_path.parent / "rcrs_NC_012920.fasta"
    rcrs_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(rcrs_cache, "w") as fh:
        fh.write(f">{rec.id} {rec.description}\n{seq}\n")
    log.info("Cached rCRS at %s", rcrs_cache)
    return seq


def apply_snp(rcrs: str, pos_1based: int, alt: str) -> str:
    """Return a copy of rcrs with alt substituted at pos_1based (1-indexed)."""
    i = pos_1based - 1
    return rcrs[:i] + alt.upper() + rcrs[i + 1 :]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_variants(df, rcrs: str, embedder: MtDNAEmbedder) -> np.ndarray:
    """
    Embed every variant in df using the pre-trained encoder's variant-token hidden state.
    df must have columns: pos (1-based int), alt (str).
    Returns ndarray of shape (n_variants, hidden_size).
    """
    embeddings: list[np.ndarray] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding variants", unit="var"):
        mutated = apply_snp(rcrs, int(row["pos"]), str(row["alt"]))
        pos_0 = int(row["pos"]) - 1
        vec = embedder.embed_variant(mutated, pos_0, pooling="token")
        embeddings.append(vec)
    return np.stack(embeddings, axis=0)


# ---------------------------------------------------------------------------
# k-NN cross-validation
# ---------------------------------------------------------------------------


def run_knn_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    k: int = 5,
) -> np.ndarray:
    """
    Stratified k-fold CV with cosine k-NN.
    Returns out-of-fold probability scores (n_samples,) for AUROC computation.
    Aggregating across folds before computing AUROC is more stable than
    averaging per-fold AUROCs on small datasets.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize

    X_norm = normalize(X, norm="l2")
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = np.zeros(len(y), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_norm, y), 1):
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
        knn.fit(X_norm[train_idx], y[train_idx])
        proba = knn.predict_proba(X_norm[test_idx])
        pos_col = list(knn.classes_).index(1)
        scores[test_idx] = proba[:, pos_col]
        log.info("  Fold %d/%d done", fold_idx, n_splits)

    return scores


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) 95% bootstrap CI for AUROC."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        try:
            aurocs.append(float(roc_auc_score(y_true[idx], y_score[idx])))
        except Exception:
            continue
    lo = float(np.percentile(aurocs, 100 * alpha / 2))
    hi = float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()

    # ── 1. Download ───────────────────────────────────────────────────────────
    log.info("=== Step 1/7: Download variant data ===")
    clinvar_vcf = download_clinvar_chrm(CLINVAR_DIR)
    gnomad_vcf = download_gnomad_no_tabix(GNOMAD_DIR)

    # ── 2. Parse VCFs ─────────────────────────────────────────────────────────
    log.info("=== Step 2/7: Parse VCFs ===")
    pathogenic_df = parse_clinvar_chrm_vcf(clinvar_vcf)
    gnomad_df = parse_gnomad_chrm_vcf(gnomad_vcf)
    labeled_df = add_benign_proxies(pathogenic_df, gnomad_df, af_threshold=0.01)

    n_pos = int((labeled_df["label"] == 1).sum())
    n_neg = int((labeled_df["label"] == 0).sum())
    log.info("Dataset: %d pathogenic, %d benign (%d total)", n_pos, n_neg, len(labeled_df))

    if n_pos < 10:
        log.error("Only %d pathogenic variants — too few for a meaningful evaluation.", n_pos)
        sys.exit(1)

    # ── 3. Load rCRS ──────────────────────────────────────────────────────────
    log.info("=== Step 3/7: Load rCRS reference ===")
    rcrs = load_rcrs(NCBI_FASTA)

    # ── 4. Load encoder ───────────────────────────────────────────────────────
    log.info("=== Step 4/7: Load pre-trained encoder (%s) ===", MODEL_PATH)
    embedder = MtDNAEmbedder.from_pretrained(str(MODEL_PATH))
    log.info("Encoder ready  device=%s", embedder.device)

    # ── 5. Embed variants ─────────────────────────────────────────────────────
    log.info("=== Step 5/7: Embed %d variants ===", len(labeled_df))
    if EMBEDDINGS_NPZ.exists():
        log.info("Loading cached embeddings from %s", EMBEDDINGS_NPZ)
        cached = np.load(EMBEDDINGS_NPZ)
        X = cached["X"]
        y = cached["y"]
        positions = cached["positions"]
    else:
        X = embed_variants(labeled_df, rcrs, embedder)
        y = labeled_df["label"].values.astype(int)
        positions = labeled_df["pos"].values.astype(int)
        np.savez(EMBEDDINGS_NPZ, X=X, y=y, positions=positions)
        log.info("Embeddings cached to %s", EMBEDDINGS_NPZ)

    # ── 6. k-NN cross-validation ──────────────────────────────────────────────
    n_splits = min(5, n_pos)
    log.info("=== Step 6/7: %d-fold stratified k-NN (k=5, cosine) ===", n_splits)
    scores = run_knn_cv(X, y, n_splits=n_splits, k=5)

    # ── 7. Metrics ────────────────────────────────────────────────────────────
    log.info("=== Step 7/7: Compute metrics ===")
    metrics = compute_metrics(y, scores, positions=positions.tolist())
    ci_lo, ci_hi = bootstrap_auroc_ci(y, scores)

    auroc = metrics["auroc"]
    auprc = metrics["auprc"]
    random_auprc = n_pos / (n_pos + n_neg)
    elapsed = time.time() - t0

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  ZERO-SHOT k-NN PATHOGENICITY EVALUATION")
    print("=" * 62)
    print(f"  Dataset:  {n_pos} pathogenic  |  {n_neg} benign  ({n_pos + n_neg} total)")
    print(f"  AUROC:    {auroc:.3f}  (95% CI: {ci_lo:.3f}–{ci_hi:.3f})")
    print(f"  AUPRC:    {auprc:.3f}")
    print(f"  Baseline: AUROC=0.500  AUPRC={random_auprc:.3f}  (random)")
    print()
    per_type = metrics.get("per_type", {})
    if per_type:
        print("  Per-variant-type AUROC:")
        for vtype, stats in sorted(per_type.items()):
            a = stats.get("auroc")
            if a is not None:
                print(
                    f"    {vtype:10s}  AUROC={a:.3f}"
                    f"  (n_pos={stats['n_pos']}, n_neg={stats['n_neg']})"
                )
    print("=" * 62)
    print(f"  Elapsed:  {elapsed / 60:.1f} min")
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "auroc": auroc,
        "auprc": auprc,
        "auroc_ci_95_lo": ci_lo,
        "auroc_ci_95_hi": ci_hi,
        "n_pathogenic": n_pos,
        "n_benign": n_neg,
        "knn_k": 5,
        "knn_metric": "cosine",
        "cv_folds": n_splits,
        "model": str(MODEL_PATH),
        "random_auroc": 0.5,
        "random_auprc": random_auprc,
        "per_type": per_type,
        "roc_curve": metrics.get("roc_curve"),
        "pr_curve": metrics.get("pr_curve"),
        "elapsed_seconds": round(elapsed, 1),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2))
    log.info("Results saved → %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
