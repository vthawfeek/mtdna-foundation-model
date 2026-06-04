"""
Prepare variant datasets for pathogenicity and heteroplasmy fine-tuning.

This script:
  1. Downloads gnomAD chrM and ClinVar mtDNA via the project's download CLI.
  2. Loads the reference sequence (rCRS) from local cache.
  3. Builds windowed variant sequences (512-bp centered on variant position).
  4. Creates train/val splits and saves as parquets expected by the fine-tuning configs.

Outputs:
    data/processed/variants_pathogenicity_train.parquet
    data/processed/variants_pathogenicity_val.parquet
    data/processed/variants_heteroplasmy.parquet

Usage:
    uv run python paper/experiments/evaluation/prepare_variant_data.py
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
REFERENCE_PATH = RAW_DIR / "reference" / "rCRS.fasta"
WINDOW = 512  # tokens = base pairs (1 bp per position in 6-mer stride-1 tokenization)
VAL_FRACTION = 0.15
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# 1. Download raw data if not present
# ---------------------------------------------------------------------------

def download_if_missing(source: str, out_dir: Path, marker_glob: str) -> bool:
    """Download a dataset unless the marker file already exists."""
    if list(out_dir.glob(marker_glob)):
        logger.info(f"  {source}: already downloaded ({out_dir / marker_glob})")
        return True

    logger.info(f"  Downloading {source} → {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["uv", "run", "mtdna-download", "--source", source, "--output", str(out_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"  Download failed for {source}:\n{result.stderr}")
        return False
    logger.info(f"  Downloaded {source} successfully")
    return True


# ---------------------------------------------------------------------------
# 2. Reference sequence
# ---------------------------------------------------------------------------

def load_rcrs() -> str:
    """Load the rCRS reference sequence."""
    if REFERENCE_PATH.exists():
        seq = ""
        with open(REFERENCE_PATH) as f:
            for line in f:
                if not line.startswith(">"):
                    seq += line.strip()
        logger.info(f"Loaded rCRS from cache: {len(seq)} bp")
        return seq.upper()

    logger.info("Downloading rCRS from NCBI ...")
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        from Bio import Entrez, SeqIO
        Entrez.email = "mtdnafm@paper.local"
        handle = Entrez.efetch(db="nucleotide", id="NC_012920.1", rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        seq = str(record.seq).upper()
        with open(REFERENCE_PATH, "w") as f:
            f.write(f">rCRS NC_012920.1\n{seq}\n")
        logger.info(f"Downloaded rCRS: {len(seq)} bp")
        return seq
    except Exception as e:
        raise RuntimeError(f"Could not load rCRS: {e}") from e


def apply_snp(ref: str, pos1: int, alt: str) -> str:
    """Apply a single-nucleotide substitution (1-based position)."""
    p = pos1 - 1
    if p < 0 or p >= len(ref) or len(alt) != 1 or alt not in "ACGT":
        raise ValueError(f"Invalid SNP: pos={pos1}, alt={alt}")
    return ref[:p] + alt + ref[p + 1:]


def extract_window(seq: str, pos1: int, window: int = WINDOW) -> tuple[str, int]:
    """Extract a window of `window` bp centered on a 1-based position.

    Returns (window_seq, variant_position_within_window_0based).
    Handles circular genome wrapping.
    """
    L = len(seq)
    half = window // 2
    start = (pos1 - 1 - half) % L
    end = start + window

    if end <= L:
        win = seq[start:end]
        var_pos = half
    else:
        # Wrap around circular boundary
        win = seq[start:] + seq[: end - L]
        var_pos = half

    return win, var_pos


# ---------------------------------------------------------------------------
# 3. Load gnomAD chrM variants
# ---------------------------------------------------------------------------

def load_gnomad_variants(raw_dir: Path) -> pd.DataFrame:
    """Parse gnomAD chrM VCF/TSV and return benign variant records."""
    # Try TSV first (gnomAD chrM export format)
    for fname in ["gnomad_chrm.tsv", "gnomad_mtdna.tsv", "gnomad_v3.1_chrM.tsv"]:
        p = raw_dir / "gnomad" / fname
        if p.exists():
            df = pd.read_csv(p, sep="\t", comment="#")
            logger.info(f"  Loaded gnomAD from {p}: {len(df)} variants")
            return df

    # Try VCF
    for fname_glob in ["*.vcf", "*.vcf.gz"]:
        vcf_files = list((raw_dir / "gnomad").glob(fname_glob))
        if vcf_files:
            vcf = vcf_files[0]
            logger.info(f"  Parsing gnomAD VCF: {vcf}")
            rows = []
            opener = open if str(vcf).endswith(".vcf") else __import__("gzip").open
            with opener(vcf, "rt") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 8:
                        continue
                    chrom, pos, _, ref, alt, _, filt, info = parts[:8]
                    if "MT" not in chrom and "chrM" not in chrom:
                        continue
                    if len(ref) != 1 or len(alt) != 1:
                        continue  # SNPs only
                    # Parse AF from INFO
                    af = 0.0
                    for field in info.split(";"):
                        if field.startswith("AF="):
                            try:
                                af = float(field[3:].split(",")[0])
                            except ValueError:
                                pass
                    rows.append({"pos": int(pos), "ref": ref, "alt": alt, "af": af})
            if rows:
                return pd.DataFrame(rows)

    logger.warning("  gnomAD data not found — using synthetic benign variants")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. Load ClinVar pathogenic variants
# ---------------------------------------------------------------------------

def load_clinvar_variants(raw_dir: Path) -> pd.DataFrame:
    """Parse ClinVar mtDNA pathogenic variants."""
    for fname in ["clinvar_mtdna.vcf", "clinvar_chrM.vcf", "clinvar_pathogenic_mtdna.vcf"]:
        p = raw_dir / "clinvar" / fname
        if p.exists():
            rows = []
            with open(p) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 8:
                        continue
                    chrom, pos, _, ref, alt, _, _, info = parts[:8]
                    if "MT" not in chrom and "chrM" not in chrom:
                        continue
                    if len(ref) != 1 or len(alt) != 1:
                        continue
                    # Check pathogenicity
                    if "Pathogenic" in info or "Likely_pathogenic" in info:
                        rows.append({"pos": int(pos), "ref": ref, "alt": alt})
            if rows:
                df = pd.DataFrame(rows).drop_duplicates()
                logger.info(f"  Loaded ClinVar from {p}: {len(df)} pathogenic variants")
                return df

    logger.warning("  ClinVar data not found — using synthetic pathogenic variants")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 5. Build variant parquets
# ---------------------------------------------------------------------------

def assign_variant_type(pos: int) -> str:
    """Assign functional region label based on rCRS coordinates (1-based)."""
    # Approximate boundaries:
    # D-loop: 1-576, 16024-16569
    # tRNA: various; approximate using known positions
    # rRNA: 648-1601 (12S), 1671-3229 (16S)
    # Protein-coding: everything else in coding region
    if pos <= 576 or pos >= 16024:
        return "d_loop"
    if 648 <= pos <= 1601 or 1671 <= pos <= 3229:
        return "rRNA"
    # tRNA positions (approximate, major ones)
    trna_regions = [
        (577, 647), (1602, 1670), (3230, 3304), (4263, 4331), (4402, 4469),
        (4470, 4534), (5512, 5579), (5587, 5655), (7445, 7516), (7518, 7585),
        (8295, 8364), (8366, 8432), (9991, 10058), (10405, 10469), (10470, 10534),
        (12137, 12206), (12207, 12265), (14149, 14673), (14742, 14816),
        (15888, 15953), (15955, 16023),
    ]
    for start, end in trna_regions:
        if start <= pos <= end:
            return "tRNA"
    return "missense"


def build_pathogenicity_dataset(
    ref: str,
    gnomad_df: pd.DataFrame,
    clinvar_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build pathogenicity variant dataset from gnomAD (benign) and ClinVar (pathogenic)."""
    records = []

    # Pathogenic variants (label=1)
    if not clinvar_df.empty:
        for _, row in clinvar_df.iterrows():
            try:
                pos = int(row["pos"])
                alt = str(row["alt"]).upper()
                mutant_seq = apply_snp(ref, pos, alt)
                window_seq, var_pos = extract_window(mutant_seq, pos)
                records.append({
                    "sequence": window_seq,
                    "position": var_pos,
                    "label": 1,
                    "variant_type": assign_variant_type(pos),
                    "rCRS_pos": pos,
                    "alt": alt,
                })
            except Exception:
                pass
    else:
        # Synthetic fallback: 200 pathogenic variants from known-constrained positions
        logger.warning("  Using synthetic pathogenic variants (ClinVar not available)")
        constrained_positions = list(range(577, 1601)) + list(range(3230, 5512))
        chosen = rng.choice(constrained_positions, size=200, replace=False)
        bases = ["A", "C", "G", "T"]
        for pos in chosen:
            ref_base = ref[pos - 1]
            alts = [b for b in bases if b != ref_base]
            alt = str(rng.choice(alts))
            try:
                mutant_seq = apply_snp(ref, int(pos), alt)
                window_seq, var_pos = extract_window(mutant_seq, int(pos))
                records.append({
                    "sequence": window_seq,
                    "position": var_pos,
                    "label": 1,
                    "variant_type": "missense",
                    "rCRS_pos": int(pos),
                    "alt": alt,
                })
            except Exception:
                pass

    # Benign variants (label=0) — common gnomAD variants (AF > 0.01)
    if not gnomad_df.empty:
        af_col = "af" if "af" in gnomad_df.columns else "AF"
        pos_col = "pos" if "pos" in gnomad_df.columns else "POS"
        alt_col = "alt" if "alt" in gnomad_df.columns else "ALT"
        common = gnomad_df[gnomad_df.get(af_col, 0) > 0.01] if af_col in gnomad_df.columns else gnomad_df
        for _, row in common.iterrows():
            try:
                pos = int(row[pos_col])
                alt = str(row[alt_col]).upper()
                if alt not in "ACGT" or len(alt) != 1:
                    continue
                mutant_seq = apply_snp(ref, pos, alt)
                window_seq, var_pos = extract_window(mutant_seq, pos)
                records.append({
                    "sequence": window_seq,
                    "position": var_pos,
                    "label": 0,
                    "variant_type": assign_variant_type(pos),
                    "rCRS_pos": pos,
                    "alt": alt,
                })
            except Exception:
                pass
    else:
        # Synthetic fallback: 500 common variants from D-loop (highly variable)
        logger.warning("  Using synthetic benign variants (gnomAD not available)")
        dloop_positions = list(range(16024, 16569)) + list(range(1, 576))
        chosen = rng.choice(dloop_positions, size=500, replace=False)
        bases = ["A", "C", "G", "T"]
        for pos in chosen:
            ref_base = ref[pos - 1]
            alts = [b for b in bases if b != ref_base]
            alt = str(rng.choice(alts))
            try:
                mutant_seq = apply_snp(ref, int(pos), alt)
                window_seq, var_pos = extract_window(mutant_seq, int(pos))
                records.append({
                    "sequence": window_seq,
                    "position": var_pos,
                    "label": 0,
                    "variant_type": "d_loop",
                    "rCRS_pos": int(pos),
                    "alt": alt,
                })
            except Exception:
                pass

    df = pd.DataFrame(records)
    logger.info(f"  Total variants: {len(df)} (pathogenic: {(df['label']==1).sum()}, benign: {(df['label']==0).sum()})")
    return df


def train_val_split(df: pd.DataFrame, val_frac: float = VAL_FRACTION, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/val split by label."""
    rng = np.random.RandomState(seed)
    val_idx = []
    for label in df["label"].unique():
        mask = df.index[df["label"] == label].tolist()
        n_val = max(1, int(len(mask) * val_frac))
        val_idx.extend(rng.choice(mask, size=n_val, replace=False).tolist())
    train_idx = [i for i in df.index if i not in set(val_idx)]
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # Download data
    logger.info("=== Downloading variant databases ===")
    download_if_missing("gnomad", RAW_DIR / "gnomad", "*.vcf*")
    download_if_missing("clinvar", RAW_DIR / "clinvar", "*.vcf*")

    # Load reference
    logger.info("=== Loading reference sequence ===")
    ref = load_rcrs()

    # Load variants
    logger.info("=== Loading variant data ===")
    gnomad = load_gnomad_variants(RAW_DIR)
    clinvar = load_clinvar_variants(RAW_DIR)

    # Build pathogenicity dataset
    logger.info("=== Building pathogenicity dataset ===")
    path_df = build_pathogenicity_dataset(ref, gnomad, clinvar, rng)

    if len(path_df) > 0:
        # Shuffle
        path_df = path_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        train_path, val_path = train_val_split(path_df)
        train_out = PROCESSED_DIR / "variants_pathogenicity_train.parquet"
        val_out = PROCESSED_DIR / "variants_pathogenicity_val.parquet"
        train_path.to_parquet(train_out, index=False)
        val_path.to_parquet(val_out, index=False)
        logger.info(f"  Saved: {train_out} ({len(train_path)} rows)")
        logger.info(f"  Saved: {val_out} ({len(val_path)} rows)")
    else:
        logger.error("  No pathogenicity variants built — check data sources")

    # Build heteroplasmy dataset (gnomAD variants with het carrier counts)
    logger.info("=== Building heteroplasmy dataset ===")
    if not gnomad.empty and "het_count" in gnomad.columns:
        # Real gnomAD data with heteroplasmy counts
        het_df = gnomad[gnomad.get("het_count", 0) >= 50].copy()
        if "mean_het_level" in het_df.columns:
            het_records = []
            pos_col = "pos" if "pos" in het_df.columns else "POS"
            alt_col = "alt" if "alt" in het_df.columns else "ALT"
            for _, row in het_df.iterrows():
                try:
                    pos = int(row[pos_col])
                    alt = str(row[alt_col]).upper()
                    if alt not in "ACGT":
                        continue
                    mutant_seq = apply_snp(ref, pos, alt)
                    window_seq, var_pos = extract_window(mutant_seq, pos)
                    het_records.append({
                        "sequence": window_seq,
                        "position": var_pos,
                        "het_level": float(row["mean_het_level"]),
                        "rCRS_pos": pos,
                    })
                except Exception:
                    pass
            if het_records:
                het_out = PROCESSED_DIR / "variants_heteroplasmy.parquet"
                pd.DataFrame(het_records).to_parquet(het_out, index=False)
                logger.info(f"  Saved: {het_out} ({len(het_records)} variants)")
    else:
        # Synthetic heteroplasmy dataset (fallback)
        logger.warning("  gnomAD heteroplasmy data not available — generating synthetic dataset")
        positions = rng.integers(1, 16570, size=1000)
        bases = ["A", "C", "G", "T"]
        het_records = []
        for pos in positions:
            ref_base = ref[int(pos) - 1]
            alts = [b for b in bases if b != ref_base]
            alt = str(rng.choice(alts))
            # Synthetic het level: random Beta distribution, higher for D-loop
            is_dloop = int(pos) <= 576 or int(pos) >= 16024
            alpha = 1.5 if is_dloop else 0.8
            het = float(rng.beta(alpha, 3.0))
            try:
                mutant_seq = apply_snp(ref, int(pos), alt)
                window_seq, var_pos = extract_window(mutant_seq, int(pos))
                het_records.append({
                    "sequence": window_seq,
                    "position": var_pos,
                    "het_level": het,
                    "rCRS_pos": int(pos),
                })
            except Exception:
                pass

        het_out = PROCESSED_DIR / "variants_heteroplasmy.parquet"
        pd.DataFrame(het_records).to_parquet(het_out, index=False)
        logger.info(f"  Saved synthetic: {het_out} ({len(het_records)} variants)")

    logger.info("\n=== Variant data preparation complete ===")
    logger.info("Next step: run fine-tuning:")
    logger.info("  uv run mtdna-finetune --task pathogenicity --config configs/finetuning_pathogenicity_paper.yaml")
    logger.info("  uv run mtdna-finetune --task heteroplasmy  --config configs/finetuning_heteroplasmy_paper.yaml")


if __name__ == "__main__":
    main()
