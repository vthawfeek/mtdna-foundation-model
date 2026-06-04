"""
G8: External pathogenicity validation on MITOMAP and HelixMTdb.

Downloads MITOMAP confirmed pathogenic variants and HelixMTdb population variants,
then evaluates the pathogenicity model AUROC on this independent set (no overlap
with ClinVar/gnomAD used in training).

Usage:
    uv run python paper/experiments/evaluation/external_pathogenicity_eval.py

Outputs:
    paper/experiments/evaluation/external_pathogenicity_results.json
    paper/experiments/evaluation/mitomap_variants.parquet
    paper/experiments/evaluation/helixmtdb_variants.parquet
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

MITOMAP_URL = "https://www.mitomap.org/foswiki/bin/view/MITOMAP/PolymorphismsConfirmed"
HELIXMTDB_URL = "https://helix.com/helixmtdb"  # check current URL in paper

# rCRS reference sequence (will be fetched from local cache or NCBI)
RCRS_ACCESSION = "NC_012920.1"


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def load_rcrs_sequence() -> str:
    """Load the rCRS reference sequence from local cache or download from NCBI."""
    rcrs_path = Path("data/raw/reference/rCRS.fasta")
    if rcrs_path.exists():
        seq = ""
        with open(rcrs_path) as f:
            for line in f:
                if not line.startswith(">"):
                    seq += line.strip()
        logger.info(f"Loaded rCRS from {rcrs_path} ({len(seq)} bp)")
        return seq.upper()

    logger.info("Downloading rCRS from NCBI...")
    from Bio import Entrez, SeqIO
    Entrez.email = "user@example.com"
    handle = Entrez.efetch(db="nucleotide", id=RCRS_ACCESSION, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    seq = str(record.seq).upper()
    rcrs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rcrs_path, "w") as f:
        f.write(f">rCRS {RCRS_ACCESSION}\n{seq}\n")
    logger.info(f"rCRS downloaded: {len(seq)} bp")
    return seq


def apply_variant_to_sequence(reference: str, position: int, alt: str) -> str:
    """Apply a SNP to the reference sequence.

    position: 1-based rCRS position (converted to 0-based internally)
    alt: alternate allele (single base for SNPs)
    """
    pos0 = position - 1
    if pos0 < 0 or pos0 >= len(reference):
        raise ValueError(f"Position {position} out of range [1, {len(reference)}]")
    return reference[:pos0] + alt + reference[pos0 + 1:]


def load_mitomap_pathogenic() -> pd.DataFrame:
    """Load MITOMAP confirmed pathogenic variants.

    MITOMAP provides a curated list of variants with confirmed disease associations.
    Download from: https://www.mitomap.org/foswiki/bin/view/MITOMAP/PolymorphismsConfirmed

    The downloaded file is expected at:
        data/raw/mitomap/mitomap_confirmed_pathogenic.tsv
    """
    local_path = Path("data/raw/mitomap/mitomap_confirmed_pathogenic.tsv")
    if not local_path.exists():
        logger.warning(
            f"MITOMAP data not found at {local_path}.\n"
            "Download manually from https://www.mitomap.org and save to that path.\n"
            "Expected columns: Position, Reference, Allele, Disease, Status"
        )
        return pd.DataFrame()

    df = pd.read_csv(local_path, sep="\t")
    logger.info(f"Loaded {len(df)} MITOMAP variants")

    # Standardize columns
    col_map = {
        "Position": "position",
        "Reference": "ref",
        "Allele": "alt",
        "Disease": "disease",
        "Status": "status",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Filter to confirmed pathogenic (excluding "Reported" which may be preliminary)
    if "status" in df.columns:
        confirmed = df[df["status"].str.contains("Confirmed", case=False, na=False)]
        logger.info(f"  Confirmed pathogenic: {len(confirmed)}")
        return confirmed

    return df


def load_helixmtdb_common() -> pd.DataFrame:
    """Load HelixMTdb common variants (AF > 0.001) as benign examples.

    HelixMTdb is available from: https://helix.com/helixmtdb
    Expected format: TSV with columns pos, ref, alt, AF_hom, AF_het

    Save to: data/raw/helixmtdb/helixmtdb_variants.tsv
    """
    local_path = Path("data/raw/helixmtdb/helixmtdb_variants.tsv")
    if not local_path.exists():
        logger.warning(
            f"HelixMTdb data not found at {local_path}.\n"
            "Download from https://helix.com/helixmtdb and save to that path."
        )
        return pd.DataFrame()

    df = pd.read_csv(local_path, sep="\t")
    logger.info(f"Loaded {len(df)} HelixMTdb variants")

    # Use high-AF variants (likely benign, no known pathogenic association)
    if "AF_hom" in df.columns and "AF_het" in df.columns:
        df["AF"] = df["AF_hom"] + df["AF_het"]
    elif "AF" not in df.columns:
        df["AF"] = 0.0

    common = df[df["AF"] > 0.001]
    logger.info(f"  Common variants (AF > 0.001): {len(common)}")

    # Exclude any variants that appear in ClinVar pathogenic set
    clinvar_path = Path("data/raw/clinvar/clinvar_mtdna_pathogenic.vcf")
    if clinvar_path.exists():
        clinvar_positions = set()
        with open(clinvar_path) as f:
            for line in f:
                if not line.startswith("#"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        clinvar_positions.add((int(parts[1]), parts[3], parts[4]))
        if "pos" in common.columns:
            before = len(common)
            common = common[
                ~common.apply(
                    lambda r: (int(r["pos"]), str(r.get("ref", "")), str(r.get("alt", "")))
                    in clinvar_positions,
                    axis=1,
                )
            ]
            logger.info(f"  After excluding ClinVar overlap: {len(common)} (removed {before - len(common)})")

    return common


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def build_variant_sequences(
    variants_df: pd.DataFrame,
    reference: str,
    label: int,
    pos_col: str = "position",
    alt_col: str = "alt",
    context_radius: int = 512,
) -> pd.DataFrame:
    """Build sequence context for each variant by applying it to the reference.

    Returns DataFrame with columns: sequence, position, label
    """
    records = []
    for _, row in variants_df.iterrows():
        try:
            pos = int(row[pos_col])
            alt = str(row[alt_col]).strip().upper()
            if len(alt) != 1 or alt not in "ACGT":
                continue  # skip non-SNP variants

            mutant_seq = apply_variant_to_sequence(reference, pos, alt)
            # Center context window on variant position
            start = max(0, pos - context_radius // 2)
            end = min(len(mutant_seq), pos + context_radius // 2)
            context = mutant_seq[start:end]

            records.append({
                "sequence": context,
                "position": pos - start,  # variant position within context
                "label": label,
                "rCRS_position": pos,
                "ref_allele": reference[pos - 1],
                "alt_allele": alt,
            })
        except Exception as e:
            logger.debug(f"Skipping variant at pos {row.get(pos_col, '?')}: {e}")

    return pd.DataFrame(records)


def run_pathogenicity_model(sequences: list[str], positions: list[int]) -> np.ndarray:
    """Run the pathogenicity model and return probability scores."""
    import torch
    from mtdna_fm.tokenizer import KmerVocabulary
    from mtdna_fm.finetune.pathogenicity import MtDNAForVariantPathogenicity

    vocab = KmerVocabulary.from_pretrained("models/vocabulary")
    model_path = "models/finetune_pathogenicity_v1"
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Pathogenicity model not found: {model_path}\n"
            "Run: uv run mtdna-finetune --task pathogenicity first"
        )
    model = MtDNAForVariantPathogenicity.from_pretrained(model_path)
    model.eval()

    scores = []
    with torch.no_grad():
        for seq, pos in zip(sequences, positions):
            tokens = vocab.tokenize(seq)
            input_ids = torch.tensor(tokens).unsqueeze(0)
            variant_positions = torch.tensor([[pos]])
            out = model(input_ids=input_ids, variant_positions=variant_positions)
            prob = torch.sigmoid(out.logits).item()
            scores.append(prob)

    return np.array(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from sklearn.metrics import roc_auc_score, average_precision_score

    reference = load_rcrs_sequence()

    mitomap_df = load_mitomap_pathogenic()
    helixmtdb_df = load_helixmtdb_common()

    if mitomap_df.empty and helixmtdb_df.empty:
        logger.error("No external data available. Download MITOMAP and HelixMTdb first.")
        return

    records = []
    if not mitomap_df.empty:
        pos_col = "position" if "position" in mitomap_df.columns else "Position"
        alt_col = "alt" if "alt" in mitomap_df.columns else "Allele"
        pathogenic_seqs = build_variant_sequences(mitomap_df, reference, label=1, pos_col=pos_col, alt_col=alt_col)
        pathogenic_seqs.to_parquet(EVAL_DIR / "mitomap_variants.parquet", index=False)
        records.append(pathogenic_seqs)
        logger.info(f"Built {len(pathogenic_seqs)} pathogenic variant sequences from MITOMAP")

    if not helixmtdb_df.empty:
        pos_col = "pos" if "pos" in helixmtdb_df.columns else "position"
        alt_col = "alt" if "alt" in helixmtdb_df.columns else "ALT"
        benign_seqs = build_variant_sequences(helixmtdb_df, reference, label=0, pos_col=pos_col, alt_col=alt_col)
        benign_seqs.to_parquet(EVAL_DIR / "helixmtdb_variants.parquet", index=False)
        records.append(benign_seqs)
        logger.info(f"Built {len(benign_seqs)} benign variant sequences from HelixMTdb")

    if not records:
        logger.error("No variant sequences built. Check input data formats.")
        return

    eval_df = pd.concat(records, ignore_index=True)
    logger.info(f"Total evaluation variants: {len(eval_df)} ({eval_df['label'].sum()} pathogenic)")

    try:
        scores = run_pathogenicity_model(eval_df["sequence"].tolist(), eval_df["position"].tolist())
        y_true = eval_df["label"].values

        auroc = roc_auc_score(y_true, scores)
        aupr = average_precision_score(y_true, scores)
        logger.info(f"External AUROC: {auroc:.4f}")
        logger.info(f"External AUPR:  {aupr:.4f}")

        results = {
            "status": "completed",
            "n_pathogenic": int(y_true.sum()),
            "n_benign": int((1 - y_true).sum()),
            "auroc": float(auroc),
            "aupr": float(aupr),
            "sources": {
                "pathogenic": "MITOMAP confirmed",
                "benign": "HelixMTdb (AF > 0.001)",
            },
        }
    except FileNotFoundError as e:
        logger.warning(str(e))
        results = {"status": "model_not_found"}
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        results = {"status": "error", "error": str(e)}

    output_path = EVAL_DIR / "external_pathogenicity_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
