"""
Preprocessing pipeline for mtDNA sequences.

Each step is a separate function so individual steps are testable and reusable.
The pipeline runs: clean -> normalize_length -> stratified_split -> save_splits.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# Human rCRS (revised Cambridge Reference Sequence) length
RCRS_LENGTH = 16569
# D-loop start position (0-indexed); padding inserted here so protein-coding
# gene coordinates remain at their canonical positions after padding.
DLOOP_PAD_POSITION = 576
# Number of bases to check for trailing circular junction duplicate
JUNCTION_DUPLICATE_CHECK_BASES = 200


def clean_sequence(seq: str) -> str:
    """
    Sanitise a raw sequence string.

    Steps:
    1. Uppercase.
    2. Replace any character that is not A/C/G/T/N with N.
    3. Strip trailing junction duplicate: some databases (HmtDB included)
       append the first JUNCTION_DUPLICATE_CHECK_BASES bases to the end so
       the circular junction is covered by every analysis window. We remove
       that suffix when detected.
    """
    seq = seq.upper()
    seq = re.sub(r"[^ACGTN]", "N", seq)
    n = JUNCTION_DUPLICATE_CHECK_BASES
    if len(seq) > 2 * n and seq[:n] == seq[-n:]:
        seq = seq[:-n]
    return seq


def normalize_length(
    seq: str,
    target_length: int = RCRS_LENGTH,
    pad_position: int = DLOOP_PAD_POSITION,
) -> str:
    """
    Pad or trim a sequence to exactly target_length.

    Padding (N characters) is inserted at pad_position (the D-loop start)
    rather than appended to the end, so protein-coding gene coordinates
    remain at their canonical positions. Excess length is trimmed from the
    3' end.
    """
    current = len(seq)
    if current == target_length:
        return seq
    if current < target_length:
        n_pad = target_length - current
        insert_at = min(pad_position, current)
        seq = seq[:insert_at] + "N" * n_pad + seq[insert_at:]
    else:
        seq = seq[:target_length]
    return seq


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "haplogroup",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add a 'split' column ('train'/'val'/'test') to df.

    Rows with a non-null, non-empty label are split with StratifiedShuffleSplit
    so each partition has proportional haplogroup representation. Rows without
    a haplogroup label (cross-species sequences for Phase 1 MLM pre-training)
    always go to 'train'.
    """
    df = df.copy()
    df["split"] = "train"

    has_label = df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")
    labeled_idx = df.index[has_label].to_numpy()

    if len(labeled_idx) < 10:
        return df

    labels = df.loc[labeled_idx, label_col].astype(str).to_numpy()

    # Merge rare classes (< 2 samples) so StratifiedShuffleSplit doesn't error
    counts = pd.Series(labels).value_counts()
    rare = set(counts[counts < 2].index.tolist())
    strat_labels = (
        np.where(np.isin(labels, list(rare)), "_rare", labels) if rare else labels
    )

    test_frac = round(1.0 - train_frac - val_frac, 10)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=round(val_frac + test_frac, 10), random_state=random_state
    )
    train_pos, tmp_pos = next(sss1.split(np.zeros(len(labeled_idx)), strat_labels))

    tmp_labels = strat_labels[tmp_pos]
    relative_test = round(test_frac / (val_frac + test_frac), 10)
    try:
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=relative_test, random_state=random_state
        )
        val_pos, test_pos = next(sss2.split(np.zeros(len(tmp_pos)), tmp_labels))
    except ValueError:
        mid = len(tmp_pos) // 2
        val_pos = np.arange(mid)
        test_pos = np.arange(mid, len(tmp_pos))

    df.loc[labeled_idx[tmp_pos[val_pos]], "split"] = "val"
    df.loc[labeled_idx[tmp_pos[test_pos]], "split"] = "test"

    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()
    n_test = (df["split"] == "test").sum()
    logger.info("Split: train=%d, val=%d, test=%d", n_train, n_val, n_test)
    return df


def build_record_dataframe(
    fasta_path: str | Path,
    metadata_df: pd.DataFrame | None = None,
    default_species: str = "homo_sapiens",
) -> pd.DataFrame:
    """
    Parse a FASTA file into a DataFrame with the canonical schema columns.

    Merges with metadata_df on 'accession' if provided. The merged frame
    prefers metadata values for haplogroup, species, and geographic_origin.

    Columns returned:
        accession, sequence, haplogroup, species, geographic_origin, het_level_vector
    """
    fasta_path = Path(fasta_path)
    rows = [
        {"accession": rec.id, "sequence": str(rec.seq)}
        for rec in SeqIO.parse(fasta_path, "fasta")
    ]
    if not rows:
        raise ValueError(f"No sequences found in {fasta_path}")

    df = pd.DataFrame(rows)
    df["haplogroup"] = None
    df["species"] = default_species
    df["geographic_origin"] = None
    df["het_level_vector"] = None  # populated from gnomAD on Day 5

    if metadata_df is not None and not metadata_df.empty:
        # Normalise accession column name
        if "accession" not in metadata_df.columns:
            metadata_df = metadata_df.rename(columns={metadata_df.columns[0]: "accession"})
        merge_cols = ["accession"] + [
            c
            for c in ("haplogroup", "species", "geographic_origin")
            if c in metadata_df.columns
        ]
        if len(merge_cols) > 1:
            df = df.drop(columns=[c for c in merge_cols[1:]], errors="ignore")
            df = df.merge(metadata_df[merge_cols], on="accession", how="left")

    # Ensure all schema columns exist
    for col in ("haplogroup", "species", "geographic_origin", "het_level_vector"):
        if col not in df.columns:
            df[col] = None

    return df[
        ["accession", "sequence", "haplogroup", "species", "geographic_origin", "het_level_vector"]
    ]


def preprocess_sequences(
    df: pd.DataFrame,
    target_length: int = RCRS_LENGTH,
    pad_position: int = DLOOP_PAD_POSITION,
    min_n_fraction: float = 0.1,
) -> pd.DataFrame:
    """
    Run clean_sequence and normalize_length on every row.

    Adds:
    - length_raw: sequence length before normalization
    - qc_pass: True if the normalized sequence has <= min_n_fraction N characters

    Sequences that fail QC are kept (exclusion is a downstream decision).
    """
    df = df.copy()
    df["sequence"] = df["sequence"].map(clean_sequence)
    df["length_raw"] = df["sequence"].str.len()
    df["sequence"] = df["sequence"].map(
        lambda s: normalize_length(s, target_length=target_length, pad_position=pad_position)
    )
    n_frac = df["sequence"].map(lambda s: s.count("N") / len(s))
    df["qc_pass"] = n_frac <= min_n_fraction
    n_fail = (~df["qc_pass"]).sum()
    if n_fail:
        logger.info(
            "%d sequences flagged qc_pass=False (>%.0f%% N content)",
            n_fail,
            min_n_fraction * 100,
        )
    return df


def save_splits(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Write train.parquet, val.parquet, test.parquet from df.

    Requires a 'split' column (added by stratified_split). Returns a dict
    mapping split name to the written Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drop_cols = [c for c in ("split", "length_raw") if c in df.columns]
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        subset = df[df["split"] == split].drop(columns=drop_cols)
        out_path = output_dir / f"{split}.parquet"
        subset.to_parquet(out_path, index=False)
        logger.info("Wrote %d rows to %s", len(subset), out_path)
        paths[split] = out_path
    return paths
