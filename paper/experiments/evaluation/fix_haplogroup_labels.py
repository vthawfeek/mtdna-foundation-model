"""
Fix haplogroup label mapping in processed parquets.

The fine-tuning dataset (HaplogroupWindowDataset) only accepts exact major haplogroup
labels ("A", "B", ..., "L0"..."L5"). But train/val/test.parquet has detailed sub-haplogroup
labels like "L5b1a", "HV1d", "J1c5d" which get silently dropped.

This script adds a `major_haplogroup` column by mapping detailed labels to their
major haplogroup ancestor. After running this, use `label_column: major_haplogroup`
in fine-tuning configs.

Mapping rules:
  L0...L5  → any haplogroup starting with "L" + digit 0-5
  HV       → any haplogroup starting with "HV" (before "H" check)
  A-Z      → first letter (single-letter major haplogroups)

Usage:
    uv run python paper/experiments/evaluation/fix_haplogroup_labels.py

Modifies in place:
    data/processed/train.parquet  (adds major_haplogroup column)
    data/processed/val.parquet
    data/processed/test.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAJOR_HAPLOGROUPS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "HV", "I",
    "J", "K", "L0", "L1", "L2", "L3", "L4", "L5", "M",
    "N", "R", "T", "U", "V", "W", "X",
]


def map_to_major(hap: str | None) -> str | None:
    """Map a detailed haplogroup label to the nearest major haplogroup.

    Examples:
        "L5b1a"  → "L5"
        "L3i2"   → "L3"
        "HV1d"   → "HV"
        "H13a2b" → "H"
        "J1c5d"  → "J"
        "M7b1"   → "M"
        "H"      → "H"  (already major)
    """
    if not hap or not isinstance(hap, str):
        return None

    hap = hap.strip()

    # L-clades: L0 through L5 are major; L6 and plain "L" fall back
    if hap.startswith("L") and len(hap) > 1 and hap[1].isdigit():
        clade = "L" + hap[1]
        if clade in MAJOR_HAPLOGROUPS:
            return clade
        # L6+ not in our 26 major groups — skip
        return None

    # HV must be checked before H (it starts with H)
    if hap.startswith("HV"):
        return "HV"

    # All other haplogroups: first letter is the major group
    first = hap[0].upper()
    if first in MAJOR_HAPLOGROUPS:
        return first

    return None


def fix_parquet(path: Path) -> None:
    """Add major_haplogroup column to a parquet file."""
    if not path.exists():
        logger.warning(f"Skipping (not found): {path}")
        return

    df = pd.read_parquet(path)

    if "haplogroup" not in df.columns:
        logger.warning(f"No 'haplogroup' column in {path} — skipping")
        return

    df["major_haplogroup"] = df["haplogroup"].apply(map_to_major)

    before = len(df)
    matched = df["major_haplogroup"].notna().sum()
    logger.info(
        f"{path.name}: {before} rows → {matched} ({matched/before:.1%}) "
        f"map to a major haplogroup"
    )

    dist = df["major_haplogroup"].value_counts().sort_index()
    logger.info(f"  Distribution:\n{dist.to_string()}")

    df.to_parquet(path, index=False)
    logger.info(f"  Saved (with major_haplogroup column): {path}")


def main() -> None:
    base = Path("data/processed")
    for split in ["train", "val", "test"]:
        fix_parquet(base / f"{split}.parquet")

    logger.info(
        "\nDone. Now update fine-tuning configs to use:\n"
        "  label_column: major_haplogroup\n"
        "  base_model: models/phase1_v1\n"
        "And re-run: uv run mtdna-finetune --task haplogroup "
        "--config configs/finetuning_haplogroup_paper.yaml"
    )


if __name__ == "__main__":
    main()
