"""
G2: Create proper held-out evaluation splits from data/processed/test.parquet.

Uses the actual test set (1,263 human mtDNA sequences) stratified by major haplogroup.
This replaces the synthetic 260-sequence evaluation used in the original eval reports.

Prerequisites:
    uv run python paper/experiments/evaluation/fix_haplogroup_labels.py

Usage:
    uv run python paper/experiments/evaluation/create_eval_splits.py

Outputs:
    paper/experiments/evaluation/held_out_test.parquet
    paper/experiments/evaluation/held_out_train.parquet
    paper/experiments/evaluation/split_stats.json
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

HAPLOGROUPS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "HV", "I",
    "J", "K", "L0", "L1", "L2", "L3", "L4", "L5", "M",
    "N", "R", "T", "U", "V", "W", "X",
]
LABEL2IDX = {h: i for i, h in enumerate(HAPLOGROUPS)}

MAX_PER_CLASS_TEST = 50
MIN_PER_CLASS_TEST = 5
RANDOM_SEED = 42


def map_to_major(hap: str | None) -> str | None:
    if not hap:
        return None
    hap = hap.strip()
    if hap.startswith("L") and len(hap) > 1 and hap[1].isdigit():
        clade = "L" + hap[1]
        return clade if clade in LABEL2IDX else None
    if hap.startswith("HV"):
        return "HV"
    first = hap[0].upper()
    return first if first in LABEL2IDX else None


def main() -> None:
    rng = np.random.RandomState(RANDOM_SEED)

    # Load test.parquet (already held out from pre-training)
    test_path = Path("data/processed/test.parquet")
    if not test_path.exists():
        logger.error(f"Test parquet not found: {test_path}")
        return

    df = pd.read_parquet(test_path)
    logger.info(f"Loaded test.parquet: {len(df)} sequences")

    # Map to major haplogroup (use existing column if fix_haplogroup_labels.py was run)
    if "major_haplogroup" in df.columns:
        logger.info("Using existing major_haplogroup column")
    else:
        logger.info("Computing major_haplogroup (run fix_haplogroup_labels.py for a permanent fix)")
        df["major_haplogroup"] = df["haplogroup"].apply(map_to_major)

    df = df[df["major_haplogroup"].notna()].copy()
    df["haplogroup_id"] = df["major_haplogroup"].map(LABEL2IDX)
    logger.info(f"After major-haplogroup mapping: {len(df)} sequences, {df['major_haplogroup'].nunique()} groups")

    # Stratified sample for paper test set
    test_parts, train_parts = [], []
    stats = {}
    for group in sorted(df["major_haplogroup"].unique()):
        group_df = df[df["major_haplogroup"] == group]
        n = len(group_df)
        if n < MIN_PER_CLASS_TEST:
            logger.info(f"  Skipping {group}: only {n} sequences")
            stats[group] = {"n": n, "n_test": 0, "included": False}
            continue
        n_test = min(n, MAX_PER_CLASS_TEST)
        test_idx = rng.choice(n, size=n_test, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[test_idx] = True
        test_parts.append(group_df.iloc[mask])
        train_parts.append(group_df.iloc[~mask])
        stats[group] = {"n": n, "n_test": n_test, "n_train": n - n_test, "included": True}
        logger.info(f"  {group}: {n_test} test, {n - n_test} train")

    test_df = pd.concat(test_parts, ignore_index=True)
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()

    test_df.to_parquet(EVAL_DIR / "held_out_test.parquet", index=False)
    logger.info(f"Saved held_out_test.parquet: {len(test_df)} sequences")
    if len(train_df) > 0:
        train_df.to_parquet(EVAL_DIR / "held_out_train.parquet", index=False)
        logger.info(f"Saved held_out_train.parquet: {len(train_df)} sequences")

    (EVAL_DIR / "split_stats.json").write_text(json.dumps({
        "total_test": len(test_df),
        "total_train": len(train_df),
        "n_haplogroups": int(test_df["major_haplogroup"].nunique()),
        "per_class": stats,
    }, indent=2))

    print(f"\nTest set: {len(test_df)} sequences, {test_df['major_haplogroup'].nunique()} haplogroups")
    print(test_df["major_haplogroup"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
