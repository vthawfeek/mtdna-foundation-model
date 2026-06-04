"""
Baseline G3: k-mer frequency + LR/SVM on haplogroup and pathogenicity tasks.

Makes the "k-mer + LR" claim quantitative with proper 5-fold CV on real data.

Usage:
    uv run python paper/experiments/baselines/kmer_frequency_baseline.py

Outputs:
    paper/experiments/baselines/results/kmer_frequency_baseline.json
"""

from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("paper/experiments/baselines/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HAPLOGROUPS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "HV", "I",
    "J", "K", "L0", "L1", "L2", "L3", "L4", "L5", "M",
    "N", "R", "T", "U", "V", "W", "X",
]
LABEL2IDX = {h: i for i, h in enumerate(HAPLOGROUPS)}


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


def build_kmer_vocab(k: int) -> dict[str, int]:
    return {"".join(m): i for i, m in enumerate(product("ACGT", repeat=k))}


def kmer_freq(seq: str, k: int, vocab: dict) -> np.ndarray:
    freq = np.zeros(len(vocab), dtype=np.float32)
    seq = seq.upper()
    n = 0
    for i in range(len(seq) - k + 1):
        km = seq[i: i + k]
        if km in vocab:
            freq[vocab[km]] += 1
            n += 1
    if n > 0:
        freq /= n
    return freq


def featurize(sequences: list[str], k: int) -> np.ndarray:
    vocab = build_kmer_vocab(k)
    out = np.array([kmer_freq(s, k, vocab) for s in sequences])
    logger.info(f"  k={k}: {out.shape}")
    return out


# ---------------------------------------------------------------------------
# Haplogroup
# ---------------------------------------------------------------------------

def run_haplogroup(k: int) -> dict:
    logger.info(f"=== Haplogroup baseline (k={k}) ===")
    # Use held-out test set if available, else fall back to test.parquet
    test_path = Path("paper/experiments/evaluation/held_out_test.parquet")
    if not test_path.exists():
        test_path = Path("data/processed/test.parquet")
    if not test_path.exists():
        return {"status": "no_data"}

    df = pd.read_parquet(test_path)
    if "major_haplogroup" not in df.columns:
        df["major_haplogroup"] = df["haplogroup"].apply(map_to_major)
    df = df[df["major_haplogroup"].notna()].copy()
    df["haplogroup_id"] = df["major_haplogroup"].map(LABEL2IDX)
    logger.info(f"  {len(df)} sequences, {df['major_haplogroup'].nunique()} haplogroups")

    X = featurize(df["sequence"].tolist(), k)
    y = df["haplogroup_id"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0))])
        pipe.fit(X[tr], y[tr])
        lr_scores.append(accuracy_score(y[va], pipe.predict(X[va])))
        logger.info(f"  Fold {fold+1}: LR={lr_scores[-1]:.4f}")

    return {
        "status": "completed", "k": k,
        "n_sequences": len(df), "n_classes": int(y.max()) + 1,
        "lr_5fold": {"mean": float(np.mean(lr_scores)), "std": float(np.std(lr_scores))},
    }


# ---------------------------------------------------------------------------
# Pathogenicity
# ---------------------------------------------------------------------------

def run_pathogenicity(k: int) -> dict:
    logger.info(f"=== Pathogenicity baseline (k={k}) ===")
    # Try val parquet (same split used for model evaluation)
    for p in [
        "data/processed/variants_pathogenicity_val.parquet",
        "reports/eval_variant_predictions.parquet",
    ]:
        if Path(p).exists():
            df = pd.read_parquet(p)
            logger.info(f"  Loaded {p}: {len(df)} rows")
            break
    else:
        logger.warning("  No pathogenicity data found. Run prepare_variant_data.py first.")
        return {"status": "no_data"}

    if "sequence" not in df.columns:
        logger.warning("  Parquet has no 'sequence' column — cannot compute k-mer features")
        return {"status": "no_sequence_column"}

    X = featurize(df["sequence"].tolist(), k)
    y = df["label"].values
    logger.info(f"  {len(df)} variants (pathogenic: {y.sum()})")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs, auprs = [], []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
        ])
        pipe.fit(X[tr], y[tr])
        probs = pipe.predict_proba(X[va])[:, 1]
        aurocs.append(roc_auc_score(y[va], probs))
        auprs.append(average_precision_score(y[va], probs))
        logger.info(f"  Fold {fold+1}: AUROC={aurocs[-1]:.4f}, AUPR={auprs[-1]:.4f}")

    return {
        "status": "completed", "k": k, "n_variants": len(df),
        "auroc_5fold": {"mean": float(np.mean(aurocs)), "std": float(np.std(aurocs))},
        "aupr_5fold": {"mean": float(np.mean(auprs)), "std": float(np.std(auprs))},
    }


def main() -> None:
    results = {}
    for k in [4, 6]:
        results[f"k{k}_haplogroup"] = run_haplogroup(k)
        results[f"k{k}_pathogenicity"] = run_pathogenicity(k)

    out = RESULTS_DIR / "kmer_frequency_baseline.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info(f"Results → {out}")

    print("\n=== k-mer Frequency Baseline ===")
    for key, res in results.items():
        if res.get("status") == "completed":
            if "haplogroup" in key:
                v = res["lr_5fold"]
                print(f"{key:<30} acc {v['mean']:.4f} ± {v['std']:.4f}")
            else:
                v = res["auroc_5fold"]
                print(f"{key:<30} auc {v['mean']:.4f} ± {v['std']:.4f}")
        else:
            print(f"{key:<30} {res.get('status', '?')}")


if __name__ == "__main__":
    main()
