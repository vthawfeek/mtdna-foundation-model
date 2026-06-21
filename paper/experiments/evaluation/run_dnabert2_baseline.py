"""
Zero-shot 26-class haplogroup k-NN evaluation using DNABERT-2 as a baseline.

Same protocol as paper/run_zeroshot_haplogroup.py:
  - Same data source (data/hmtdb_labeled/)
  - Same stratified 80/10/10 split (random_state=42)
  - Same balanced sampling (60 train / 40 test per class)
  - Same cosine 5-NN evaluation
  - Same 1,000-replicate bootstrap CI

DNABERT-2 difference: each 16,569 bp genome is truncated to TRUNCATE_NT=3000 nt
before BPE tokenization. After tokenization the 512-token context limit applies.
DNABERT-2 therefore sees at most ~18% of each complete mitochondrial genome.
CLS token embedding from the final transformer layer is used as the sequence
representation.

Prerequisites
-------------
    pip install "transformers>=4.40" einops
    uv run python paper/download_labeled_mtdna.py   # data must exist

Usage
-----
    uv run python paper/experiments/evaluation/run_dnabert2_baseline.py

Output
------
    reports/dnabert2_haplogroup_knn.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

# Reuse Windows HuggingFace cache when running under WSL2 to avoid re-downloading
_win_hf = Path("/mnt/c/Users/vthawfeek.Shajitha/.cache/huggingface")
if _win_hf.exists() and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_win_hf)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR_LABELED = ROOT / "data" / "hmtdb_labeled"
REPORTS_DIR      = ROOT / "reports"
OUTPUT_JSON      = REPORTS_DIR / "dnabert2_haplogroup_knn.json"

DNABERT2_REPO   = "zhihan1996/DNABERT-2-117M"
TRUNCATE_NT     = 3000   # safe upper bound; BPE max_length=512 applies after tokenization

TEST_PER_CLASS  = 40
TRAIN_PER_CLASS = 60
KNN_K           = 5
BOOTSTRAP_N     = 1_000
RANDOM_STATE    = 42

MAJOR_26 = {
    "A", "B", "C", "D", "E", "F", "G", "H", "HV", "I", "J", "K",
    "L0", "L1", "L2", "L3", "L4", "L5", "M", "N", "R", "T", "U", "V", "W", "X",
}


# ── Haplogroup mapping (verbatim from run_zeroshot_haplogroup.py) ──────────────

def map_haplogroup(raw: object) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "", "?", "-"):
        return None
    upper = s.upper()
    if upper.startswith("HV"):
        return "HV"
    if upper.startswith("L") and len(s) > 1 and s[1].isdigit():
        major = upper[:2]
        return major if major in MAJOR_26 else None
    major = upper[0]
    return major if major in MAJOR_26 else None


# ── Step 1: Load data (verbatim from run_zeroshot_haplogroup.py) ───────────────

def load_data() -> "pd.DataFrame":
    import pandas as pd
    from Bio import SeqIO
    from mtdna_fm.data.preprocessor import clean_sequence, normalize_length

    fasta_path = DATA_DIR_LABELED / "sequences.fasta"
    meta_path  = DATA_DIR_LABELED / "metadata.parquet"

    if not fasta_path.exists() or not meta_path.exists():
        log.error(
            "Data not found at %s. Run paper/download_labeled_mtdna.py first.",
            DATA_DIR_LABELED,
        )
        sys.exit(1)

    log.info("Loading metadata ...")
    meta_df = pd.read_parquet(meta_path)
    log.info("  %d metadata rows, columns: %s", len(meta_df), list(meta_df.columns))

    log.info("Parsing FASTA ...")
    rows = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        rows.append({"accession": rec.id, "sequence": str(rec.seq)})
    seq_df = pd.DataFrame(rows)
    log.info("  %d sequences in FASTA", len(seq_df))

    df = seq_df.merge(
        meta_df[["accession", "haplogroup"]],
        on="accession", how="left",
    )

    df["major_hg"] = df["haplogroup"].apply(map_haplogroup)
    labeled = df[df["major_hg"].notna()].copy()
    log.info("  %d sequences with a valid major haplogroup label", len(labeled))

    import collections
    dist = collections.Counter(labeled["major_hg"].tolist())
    log.info("  Classes found: %d  |  top 10: %s",
             len(dist), sorted(dist.items(), key=lambda x: -x[1])[:10])

    log.info("Preprocessing sequences (clean + normalize to 16,569 bp) ...")
    labeled = labeled.copy()
    labeled["sequence"] = labeled["sequence"].apply(clean_sequence)
    labeled["sequence"] = labeled["sequence"].apply(normalize_length)

    n_frac = labeled["sequence"].apply(lambda s: s.count("N") / len(s))
    labeled = labeled[n_frac <= 0.10].copy()
    log.info("  After QC: %d sequences", len(labeled))

    return labeled


# ── Step 2: Split (verbatim from run_zeroshot_haplogroup.py) ──────────────────

def prepare_splits(df: "pd.DataFrame") -> tuple["pd.DataFrame", "pd.DataFrame"]:
    from mtdna_fm.data.preprocessor import stratified_split

    log.info("Stratified 80/10/10 split (random_state=%d) ...", RANDOM_STATE)
    df = stratified_split(df, label_col="major_hg", random_state=RANDOM_STATE)

    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()

    common = sorted(set(train_df["major_hg"]) & set(test_df["major_hg"]))
    log.info("  Train: %d   Test: %d   Shared classes: %d",
             len(train_df), len(test_df), len(common))

    train_df = train_df[train_df["major_hg"].isin(common)]
    test_df  = test_df[test_df["major_hg"].isin(common)]
    return train_df, test_df


def sample_balanced(df: "pd.DataFrame", n_per_class: int, seed: int,
                    label_col: str = "major_hg") -> "pd.DataFrame":
    import pandas as pd
    frames = []
    for _, group in df.groupby(label_col):
        n = min(n_per_class, len(group))
        frames.append(group.sample(n, random_state=seed))
    out = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)
    log.info("  Balanced sample: %d seqs across %d classes",
             len(out), out[label_col].nunique())
    return out


# ── Step 3: k-NN evaluation (verbatim from run_zeroshot_haplogroup.py) ─────────

def cosine_knn_predict(
    train_emb: np.ndarray, train_labels: np.ndarray,
    test_emb: np.ndarray, k: int = 5,
) -> np.ndarray:
    train_norm = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-12)
    test_norm  = test_emb  / (np.linalg.norm(test_emb,  axis=1, keepdims=True) + 1e-12)
    sim = test_norm @ train_norm.T

    unique_labels = sorted(set(train_labels.tolist()))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
    train_idx     = np.array([label_to_idx[l] for l in train_labels])

    predictions = []
    for i in range(len(test_emb)):
        top_k  = np.argsort(-sim[i])[:k]
        votes  = np.bincount(train_idx[top_k], minlength=len(unique_labels))
        predictions.append(unique_labels[int(np.argmax(votes))])

    return np.array(predictions)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    classes: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    acc      = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro",
                              zero_division=0, labels=classes))
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)

    per_class = [
        {"label": cls, "precision": round(float(p[i]), 4),
         "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4),
         "support": int(sup[i])}
        for i, cls in enumerate(classes)
    ]
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "per_class": per_class}


def bootstrap_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                        n: int = BOOTSTRAP_N, seed: int = RANDOM_STATE
                        ) -> tuple[float, float]:
    rng     = np.random.default_rng(seed)
    correct = (y_true == y_pred).astype(float)
    boot    = [correct[rng.integers(0, len(correct), size=len(correct))].mean()
               for _ in range(n)]
    lo, hi  = np.percentile(boot, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


# ── Step 4: DNABERT-2 model loading and embedding ─────────────────────────────

def load_dnabert2():
    """Load DNABERT-2 via dynamic module class to bypass AutoModel config-class mismatch."""
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    log.info("Loading DNABERT-2 (%s) ...", DNABERT2_REPO)
    tok = AutoTokenizer.from_pretrained(DNABERT2_REPO, trust_remote_code=True)
    config = AutoConfig.from_pretrained(DNABERT2_REPO, trust_remote_code=True)
    BertModelClass = get_class_from_dynamic_module("bert_layers.BertModel", DNABERT2_REPO)
    mdl = BertModelClass.from_pretrained(DNABERT2_REPO, config=config)
    mdl.eval()

    n = sum(p.numel() for p in mdl.parameters())
    log.info("  Loaded. Parameters: %.1f M (Flash Attention disabled: no triton)", n / 1e6)
    return tok, mdl


def embed_dnabert2_single(tok, mdl, sequence: str) -> np.ndarray:
    import torch

    seq = sequence[:TRUNCATE_NT]
    inputs = tok(seq, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = mdl(**inputs)
    # DNABERT-2 returns a tuple (sequence_output, pooler_output); CLS is position 0
    seq_out = out[0] if isinstance(out, tuple) else out.last_hidden_state
    return seq_out[:, 0, :].squeeze(0).numpy().astype(np.float32)


def embed_sequences_dnabert2(tok, mdl, df: "pd.DataFrame", desc: str = "") -> np.ndarray:
    seqs = df["sequence"].tolist()
    n    = len(seqs)
    embs = []
    t0   = time.time()

    for i, seq in enumerate(seqs):
        embs.append(embed_dnabert2_single(tok, mdl, seq))
        if (i + 1) % 20 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (n - i - 1) / rate if rate > 0 else 0
            log.info("  %s  %d/%d  (%.1f seq/min, ETA %.0f min)",
                     desc, i + 1, n, rate * 60, eta / 60)

    return np.stack(embs, axis=0).astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train_df, test_df = prepare_splits(df)
    classes = sorted(set(train_df["major_hg"]) & set(test_df["major_hg"]))
    n_classes = len(classes)
    random_baseline = round(1.0 / n_classes, 4)
    log.info("Classes: %d   Random baseline: %.1f%%", n_classes, random_baseline * 100)

    train_lib = sample_balanced(train_df, TRAIN_PER_CLASS, RANDOM_STATE)
    test_q    = sample_balanced(test_df,  TEST_PER_CLASS,  RANDOM_STATE + 1)
    log.info("Train library: %d   Test queries: %d", len(train_lib), len(test_q))

    tok, mdl = load_dnabert2()

    log.info("Embedding train library (%d sequences, truncated to %d nt) ...",
             len(train_lib), TRUNCATE_NT)
    t0 = time.time()
    train_emb = embed_sequences_dnabert2(tok, mdl, train_lib, desc="train")
    log.info("  Train embedding done in %.1f min", (time.time() - t0) / 60)

    log.info("Embedding test queries (%d sequences, truncated to %d nt) ...",
             len(test_q), TRUNCATE_NT)
    t0 = time.time()
    test_emb = embed_sequences_dnabert2(tok, mdl, test_q, desc="test ")
    log.info("  Test embedding done in %.1f min", (time.time() - t0) / 60)

    log.info("Running %d-NN (cosine) ...", KNN_K)
    train_labels = train_lib["major_hg"].to_numpy()
    test_labels  = test_q["major_hg"].to_numpy()
    y_pred = cosine_knn_predict(train_emb, train_labels, test_emb, k=KNN_K)

    metrics      = compute_metrics(test_labels, y_pred, classes=classes)
    ci_lo, ci_hi = bootstrap_accuracy(test_labels, y_pred)
    lift = round(metrics["accuracy"] / random_baseline, 2) if random_baseline > 0 else None

    log.info("=" * 60)
    log.info("RESULTS -- DNABERT-2 zero-shot %d-class %d-NN", n_classes, KNN_K)
    log.info("  Accuracy  : %.1f%%  (95%% CI %.1f%%--%.1f%%)",
             metrics["accuracy"] * 100, ci_lo * 100, ci_hi * 100)
    log.info("  Random    : %.1f%%  (1/%d)", random_baseline * 100, n_classes)
    log.info("  Lift      : %.1fx", lift)
    log.info("  Macro-F1  : %.4f", metrics["macro_f1"])
    log.info("  Truncation: %d nt (BPE max_length=512)", TRUNCATE_NT)
    log.info("=" * 60)

    result = {
        "method":                "zero-shot cosine 5-NN",
        "model":                 DNABERT2_REPO,
        "n_classes":             n_classes,
        "classes":               classes,
        "n_train_library":       int(len(train_lib)),
        "n_test_queries":        int(len(test_q)),
        "train_per_class":       TRAIN_PER_CLASS,
        "test_per_class":        TEST_PER_CLASS,
        "random_baseline":       random_baseline,
        "accuracy":              metrics["accuracy"],
        "accuracy_ci_95_lo":     ci_lo,
        "accuracy_ci_95_hi":     ci_hi,
        "lift_over_random":      lift,
        "macro_f1":              metrics["macro_f1"],
        "per_class":             metrics["per_class"],
        "knn_k":                 KNN_K,
        "random_state":          RANDOM_STATE,
        "bootstrap_n":           BOOTSTRAP_N,
        "sequence_truncation_nt": TRUNCATE_NT,
        "max_bpe_tokens":        512,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved -> %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
