"""
Baseline G3: DNABERT2 on haplogroup classification and pathogenicity prediction.

Extracts DNABERT2 embeddings and evaluates on the same held-out test sets used
for mtDNA-FM, providing a direct apples-to-apples comparison.

Requirements:
    pip install transformers torch

Usage:
    uv run python paper/experiments/baselines/dnabert2_baseline.py

Outputs:
    paper/experiments/baselines/results/dnabert2_baseline.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("paper/experiments/baselines/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DNABERT2_MODEL = "zhihan1996/DNABERT-2-117M"
MAX_LENGTH = 512  # DNABERT2 max context window
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# DNABERT2 embedding extraction
# ---------------------------------------------------------------------------

def load_dnabert2():
    """Load DNABERT2 model and tokenizer from HuggingFace Hub."""
    from transformers import AutoTokenizer, AutoModel
    logger.info(f"Loading DNABERT2 from {DNABERT2_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(DNABERT2_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(DNABERT2_MODEL, trust_remote_code=True).to(DEVICE)
    model.eval()
    logger.info(f"DNABERT2 loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return tokenizer, model


def extract_cls_embeddings(
    tokenizer, model, sequences: list[str], batch_size: int = 8
) -> np.ndarray:
    """Extract CLS token embeddings from DNABERT2 for a list of sequences.

    Sequences longer than MAX_LENGTH tokens are truncated (center crop).
    """
    embeddings = []
    n = len(sequences)
    for i in range(0, n, batch_size):
        batch_seqs = sequences[i : i + batch_size]
        # DNABERT2 expects space-separated k-mers or raw sequence with special tokenizer
        encoded = tokenizer(
            batch_seqs,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        # Use CLS token (position 0)
        cls_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embs)
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Embedded {min(i + batch_size, n)}/{n} sequences")
    return np.vstack(embeddings)


def extract_position_embedding(
    tokenizer, model, sequence: str, position: int
) -> np.ndarray:
    """Extract embedding at a specific nucleotide position.

    Used for variant pathogenicity: get hidden state at the variant position.
    """
    encoded = tokenizer(
        sequence,
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    # Map nucleotide position to token position (approximate for BPE)
    token_position = min(position // 3, MAX_LENGTH - 2)  # rough BPE mapping
    with torch.no_grad():
        outputs = model(**encoded)
    return outputs.last_hidden_state[0, token_position, :].cpu().numpy()


# ---------------------------------------------------------------------------
# Haplogroup classification baseline
# ---------------------------------------------------------------------------

def run_haplogroup_baseline(tokenizer, model) -> dict:
    """5-fold k-NN on DNABERT2 CLS embeddings for haplogroup classification."""
    logger.info("=== Haplogroup Classification Baseline ===")

    held_out_path = Path("paper/experiments/evaluation/held_out_test.parquet")
    if not held_out_path.exists():
        logger.warning("Held-out test set not found. Run create_eval_splits.py first.")
        return {"status": "no_eval_data"}

    df = pd.read_parquet(held_out_path)
    sequences = df["sequence"].tolist()
    labels = df["haplogroup_id"].values

    logger.info(f"Extracting embeddings for {len(sequences)} sequences...")
    embeddings = extract_cls_embeddings(tokenizer, model, sequences)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_scores = []
    lr_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # k-NN
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X_train, y_train)
        knn_scores.append(accuracy_score(y_val, knn.predict(X_val)))

        # Logistic regression (fine-tuned proxy)
        scaler = StandardScaler()
        lr = LogisticRegression(max_iter=500, C=1.0)
        lr.fit(scaler.fit_transform(X_train), y_train)
        lr_scores.append(accuracy_score(y_val, lr.predict(scaler.transform(X_val))))

        logger.info(
            f"  Fold {fold+1}: k-NN={knn_scores[-1]:.4f}, LR={lr_scores[-1]:.4f}"
        )

    return {
        "status": "completed",
        "n_sequences": len(sequences),
        "n_classes": int(labels.max()) + 1,
        "knn_5fold": {"mean": float(np.mean(knn_scores)), "std": float(np.std(knn_scores))},
        "lr_5fold": {"mean": float(np.mean(lr_scores)), "std": float(np.std(lr_scores))},
    }


# ---------------------------------------------------------------------------
# Pathogenicity prediction baseline
# ---------------------------------------------------------------------------

def run_pathogenicity_baseline(tokenizer, model) -> dict:
    """Logistic regression on DNABERT2 position embeddings for pathogenicity."""
    logger.info("=== Variant Pathogenicity Baseline ===")

    # Load the same ClinVar/gnomAD variants used for mtDNA-FM evaluation
    path = Path("data/processed/variants_test.parquet")
    if not path.exists():
        logger.warning(f"Variant test data not found: {path}")
        return {"status": "no_eval_data"}

    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} variants (pathogenic: {df['label'].sum()})")

    sequences = df["sequence"].tolist()
    positions = df["position"].tolist()
    labels = df["label"].values

    logger.info("Extracting position embeddings...")
    embeddings = []
    for seq, pos in zip(sequences, positions):
        emb = extract_position_embedding(tokenizer, model, seq, pos)
        embeddings.append(emb)
    embeddings = np.array(embeddings)

    # Also include CLS for comparison
    logger.info("Extracting CLS embeddings for comparison...")
    cls_embs = extract_cls_embeddings(tokenizer, model, sequences)

    results = {}
    for emb_name, embs in [("position_token", embeddings), ("cls_token", cls_embs)]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aurocs, auprs = [], []
        for train_idx, val_idx in skf.split(embs, labels):
            X_train, X_val = embs[train_idx], embs[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            lr = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
            lr.fit(scaler.fit_transform(X_train), y_train)
            probs = lr.predict_proba(scaler.transform(X_val))[:, 1]
            aurocs.append(roc_auc_score(y_val, probs))
            auprs.append(average_precision_score(y_val, probs))

        results[emb_name] = {
            "auroc_5fold": {"mean": float(np.mean(aurocs)), "std": float(np.std(aurocs))},
            "aupr_5fold": {"mean": float(np.mean(auprs)), "std": float(np.std(auprs))},
        }
        logger.info(
            f"  {emb_name}: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}, "
            f"AUPR={np.mean(auprs):.4f}±{np.std(auprs):.4f}"
        )

    return {"status": "completed", "n_variants": len(labels), **results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        tokenizer, model = load_dnabert2()
    except Exception as e:
        logger.error(f"Failed to load DNABERT2: {e}")
        logger.error("Install: pip install transformers torch")
        return

    results = {
        "model": DNABERT2_MODEL,
        "device": str(DEVICE),
        "haplogroup": run_haplogroup_baseline(tokenizer, model),
        "pathogenicity": run_pathogenicity_baseline(tokenizer, model),
    }

    output_path = RESULTS_DIR / "dnabert2_baseline.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results written to {output_path}")

    print("\n=== DNABERT2 Baseline Summary ===")
    hap = results["haplogroup"]
    if hap.get("status") == "completed":
        print(f"Haplogroup k-NN:   {hap['knn_5fold']['mean']:.4f} ± {hap['knn_5fold']['std']:.4f}")
        print(f"Haplogroup LR:     {hap['lr_5fold']['mean']:.4f} ± {hap['lr_5fold']['std']:.4f}")
    path_res = results["pathogenicity"]
    if path_res.get("status") == "completed":
        pt = path_res["position_token"]
        print(f"Pathogenicity AUROC (pos token): {pt['auroc_5fold']['mean']:.4f} ± {pt['auroc_5fold']['std']:.4f}")


if __name__ == "__main__":
    main()
