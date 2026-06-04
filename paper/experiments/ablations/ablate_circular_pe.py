"""
Ablation G1-A1: Circular PE vs standard sinusoidal/learnable PE.

Trains two identical models differing only in positional encoding type, then
evaluates on zero-shot k-NN haplogroup accuracy and fine-tuned accuracy.

Usage:
    uv run python paper/experiments/ablations/ablate_circular_pe.py

Outputs:
    paper/experiments/ablations/results/circular_pe_ablation.json
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("paper/experiments/ablations/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Positional encoding variants
# ---------------------------------------------------------------------------

class CircularPositionalEncoding(nn.Module):
    """Circular sinusoidal PE: positions 0 and genome_length map identically."""

    def __init__(self, hidden_size: int, genome_length: int = 16_569, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(genome_length, hidden_size)
        angle = torch.arange(genome_length, dtype=torch.float) / genome_length * 2 * math.pi
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(angle.unsqueeze(1) * div_term.unsqueeze(0))
        pe[:, 1::2] = torch.cos(angle.unsqueeze(1) * div_term.unsqueeze(0))
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len])


class StandardSinusoidalPE(nn.Module):
    """Standard sinusoidal PE from Vaswani et al. 2017 (linear positions)."""

    def __init__(self, hidden_size: int, max_len: int = 16_569, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len])


class LearnablePE(nn.Module):
    """Learnable absolute positional embedding."""

    def __init__(self, hidden_size: int, max_len: int = 16_569, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_len, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return self.dropout(x + self.embedding(positions).unsqueeze(0))


# ---------------------------------------------------------------------------
# Training helpers (stub — fill in when running the full ablation)
# ---------------------------------------------------------------------------

def load_haplogroup_dataset(split: str = "test") -> tuple[np.ndarray, np.ndarray]:
    """Load haplogroup sequences and labels from the held-out evaluation split.

    Returns:
        sequences: array of DNA strings
        labels: array of haplogroup label integers
    """
    path = Path(f"paper/experiments/evaluation/held_out_{split}.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Held-out split not found: {path}\n"
            "Run: uv run python paper/experiments/evaluation/create_eval_splits.py first"
        )
    import pandas as pd
    df = pd.read_parquet(path)
    return df["sequence"].values, df["haplogroup_id"].values


def embed_sequences_with_model(model, tokenizer, sequences: list[str]) -> np.ndarray:
    """Extract CLS embeddings from a model for a list of sequences.

    Args:
        model: MtDNAModel (or any model with .forward() returning last_hidden_state)
        tokenizer: KmerVocabulary instance
        sequences: list of DNA strings

    Returns:
        embeddings: (N, hidden_size) numpy array
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            tokens = tokenizer.tokenize(seq)
            input_ids = torch.tensor(tokens).unsqueeze(0)
            outputs = model(input_ids=input_ids)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            embeddings.append(cls_emb)
    return np.array(embeddings)


def zero_shot_knn_accuracy(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    k: int = 5,
    n_splits: int = 5,
) -> dict:
    """5-fold stratified k-NN accuracy on embeddings.

    Returns dict with mean, std, and per-fold scores.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_embeddings = np.vstack([train_embeddings, test_embeddings])
    all_labels = np.concatenate([train_labels, test_labels])

    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_embeddings, all_labels)):
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(all_embeddings[train_idx], all_labels[train_idx])
        preds = knn.predict(all_embeddings[val_idx])
        score = accuracy_score(all_labels[val_idx], preds)
        scores.append(score)
        logger.info(f"  Fold {fold+1}/{n_splits}: accuracy={score:.4f}")

    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "folds": scores}


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(pe_type: str) -> dict:
    """Train a model with the specified PE type and evaluate.

    pe_type: one of 'circular', 'sinusoidal', 'learnable'
    """
    logger.info(f"=== Ablation: PE type = {pe_type} ===")

    # Import project modules
    from mtdna_fm.tokenizer import KmerVocabulary
    from mtdna_fm.model.config import MtDNAConfig
    from mtdna_fm.model.model import MtDNAModel

    vocab = KmerVocabulary.from_pretrained("models/vocabulary")
    config = MtDNAConfig(
        vocab_size=len(vocab),
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=514,
        positional_encoding_type=pe_type,  # pass to config if supported, else patch below
    )

    # Patch the model's PE based on pe_type
    model = MtDNAModel(config)
    if pe_type == "sinusoidal":
        model.embeddings.position_embeddings = StandardSinusoidalPE(config.hidden_size)
    elif pe_type == "learnable":
        model.embeddings.position_embeddings = LearnablePE(config.hidden_size)
    # 'circular' uses the default CircularPositionalEncoding already in the model

    # NOTE: Full pre-training (~8–12h per model) is required here.
    # For the ablation, we load from a checkpoint if available, otherwise train.
    checkpoint_path = Path(f"models/ablation_pe_{pe_type}")
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model = MtDNAModel.from_pretrained(str(checkpoint_path))
    else:
        logger.warning(
            f"No checkpoint found at {checkpoint_path}. "
            "Train the model first with the pre-training CLI:\n"
            f"  uv run mtdna-pretrain --phase 1 --pe-type {pe_type} "
            f"--output models/ablation_pe_{pe_type}"
        )
        return {"pe_type": pe_type, "status": "not_trained"}

    # Load evaluation data
    sequences, labels = load_haplogroup_dataset("test")
    train_seqs, train_labels = load_haplogroup_dataset("train")

    logger.info("Extracting embeddings...")
    test_embs = embed_sequences_with_model(model, vocab, sequences)
    train_embs = embed_sequences_with_model(model, vocab, train_seqs)

    logger.info("Running zero-shot k-NN...")
    knn_results = zero_shot_knn_accuracy(train_embs, train_labels, test_embs, labels)
    logger.info(f"Zero-shot k-NN: {knn_results['mean']:.4f} ± {knn_results['std']:.4f}")

    return {
        "pe_type": pe_type,
        "status": "completed",
        "zero_shot_knn": knn_results,
        # fine-tuned accuracy would be added here after fine-tuning
    }


def main():
    results = {}
    for pe_type in ["circular", "sinusoidal", "learnable"]:
        try:
            results[pe_type] = run_ablation(pe_type)
        except Exception as e:
            logger.error(f"Ablation failed for pe_type={pe_type}: {e}")
            results[pe_type] = {"pe_type": pe_type, "status": "error", "error": str(e)}

    output_path = RESULTS_DIR / "circular_pe_ablation.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results written to {output_path}")

    # Print summary table
    print("\n=== Circular PE Ablation Results ===")
    print(f"{'PE Type':<15} {'Zero-shot k-NN':>18} {'Status':<15}")
    print("-" * 50)
    for pe_type, res in results.items():
        if res.get("status") == "completed":
            acc = res["zero_shot_knn"]["mean"]
            std = res["zero_shot_knn"]["std"]
            print(f"{pe_type:<15} {acc:.4f} ± {std:.4f}    {res['status']:<15}")
        else:
            print(f"{pe_type:<15} {'N/A':>18}    {res.get('status', 'unknown'):<15}")


if __name__ == "__main__":
    main()
