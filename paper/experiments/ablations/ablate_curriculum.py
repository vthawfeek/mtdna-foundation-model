"""
Ablation G1-A2: Two-phase curriculum vs single-phase training.

Compares the two-phase curriculum (Phase 1: cross-species, Phase 2: human-only)
against a single-phase baseline trained on the combined dataset for the same total
number of steps.

Metrics:
  - Final MLM loss on human validation set
  - Zero-shot k-NN haplogroup accuracy (no fine-tuning labels)
  - Fine-tuned haplogroup accuracy (with LoRA)

Usage:
    uv run python paper/experiments/ablations/ablate_curriculum.py

Outputs:
    paper/experiments/ablations/results/curriculum_ablation.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("paper/experiments/ablations/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CURRICULUM_CONFIG = {
    "phase1_steps": 50_000,
    "phase2_steps": 25_000,
    "phase1_data": "data/processed/train_cross_species.parquet",
    "phase2_data": "data/processed/train_human.parquet",
    "checkpoint_dir": "models/phase1_v1",
    "phase2_checkpoint_dir": "models/phase2_v1",
}

SINGLE_PHASE_CONFIG = {
    "total_steps": 75_000,  # same total as Phase1 + Phase2
    "data": "data/processed/train_combined.parquet",  # merged cross-species + human
    "checkpoint_dir": "models/ablation_single_phase",
}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_mlm_loss(model, dataloader) -> float:
    """Evaluate MLM loss on a held-out validation set."""
    import torch
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def zero_shot_knn(model, tokenizer, test_seqs, test_labels, k: int = 5) -> dict:
    """5-fold k-NN accuracy on CLS embeddings."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    import torch

    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in test_seqs:
            tokens = tokenizer.tokenize(seq)
            input_ids = torch.tensor(tokens).unsqueeze(0)
            out = model(input_ids=input_ids)
            emb = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(emb)
    embeddings = np.array(embeddings)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(embeddings, test_labels):
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(embeddings[train_idx], test_labels[train_idx])
        scores.append(accuracy_score(test_labels[val_idx], knn.predict(embeddings[val_idx])))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir: str):
    """Load MtDNAModel from a checkpoint directory."""
    from mtdna_fm.model.model import MtDNAModel
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Train the model before running this ablation.\n\n"
            "Two-phase training:\n"
            "  uv run mtdna-pretrain --phase 1 --output models/phase1_v1\n"
            "  uv run mtdna-pretrain --phase 2 --output models/phase2_v1\n\n"
            "Single-phase training:\n"
            "  uv run mtdna-pretrain --phase combined --output models/ablation_single_phase"
        )
    return MtDNAModel.from_pretrained(str(ckpt))


def evaluate_model(label: str, checkpoint_dir: str) -> dict:
    logger.info(f"Evaluating {label} from {checkpoint_dir}")
    from mtdna_fm.tokenizer import KmerVocabulary
    import pandas as pd

    try:
        model = load_model(checkpoint_dir)
    except FileNotFoundError as e:
        logger.warning(str(e))
        return {"label": label, "status": "not_trained"}

    vocab = KmerVocabulary.from_pretrained("models/vocabulary")

    held_out_path = Path("paper/experiments/evaluation/held_out_test.parquet")
    if not held_out_path.exists():
        logger.warning("Held-out test set not found. Run create_eval_splits.py first.")
        return {"label": label, "status": "no_eval_data"}

    df = pd.read_parquet(held_out_path)
    sequences = df["sequence"].tolist()
    labels = df["haplogroup_id"].values

    knn = zero_shot_knn(model, vocab, sequences, labels)
    logger.info(f"{label} zero-shot k-NN: {knn['mean']:.4f} ± {knn['std']:.4f}")
    return {
        "label": label,
        "status": "completed",
        "checkpoint": checkpoint_dir,
        "zero_shot_knn": knn,
        # mlm_loss would be added with a proper validation dataloader
    }


def main():
    results = {}

    # Two-phase curriculum: evaluate Phase 2 checkpoint
    results["two_phase"] = evaluate_model(
        label="Two-phase curriculum (Phase1→Phase2)",
        checkpoint_dir=CURRICULUM_CONFIG["phase2_checkpoint_dir"],
    )

    # Single-phase: evaluate combined checkpoint
    results["single_phase"] = evaluate_model(
        label="Single-phase (combined data)",
        checkpoint_dir=SINGLE_PHASE_CONFIG["checkpoint_dir"],
    )

    # Also compare Phase 1 alone (before Phase 2 specialization)
    results["phase1_only"] = evaluate_model(
        label="Phase 1 only (cross-species)",
        checkpoint_dir=CURRICULUM_CONFIG["checkpoint_dir"],
    )

    output_path = RESULTS_DIR / "curriculum_ablation.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results written to {output_path}")

    print("\n=== Curriculum Ablation Results ===")
    print(f"{'Model':<40} {'Zero-shot k-NN':>18} {'Status'}")
    print("-" * 70)
    for key, res in results.items():
        label = res.get("label", key)[:38]
        if res.get("status") == "completed":
            knn = res["zero_shot_knn"]
            print(f"{label:<40} {knn['mean']:.4f} ± {knn['std']:.4f}    {res['status']}")
        else:
            print(f"{label:<40} {'N/A':>18}    {res.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
