"""
Loss functions for mtDNA-FM pre-training.

WHY a combined MLM + heteroplasmy loss:
  The pre-training objective has two terms:
    1. MLM cross-entropy (kmer_logits vs kmer_labels): standard masked LM
       objective that teaches the model to predict masked k-mers from context.
    2. Heteroplasmy MSE (het_preds vs het_labels): regression head that learns
       to predict the observed heteroplasmy level at each position.
  het_weight=0.0 during Phase 1 (cross-species corpus has no het data).
  het_weight=0.3 during Phase 2 (human HmtDB corpus with gnomAD het levels).

WHY ignore_index=-100:
  CE loss is computed ONLY at positions where kmer_labels != -100.
  At non-masked positions the label is -100 (set by MtDNAMaskingCollator),
  so they contribute no gradient. This is the standard BERT masking convention.

WHY het MSE uses labels != -1:
  het_labels == -1 at positions with no heteroplasmy data (most positions).
  MSE is computed only at masked positions where gnomAD provides het levels.
  Using -1 as a sentinel (not -100) avoids confusion with token labels.

WHY MSE not Huber for pre-training het loss:
  Huber is reserved for the dedicated heteroplasmy regression fine-tuning task
  (Day 18) where the dataset is small and individual outliers matter more.
  For pre-training the het signal is just a regulariser on the encoder; MSE
  is simpler and the variance of gnomAD-level estimates is low enough that
  Huber's robustness is not needed here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mtdna_mlm_loss(
    kmer_logits: torch.Tensor,
    kmer_labels: torch.Tensor,
    het_preds: torch.Tensor | None = None,
    het_labels: torch.Tensor | None = None,
    mlm_weight: float = 1.0,
    het_weight: float = 0.0,
) -> torch.Tensor:
    """
    Combined masked k-mer language modelling + heteroplasmy MSE loss.

    Parameters
    ----------
    kmer_logits:
        Shape (batch, seq_len, vocab_size). Raw logits from the k-mer prediction
        head (not softmaxed).
    kmer_labels:
        Shape (batch, seq_len). Original token IDs at masked positions, -100
        elsewhere. CE loss is computed only where label != -100.
    het_preds:
        Shape (batch, seq_len) or (batch, seq_len, 1). Sigmoid-bounded het
        level predictions from het_prediction_head. Required when het_weight > 0.
    het_labels:
        Shape (batch, seq_len). Observed het levels at masked positions, -1.0
        elsewhere. MSE is computed only where label != -1.
    mlm_weight:
        Scalar weight for the MLM loss term. Default 1.0.
    het_weight:
        Scalar weight for the heteroplasmy MSE term. Default 0.0 (Phase 1).
        Set to 0.3 for Phase 2 training on human HmtDB.

    Returns
    -------
    torch.Tensor
        Combined scalar loss.
    """
    mlm_loss = F.cross_entropy(
        kmer_logits.view(-1, kmer_logits.size(-1)),
        kmer_labels.view(-1),
        ignore_index=-100,
    )

    if het_weight > 0 and het_preds is not None and het_labels is not None:
        # Flatten het_preds in case it has a trailing dim-1 from the prediction head
        het_preds_flat = het_preds.squeeze(-1)  # (batch, seq_len)
        valid = het_labels != -1
        if valid.any():
            het_loss = F.mse_loss(
                het_preds_flat[valid],
                het_labels[valid].float(),
            )
            return mlm_weight * mlm_loss + het_weight * het_loss

    return mlm_weight * mlm_loss
