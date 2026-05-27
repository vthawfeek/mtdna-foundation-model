"""
Data collator that applies masked k-mer modeling to a batch of tokenised mtDNA windows.

WHY BERT's 80/10/10 masking strategy:
  Of the 15% of k-mer tokens selected for masking:
    - 80% are replaced with [MASK] token:
        The primary masking signal — forces the model to predict the k-mer from
        its surrounding genomic context.
    - 10% are replaced with a random k-mer token:
        Prevents the shortcut: "if I see [MASK], I must predict something; otherwise
        I can ignore that position." By occasionally replacing with random tokens,
        the model must maintain good representations of ALL positions to detect
        sequence inconsistencies.
    - 10% are kept as the original token:
        Ensures representations are useful at inference time when no [MASK] tokens
        are present. The model must produce sensible embeddings even for unmasked
        positions.
  This was the key innovation in BERT that made MLM work well and has been
  validated across genomic language models.

WHY mask_prob=15%:
  Too low: few prediction targets per forward pass, slow learning.
  Too high: too much context removed, neighbours can't reconstruct the identity.
  15% is the empirically established sweet spot from BERT, validated on all
  subsequent MLM pre-training work including genomic LMs.

WHY the C-tract blacklist (positions 303-315):
  The homopolymeric C-stretch in the D-loop (approximately positions 303-315 of
  the rCRS reference) is a known sequencing artefact region:
    - Sequencers systematically over- or under-call C-count in homopolymer runs
    - Most mtDNA databases contain population-level noise at these positions
    - The true biological sequence is not reliably recoverable from short reads
  Teaching the model to predict these positions would cause it to model
  sequencing noise rather than biological signal. The blacklist ensures they
  are never selected as masking targets.
  This is the key biological customisation that distinguishes MtDNAMaskingCollator
  from a generic BERT collator.

WHY CLS and PAD are never masked:
  CLS (token ID 1) is a synthetic aggregation token; predicting its identity
  is meaningless and would corrupt genome-level embeddings.
  PAD tokens (attention_mask=0) are not real sequence positions.

WHY masking is applied in the collator (not pre-computed):
  Pre-computing masks would require storing 15% of each sequence's tokens in
  a separate label array, wasting memory. More importantly, applying masks at
  collation time means each epoch sees DIFFERENT masks over the same windows,
  which is equivalent to 1/mask_prob epochs of unique training signal per
  sequence — a significant data augmentation effect for free.
"""

from __future__ import annotations

import numpy as np
import torch

from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

# Homopolymeric C-tract in the D-loop: positions 303-315 (0-indexed, rCRS coordinates)
# This region is excluded from masking because it is dominated by sequencing noise.
_DEFAULT_BLACKLIST: tuple[int, ...] = tuple(range(303, 316))


class MtDNAMaskingCollator:
    """
    Collates a list of tokenised mtDNA window dicts into a masked batch.

    Applied per-training-step (not pre-cached) so each epoch sees different
    masks over the same windows.

    Parameters
    ----------
    vocabulary:
        KmerVocabulary providing special token IDs and vocab size.
    mask_prob:
        Fraction of eligible tokens to select for masking. Default 0.15.
    blacklist_positions:
        Genomic positions (0-indexed, rCRS coordinates) that must never be
        selected as masking targets. Defaults to the D-loop C-tract (303-315).
    seed:
        Optional RNG seed for reproducibility in tests.
    """

    def __init__(
        self,
        vocabulary: KmerVocabulary,
        mask_prob: float = 0.15,
        blacklist_positions: tuple[int, ...] = _DEFAULT_BLACKLIST,
        seed: int | None = None,
    ) -> None:
        self.vocab = vocabulary
        self.mask_prob = mask_prob
        self.blacklist_positions: frozenset[int] = frozenset(blacklist_positions)

        # Special token IDs (always ineligible for masking)
        self.mask_token_id: int = vocabulary.mask_token_id   # 2
        self.cls_token_id: int = vocabulary.cls_token_id     # 1
        self.pad_token_id: int = vocabulary.pad_token_id     # 0
        self.vocab_size: int = len(vocabulary)

        # Number of special tokens (random replacement uses only real k-mer IDs)
        self.n_special: int = vocabulary.n_special  # 6

        self._rng = np.random.default_rng(seed)

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Pad and mask a list of tokenised window dicts.

        Input:
            list of dicts, each containing:
              - input_ids:      (seq_len,) int64 k-mer token IDs
              - position_ids:   (seq_len,) int64 absolute genomic coordinates
              - attention_mask: (seq_len,) int64  1=real token, 0=PAD
              - het_values:     (seq_len,) float32 heteroplasmy level [0,1]
                                (optional — zeros used if absent)

        Output:
            dict of batched tensors (batch_size, seq_len):
              - input_ids:      masked k-mer IDs ([MASK]/random/original)
              - position_ids:   unchanged
              - attention_mask: unchanged
              - het_values:     unchanged
              - kmer_labels:    original IDs at masked positions, -100 elsewhere
              - het_labels:     het_values at masked positions,  -1   elsewhere
        """
        # ── Stack inputs ───────────────────────────────────────────────────────
        input_ids = torch.stack([s["input_ids"] for s in batch])
        position_ids = torch.stack([s["position_ids"] for s in batch])
        attention_mask = torch.stack([s["attention_mask"] for s in batch])

        if "het_values" in batch[0]:
            het_values = torch.stack([s["het_values"] for s in batch])
        else:
            het_values = torch.zeros_like(input_ids, dtype=torch.float32)

        # ── Prepare label tensors ──────────────────────────────────────────────
        original_ids = input_ids.clone()
        masked_input_ids = input_ids.clone()

        batch_size, seq_len = input_ids.shape

        # -100 = CE ignore_index (cross_entropy skips these positions)
        kmer_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        # -1 = "no het data" sentinel (het MSE loss skips positions where label == -1)
        het_labels = torch.full((batch_size, seq_len), -1.0, dtype=torch.float32)

        # ── Per-sample masking ─────────────────────────────────────────────────
        for i in range(batch_size):
            # Eligible: real tokens that are not special tokens
            token_eligible = (
                (attention_mask[i] == 1)
                & (original_ids[i] != self.cls_token_id)
                & (original_ids[i] != self.pad_token_id)
            ).numpy()

            # Additionally exclude blacklisted genomic positions
            pos_array = position_ids[i].numpy()
            blacklisted = np.isin(pos_array, list(self.blacklist_positions))
            eligible = token_eligible & ~blacklisted

            eligible_indices = np.where(eligible)[0]
            if len(eligible_indices) == 0:
                continue

            # Sample ~mask_prob fraction; always at least 1 token
            n_mask = max(1, int(round(len(eligible_indices) * self.mask_prob)))
            mask_indices = self._rng.choice(
                eligible_indices, size=min(n_mask, len(eligible_indices)), replace=False
            )

            for pos in mask_indices:
                # Record original token as prediction target
                kmer_labels[i, pos] = int(original_ids[i, pos].item())
                het_labels[i, pos] = het_values[i, pos].item()

                # 80/10/10 replacement strategy
                r = self._rng.random()
                if r < 0.80:
                    # 80 %: replace with [MASK]
                    masked_input_ids[i, pos] = self.mask_token_id
                elif r < 0.90:
                    # 10 %: replace with a random real k-mer (ID >= n_special)
                    random_id = self._rng.integers(self.n_special, self.vocab_size)
                    masked_input_ids[i, pos] = int(random_id)
                # else 10 %: keep original (no change)

        return {
            "input_ids": masked_input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "het_values": het_values,
            "kmer_labels": kmer_labels,
            "het_labels": het_labels,
        }
