"""
PyTorch Dataset for mtDNA pre-training.

Each 16,569-bp genome is tokenised once (stride=1, circular=True → genome_length
tokens).  The token stream is then split into overlapping windows of
`window_size` tokens with step `stride`.  Windows wrap circularly so the
16568/0 junction is always covered by at least one window.

Position IDs are absolute genomic coordinates (not window-relative), which
is required for circular positional encoding to map tokens correctly to the
genome coordinate system.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from mtdna_fm.tokenizer.tokenize import tokenize_sequence
from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

if TYPE_CHECKING:
    import pandas as pd


class MtDNADataset(Dataset):
    """
    Windowed Dataset for mitochondrial genome pre-training.

    Parameters
    ----------
    sequences:
        List of DNA strings, each expected to be `genome_length` bases.
    vocabulary:
        KmerVocabulary built with KmerVocabulary.build(k=k).
    k:
        K-mer size; must match the vocabulary.
    window_size:
        Number of tokens per training window (default 512).
    stride:
        Step between consecutive window starts in tokens (default 256).
    genome_length:
        Full genome length in base pairs (default 16,569).
    het_level_vectors:
        Optional per-base heteroplasmy arrays, one per sequence.
        Each array must be genome_length floats in [0.0, 1.0].
    labels:
        Optional per-sequence integer label.  Each window inherits its
        parent sequence label.
    """

    def __init__(
        self,
        sequences: list[str],
        vocabulary: KmerVocabulary,
        k: int = 6,
        window_size: int = 512,
        stride: int = 256,
        genome_length: int = 16569,
        het_level_vectors: list[np.ndarray | None] | None = None,
        labels: list[int | None] | None = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.k = k
        self.window_size = window_size
        self.stride = stride
        self.genome_length = genome_length

        # Pre-tokenise every sequence: stride=1, circular=True → genome_length tokens
        self._tokens: list[dict] = []
        for i, seq in enumerate(sequences):
            het = het_level_vectors[i] if het_level_vectors is not None else None
            tokens = tokenize_sequence(
                seq,
                vocabulary,
                k=k,
                stride=1,
                max_seq_len=genome_length,
                circular=True,
                het_levels=het,
            )
            self._tokens.append(tokens)

        self._labels = labels

        # Build flat index: (seq_idx, window_start_token)
        # Windows start at 0, stride, 2*stride, ... up to genome_length - 1.
        # The last window wraps circularly, covering the 16568/0 boundary.
        self._index: list[tuple[int, int]] = []
        for seq_idx in range(len(sequences)):
            for start in range(0, genome_length, stride):
                self._index.append((seq_idx, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        seq_idx, window_start = self._index[idx]
        tokens = self._tokens[seq_idx]
        n = len(tokens["input_ids"])

        # Circular window: token indices wrap at n (= genome_length)
        indices = [(window_start + i) % n for i in range(self.window_size)]

        result: dict = {
            "input_ids": torch.tensor([tokens["input_ids"][j] for j in indices], dtype=torch.long),
            "attention_mask": torch.tensor(
                [tokens["attention_mask"][j] for j in indices], dtype=torch.long
            ),
            "position_ids": torch.tensor(
                [tokens["position_ids"][j] for j in indices], dtype=torch.long
            ),
            "het_values": torch.tensor(
                [tokens["het_values"][j] for j in indices], dtype=torch.float
            ),
        }

        if self._labels is not None:
            label = self._labels[seq_idx]
            result["labels"] = torch.tensor(-1 if label is None else label, dtype=torch.long)

        return result

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        vocabulary: KmerVocabulary,
        sequence_col: str = "sequence",
        het_col: str | None = "het_level_vector",
        label_col: str | None = None,
        **kwargs,
    ) -> MtDNADataset:
        """Construct a dataset directly from a processed parquet DataFrame."""
        sequences = df[sequence_col].tolist()

        het_vectors: list[np.ndarray | None] | None = None
        if het_col is not None and het_col in df.columns:
            het_vectors = [
                np.asarray(v, dtype=np.float32) if v is not None else None for v in df[het_col]
            ]

        labels: list[int | None] | None = None
        if label_col is not None and label_col in df.columns:
            labels = [
                int(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
                for v in df[label_col]
            ]

        return cls(sequences, vocabulary, het_level_vectors=het_vectors, labels=labels, **kwargs)
