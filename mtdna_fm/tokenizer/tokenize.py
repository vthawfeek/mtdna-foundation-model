"""
K-mer tokenization for DNA sequences.

tokenize_sequence handles both linear and circular genomes. For circular
genomes (circular=True) k-mers wrap at the boundary so every genomic
position is covered by exactly k overlapping tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mtdna_fm.tokenizer.vocabulary import UNK_TOKEN_ID, KmerVocabulary

if TYPE_CHECKING:
    pass


def tokenize_sequence(
    seq: str,
    vocabulary: KmerVocabulary,
    k: int = 6,
    stride: int = 1,
    max_seq_len: int = 512,
    circular: bool = True,
    het_levels: np.ndarray | None = None,
) -> dict:
    """
    Tokenize a DNA sequence into k-mer token IDs.

    Parameters
    ----------
    seq:
        DNA string (A/C/G/T/N). Case-insensitive.
    vocabulary:
        KmerVocabulary instance used for encoding.
    k:
        K-mer size. Must match the vocabulary's k.
    stride:
        Step between consecutive k-mer start positions.
    max_seq_len:
        Maximum number of tokens returned (truncates, does not pad).
    circular:
        When True, k-mers wrap around the sequence boundary so the junction
        between the last and first bases is covered. Produces len(seq)//stride
        tokens before truncation for stride=1.
    het_levels:
        Optional per-position heteroplasmy levels in [0.0, 1.0]. Must be the
        same length as seq. Defaults to all-zeros.

    Returns
    -------
    dict with keys:
        input_ids       : list[int] — k-mer token IDs
        attention_mask  : list[int] — 1 for real tokens (no padding)
        position_ids    : list[int] — genomic start position of each k-mer
        het_values      : list[float] — heteroplasmy level at each k-mer start
    """
    seq = seq.upper()
    L = len(seq)

    input_ids: list[int] = []
    position_ids: list[int] = []

    positions = range(0, L, stride)

    for p in positions:
        if len(input_ids) >= max_seq_len:
            break

        if circular:
            kmer = "".join(seq[(p + j) % L] for j in range(k))
        else:
            if p + k > L:
                break
            kmer = seq[p : p + k]

        tid = UNK_TOKEN_ID if "N" in kmer else vocabulary.encode(kmer)

        input_ids.append(tid)
        position_ids.append(p)

    n = len(input_ids)
    attention_mask = [1] * n

    if het_levels is not None:
        het_arr = np.asarray(het_levels, dtype=np.float32)
        het_values = [float(het_arr[p % L]) for p in position_ids[:n]]
    else:
        het_values = [0.0] * n

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "het_values": het_values,
    }
