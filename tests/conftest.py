"""
Shared test fixtures for mtDNA-FM.

Uses tiny configs and synthetic sequences so tests run in milliseconds
without requiring real data or GPU.
"""

import numpy as np
import pytest

from mtdna_fm.model.config import MtDNAConfig
from mtdna_fm.tokenizer.vocabulary import KmerVocabulary


@pytest.fixture()
def synthetic_sequence() -> str:
    """100 bp synthetic circular DNA sequence (fixed seed for reproducibility)."""
    rng = np.random.default_rng(42)
    return "".join(rng.choice(list("ACGT"), size=100))


@pytest.fixture()
def synthetic_sequence_16569() -> str:
    """Full-length synthetic mtDNA sequence (16569 bp, rCRS length)."""
    rng = np.random.default_rng(42)
    return "".join(rng.choice(list("ACGT"), size=16569))


@pytest.fixture()
def tiny_config() -> MtDNAConfig:
    """
    Minimal MtDNAConfig for fast unit tests.

    Uses k=3 vocabulary (64 3-mers + 6 special = 70 tokens) and
    a 100-bp synthetic genome so all components run in milliseconds
    without a GPU.
    """
    return MtDNAConfig(
        vocab_size=70,  # 64 3-mers + 6 special tokens
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_seq_len=12,
        genome_length=100,
        use_circular_encoding=True,
        use_het_projection=True,
    )


@pytest.fixture()
def tiny_vocabulary() -> KmerVocabulary:
    """3-mer vocabulary (64 k-mers + 6 special = 70 tokens) — fast for tests."""
    return KmerVocabulary.build(k=3)
