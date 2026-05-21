"""
Shared test fixtures for mtDNA-FM.

Uses tiny configs and synthetic sequences so tests run in milliseconds
without requiring real data or GPU.
"""

import numpy as np
import pytest


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
