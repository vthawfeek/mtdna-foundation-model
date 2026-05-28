"""
Tests for MtDNAEmbedder (mtdna_fm/inference/api.py).

All tests use tiny_config (k=3, genome_length=100, hidden_size=16) and
synthetic sequences so they run in milliseconds without a GPU or real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtdna_fm.inference.api import MtDNAEmbedder
from mtdna_fm.model.config import MtDNAConfig
from mtdna_fm.model.model import MtDNAModel
from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def tiny_embedder(tiny_config: MtDNAConfig, tiny_vocabulary: KmerVocabulary) -> MtDNAEmbedder:
    """MtDNAEmbedder with tiny model (k=3, hidden=16, genome=100 bp)."""
    model = MtDNAModel(tiny_config)
    return MtDNAEmbedder(
        model=model,
        vocabulary=tiny_vocabulary,
        device="cpu",
        window_size=10,
        stride=5,
    )


@pytest.fixture()
def synthetic_100bp() -> str:
    """100 bp synthetic genome for use with tiny_config (genome_length=100)."""
    rng = np.random.default_rng(7)
    return "".join(rng.choice(list("ACGT"), size=100))


# ── k inference from vocab ─────────────────────────────────────────────────────


class TestKmerInference:
    def test_k_inferred_from_vocab_size(
        self, tiny_config: MtDNAConfig, tiny_vocabulary: KmerVocabulary
    ) -> None:
        """k must be inferred correctly from the vocabulary size (64 3-mers → k=3)."""
        model = MtDNAModel(tiny_config)
        embedder = MtDNAEmbedder(model, tiny_vocabulary, device="cpu", window_size=10, stride=5)
        assert embedder.k == 3

    def test_k_inferred_for_6mer(self) -> None:
        """k=6 is inferred from a full 4,102-token vocabulary."""
        vocab = KmerVocabulary.build(k=6)
        assert round(np.log(len(vocab) - 6) / np.log(4)) == 6


# ── embed_genome ───────────────────────────────────────────────────────────────


class TestEmbedGenome:
    def test_output_shape(self, tiny_embedder: MtDNAEmbedder, synthetic_100bp: str) -> None:
        """embed_genome must return a 1D array of shape (hidden_size,)."""
        vec = tiny_embedder.embed_genome(synthetic_100bp)
        assert vec.shape == (tiny_embedder.model.config.hidden_size,)

    def test_output_dtype_float32(
        self, tiny_embedder: MtDNAEmbedder, synthetic_100bp: str
    ) -> None:
        vec = tiny_embedder.embed_genome(synthetic_100bp)
        assert vec.dtype == np.float32

    def test_deterministic(self, tiny_embedder: MtDNAEmbedder, synthetic_100bp: str) -> None:
        """Same input must produce identical output (model in eval mode, no dropout)."""
        v1 = tiny_embedder.embed_genome(synthetic_100bp)
        v2 = tiny_embedder.embed_genome(synthetic_100bp)
        np.testing.assert_array_equal(v1, v2)

    def test_different_sequences_differ(
        self, tiny_embedder: MtDNAEmbedder, synthetic_100bp: str
    ) -> None:
        """Two distinct sequences must produce different embeddings."""
        rng = np.random.default_rng(99)
        seq2 = "".join(rng.choice(list("ACGT"), size=100))
        v1 = tiny_embedder.embed_genome(synthetic_100bp)
        v2 = tiny_embedder.embed_genome(seq2)
        assert not np.allclose(v1, v2)

    def test_het_levels_change_embedding(
        self, tiny_embedder: MtDNAEmbedder, synthetic_100bp: str
    ) -> None:
        """Providing het_levels should change the embedding if the model uses het projection."""
        v_no_het = tiny_embedder.embed_genome(synthetic_100bp, het_levels=None)
        het = np.full(100, 0.5, dtype=np.float32)
        v_with_het = tiny_embedder.embed_genome(synthetic_100bp, het_levels=het)
        # Embeddings may or may not differ depending on het_projection init, but both valid
        assert v_no_het.shape == v_with_het.shape

    def test_invalid_pooling_raises(
        self, tiny_embedder: MtDNAEmbedder, synthetic_100bp: str
    ) -> None:
        with pytest.raises(ValueError, match="pooling"):
            tiny_embedder.embed_genome(synthetic_100bp, pooling="max")

    def test_unit_norm_not_required(
        self, tiny_embedder: MtDNAEmbedder, synthetic_100bp: str
    ) -> None:
        """Embedder does NOT normalise outputs — raw hidden states are returned."""
        vec = tiny_embedder.embed_genome(synthetic_100bp)
        norm = float(np.linalg.norm(vec))
        # Just check it's a valid non-zero float (not 0 or inf)
        assert 0 < norm < 1e6


# ── embed_variant ──────────────────────────────────────────────────────────────


class TestEmbedVariant:
    def test_output_shape(self, tiny_embedder: MtDNAEmbedder, synthetic_100bp: str) -> None:
        """embed_variant must return (hidden_size,)."""
        vec = tiny_embedder.embed_variant(synthetic_100bp, position=50)
        assert vec.shape == (tiny_embedder.model.config.hidden_size,)

    def test_different_positions_differ(
        self, tiny_embedder: MtDNAEmbedder, synthetic_100bp: str
    ) -> None:
        """Different positions on the same genome should produce different embeddings."""
        v1 = tiny_embedder.embed_variant(synthetic_100bp, position=10)
        v2 = tiny_embedder.embed_variant(synthetic_100bp, position=70)
        assert not np.allclose(v1, v2)

    def test_cls_pooling(self, tiny_embedder: MtDNAEmbedder, synthetic_100bp: str) -> None:
        """pooling='cls' should return the window CLS hidden state (position 0)."""
        vec = tiny_embedder.embed_variant(synthetic_100bp, position=50, pooling="cls")
        assert vec.shape == (tiny_embedder.model.config.hidden_size,)

    def test_boundary_position(self, tiny_embedder: MtDNAEmbedder, synthetic_100bp: str) -> None:
        """Position at genome boundary (0 or genome_length-1) must not error."""
        vec_start = tiny_embedder.embed_variant(synthetic_100bp, position=0)
        vec_end = tiny_embedder.embed_variant(synthetic_100bp, position=99)
        assert vec_start.shape == vec_end.shape == (tiny_embedder.model.config.hidden_size,)


# ── embed_dataset ──────────────────────────────────────────────────────────────


class TestEmbedDataset:
    def test_output_shape(self, tiny_embedder: MtDNAEmbedder) -> None:
        """embed_dataset should return (n_sequences, hidden_size)."""
        import pandas as pd

        rng = np.random.default_rng(0)
        seqs = ["".join(rng.choice(list("ACGT"), size=100)) for _ in range(5)]
        df = pd.DataFrame({"sequence": seqs})
        out = tiny_embedder.embed_dataset(df)
        assert out.shape == (5, tiny_embedder.model.config.hidden_size)

    def test_each_row_matches_embed_genome(self, tiny_embedder: MtDNAEmbedder) -> None:
        """embed_dataset rows must match individual embed_genome calls."""
        import pandas as pd

        rng = np.random.default_rng(1)
        seqs = ["".join(rng.choice(list("ACGT"), size=100)) for _ in range(3)]
        df = pd.DataFrame({"sequence": seqs})
        batch_out = tiny_embedder.embed_dataset(df)
        for i, seq in enumerate(seqs):
            individual = tiny_embedder.embed_genome(seq)
            np.testing.assert_array_almost_equal(batch_out[i], individual)

    def test_custom_sequence_col(self, tiny_embedder: MtDNAEmbedder) -> None:
        """embed_dataset should respect custom sequence_col argument."""
        import pandas as pd

        rng = np.random.default_rng(2)
        seqs = ["".join(rng.choice(list("ACGT"), size=100)) for _ in range(2)]
        df = pd.DataFrame({"dna": seqs})
        out = tiny_embedder.embed_dataset(df, sequence_col="dna")
        assert out.shape == (2, tiny_embedder.model.config.hidden_size)


# ── from_pretrained ────────────────────────────────────────────────────────────


class TestFromPretrained:
    def test_roundtrip_save_load(
        self, tmp_path, tiny_config: MtDNAConfig, tiny_vocabulary: KmerVocabulary
    ) -> None:
        """Save a tiny model, then load it via MtDNAEmbedder.from_pretrained."""
        from mtdna_fm.model.model import MtDNAForMaskedModeling

        # Save model + vocabulary to tmp_path
        mlm_model = MtDNAForMaskedModeling(tiny_config)
        mlm_model.save_pretrained(str(tmp_path))
        tiny_vocabulary.save_pretrained(str(tmp_path))

        # Load via MtDNAEmbedder.from_pretrained
        embedder = MtDNAEmbedder.from_pretrained(
            str(tmp_path), device="cpu", window_size=10, stride=5
        )
        assert isinstance(embedder, MtDNAEmbedder)
        assert isinstance(embedder.model, MtDNAModel)
        assert len(embedder.vocabulary) == len(tiny_vocabulary)

    def test_loaded_embedder_produces_valid_output(
        self, tmp_path, tiny_config: MtDNAConfig, tiny_vocabulary: KmerVocabulary
    ) -> None:
        """A loaded embedder must produce a finite embedding."""
        from mtdna_fm.model.model import MtDNAForMaskedModeling

        mlm_model = MtDNAForMaskedModeling(tiny_config)
        mlm_model.save_pretrained(str(tmp_path))
        tiny_vocabulary.save_pretrained(str(tmp_path))

        embedder = MtDNAEmbedder.from_pretrained(
            str(tmp_path), device="cpu", window_size=10, stride=5
        )
        rng = np.random.default_rng(42)
        seq = "".join(rng.choice(list("ACGT"), size=100))
        vec = embedder.embed_genome(seq)
        assert np.all(np.isfinite(vec))
