"""
Tests for Day 20: ancient DNA download utilities and UMAP visualisation.

All tests mock NCBI calls and use tiny synthetic data, so they run fast
without requiring a network connection or the real model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── TestAncientAccessions ──────────────────────────────────────────────────────


class TestAncientAccessions:
    def test_accessions_dict_keys(self):
        """ANCIENT_ACCESSIONS must have neanderthal and denisovan entries."""
        from mtdna_fm.data.ancient_dna import ANCIENT_ACCESSIONS

        assert "neanderthal" in ANCIENT_ACCESSIONS
        assert "denisovan" in ANCIENT_ACCESSIONS

    def test_accession_fields(self):
        """Each entry must have accession, label, and site fields."""
        from mtdna_fm.data.ancient_dna import ANCIENT_ACCESSIONS

        for name, meta in ANCIENT_ACCESSIONS.items():
            assert "accession" in meta, f"{name} missing accession"
            assert "label" in meta, f"{name} missing label"
            assert "site" in meta, f"{name} missing site"

    def test_neanderthal_accession(self):
        """NC_011137.1 is the canonical Neanderthal mtDNA accession."""
        from mtdna_fm.data.ancient_dna import ANCIENT_ACCESSIONS

        assert ANCIENT_ACCESSIONS["neanderthal"]["accession"] == "NC_011137.1"

    def test_denisovan_accession(self):
        """FR695060.1 is the canonical Denisovan mtDNA accession."""
        from mtdna_fm.data.ancient_dna import ANCIENT_ACCESSIONS

        assert ANCIENT_ACCESSIONS["denisovan"]["accession"] == "FR695060.1"


# ── TestDownloadAncientAccession ───────────────────────────────────────────────


class TestDownloadAncientAccession:
    def test_idempotent_skips_existing(self, tmp_path):
        """download_ancient_accession skips download if file already exists."""
        from mtdna_fm.data.ancient_dna import download_ancient_accession

        # Pre-create the expected file
        fasta_path = tmp_path / "NC_011137.1.fasta"
        fasta_path.write_text(">NC_011137.1\nATGCATGCATGC\n")

        with patch("mtdna_fm.data.ancient_dna.Entrez.efetch") as mock_efetch:
            result = download_ancient_accession("NC_011137.1", output_dir=tmp_path)
            # efetch should NOT have been called
            mock_efetch.assert_not_called()

        assert result == fasta_path

    def test_downloads_when_missing(self, tmp_path):
        """download_ancient_accession calls efetch when file is absent."""
        from mtdna_fm.data.ancient_dna import download_ancient_accession

        fasta_content = ">NC_011137.1 Neanderthal\nATGCATGCATGC\n"

        mock_handle = MagicMock()
        mock_handle.read.return_value = fasta_content
        mock_handle.__enter__ = lambda self: self
        mock_handle.__exit__ = MagicMock(return_value=False)

        with (
            patch("mtdna_fm.data.ancient_dna.Entrez.efetch", return_value=mock_handle),
            patch("mtdna_fm.data.ancient_dna.time.sleep"),
        ):
            result = download_ancient_accession("NC_011137.1", output_dir=tmp_path)

        assert result.exists()
        assert result.read_text() == fasta_content

    def test_force_redownloads(self, tmp_path):
        """force=True triggers a fresh download even if file exists."""
        from mtdna_fm.data.ancient_dna import download_ancient_accession

        fasta_path = tmp_path / "NC_011137.1.fasta"
        fasta_path.write_text(">old\nAAAA\n")

        new_content = ">NC_011137.1\nATGCATGCATGC\n"
        mock_handle = MagicMock()
        mock_handle.read.return_value = new_content

        with (
            patch("mtdna_fm.data.ancient_dna.Entrez.efetch", return_value=mock_handle),
            patch("mtdna_fm.data.ancient_dna.time.sleep"),
        ):
            result = download_ancient_accession(
                "NC_011137.1", output_dir=tmp_path, force=True
            )

        assert result.read_text() == new_content

    def test_output_dir_created(self, tmp_path):
        """download_ancient_accession creates the output directory if needed."""
        from mtdna_fm.data.ancient_dna import download_ancient_accession

        nested_dir = tmp_path / "a" / "b" / "c"
        assert not nested_dir.exists()

        fasta_content = ">ACC\nATGC\n"
        mock_handle = MagicMock()
        mock_handle.read.return_value = fasta_content

        with (
            patch("mtdna_fm.data.ancient_dna.Entrez.efetch", return_value=mock_handle),
            patch("mtdna_fm.data.ancient_dna.time.sleep"),
        ):
            download_ancient_accession("ACC", output_dir=nested_dir)

        assert nested_dir.exists()


# ── TestLoadAncientSequence ────────────────────────────────────────────────────


class TestLoadAncientSequence:
    def test_loads_single_record(self, tmp_path):
        """load_ancient_sequence returns the first SeqRecord from a FASTA file."""
        from mtdna_fm.data.ancient_dna import load_ancient_sequence

        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq1 description\nATGCATGCATGCATGCATGC\n")

        record = load_ancient_sequence(fasta)
        assert str(record.seq) == "ATGCATGCATGCATGCATGC"
        assert record.id == "seq1"

    def test_raises_on_empty_fasta(self, tmp_path):
        """load_ancient_sequence raises ValueError on an empty FASTA file."""
        from mtdna_fm.data.ancient_dna import load_ancient_sequence

        empty = tmp_path / "empty.fasta"
        empty.write_text("")

        with pytest.raises(ValueError, match="No sequences found"):
            load_ancient_sequence(empty)

    def test_returns_first_record_when_multiple(self, tmp_path):
        """If the FASTA has multiple records, only the first is returned."""
        from mtdna_fm.data.ancient_dna import load_ancient_sequence

        fasta = tmp_path / "multi.fasta"
        fasta.write_text(">first\nAAAA\n>second\nCCCC\n")

        record = load_ancient_sequence(fasta)
        assert str(record.seq) == "AAAA"


# ── TestDownloadAllAncient ─────────────────────────────────────────────────────


class TestDownloadAllAncient:
    def test_returns_both_paths(self, tmp_path):
        """download_all_ancient returns paths for both neanderthal and denisovan."""
        from mtdna_fm.data.ancient_dna import download_all_ancient

        mock_handle = MagicMock()
        mock_handle.read.return_value = ">ACC\nATGC\n"

        with (
            patch("mtdna_fm.data.ancient_dna.Entrez.efetch", return_value=mock_handle),
            patch("mtdna_fm.data.ancient_dna.time.sleep"),
        ):
            paths = download_all_ancient(output_dir=tmp_path)

        assert "neanderthal" in paths
        assert "denisovan" in paths
        for path in paths.values():
            assert isinstance(path, Path)

    def test_both_files_exist(self, tmp_path):
        """Both FASTA files are written to disk."""
        from mtdna_fm.data.ancient_dna import download_all_ancient

        mock_handle = MagicMock()
        mock_handle.read.return_value = ">ACC\nATGC\n"

        with (
            patch("mtdna_fm.data.ancient_dna.Entrez.efetch", return_value=mock_handle),
            patch("mtdna_fm.data.ancient_dna.time.sleep"),
        ):
            paths = download_all_ancient(output_dir=tmp_path)

        for path in paths.values():
            assert path.exists()


# ── TestPlotUmapWithAncientDna ─────────────────────────────────────────────────


class TestPlotUmapWithAncientDna:
    @pytest.fixture()
    def synthetic_embeddings(self):
        """Return tiny synthetic modern + ancient embeddings for plot testing."""
        rng = np.random.default_rng(42)
        modern = rng.standard_normal((20, 8)).astype(np.float32)
        ancient = rng.standard_normal((2, 8)).astype(np.float32)
        modern_labels = [f"H{i}" for i in range(10)] * 2
        ancient_labels = ["Neanderthal", "Denisovan"]
        return modern, modern_labels, ancient, ancient_labels

    def test_returns_figure(self, synthetic_embeddings):
        """plot_umap_with_ancient_dna must return a matplotlib Figure."""
        import matplotlib
        matplotlib.use("Agg")

        from mtdna_fm.evaluation.viz import plot_umap_with_ancient_dna

        modern, m_labels, ancient, a_labels = synthetic_embeddings
        fig = plot_umap_with_ancient_dna(
            modern_embeddings=modern,
            modern_labels=m_labels,
            ancient_embeddings=ancient,
            ancient_labels=a_labels,
            n_neighbors=3,
        )
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_saves_without_error(self, synthetic_embeddings, tmp_path):
        """Figure can be saved to disk."""
        import matplotlib
        matplotlib.use("Agg")

        from mtdna_fm.evaluation.viz import plot_umap_with_ancient_dna

        modern, m_labels, ancient, a_labels = synthetic_embeddings
        fig = plot_umap_with_ancient_dna(
            modern_embeddings=modern,
            modern_labels=m_labels,
            ancient_embeddings=ancient,
            ancient_labels=a_labels,
            n_neighbors=3,
        )
        out = tmp_path / "ancient_umap.png"
        fig.savefig(str(out))
        assert out.exists()

    def test_single_ancient_works(self, synthetic_embeddings):
        """Works when only one ancient sequence is provided."""
        import matplotlib
        matplotlib.use("Agg")

        from mtdna_fm.evaluation.viz import plot_umap_with_ancient_dna

        modern, m_labels, ancient, a_labels = synthetic_embeddings
        fig = plot_umap_with_ancient_dna(
            modern_embeddings=modern,
            modern_labels=m_labels,
            ancient_embeddings=ancient[:1],
            ancient_labels=a_labels[:1],
            n_neighbors=3,
        )
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)


# ── TestEmbedGenomeLengthNormalization ─────────────────────────────────────────


class TestEmbedGenomeLengthNormalization:
    """embed_genome must handle sequences of arbitrary length gracefully."""

    @pytest.fixture()
    def tiny_embedder(self, tiny_config, tiny_vocabulary):
        from mtdna_fm.inference.api import MtDNAEmbedder
        from mtdna_fm.model.model import MtDNAModel

        model = MtDNAModel(tiny_config)
        return MtDNAEmbedder(
            model=model,
            vocabulary=tiny_vocabulary,
            device="cpu",
            window_size=10,
            stride=5,
        )

    def test_longer_sequence_truncated(self, tiny_embedder):
        """Sequences longer than genome_length should be truncated."""
        rng = np.random.default_rng(0)
        # genome_length=100; give 105 bp
        seq = "".join(rng.choice(list("ACGT"), size=105))
        vec = tiny_embedder.embed_genome(seq)
        assert vec.shape == (tiny_embedder.model.config.hidden_size,)

    def test_shorter_sequence_padded(self, tiny_embedder):
        """Sequences shorter than genome_length should be N-padded."""
        rng = np.random.default_rng(0)
        # genome_length=100; give 95 bp
        seq = "".join(rng.choice(list("ACGT"), size=95))
        vec = tiny_embedder.embed_genome(seq)
        assert vec.shape == (tiny_embedder.model.config.hidden_size,)

    def test_exact_length_unchanged(self, tiny_embedder):
        """Sequences exactly genome_length should produce the same result."""
        rng = np.random.default_rng(0)
        seq = "".join(rng.choice(list("ACGT"), size=100))
        v1 = tiny_embedder.embed_genome(seq)
        v2 = tiny_embedder.embed_genome(seq)
        np.testing.assert_array_equal(v1, v2)
