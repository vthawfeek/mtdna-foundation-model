"""
Tests for data download clients and PyTorch Dataset classes.

All tests here are unit tests that run without network access. Integration
tests (which actually hit HmtDB or NCBI) are marked with
@pytest.mark.integration and skipped in CI / the normal test run.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# hmtdb_client tests
# ---------------------------------------------------------------------------


class TestHmtdbClient:
    def test_skips_download_when_outputs_exist(self, tmp_path: Path) -> None:
        """download_hmtdb returns immediately if both output files are present."""
        from mtdna_fm.data.hmtdb_client import (
            FASTA_FILENAME,
            METADATA_FILENAME,
            download_hmtdb,
        )

        fasta_path = tmp_path / FASTA_FILENAME
        metadata_path = tmp_path / METADATA_FILENAME
        fasta_path.write_text(">seq1\nACGT\n")
        metadata_path.write_bytes(b"fake parquet")

        # Patch _validate_fasta so we don't need real sequence data
        with patch("mtdna_fm.data.hmtdb_client._validate_fasta"):
            result_fasta, result_meta = download_hmtdb(tmp_path, force=False)

        assert result_fasta == fasta_path
        assert result_meta == metadata_path

    def test_force_triggers_redownload(self, tmp_path: Path) -> None:
        """force=True should attempt a download even if outputs exist."""
        from mtdna_fm.data.hmtdb_client import (
            FASTA_FILENAME,
            METADATA_FILENAME,
            download_hmtdb,
        )

        fasta_path = tmp_path / FASTA_FILENAME
        metadata_path = tmp_path / METADATA_FILENAME
        fasta_path.write_text(">seq1\nACGT\n")
        metadata_path.write_bytes(b"fake parquet")

        with (
            patch("mtdna_fm.data.hmtdb_client._download_fasta_from_hmtdb") as mock_dl,
            patch("mtdna_fm.data.hmtdb_client._download_metadata_from_hmtdb") as mock_meta,
            patch("mtdna_fm.data.hmtdb_client._validate_fasta"),
        ):
            mock_dl.return_value = None
            mock_meta.return_value = None
            download_hmtdb(tmp_path, force=True)

        mock_dl.assert_called_once()
        mock_meta.assert_called_once()

    def test_sha256_mismatch_raises(self, tmp_path: Path) -> None:
        """SHA256 mismatch during zip extraction raises ValueError."""
        from mtdna_fm.data.hmtdb_client import extract_zip_fasta

        fake_zip = tmp_path / "test.zip"
        fake_zip.write_bytes(b"not a real zip")
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            extract_zip_fasta(fake_zip, tmp_path, expected_sha256="aabbcc")

    def test_output_dir_created_if_missing(self, tmp_path: Path) -> None:
        """download_hmtdb creates the output directory if it does not exist."""
        from mtdna_fm.data.hmtdb_client import download_hmtdb

        new_dir = tmp_path / "nested" / "hmtdb"
        assert not new_dir.exists()

        with (
            patch("mtdna_fm.data.hmtdb_client._download_fasta_from_hmtdb"),
            patch("mtdna_fm.data.hmtdb_client._download_metadata_from_hmtdb"),
            patch("mtdna_fm.data.hmtdb_client._validate_fasta"),
        ):
            download_hmtdb(new_dir)

        assert new_dir.exists()

    def test_fallback_called_on_network_error(self, tmp_path: Path) -> None:
        """Network failure during HmtDB download triggers NCBI fallback."""
        import requests

        from mtdna_fm.data.hmtdb_client import download_hmtdb

        with (
            patch(
                "mtdna_fm.data.hmtdb_client._download_fasta_from_hmtdb",
                side_effect=requests.ConnectionError("offline"),
            ),
            patch(
                "mtdna_fm.data.hmtdb_client._ncbi_fallback",
                return_value=(tmp_path / "sequences.fasta", tmp_path / "metadata.parquet"),
            ) as mock_fallback,
            patch("mtdna_fm.data.hmtdb_client._validate_fasta"),
        ):
            download_hmtdb(tmp_path)

        mock_fallback.assert_called_once()


# ---------------------------------------------------------------------------
# ncbi_client tests
# ---------------------------------------------------------------------------


class TestNcbiClient:
    def test_progress_file_initialises_empty(self, tmp_path: Path) -> None:
        from mtdna_fm.data.ncbi_client import _load_progress

        non_existent = tmp_path / "progress.json"
        assert _load_progress(non_existent) == {}

    def test_progress_file_save_load_roundtrip(self, tmp_path: Path) -> None:
        from mtdna_fm.data.ncbi_client import _load_progress, _save_progress

        progress_path = tmp_path / "prog.json"
        data = {"0": "done", "1": "done", "2": "done"}
        _save_progress(progress_path, data)
        loaded = _load_progress(progress_path)
        assert loaded == data

    def test_progress_complete_all_done(self) -> None:
        from mtdna_fm.data.ncbi_client import _progress_complete

        progress = {str(i): "done" for i in range(4)}
        assert _progress_complete(progress, total=1500, batch_size=500)

    def test_progress_incomplete_missing_batch(self) -> None:
        from mtdna_fm.data.ncbi_client import _progress_complete

        progress = {"0": "done", "1": "done"}  # batch 2 missing
        assert not _progress_complete(progress, total=1500, batch_size=500)

    def test_skips_all_batches_when_complete(self, tmp_path: Path) -> None:
        """download_ncbi_mtdna returns early if FASTA exists and progress is complete."""
        from mtdna_fm.data.ncbi_client import download_ncbi_mtdna

        fasta_path = tmp_path / "vertebrate_mtdna.fasta"
        fasta_path.write_text(">seq1\nACGT\n")

        # Simulate a completed 2-batch download
        progress_path = tmp_path / "vertebrate_mtdna.fasta.progress.json"
        progress = {"0": "done", "1": "done"}
        with open(progress_path, "w") as f:
            json.dump(progress, f)

        mock_total = 2
        with patch(
            "mtdna_fm.data.ncbi_client._esearch",
            return_value=(mock_total, "webenv_xxx", "1"),
        ):
            result = download_ncbi_mtdna(
                query="test query",
                output_dir=tmp_path,
                output_filename="vertebrate_mtdna.fasta",
                batch_size=1,
            )

        assert result == fasta_path

    def test_output_dir_created(self, tmp_path: Path) -> None:
        """download_ncbi_mtdna creates the output directory if missing."""
        from mtdna_fm.data.ncbi_client import download_ncbi_mtdna

        new_dir = tmp_path / "nested" / "ncbi"
        fasta_name = "test.fasta"
        fasta_path = new_dir / fasta_name

        # Simulate a single completed batch so the fetch loop body is skipped
        with (
            patch(
                "mtdna_fm.data.ncbi_client._esearch",
                return_value=(1, "webenv", "1"),
            ),
            patch(
                "mtdna_fm.data.ncbi_client._efetch_batch",
                return_value=">seq1\nACGT\n",
            ),
            patch("mtdna_fm.data.ncbi_client._save_progress"),
        ):
            # Pre-create fasta so the complete check passes
            new_dir.mkdir(parents=True)
            fasta_path.write_text(">seq1\nACGT\n")
            prog = new_dir / f"{fasta_name}.progress.json"
            with open(prog, "w") as f:
                json.dump({"0": "done"}, f)

            result = download_ncbi_mtdna(
                query="test",
                output_dir=new_dir,
                output_filename=fasta_name,
                batch_size=500,
            )

        assert new_dir.exists()
        assert result == fasta_path

    def test_count_fasta_records(self, tmp_path: Path) -> None:
        from mtdna_fm.data.ncbi_client import count_fasta_records

        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq1\nACGT\n>seq2\nTTTT\n>seq3\nGGGG\n")
        assert count_fasta_records(fasta) == 3


# ---------------------------------------------------------------------------
# download CLI tests
# ---------------------------------------------------------------------------


class TestDownloadCLI:
    def test_invalid_source_exits_with_error(self) -> None:
        from typer.testing import CliRunner

        from mtdna_fm.scripts.download import app

        runner = CliRunner()
        result = runner.invoke(app, ["--source", "invalid-source"])
        assert result.exit_code != 0

    def test_valid_source_gnomad_calls_download(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from mtdna_fm.scripts.download import app

        runner = CliRunner()
        with patch("mtdna_fm.scripts.download._run_gnomad") as mock_run:
            result = runner.invoke(app, ["--source", "gnomad", "--output", str(tmp_path)])
        mock_run.assert_called_once_with(tmp_path, False)
        assert result.exit_code == 0

    def test_valid_source_clinvar_calls_download(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from mtdna_fm.scripts.download import app

        runner = CliRunner()
        with patch("mtdna_fm.scripts.download._run_clinvar") as mock_run:
            result = runner.invoke(app, ["--source", "clinvar", "--output", str(tmp_path)])
        mock_run.assert_called_once_with(tmp_path, False)
        assert result.exit_code == 0

    def test_valid_source_phylotree_calls_download(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from mtdna_fm.scripts.download import app

        runner = CliRunner()
        with patch("mtdna_fm.scripts.download._run_phylotree") as mock_run:
            result = runner.invoke(app, ["--source", "phylotree", "--output", str(tmp_path)])
        mock_run.assert_called_once_with(tmp_path, False)
        assert result.exit_code == 0

    def test_valid_source_hmtdb_calls_download(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from mtdna_fm.scripts.download import app

        runner = CliRunner()
        with patch("mtdna_fm.scripts.download._run_hmtdb") as mock_run:
            result = runner.invoke(app, ["--source", "hmtdb", "--output", str(tmp_path)])
        mock_run.assert_called_once_with(tmp_path, False)
        assert result.exit_code == 0

    def test_valid_source_ncbi_calls_download(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from mtdna_fm.scripts.download import app

        runner = CliRunner()
        with patch("mtdna_fm.scripts.download._run_ncbi_refseq") as mock_run:
            result = runner.invoke(app, ["--source", "ncbi-refseq", "--output", str(tmp_path)])
        mock_run.assert_called_once_with(tmp_path, False)
        assert result.exit_code == 0

    def test_force_flag_passed_through(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from mtdna_fm.scripts.download import app

        runner = CliRunner()
        with patch("mtdna_fm.scripts.download._run_ncbi_refseq") as mock_run:
            runner.invoke(
                app,
                ["--source", "ncbi-refseq", "--output", str(tmp_path), "--force"],
            )
        mock_run.assert_called_once_with(tmp_path, True)


# ---------------------------------------------------------------------------
# preprocessor tests
# ---------------------------------------------------------------------------


class TestCleanSequence:
    def test_uppercase(self) -> None:
        from mtdna_fm.data.preprocessor import clean_sequence

        assert clean_sequence("acgtacgt") == "ACGTACGT"

    def test_non_acgtn_replaced_with_n(self) -> None:
        from mtdna_fm.data.preprocessor import clean_sequence

        # IUPAC ambiguity codes (r, y, w, s, k, m) are not ACGTN → each becomes N
        result = clean_sequence("ACGTrywskm")
        assert result == "ACGTNNNNNN"  # 6 non-ACGTN chars → 6 Ns

    def test_trailing_junction_duplicate_removed(self) -> None:
        from mtdna_fm.data.preprocessor import JUNCTION_DUPLICATE_CHECK_BASES, clean_sequence

        n = JUNCTION_DUPLICATE_CHECK_BASES
        prefix = "A" * n
        middle = "G" * 300
        seq = prefix + middle + prefix  # last n bases == first n bases
        cleaned = clean_sequence(seq)
        assert len(cleaned) == n + 300
        assert not cleaned.endswith("A" * n + "A")  # trailing dup gone

    def test_no_false_positive_removal(self) -> None:
        from mtdna_fm.data.preprocessor import JUNCTION_DUPLICATE_CHECK_BASES, clean_sequence

        n = JUNCTION_DUPLICATE_CHECK_BASES
        # Last n bases differ from first n bases — nothing should be stripped
        seq = "A" * n + "G" * n + "T" * 100
        cleaned = clean_sequence(seq)
        assert len(cleaned) == len(seq)

    def test_n_passthrough(self) -> None:
        from mtdna_fm.data.preprocessor import clean_sequence

        assert clean_sequence("NNNN") == "NNNN"


class TestNormalizeLength:
    def test_exact_length_unchanged(self) -> None:
        from mtdna_fm.data.preprocessor import normalize_length

        seq = "A" * 16569
        assert normalize_length(seq) == seq

    def test_short_padded_to_target(self) -> None:
        from mtdna_fm.data.preprocessor import normalize_length

        result = normalize_length("A" * 1000)
        assert len(result) == 16569

    def test_long_trimmed_to_target(self) -> None:
        from mtdna_fm.data.preprocessor import normalize_length

        result = normalize_length("A" * 20000)
        assert len(result) == 16569

    def test_padding_inserted_at_dloop_position(self) -> None:
        from mtdna_fm.data.preprocessor import normalize_length

        # Use a small target so we can reason about character positions easily
        seq = "G" * 100
        result = normalize_length(seq, target_length=200, pad_position=50)
        assert len(result) == 200
        # First 50 chars are G (original bases before insert point)
        assert result[:50] == "G" * 50
        # Insert point onwards should contain Ns
        assert "N" in result[50:]

    def test_padded_sequence_ends_with_original_suffix(self) -> None:
        from mtdna_fm.data.preprocessor import normalize_length

        seq = "ACGT" * 25  # 100 bp
        result = normalize_length(seq, target_length=200, pad_position=50)
        # Original bases after pad_position should be preserved at the end
        assert result[200 - 50 :] == seq[50:]

    def test_custom_target_length(self) -> None:
        from mtdna_fm.data.preprocessor import normalize_length

        for target in (100, 500, 1000):
            assert len(normalize_length("A" * 50, target_length=target, pad_position=25)) == target


class TestStratifiedSplit:
    def test_split_column_added(self) -> None:
        from mtdna_fm.data.preprocessor import stratified_split

        df = pd.DataFrame(
            {
                "accession": [f"s{i}" for i in range(100)],
                "haplogroup": ["H"] * 50 + ["L"] * 50,
            }
        )
        result = stratified_split(df)
        assert "split" in result.columns

    def test_approximate_fractions(self) -> None:
        from mtdna_fm.data.preprocessor import stratified_split

        n = 1000
        df = pd.DataFrame(
            {
                "accession": [f"s{i}" for i in range(n)],
                "haplogroup": ["H"] * (n // 2) + ["L"] * (n // 2),
            }
        )
        result = stratified_split(df, train_frac=0.8, val_frac=0.1)
        counts = result["split"].value_counts()
        assert abs(counts.get("train", 0) / n - 0.8) < 0.05
        assert abs(counts.get("val", 0) / n - 0.1) < 0.05
        assert abs(counts.get("test", 0) / n - 0.1) < 0.05

    def test_unlabelled_rows_go_to_train(self) -> None:
        from mtdna_fm.data.preprocessor import stratified_split

        df = pd.DataFrame(
            {
                "accession": [f"s{i}" for i in range(100)],
                "haplogroup": ["H"] * 50 + [None] * 50,
            }
        )
        result = stratified_split(df)
        assert (result[result["haplogroup"].isna()]["split"] == "train").all()

    def test_reproducible_with_same_seed(self) -> None:
        from mtdna_fm.data.preprocessor import stratified_split

        df = pd.DataFrame(
            {
                "accession": [f"s{i}" for i in range(200)],
                "haplogroup": ["H"] * 100 + ["L"] * 100,
            }
        )
        r1 = stratified_split(df, random_state=42)
        r2 = stratified_split(df, random_state=42)
        assert (r1["split"] == r2["split"]).all()

    def test_different_seeds_give_different_splits(self) -> None:
        from mtdna_fm.data.preprocessor import stratified_split

        df = pd.DataFrame(
            {
                "accession": [f"s{i}" for i in range(500)],
                "haplogroup": ["H"] * 250 + ["L"] * 250,
            }
        )
        r1 = stratified_split(df, random_state=1)
        r2 = stratified_split(df, random_state=99)
        assert not (r1["split"] == r2["split"]).all()

    def test_few_labeled_rows_fallback(self) -> None:
        from mtdna_fm.data.preprocessor import stratified_split

        # < 10 labeled rows: all go to train, no error
        df = pd.DataFrame(
            {
                "accession": [f"s{i}" for i in range(5)],
                "haplogroup": ["H"] * 5,
            }
        )
        result = stratified_split(df)
        assert (result["split"] == "train").all()


class TestPreprocessSequences:
    def test_sequences_normalized_to_target_length(self) -> None:
        from mtdna_fm.data.preprocessor import preprocess_sequences

        df = pd.DataFrame(
            {
                "accession": ["s1", "s2"],
                "sequence": ["ACGT" * 25, "A" * 20000],  # short and long
                "haplogroup": [None, None],
                "species": ["homo_sapiens", "homo_sapiens"],
                "geographic_origin": [None, None],
                "het_level_vector": [None, None],
            }
        )
        result = preprocess_sequences(df, target_length=16569)
        assert (result["sequence"].str.len() == 16569).all()

    def test_qc_pass_column_added(self) -> None:
        from mtdna_fm.data.preprocessor import preprocess_sequences

        df = pd.DataFrame(
            {
                "accession": ["good", "bad"],
                "sequence": ["ACGT" * 4000 + "A" * 569, "N" * 16569],
                "haplogroup": [None, None],
                "species": ["homo_sapiens", "homo_sapiens"],
                "geographic_origin": [None, None],
                "het_level_vector": [None, None],
            }
        )
        result = preprocess_sequences(df, target_length=16569, min_n_fraction=0.1)
        assert "qc_pass" in result.columns
        assert result.loc[result["accession"] == "good", "qc_pass"].iloc[0]
        assert not result.loc[result["accession"] == "bad", "qc_pass"].iloc[0]

    def test_length_raw_column_reflects_pre_norm_length(self) -> None:
        from mtdna_fm.data.preprocessor import preprocess_sequences

        df = pd.DataFrame(
            {
                "accession": ["s1"],
                "sequence": ["ACGT" * 25],  # 100 bp
                "haplogroup": [None],
                "species": ["homo_sapiens"],
                "geographic_origin": [None],
                "het_level_vector": [None],
            }
        )
        result = preprocess_sequences(df, target_length=16569)
        assert result["length_raw"].iloc[0] == 100


class TestBuildRecordDataframe:
    def test_parses_fasta_into_dataframe(self, tmp_path: Path) -> None:
        from mtdna_fm.data.preprocessor import build_record_dataframe

        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq1 description\nACGTACGT\n>seq2\nTTTTGGGG\n")
        df = build_record_dataframe(fasta)
        assert len(df) == 2
        assert list(df["accession"]) == ["seq1", "seq2"]

    def test_schema_columns_present(self, tmp_path: Path) -> None:
        from mtdna_fm.data.preprocessor import build_record_dataframe

        fasta = tmp_path / "test.fasta"
        fasta.write_text(">s1\nACGT\n")
        df = build_record_dataframe(fasta)
        for col in (
            "accession",
            "sequence",
            "haplogroup",
            "species",
            "geographic_origin",
            "het_level_vector",
        ):
            assert col in df.columns

    def test_metadata_merge(self, tmp_path: Path) -> None:
        from mtdna_fm.data.preprocessor import build_record_dataframe

        fasta = tmp_path / "test.fasta"
        fasta.write_text(">s1\nACGT\n>s2\nTTTT\n")
        meta = pd.DataFrame({"accession": ["s1"], "haplogroup": ["H1a"]})
        df = build_record_dataframe(fasta, metadata_df=meta)
        assert df.loc[df["accession"] == "s1", "haplogroup"].iloc[0] == "H1a"
        # s2 has no metadata entry — haplogroup should be NaN
        assert pd.isna(df.loc[df["accession"] == "s2", "haplogroup"].iloc[0])

    def test_empty_fasta_raises(self, tmp_path: Path) -> None:
        from mtdna_fm.data.preprocessor import build_record_dataframe

        fasta = tmp_path / "empty.fasta"
        fasta.write_text("")
        with pytest.raises(ValueError, match="No sequences found"):
            build_record_dataframe(fasta)


class TestSaveSplits:
    def test_creates_three_parquet_files(self, tmp_path: Path) -> None:
        from mtdna_fm.data.preprocessor import save_splits

        df = pd.DataFrame(
            {
                "accession": [f"s{i}" for i in range(10)],
                "sequence": ["ACGT"] * 10,
                "haplogroup": [None] * 10,
                "species": ["homo_sapiens"] * 10,
                "geographic_origin": [None] * 10,
                "het_level_vector": [None] * 10,
                "split": ["train"] * 8 + ["val"] * 1 + ["test"] * 1,
            }
        )
        paths = save_splits(df, tmp_path)
        assert set(paths.keys()) == {"train", "val", "test"}
        for path in paths.values():
            assert path.exists()

    def test_split_column_excluded_from_output(self, tmp_path: Path) -> None:
        from mtdna_fm.data.preprocessor import save_splits

        df = pd.DataFrame(
            {
                "accession": ["s1"],
                "sequence": ["ACGT"],
                "split": ["train"],
            }
        )
        paths = save_splits(df, tmp_path)
        loaded = pd.read_parquet(paths["train"])
        assert "split" not in loaded.columns


# ---------------------------------------------------------------------------
# variant_processor tests
# ---------------------------------------------------------------------------


def _make_vcf(tmp_path: Path, filename: str, lines: list[str]) -> Path:
    """Write a minimal VCF file for testing."""
    path = tmp_path / filename
    header = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    path.write_text(header + "\n".join(lines) + "\n")
    return path


class TestGnomadParser:
    def test_parses_pass_snp(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_gnomad_chrm_vcf

        vcf = _make_vcf(
            tmp_path,
            "gnomad.vcf",
            ["chrM\t73\t.\tA\tG\t.\tPASS\tAF=0.987;mean_hl=0.99;n_het=12345;n_hom_var=55000"],
        )
        df = parse_gnomad_chrm_vcf(vcf)
        assert len(df) == 1
        assert df.iloc[0]["pos"] == 73
        assert df.iloc[0]["ref"] == "A"
        assert df.iloc[0]["alt"] == "G"
        assert abs(df.iloc[0]["af"] - 0.987) < 1e-6
        assert df.iloc[0]["n_het"] == 12345
        assert df.iloc[0]["n_hom"] == 55000

    def test_skips_non_pass(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_gnomad_chrm_vcf

        vcf = _make_vcf(
            tmp_path,
            "gnomad.vcf",
            [
                "chrM\t73\t.\tA\tG\t.\tPASS\tAF=0.5",
                "chrM\t152\t.\tT\tC\t.\tFAIL\tAF=0.1",
            ],
        )
        df = parse_gnomad_chrm_vcf(vcf)
        assert len(df) == 1
        assert df.iloc[0]["pos"] == 73

    def test_skips_indels(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_gnomad_chrm_vcf

        vcf = _make_vcf(
            tmp_path,
            "gnomad.vcf",
            [
                "chrM\t73\t.\tA\tGG\t.\tPASS\tAF=0.5",
                "chrM\t100\t.\tAT\tA\t.\tPASS\tAF=0.2",
            ],
        )
        df = parse_gnomad_chrm_vcf(vcf)
        assert len(df) == 0

    def test_empty_vcf_returns_empty_dataframe(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_gnomad_chrm_vcf

        vcf = tmp_path / "empty.vcf"
        vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        df = parse_gnomad_chrm_vcf(vcf)
        assert len(df) == 0
        assert set(df.columns) == {"pos", "ref", "alt", "af", "het_level", "n_het", "n_hom"}

    def test_schema_columns(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_gnomad_chrm_vcf

        vcf = _make_vcf(tmp_path, "gnomad.vcf", ["chrM\t73\t.\tA\tG\t.\tPASS\tAF=0.9"])
        df = parse_gnomad_chrm_vcf(vcf)
        assert set(df.columns) == {"pos", "ref", "alt", "af", "het_level", "n_het", "n_hom"}


class TestClinvarParser:
    def test_extracts_pathogenic_label_1(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_clinvar_chrm_vcf

        vcf = _make_vcf(
            tmp_path,
            "clinvar.vcf",
            ["chrM\t3243\t.\tA\tG\t.\t.\tCLNSIG=Pathogenic"],
        )
        df = parse_clinvar_chrm_vcf(vcf)
        assert len(df) == 1
        assert df.iloc[0]["label"] == 1
        assert df.iloc[0]["pos"] == 3243

    def test_skips_vus_and_benign(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_clinvar_chrm_vcf

        vcf = _make_vcf(
            tmp_path,
            "clinvar.vcf",
            [
                "chrM\t3243\t.\tA\tG\t.\t.\tCLNSIG=Pathogenic",
                "chrM\t100\t.\tT\tC\t.\t.\tCLNSIG=Uncertain_significance",
                "chrM\t200\t.\tG\tA\t.\t.\tCLNSIG=Benign",
            ],
        )
        df = parse_clinvar_chrm_vcf(vcf)
        assert len(df) == 1
        assert df.iloc[0]["pos"] == 3243

    def test_add_benign_proxies(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import add_benign_proxies

        pathogenic = pd.DataFrame({"pos": [3243], "ref": ["A"], "alt": ["G"], "label": [1]})
        gnomad = pd.DataFrame(
            {
                "pos": [73, 3243, 263],
                "ref": ["A", "A", "A"],
                "alt": ["G", "G", "G"],
                "af": [0.987, 0.0001, 0.95],
                "het_level": [0.99, 0.5, 0.98],
                "n_het": [1000, 5, 2000],
                "n_hom": [50000, 1, 40000],
            }
        )
        result = add_benign_proxies(pathogenic, gnomad, af_threshold=0.01)
        # pathogenic 3243 stays, common gnomad 73 + 263 added as benign
        # gnomad 3243 has af=0.0001 < 0.01 threshold, excluded
        # gnomad 3243 (af=0.0001) excluded by threshold; gnomad 3243 with
        # af > threshold would be excluded by pathogenic overlap check
        labels = dict(zip(result["pos"], result["label"], strict=False))
        assert labels[3243] == 1
        assert labels[73] == 0
        assert labels[263] == 0

    def test_benign_proxies_do_not_duplicate_pathogenic(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import add_benign_proxies

        pathogenic = pd.DataFrame({"pos": [73], "ref": ["A"], "alt": ["G"], "label": [1]})
        gnomad = pd.DataFrame(
            {
                "pos": [73],
                "ref": ["A"],
                "alt": ["G"],
                "af": [0.987],
                "het_level": [0.99],
                "n_het": [1000],
                "n_hom": [50000],
            }
        )
        result = add_benign_proxies(pathogenic, gnomad)
        # 73 A>G is in pathogenic — it must not appear again with label=0
        assert len(result) == 1
        assert result.iloc[0]["label"] == 1

    def test_schema_columns(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_clinvar_chrm_vcf

        vcf = _make_vcf(
            tmp_path,
            "clinvar.vcf",
            ["chrM\t3243\t.\tA\tG\t.\t.\tCLNSIG=Pathogenic"],
        )
        df = parse_clinvar_chrm_vcf(vcf)
        assert set(df.columns) == {"pos", "ref", "alt", "label"}


class TestPhylotreeParser:
    def _make_csv(self, tmp_path: Path, rows: list[dict]) -> Path:
        df = pd.DataFrame(rows)
        path = tmp_path / "phylotree.csv"
        df.to_csv(path, index=False)
        return path

    def test_parses_standard_mutation_string(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_phylotree_csv

        csv = self._make_csv(
            tmp_path,
            [
                {"haplogroup": "A", "mutation": "73A>G"},
                {"haplogroup": "A", "mutation": "263A>G"},
                {"haplogroup": "H", "mutation": "3243A>G"},
            ],
        )
        df = parse_phylotree_csv(csv)
        assert len(df) == 3
        assert df.iloc[0]["pos"] == 73
        assert df.iloc[0]["ref"] == "A"
        assert df.iloc[0]["alt"] == "G"
        assert df.iloc[0]["haplogroup"] == "A"

    def test_strips_back_mutation_prefix(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_phylotree_csv

        csv = self._make_csv(
            tmp_path,
            [{"haplogroup": "B", "mutation": "!16519T>C"}],
        )
        df = parse_phylotree_csv(csv)
        assert len(df) == 1
        assert df.iloc[0]["pos"] == 16519
        assert df.iloc[0]["ref"] == "T"
        assert df.iloc[0]["alt"] == "C"

    def test_skips_non_snp_mutations(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_phylotree_csv

        csv = self._make_csv(
            tmp_path,
            [
                {"haplogroup": "A", "mutation": "73A>G"},  # valid SNP
                {"haplogroup": "A", "mutation": "315.1C"},  # insertion
                {"haplogroup": "A", "mutation": "np"},  # non-standard
            ],
        )
        df = parse_phylotree_csv(csv)
        assert len(df) == 1
        assert df.iloc[0]["pos"] == 73

    def test_wrong_columns_raises(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_phylotree_csv

        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("clade,var\nA,73A>G\n")
        with pytest.raises(ValueError, match="haplogroup.*mutation"):
            parse_phylotree_csv(bad_csv)

    def test_schema_columns(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import parse_phylotree_csv

        csv = self._make_csv(
            tmp_path,
            [{"haplogroup": "A", "mutation": "73A>G"}],
        )
        df = parse_phylotree_csv(csv)
        assert set(df.columns) == {"pos", "ref", "alt", "haplogroup"}


class TestBuildParquets:
    def _gnomad_vcf(self, tmp_path: Path) -> Path:
        return _make_vcf(
            tmp_path,
            "gnomad.vcf",
            [
                "chrM\t73\t.\tA\tG\t.\tPASS\tAF=0.987;mean_hl=0.99;n_het=12000;n_hom_var=55000",
                "chrM\t152\t.\tT\tC\t.\tPASS\tAF=0.02;mean_hl=0.99;n_het=500;n_hom_var=1000",
            ],
        )

    def _clinvar_vcf(self, tmp_path: Path) -> Path:
        return _make_vcf(
            tmp_path,
            "clinvar.vcf",
            ["chrM\t3243\t.\tA\tG\t.\t.\tCLNSIG=Pathogenic"],
        )

    def _phylotree_csv(self, tmp_path: Path) -> Path:
        df = pd.DataFrame(
            [
                {"haplogroup": "A", "mutation": "73A>G"},
                {"haplogroup": "H", "mutation": "263A>G"},
            ]
        )
        path = tmp_path / "phylotree.csv"
        df.to_csv(path, index=False)
        return path

    def test_build_gnomad_parquet_creates_file(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import GNOMAD_PARQUET, build_gnomad_parquet

        vcf = self._gnomad_vcf(tmp_path)
        out_dir = tmp_path / "processed"
        build_gnomad_parquet(vcf, out_dir)
        assert (out_dir / GNOMAD_PARQUET).exists()

    def test_build_gnomad_parquet_is_idempotent(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import GNOMAD_PARQUET, build_gnomad_parquet

        vcf = self._gnomad_vcf(tmp_path)
        out_dir = tmp_path / "processed"
        path1 = build_gnomad_parquet(vcf, out_dir)
        # second call should return immediately without re-parsing
        mtime1 = (out_dir / GNOMAD_PARQUET).stat().st_mtime
        build_gnomad_parquet(vcf, out_dir)
        mtime2 = (out_dir / GNOMAD_PARQUET).stat().st_mtime
        assert mtime1 == mtime2
        assert path1 == out_dir / GNOMAD_PARQUET

    def test_build_clinvar_parquet_creates_file(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import CLINVAR_PARQUET, build_clinvar_parquet

        vcf = self._clinvar_vcf(tmp_path)
        out_dir = tmp_path / "processed"
        build_clinvar_parquet(vcf, out_dir)
        assert (out_dir / CLINVAR_PARQUET).exists()

    def test_build_haplogroup_markers_parquet_creates_file(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import (
            HAPLOGROUP_PARQUET,
            build_haplogroup_markers_parquet,
        )

        csv = self._phylotree_csv(tmp_path)
        out_dir = tmp_path / "processed"
        build_haplogroup_markers_parquet(csv, out_dir)
        assert (out_dir / HAPLOGROUP_PARQUET).exists()

    def test_clinvar_parquet_with_gnomad_has_benign_rows(self, tmp_path: Path) -> None:
        from mtdna_fm.data.variant_processor import CLINVAR_PARQUET, build_clinvar_parquet

        vcf = self._clinvar_vcf(tmp_path)
        gnomad_vcf = self._gnomad_vcf(tmp_path)
        out_dir = tmp_path / "processed"

        # First build gnomad parquet so build_clinvar can read it
        from mtdna_fm.data.variant_processor import GNOMAD_PARQUET, build_gnomad_parquet

        build_gnomad_parquet(gnomad_vcf, out_dir)
        build_clinvar_parquet(vcf, out_dir, gnomad_parquet=out_dir / GNOMAD_PARQUET)

        result = pd.read_parquet(out_dir / CLINVAR_PARQUET)
        assert 1 in result["label"].values
        assert 0 in result["label"].values


# ---------------------------------------------------------------------------
# MtDNADataset tests (Day 6)
# ---------------------------------------------------------------------------


def _make_dataset(
    n_seqs: int = 2,
    genome_length: int = 100,
    window_size: int = 20,
    stride: int = 10,
    k: int = 3,
    with_het: bool = False,
):
    """Helper: build a tiny MtDNADataset for fast unit tests."""
    from mtdna_fm.data.dataset import MtDNADataset
    from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

    rng = np.random.default_rng(42)
    sequences = ["".join(rng.choice(list("ACGT"), size=genome_length)) for _ in range(n_seqs)]
    vocab = KmerVocabulary.build(k=k)

    het_vectors = None
    if with_het:
        het_vectors = [
            rng.uniform(0.0, 1.0, size=genome_length).astype(np.float32) for _ in range(n_seqs)
        ]

    return MtDNADataset(
        sequences=sequences,
        vocabulary=vocab,
        k=k,
        window_size=window_size,
        stride=stride,
        genome_length=genome_length,
        het_level_vectors=het_vectors,
    )


class TestMtDNADataset:
    def test_dataset_length(self) -> None:
        """Window count equals ceil(genome_length / stride) per sequence."""
        genome_length, stride, n_seqs = 100, 10, 2
        dataset = _make_dataset(n_seqs=n_seqs, genome_length=genome_length, stride=stride)
        expected = len(range(0, genome_length, stride)) * n_seqs
        assert len(dataset) == expected

    def test_all_positions_covered(self) -> None:
        """Every genomic position appears in at least one window's position_ids."""
        genome_length, stride = 100, 10
        dataset = _make_dataset(n_seqs=1, genome_length=genome_length, stride=stride)
        all_positions: set[int] = set()
        for i in range(len(dataset)):
            item = dataset[i]
            all_positions.update(item["position_ids"].tolist())
        assert all_positions == set(range(genome_length))

    def test_circular_junction_window(self) -> None:
        """At least one window spans the genome end/start boundary."""
        genome_length, window_size, stride = 100, 20, 10
        dataset = _make_dataset(
            n_seqs=1, genome_length=genome_length, window_size=window_size, stride=stride
        )
        # The window starting at token 90 wraps: positions 90-99 and 0-9
        junction_found = False
        for i in range(len(dataset)):
            item = dataset[i]
            positions = item["position_ids"].tolist()
            if max(positions) >= genome_length - 1 and min(positions) == 0:
                junction_found = True
                break
        assert junction_found, "No window spanning the circular genome junction was found"

    def test_het_values_range(self) -> None:
        """All returned het_values are in [0.0, 1.0]."""
        dataset = _make_dataset(n_seqs=2, genome_length=100, with_het=True)
        for i in range(len(dataset)):
            item = dataset[i]
            assert float(item["het_values"].min()) >= 0.0
            assert float(item["het_values"].max()) <= 1.0

    def test_output_tensor_shapes(self) -> None:
        """All returned tensors have the expected window_size length."""
        window_size = 20
        dataset = _make_dataset(n_seqs=1, genome_length=100, window_size=window_size)
        item = dataset[0]
        for key in ("input_ids", "attention_mask", "position_ids", "het_values"):
            assert item[key].shape == (window_size,), f"{key} shape mismatch"

    def test_labels_propagated(self) -> None:
        """Labels from the sequence level are inherited by every window."""
        from mtdna_fm.data.dataset import MtDNADataset
        from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

        rng = np.random.default_rng(0)
        genome_length, stride = 100, 10
        seq = "".join(rng.choice(list("ACGT"), size=genome_length))
        vocab = KmerVocabulary.build(k=3)
        dataset = MtDNADataset(
            sequences=[seq],
            vocabulary=vocab,
            k=3,
            window_size=20,
            stride=stride,
            genome_length=genome_length,
            labels=[7],
        )
        for i in range(len(dataset)):
            assert int(dataset[i]["labels"]) == 7

    def test_from_dataframe(self) -> None:
        """from_dataframe constructs a dataset with the correct length."""
        from mtdna_fm.data.dataset import MtDNADataset
        from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

        rng = np.random.default_rng(0)
        genome_length, stride = 100, 10
        n_seqs = 3
        seqs = ["".join(rng.choice(list("ACGT"), size=genome_length)) for _ in range(n_seqs)]
        df = pd.DataFrame({"sequence": seqs, "het_level_vector": [None] * n_seqs})
        vocab = KmerVocabulary.build(k=3)
        dataset = MtDNADataset.from_dataframe(
            df, vocab, k=3, window_size=20, stride=stride, genome_length=genome_length
        )
        expected = len(range(0, genome_length, stride)) * n_seqs
        assert len(dataset) == expected


# ---------------------------------------------------------------------------
# VariantDataset tests (Day 6)
# ---------------------------------------------------------------------------


def _make_variant_df(positions: list[int], label: int = 1) -> pd.DataFrame:
    """Helper: make a simple variant DataFrame with dummy SNPs."""
    return pd.DataFrame(
        {
            "pos": positions,  # 1-based VCF positions
            "ref": ["A"] * len(positions),
            "alt": ["G"] * len(positions),
            "label": [label] * len(positions),
        }
    )


class TestVariantDataset:
    def _reference(self, genome_length: int = 100) -> str:
        rng = np.random.default_rng(99)
        return "".join(rng.choice(list("ACGT"), size=genome_length))

    def test_length_equals_snp_count(self) -> None:
        from mtdna_fm.data.variant_dataset import VariantDataset
        from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

        ref = self._reference(100)
        vocab = KmerVocabulary.build(k=3)
        df = _make_variant_df(positions=[10, 50, 80])
        dataset = VariantDataset(ref, df, vocab, k=3, window_size=20, genome_length=100)
        assert len(dataset) == 3

    def test_indels_excluded(self) -> None:
        from mtdna_fm.data.variant_dataset import VariantDataset
        from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

        ref = self._reference(100)
        vocab = KmerVocabulary.build(k=3)
        df = pd.DataFrame(
            {
                "pos": [10, 20, 30],
                "ref": ["A", "AT", "A"],  # middle row is indel
                "alt": ["G", "G", "GC"],  # last row is indel
                "label": [1, 0, 0],
            }
        )
        dataset = VariantDataset(ref, df, vocab, k=3, window_size=20, genome_length=100)
        assert len(dataset) == 1  # only pos=10 A>G is a valid SNP

    def test_output_tensor_shapes(self) -> None:
        from mtdna_fm.data.variant_dataset import VariantDataset
        from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

        ref = self._reference(100)
        vocab = KmerVocabulary.build(k=3)
        df = _make_variant_df(positions=[50])
        window_size = 20
        dataset = VariantDataset(ref, df, vocab, k=3, window_size=window_size, genome_length=100)
        item = dataset[0]
        for key in ("input_ids", "attention_mask", "position_ids", "het_values"):
            assert item[key].shape == (window_size,), f"{key} shape mismatch"

    def test_label_returned(self) -> None:
        from mtdna_fm.data.variant_dataset import VariantDataset
        from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

        ref = self._reference(100)
        vocab = KmerVocabulary.build(k=3)
        df = pd.DataFrame({"pos": [10, 50], "ref": ["A", "A"], "alt": ["G", "G"], "label": [1, 0]})
        dataset = VariantDataset(ref, df, vocab, k=3, window_size=20, genome_length=100)
        assert int(dataset[0]["label"]) == 1
        assert int(dataset[1]["label"]) == 0

    def test_snp_applied_to_reference(self) -> None:
        """The mutated token at the variant position differs from the reference."""
        from mtdna_fm.data.variant_dataset import VariantDataset
        from mtdna_fm.tokenizer.tokenize import tokenize_sequence
        from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

        genome_length = 100
        ref = "A" * genome_length  # all-A reference
        vocab = KmerVocabulary.build(k=3)
        # Variant at VCF pos=50 (0-based 49): A>G
        df = pd.DataFrame({"pos": [50], "ref": ["A"], "alt": ["G"], "label": [1]})
        dataset = VariantDataset(ref, df, vocab, k=3, window_size=20, genome_length=genome_length)
        item = dataset[0]

        # The window should contain a token encoding a G-containing k-mer at position 49
        ref_tokens = tokenize_sequence(
            ref, vocab, k=3, stride=1, max_seq_len=genome_length, circular=True
        )
        variant_token_in_ref = ref_tokens["input_ids"][49]
        variant_token_in_window = item["input_ids"][item["variant_offset"].item()].item()
        # The k-mer at position 49 of the mutant differs from the reference k-mer
        assert variant_token_in_window != variant_token_in_ref
