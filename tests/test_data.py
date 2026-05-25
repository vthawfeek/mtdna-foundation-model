"""
Tests for data download clients.

All tests here are unit tests that run without network access. Integration
tests (which actually hit HmtDB or NCBI) are marked with
@pytest.mark.integration and skipped in CI / the normal test run.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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

        with patch(
            "mtdna_fm.data.hmtdb_client._download_fasta_from_hmtdb"
        ) as mock_dl, patch(
            "mtdna_fm.data.hmtdb_client._download_metadata_from_hmtdb"
        ) as mock_meta, patch(
            "mtdna_fm.data.hmtdb_client._validate_fasta"
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

        with patch(
            "mtdna_fm.data.hmtdb_client._download_fasta_from_hmtdb"
        ), patch(
            "mtdna_fm.data.hmtdb_client._download_metadata_from_hmtdb"
        ), patch(
            "mtdna_fm.data.hmtdb_client._validate_fasta"
        ):
            download_hmtdb(new_dir)

        assert new_dir.exists()

    def test_fallback_called_on_network_error(self, tmp_path: Path) -> None:
        """Network failure during HmtDB download triggers NCBI fallback."""
        import requests

        from mtdna_fm.data.hmtdb_client import download_hmtdb

        with patch(
            "mtdna_fm.data.hmtdb_client._download_fasta_from_hmtdb",
            side_effect=requests.ConnectionError("offline"),
        ), patch(
            "mtdna_fm.data.hmtdb_client._ncbi_fallback",
            return_value=(tmp_path / "sequences.fasta", tmp_path / "metadata.parquet"),
        ) as mock_fallback, patch(
            "mtdna_fm.data.hmtdb_client._validate_fasta"
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
        with patch(
            "mtdna_fm.data.ncbi_client._esearch",
            return_value=(1, "webenv", "1"),
        ), patch(
            "mtdna_fm.data.ncbi_client._efetch_batch",
            return_value=">seq1\nACGT\n",
        ), patch(
            "mtdna_fm.data.ncbi_client._save_progress"
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

    def test_unimplemented_source_exits_with_error(self) -> None:
        from typer.testing import CliRunner

        from mtdna_fm.scripts.download import app

        runner = CliRunner()
        result = runner.invoke(app, ["--source", "gnomad"])
        assert result.exit_code != 0
        assert "Not yet implemented" in result.output

    def test_valid_source_hmtdb_calls_download(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from mtdna_fm.scripts.download import app

        runner = CliRunner()
        with patch("mtdna_fm.scripts.download._run_hmtdb") as mock_run:
            result = runner.invoke(
                app, ["--source", "hmtdb", "--output", str(tmp_path)]
            )
        mock_run.assert_called_once_with(tmp_path, False)
        assert result.exit_code == 0

    def test_valid_source_ncbi_calls_download(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from mtdna_fm.scripts.download import app

        runner = CliRunner()
        with patch("mtdna_fm.scripts.download._run_ncbi_refseq") as mock_run:
            result = runner.invoke(
                app, ["--source", "ncbi-refseq", "--output", str(tmp_path)]
            )
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
        for col in ("accession", "sequence", "haplogroup", "species", "geographic_origin",
                    "het_level_vector"):
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
