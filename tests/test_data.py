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
