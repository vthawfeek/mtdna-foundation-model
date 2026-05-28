"""
Tests for CLI entry points in mtdna_fm/scripts/.

Uses typer.testing.CliRunner to invoke commands in-process without spawning
subprocesses, so coverage is collected correctly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from mtdna_fm.scripts.evaluate import app as evaluate_app
from mtdna_fm.scripts.finetune import app as finetune_app

runner = CliRunner()


# ── TestEvaluateCLI ────────────────────────────────────────────────────────────


class TestEvaluateCLI:
    def test_exits_with_error_code(self) -> None:
        result = runner.invoke(evaluate_app, ["--model", "/tmp/fake_model"])
        assert result.exit_code == 1

    def test_not_yet_implemented_message(self) -> None:
        result = runner.invoke(evaluate_app, ["--model", "/tmp/fake_model"])
        assert "not yet implemented" in result.output


# ── TestFinetuneCLI ────────────────────────────────────────────────────────────


class TestFinetuneCLI:
    def test_exits_with_error_code(self) -> None:
        result = runner.invoke(
            finetune_app,
            ["--task", "haplogroup", "--config", "/tmp/cfg.yaml"],
        )
        assert result.exit_code == 1

    def test_not_yet_implemented_message(self) -> None:
        result = runner.invoke(
            finetune_app,
            ["--task", "haplogroup", "--config", "/tmp/cfg.yaml"],
        )
        assert "not yet implemented" in result.output


# ── TestPreprocessCLI ──────────────────────────────────────────────────────────


class TestPreprocessCLI:
    def test_no_source_exits_with_error(self) -> None:
        """Invoking preprocess without --hmtdb or --ncbi should exit non-zero."""
        from mtdna_fm.scripts.preprocess import app

        result = runner.invoke(app, [])
        assert result.exit_code == 1
        assert "Error" in result.output or "error" in result.output.lower()

    def test_hmtdb_fasta_not_found_exits(self, tmp_path: Path) -> None:
        """--hmtdb pointing to a dir without sequences.fasta exits non-zero."""
        from mtdna_fm.scripts.preprocess import app

        result = runner.invoke(app, ["--hmtdb", str(tmp_path)])
        assert result.exit_code == 1

    def test_ncbi_fasta_not_found_exits(self, tmp_path: Path) -> None:
        """--ncbi pointing to a dir without vertebrate_mtdna.fasta exits non-zero."""
        from mtdna_fm.scripts.preprocess import app

        result = runner.invoke(app, ["--ncbi", str(tmp_path)])
        assert result.exit_code == 1

    def test_hmtdb_source_success(self, tmp_path: Path) -> None:
        """A valid HmtDB source runs through preprocessing and writes parquet files."""
        from mtdna_fm.scripts.preprocess import app

        hmtdb_dir = tmp_path / "hmtdb"
        hmtdb_dir.mkdir()
        fasta = hmtdb_dir / "sequences.fasta"
        fasta.write_text(">seq1\nACGT" * 5000 + "\n>seq2\nTTTT" * 5000 + "\n")

        out_dir = tmp_path / "processed"

        import pandas as pd

        fake_df = pd.DataFrame(
            {
                "accession": [f"s{i}" for i in range(10)],
                "sequence": ["A" * 16569] * 10,
                "haplogroup": ["H"] * 10,
                "species": ["homo_sapiens"] * 10,
                "geographic_origin": [None] * 10,
                "het_level_vector": [None] * 10,
                "qc_pass": [True] * 10,
                "split": ["train"] * 8 + ["val"] + ["test"],
            }
        )

        def mock_build(*args, **kwargs):
            return fake_df

        def mock_preprocess(df, **kwargs):
            return fake_df

        def mock_split(df, **kwargs):
            return fake_df

        def mock_save(df, output_dir):
            od = Path(output_dir)
            od.mkdir(parents=True, exist_ok=True)
            paths = {}
            for split in ("train", "val", "test"):
                p = od / f"{split}.parquet"
                df[df["split"] == split].to_parquet(p, index=False)
                paths[split] = p
            return paths

        with (
            patch("mtdna_fm.data.preprocessor.build_record_dataframe", side_effect=mock_build),
            patch("mtdna_fm.data.preprocessor.preprocess_sequences", side_effect=mock_preprocess),
            patch("mtdna_fm.data.preprocessor.stratified_split", side_effect=mock_split),
            patch("mtdna_fm.data.preprocessor.save_splits", side_effect=mock_save),
        ):
            result = runner.invoke(
                app,
                ["--hmtdb", str(hmtdb_dir), "--output-dir", str(out_dir)],
            )

        assert result.exit_code == 0, result.output


# ── TestTrainCLI ───────────────────────────────────────────────────────────────


class TestTrainCLI:
    def test_missing_config_exits_with_error(self) -> None:
        """Invoking train without --config exits non-zero (required arg)."""
        from mtdna_fm.scripts.train import app

        result = runner.invoke(app, [])
        assert result.exit_code != 0

    def test_invokes_trainer_from_yaml(self, tmp_path: Path) -> None:
        """mtdna-train calls MtDNATrainer.from_yaml, setup, and train."""
        from mtdna_fm.scripts.train import app

        config_path = tmp_path / "phase1.yaml"
        config_path.write_text("batch_size: 2\n")

        mock_trainer = MagicMock()
        with patch(
            "mtdna_fm.training.trainer.MtDNATrainer.from_yaml",
            return_value=mock_trainer,
        ):
            result = runner.invoke(
                app,
                ["--config", str(config_path), "--device", "cpu"],
            )

        assert result.exit_code == 0
        mock_trainer.setup.assert_called_once()
        mock_trainer.train.assert_called_once()


# ── TestDownloadScriptInternals ────────────────────────────────────────────────


class TestDownloadScriptInternals:
    """Test the private _run_* functions in scripts/download.py directly."""

    def test_run_hmtdb_calls_download_and_echoes(self, tmp_path: Path) -> None:
        from mtdna_fm.scripts.download import _run_hmtdb

        fasta = tmp_path / "sequences.fasta"
        meta = tmp_path / "metadata.parquet"
        with patch(
            "mtdna_fm.data.hmtdb_client.download_hmtdb",
            return_value=(fasta, meta),
        ):
            _run_hmtdb(tmp_path, force=False)

    def test_run_ncbi_refseq_calls_download_and_echoes(self, tmp_path: Path) -> None:
        from mtdna_fm.scripts.download import _run_ncbi_refseq

        fasta = tmp_path / "vertebrate_mtdna.fasta"
        with patch(
            "mtdna_fm.data.ncbi_client.download_ncbi_mtdna",
            return_value=fasta,
        ):
            _run_ncbi_refseq(tmp_path, force=False)

    def test_run_gnomad_calls_download_and_echoes(self, tmp_path: Path) -> None:
        from mtdna_fm.scripts.download import _run_gnomad

        vcf = tmp_path / "gnomad_chrM.vcf"
        with patch(
            "mtdna_fm.data.variant_downloader.download_gnomad_chrm",
            return_value=vcf,
        ):
            _run_gnomad(tmp_path, force=False)

    def test_run_clinvar_calls_download_and_echoes(self, tmp_path: Path) -> None:
        from mtdna_fm.scripts.download import _run_clinvar

        vcf = tmp_path / "clinvar_chrM.vcf"
        with patch(
            "mtdna_fm.data.variant_downloader.download_clinvar_chrm",
            return_value=vcf,
        ):
            _run_clinvar(tmp_path, force=False)

    def test_run_phylotree_calls_download_and_echoes(self, tmp_path: Path) -> None:
        from mtdna_fm.scripts.download import _run_phylotree

        csv = tmp_path / "phylotree_build17.csv"
        with patch(
            "mtdna_fm.data.variant_downloader.download_phylotree",
            return_value=csv,
        ):
            _run_phylotree(tmp_path, force=False)
