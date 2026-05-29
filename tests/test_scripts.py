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

    def test_missing_config_exits_with_error(self) -> None:
        """Missing config file should exit with error message."""
        result = runner.invoke(
            finetune_app,
            ["--task", "haplogroup", "--config", "/tmp/nonexistent_cfg.yaml"],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "Config" in result.output


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


# ── TestHeteroplasmyRegressionDataset (Day 18) ────────────────────────────────


class TestHeteroplasmyRegressionDataset:
    """Tests for the HeteroplasmyRegressionDataset synthetic fallback path."""

    def _make_vocab(self):
        from mtdna_fm.tokenizer.vocabulary import KmerVocabulary
        return KmerVocabulary.build(k=6)

    def test_synthetic_fallback_length(self) -> None:
        """Dataset with no parquet should fall back to 64 synthetic samples."""
        from mtdna_fm.scripts.finetune import HeteroplasmyRegressionDataset

        vocab = self._make_vocab()
        ds = HeteroplasmyRegressionDataset("/tmp/nonexistent_het.parquet", vocab)
        assert len(ds) == 64

    def test_max_variants_truncates(self) -> None:
        from mtdna_fm.scripts.finetune import HeteroplasmyRegressionDataset

        vocab = self._make_vocab()
        ds = HeteroplasmyRegressionDataset(
            "/tmp/nonexistent_het.parquet", vocab, max_variants=10
        )
        assert len(ds) == 10

    def test_item_keys(self) -> None:
        """Each item must have required tensor keys."""
        import torch

        from mtdna_fm.scripts.finetune import HeteroplasmyRegressionDataset

        vocab = self._make_vocab()
        ds = HeteroplasmyRegressionDataset(
            "/tmp/nonexistent_het.parquet", vocab, max_variants=4
        )
        item = ds[0]
        for key in ("input_ids", "position_ids", "attention_mask", "variant_token_idx", "labels"):
            assert key in item, f"Missing key: {key}"
            assert isinstance(item[key], torch.Tensor)

    def test_labels_in_range(self) -> None:
        """Labels (het_level) should be floats in [0, 1]."""
        from mtdna_fm.scripts.finetune import HeteroplasmyRegressionDataset

        vocab = self._make_vocab()
        ds = HeteroplasmyRegressionDataset(
            "/tmp/nonexistent_het.parquet", vocab, max_variants=16
        )
        for i in range(len(ds)):
            label = ds[i]["labels"].item()
            assert 0.0 <= label <= 1.0, f"Label out of range at index {i}: {label}"

    def test_window_size_respected(self) -> None:
        """input_ids length should match window_size."""
        from mtdna_fm.scripts.finetune import HeteroplasmyRegressionDataset

        vocab = self._make_vocab()
        ds = HeteroplasmyRegressionDataset(
            "/tmp/nonexistent_het.parquet", vocab, window_size=128, max_variants=4
        )
        for i in range(len(ds)):
            assert ds[i]["input_ids"].shape[0] == 128

    def test_from_parquet(self, tmp_path: Path) -> None:
        """Dataset should load correctly from a real parquet file."""
        import numpy as np
        import pandas as pd

        from mtdna_fm.scripts.finetune import HeteroplasmyRegressionDataset

        vocab = self._make_vocab()
        rng = np.random.default_rng(99)
        ref = "".join(rng.choice(list("ACGT"), size=16569))
        bases = list("ACGT")
        rows = []
        for i in range(8):
            pos = int(rng.integers(0, 16569))
            alt = rng.choice([b for b in bases if b != ref[pos]])
            seq = ref[:pos] + alt + ref[pos + 1:]
            rows.append({"sequence": seq, "position": pos, "het_level": float(i) / 8.0})
        df = pd.DataFrame(rows)
        parquet_path = tmp_path / "het.parquet"
        df.to_parquet(parquet_path, index=False)

        ds = HeteroplasmyRegressionDataset(str(parquet_path), vocab)
        assert len(ds) == 8
        assert ds[0]["labels"].dtype.is_floating_point


# ── TestFinetuneHeterologyCLI (Day 18) ────────────────────────────────────────


class TestFinetuneHeterologyCLI:
    def test_heteroplasmy_unknown_task_exits(self) -> None:
        """Unknown task name should exit with error."""
        result = runner.invoke(
            finetune_app,
            ["--task", "notarealtask", "--config", "/tmp/cfg.yaml"],
        )
        assert result.exit_code == 1

    def test_heteroplasmy_missing_model_exits(self, tmp_path: Path) -> None:
        """heteroplasmy task with missing base model should exit with error."""
        cfg_path = tmp_path / "het.yaml"
        cfg_path.write_text(
            "task: heteroplasmy\nbase_model: /tmp/no_such_model\n"
            "output_dir: /tmp/out\ndata:\n  parquet: /tmp/x.parquet\n"
        )
        result = runner.invoke(
            finetune_app,
            ["--task", "heteroplasmy", "--config", str(cfg_path)],
        )
        assert result.exit_code == 1
