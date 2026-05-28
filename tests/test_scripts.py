"""
Tests for CLI entry points in mtdna_fm/scripts/.

Uses typer.testing.CliRunner to invoke commands in-process without spawning
subprocesses, so coverage is collected correctly.
"""

from __future__ import annotations

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
