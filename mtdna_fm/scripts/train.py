"""
mtdna-train CLI: launch Phase 1 or Phase 2 pre-training.

Usage:
    # Phase 1 (cold start):
    uv run mtdna-train \\
        --config configs/pretraining_phase1.yaml \\
        --model-config configs/model_small.yaml

    # Phase 2 (load Phase 1 encoder, fresh optimizer):
    uv run mtdna-train \\
        --config configs/pretraining_phase2.yaml
        # model config is auto-loaded from resume_from checkpoint

Monitor training:
    mlflow ui --backend-store-uri mlruns

Expected loss curve (Phase 1, 50k steps):
    step    0: ~8.3 (log 4,102 — random baseline)
    step 5000: ~5.5–6.0
    step 20000: ~3.5–4.0
    step 50000: ~2.5–3.0
"""

import logging

import typer

app = typer.Typer(help="Pre-train mtDNA-FM (Phase 1 cross-species or Phase 2 human-specific).")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


@app.command()
def main(
    config: str = typer.Option(..., "--config", help="Path to training config YAML"),
    model_config: str = typer.Option(
        None,
        "--model-config",
        help="Path to model config YAML (required for Phase 1 cold start)",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="PyTorch device: cpu | cuda | mps",
    ),
) -> None:
    """Launch mtDNA-FM pre-training from YAML config files."""
    from mtdna_fm.training.trainer import MtDNATrainer

    trainer = MtDNATrainer.from_yaml(
        config_path=config,
        model_config_path=model_config,
        device=device,
    )
    trainer.setup()
    trainer.train()
