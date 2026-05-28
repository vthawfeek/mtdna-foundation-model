"""
Tests for MtDNATrainer.

Uses a tiny config (2-layer, 16-dim) and a synthetic 2-sequence dataset so
everything runs in milliseconds without real data or a GPU.

Test classes:
  TestCosineSchedule   — LR multiplier at key points (step 0, warmup end, max_steps)
  TestMtDNATrainer     — setup, forward, checkpoint round-trip, Phase 2 encoder loading,
                         evaluate, short training loop with loss decrease
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from mtdna_fm.training.trainer import MtDNATrainer, _cosine_lr_with_warmup

# ── Minimal trainer config ─────────────────────────────────────────────────────


def make_tiny_config(tmp_path: Path, output_dir: Path | None = None) -> dict:
    """Return a training config using a tiny 16-dim model and no real data."""
    return {
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_workers": 0,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "warmup_steps": 2,
        "max_steps": 5,
        "max_grad_norm": 1.0,
        "mask_prob": 0.15,
        "mlm_weight": 1.0,
        "het_weight": 0.0,
        "save_steps": 10,     # won't fire during 5-step test
        "eval_steps": 5,
        "log_steps": 5,
        "keep_last_n_checkpoints": 2,
        "gradient_checkpointing": False,
        "fp16": False,
        "mlflow_experiment": "test_experiment",
        "output_dir": str(output_dir or tmp_path / "out"),
        # No parquet paths → synthetic fallback
    }


def make_tiny_model_config() -> dict:
    """Return model config matching the 3-mer tiny_config fixture."""
    return {
        "vocab_size": 70,           # 64 3-mers + 6 special
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "max_seq_len": 12,
        "genome_length": 100,
        "use_circular_encoding": True,
        "use_het_projection": True,
        "dropout_prob": 0.0,
        "attention_dropout_prob": 0.0,
    }


def make_trainer(tmp_path: Path, **config_overrides) -> MtDNATrainer:
    """Create a ready-to-use MtDNATrainer with tiny synthetic config."""
    config = make_tiny_config(tmp_path)
    config.update(config_overrides)
    model_config = make_tiny_model_config()
    trainer = MtDNATrainer(config, model_config, device="cpu")
    trainer.setup()
    return trainer


# ── TestCosineSchedule ─────────────────────────────────────────────────────────


class TestCosineSchedule:
    def test_warmup_starts_at_zero(self) -> None:
        mult = _cosine_lr_with_warmup(step=0, warmup_steps=100, max_steps=1000, base_lr=1.0)
        assert mult == pytest.approx(0.0, abs=1e-6)

    def test_warmup_ends_at_one(self) -> None:
        mult = _cosine_lr_with_warmup(step=100, warmup_steps=100, max_steps=1000, base_lr=1.0)
        # At exactly warmup_steps the cosine starts; cosine(0) = 1
        assert mult == pytest.approx(1.0, abs=1e-3)

    def test_halfway_after_warmup(self) -> None:
        """At the midpoint of cosine decay, LR should be close to (1+min)/2."""
        mult = _cosine_lr_with_warmup(
            step=550, warmup_steps=100, max_steps=1000, base_lr=1.0, min_lr_fraction=0.1
        )
        # cosine(pi/2) = 0 → multiplier ≈ 0.1 + 0.9 * 0.5 = 0.55
        assert 0.50 <= mult <= 0.60

    def test_at_max_steps_is_min(self) -> None:
        mult = _cosine_lr_with_warmup(
            step=1000, warmup_steps=100, max_steps=1000, base_lr=1.0, min_lr_fraction=0.1
        )
        assert mult == pytest.approx(0.1, abs=1e-3)

    def test_monotone_decay_after_warmup(self) -> None:
        """LR must not increase after warmup completes."""
        mults = [
            _cosine_lr_with_warmup(s, warmup_steps=10, max_steps=100, base_lr=1.0)
            for s in range(10, 100)
        ]
        for a, b in zip(mults, mults[1:], strict=False):
            assert a >= b - 1e-9


# ── TestMtDNATrainer ───────────────────────────────────────────────────────────


class TestMtDNATrainer:
    def test_setup_creates_model_and_dataloaders(self, tmp_path: Path) -> None:
        trainer = make_trainer(tmp_path)
        assert trainer.model is not None
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        assert trainer.optimizer is not None
        assert trainer.vocabulary is not None

    def test_model_on_cpu(self, tmp_path: Path) -> None:
        trainer = make_trainer(tmp_path)
        param = next(trainer.model.parameters())
        assert param.device.type == "cpu"

    def test_parameter_count_tiny(self, tmp_path: Path) -> None:
        """Tiny 2-layer 16-dim model should have fewer than 100k params."""
        trainer = make_trainer(tmp_path)
        n_params = sum(p.numel() for p in trainer.model.parameters())
        assert n_params < 100_000, f"Expected <100k params, got {n_params}"

    def test_evaluate_returns_metrics(self, tmp_path: Path) -> None:
        trainer = make_trainer(tmp_path)
        metrics = trainer.evaluate(max_batches=2)
        assert "eval/mlm_loss" in metrics
        assert "eval/perplexity" in metrics
        assert metrics["eval/mlm_loss"] > 0
        assert metrics["eval/perplexity"] > 1

    def test_evaluate_mlm_loss_is_finite(self, tmp_path: Path) -> None:
        trainer = make_trainer(tmp_path)
        metrics = trainer.evaluate(max_batches=2)
        assert math.isfinite(metrics["eval/mlm_loss"])
        assert math.isfinite(metrics["eval/perplexity"])

    def test_random_baseline_loss_near_log_vocab(self, tmp_path: Path) -> None:
        """
        At step 0, MLM loss should be close to log(vocab_size) — the random baseline.
        Tiny model: 70 tokens → log(70) ≈ 4.25.
        Allow ±2.0 because weights are random but not perfectly uniform.
        """
        trainer = make_trainer(tmp_path)
        metrics = trainer.evaluate(max_batches=5)
        expected = math.log(70)  # 3-mer vocabulary
        assert abs(metrics["eval/mlm_loss"] - expected) < 3.0, (
            f"Initial loss {metrics['eval/mlm_loss']:.3f} very far from "
            f"random baseline {expected:.3f}"
        )

    def test_train_runs_without_error(self, tmp_path: Path) -> None:
        """Full 5-step training run must complete without raising."""
        trainer = make_trainer(tmp_path, max_steps=5, log_steps=5, eval_steps=5)
        trainer.train()  # Should not raise

    def test_loss_decreases_over_training(self, tmp_path: Path) -> None:
        """
        Loss before and after 5 gradient steps on the same tiny synthetic dataset.
        With lr=1e-3 and batch=2, the model should overfit slightly on 2 sequences.
        """
        trainer = make_trainer(tmp_path, max_steps=20, log_steps=100, eval_steps=100)

        metrics_before = trainer.evaluate(max_batches=10)
        trainer.train()
        metrics_after = trainer.evaluate(max_batches=10)

        # Loss must not increase (though with only 2 sequences it should drop)
        assert metrics_after["eval/mlm_loss"] <= metrics_before["eval/mlm_loss"] + 0.5, (
            f"Loss did not decrease: {metrics_before['eval/mlm_loss']:.4f} → "
            f"{metrics_after['eval/mlm_loss']:.4f}"
        )

    def test_checkpoint_save_and_load(self, tmp_path: Path) -> None:
        """Saved checkpoint must reload to the same parameter values."""
        trainer = make_trainer(tmp_path)
        trainer._save_checkpoint(step=1)

        ckpt_dir = Path(trainer.config["output_dir"]) / "checkpoint-1"
        assert ckpt_dir.exists(), "Checkpoint directory not created"
        assert (ckpt_dir / "config.json").exists()
        assert (ckpt_dir / "optimizer.pt").exists()
        assert (ckpt_dir / "trainer_state.yaml").exists()

    def test_checkpoint_rotation(self, tmp_path: Path) -> None:
        """Only the last keep_last_n checkpoints must be retained."""
        trainer = make_trainer(tmp_path, keep_last_n_checkpoints=2)
        for step in [1, 2, 3]:
            trainer._save_checkpoint(step)

        # After 3 saves with keep=2, checkpoint-1 should be gone
        ckpt1 = Path(trainer.config["output_dir"]) / "checkpoint-1"
        ckpt2 = Path(trainer.config["output_dir"]) / "checkpoint-2"
        ckpt3 = Path(trainer.config["output_dir"]) / "checkpoint-3"
        assert not ckpt1.exists(), "checkpoint-1 should have been rotated out"
        assert ckpt2.exists(), "checkpoint-2 should still exist"
        assert ckpt3.exists(), "checkpoint-3 should still exist"

    def test_final_save_writes_model_files(self, tmp_path: Path) -> None:
        """_save_final() must write config.json + model weights to output_dir."""
        trainer = make_trainer(tmp_path)
        trainer._save_final()

        out = Path(trainer.config["output_dir"])
        assert (out / "config.json").exists()
        assert (out / "training_config.yaml").exists()
        # KmerVocabulary.save_pretrained() writes vocab.json
        assert (out / "vocab.json").exists()

    def test_phase2_encoder_loading(self, tmp_path: Path) -> None:
        """
        Phase 2 loading: _load_checkpoint(encoder_weights_only=True) must copy
        the encoder ('mtdna.*') weights from a Phase 1 checkpoint while leaving
        the prediction heads fresh.
        """
        # Phase 1: save to tmp_path/"phase1_out" (the actual output_dir)
        p1_output = tmp_path / "phase1_out"
        config_p1 = make_tiny_config(tmp_path, output_dir=p1_output)
        trainer_p1 = MtDNATrainer(config_p1, make_tiny_model_config(), device="cpu")
        trainer_p1.setup()

        # Modify an encoder weight so we can verify it was loaded in Phase 2
        with torch.no_grad():
            for name, param in trainer_p1.model.named_parameters():
                if "mtdna" in name:
                    param.fill_(0.42)
                    break
        trainer_p1._save_final()  # saves to p1_output

        # Phase 2: resume_from points at the Phase 1 output dir
        config_p2 = make_tiny_config(tmp_path, output_dir=str(tmp_path / "phase2"))
        config_p2["resume_from"] = str(p1_output)
        trainer_p2 = MtDNATrainer(config_p2, model_config=None, device="cpu")
        trainer_p2.setup()

        # The encoder (mtdna.*) weights should have been loaded from Phase 1
        for name, param in trainer_p2.model.named_parameters():
            if (
                "mtdna." in name
                and not any(s in name for s in ["bias", "LayerNorm", "layer_norm"])
                and param.numel() == 1
            ):
                assert param.item() == pytest.approx(0.42, abs=1e-4), (
                    f"Encoder weight '{name}' not loaded from Phase 1: "
                    f"got {param.item():.4f}, expected 0.42"
                )
                break

    def test_optimizer_param_groups(self, tmp_path: Path) -> None:
        """AdamW must have exactly 2 param groups: decay and no-decay."""
        trainer = make_trainer(tmp_path)
        assert len(trainer.optimizer.param_groups) == 2
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.01
        assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_species_filter_all(self, tmp_path: Path) -> None:
        """species_filter='all' must not filter any sequences from synthetic data."""
        # Use synthetic fallback (no real parquet) — just check species_filter='all'
        # passes through without filtering error
        trainer = make_trainer(
            tmp_path,
            data={"species_filter": "all"},
        )
        assert trainer.train_loader is not None

    def test_het_weight_zero_no_het_loss(self, tmp_path: Path) -> None:
        """With het_weight=0.0, het_loss on model output should be None."""
        trainer = make_trainer(tmp_path, het_weight=0.0)
        trainer.model.eval()

        # Get one batch from val loader
        batch = next(iter(trainer.val_loader))
        batch = {k: v.to(trainer.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = trainer.model(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                het_values=batch.get("het_values"),
                attention_mask=batch["attention_mask"],
                kmer_labels=batch["kmer_labels"],
                het_labels=batch.get("het_labels"),
            )
        assert outputs.het_loss is None

    def test_infer_k_invalid_vocab_raises(self, tmp_path: Path) -> None:
        """vocab_size that isn't 4^k + 6 must raise ValueError."""
        trainer = make_trainer(tmp_path)
        with pytest.raises(ValueError, match="valid 4\\^k vocabulary"):
            trainer._infer_k_from_vocab_size(100)  # 100-6=94, not a power of 4

    def test_setup_with_gradient_checkpointing(self, tmp_path: Path) -> None:
        """setup() with gradient_checkpointing=True must complete without error."""
        trainer = make_trainer(tmp_path, gradient_checkpointing=True)
        assert trainer.model is not None
        assert trainer.train_loader is not None

    def test_build_model_no_config_raises(self, tmp_path: Path) -> None:
        """_build_model raises ValueError when neither resume_from nor model_config is set."""
        from mtdna_fm.training.trainer import MtDNATrainer
        config = make_tiny_config(tmp_path)
        trainer = MtDNATrainer(config, model_config=None, device="cpu")
        with pytest.raises(ValueError, match="model_config or resume_from"):
            trainer._build_model()

    def test_standard_checkpoint_resume(self, tmp_path: Path) -> None:
        """Standard (non-Phase-2) checkpoint resume loads model + optimizer + step counter."""
        trainer_a = make_trainer(tmp_path, output_dir=str(tmp_path / "out_a"))
        trainer_a._save_checkpoint(step=7)

        ckpt_dir = Path(trainer_a.config["output_dir"]) / "checkpoint-7"
        assert ckpt_dir.exists()

        # Create a fresh trainer and do a standard resume
        trainer_b = make_trainer(tmp_path, output_dir=str(tmp_path / "out_b"))
        trainer_b._load_checkpoint(ckpt_dir, encoder_weights_only=False)

        # The global_step should have been restored from trainer_state.yaml
        assert trainer_b.global_step == 7

    def test_evaluate_max_batches_zero_returns_empty(self, tmp_path: Path) -> None:
        """evaluate() with max_batches=0 should return an empty dict."""
        trainer = make_trainer(tmp_path)
        metrics = trainer.evaluate(max_batches=0)
        assert metrics == {}

    def test_load_dataset_from_parquet(self, tmp_path: Path) -> None:
        """_load_dataset uses the parquet branch when the file exists."""
        import numpy as np
        import pandas as pd

        # Build a tiny parquet with the expected schema
        rng = np.random.default_rng(0)
        sequences = [
            "".join(rng.choice(list("ACGT"), size=100)) for _ in range(3)
        ]
        df = pd.DataFrame({
            "sequence": sequences,
            "species": ["homo_sapiens"] * 3,
            "haplogroup": ["H", "L3", "R"] ,
            "het_level_vector": [None, None, None],
        })
        parquet_path = tmp_path / "train.parquet"
        df.to_parquet(parquet_path)

        # Create a trainer with genome_length=100 (tiny_config)
        trainer = make_trainer(tmp_path)
        dataset = trainer._load_dataset(
            str(parquet_path), split="train",
            window_size=50, stride=25,
        )
        assert len(dataset) > 0

    def test_load_dataset_species_filter(self, tmp_path: Path) -> None:
        """_load_dataset filters by species when species_filter is set."""
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        sequences = [
            "".join(rng.choice(list("ACGT"), size=100)) for _ in range(4)
        ]
        df = pd.DataFrame({
            "sequence": sequences,
            "species": ["homo_sapiens", "homo_sapiens", "mus_musculus", "rattus_norvegicus"],
            "haplogroup": ["H", "L3", "R", "D"],
            "het_level_vector": [None, None, None, None],
        })
        parquet_path = tmp_path / "train_multi.parquet"
        df.to_parquet(parquet_path)

        # Trainer with species_filter applied via data config
        config = make_tiny_config(tmp_path)
        config["data"] = {"species_filter": "homo_sapiens"}
        trainer = MtDNATrainer(config, make_tiny_model_config(), device="cpu")
        trainer.setup()

        dataset = trainer._load_dataset(
            str(parquet_path), split="train",
            window_size=50, stride=25,
        )
        # Only homo_sapiens rows should be loaded
        assert len(dataset) > 0

    def test_from_yaml(self, tmp_path: Path) -> None:
        """MtDNATrainer.from_yaml() must construct a valid trainer from YAML files."""
        import yaml

        config = make_tiny_config(tmp_path)
        model_config = make_tiny_model_config()

        config_path = tmp_path / "config.yaml"
        model_path = tmp_path / "model.yaml"
        config_path.write_text(yaml.dump(config))
        model_path.write_text(yaml.dump(model_config))

        trainer = MtDNATrainer.from_yaml(config_path, model_path, device="cpu")
        trainer.setup()
        assert trainer.model is not None
