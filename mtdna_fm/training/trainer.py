"""
Two-phase pre-training trainer for mtDNA-FM.

WHY a custom trainer instead of HuggingFace Trainer:
  The standard HF Trainer is designed for single-phase fine-tuning.
  Our two-phase curriculum (Phase 1: cross-species MLM, Phase 2: human-specific
  MLM + het regression) requires:
    1. Phase 2 loads ONLY encoder weights from Phase 1 (fresh optimizer).
       HF Trainer's resume_from_checkpoint loads the full optimizer state,
       which is wrong for a domain-shift phase.
    2. Per-phase het_weight toggling: 0.0 → 0.3 between phases.
    3. Species filtering per phase: Phase 1 uses all vertebrates, Phase 2
       filters to homo_sapiens only.
  A custom trainer lets us implement these three differences cleanly.

WHY gradient accumulation:
  Effective batch = batch_size * gradient_accumulation_steps = 16 * 8 = 128.
  On a laptop with limited VRAM, a physical batch of 16 windows fits;
  128 is large enough for stable MLM pre-training gradient estimates.
  Accumulation gives us large-batch semantics without large-batch memory.

WHY cosine LR with warmup:
  Warmup (2k steps): prevents instability in the first steps when the model's
    parameters are random — large gradients early can cause divergence.
  Cosine decay: anneals the learning rate smoothly so the model's final
    representation is not disrupted by large gradient steps near convergence.
  This schedule is standard for BERT-style pre-training.

WHY gradient checkpointing:
  Trading compute for memory: instead of storing all intermediate activations
  for the backward pass, checkpointing recomputes them during backprop.
  Roughly halves peak memory usage at the cost of ~30% more compute.
  Necessary to train a 6M-parameter transformer on a laptop.

WHY MLflow:
  Run tracking with no code changes: each invocation is automatically a new
  run with all hyperparameters, loss curves, and metrics logged. DVC reads
  MLflow for its metric tracking. This is the same setup used in scFM.

WHY save_pretrained (not torch.save):
  save_pretrained() writes config.json + model.safetensors in a format that
  MtDNAModel.from_pretrained() can load, following HF conventions.
  Safetensors is faster, safer, and smaller than pickle-based .pt files.

Phase 2 loading:
  _load_checkpoint(path, encoder_weights_only=True) loads only the encoder
  (mtdna.*) weights and discards the optimizer state. This is the key
  difference from a standard checkpoint resume.
"""

from __future__ import annotations

import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mtdna_fm.data.dataset import MtDNADataset
from mtdna_fm.model.config import MtDNAConfig
from mtdna_fm.model.model import MtDNAForMaskedModeling
from mtdna_fm.tokenizer.vocabulary import KmerVocabulary
from mtdna_fm.training.masking import MtDNAMaskingCollator

logger = logging.getLogger(__name__)


# ── LR schedule helpers ────────────────────────────────────────────────────────


def _cosine_lr_with_warmup(
    step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr_fraction: float = 0.1
) -> float:
    """
    Cosine decay with linear warmup.

    During warmup (step < warmup_steps): LR ramps from 0 → base_lr linearly.
    After warmup: LR follows cosine decay from base_lr → min_lr_fraction * base_lr.

    Returns the LR multiplier (multiply by base_lr to get the actual LR).
    """
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_fraction + (1.0 - min_lr_fraction) * cosine_factor


# ── Trainer ────────────────────────────────────────────────────────────────────


class MtDNATrainer:
    """
    Two-phase aware pre-trainer for mtDNA-FM.

    Usage (Phase 1):
        trainer = MtDNATrainer(config, model_config, device="cpu")
        trainer.train()

    Usage (Phase 2):
        trainer = MtDNATrainer(config, model_config=None, device="cpu")
        # config["resume_from"] triggers Phase 2 loading automatically
        trainer.train()

    Parameters
    ----------
    config:
        Training hyperparameters (loaded from pretraining_phase*.yaml).
    model_config:
        Model architecture dict (loaded from model_small.yaml).
        Required for Phase 1 (cold start); optional for Phase 2 (loaded
        from the Phase 1 checkpoint's config.json).
    device:
        Torch device string: "cpu", "cuda", "mps".
    """

    def __init__(
        self,
        config: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.model_config_dict = model_config
        self.device = torch.device(device)

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training hyperparameters
        self.batch_size: int = config.get("batch_size", 16)
        self.grad_accum: int = config.get("gradient_accumulation_steps", 8)
        self.lr: float = float(config.get("learning_rate", 1e-4))
        self.weight_decay: float = float(config.get("weight_decay", 0.01))
        self.warmup_steps: int = config.get("warmup_steps", 2000)
        self.max_steps: int = config.get("max_steps", 50000)
        self.max_grad_norm: float = float(config.get("max_grad_norm", 1.0))
        self.mask_prob: float = float(config.get("mask_prob", 0.15))
        self.mlm_weight: float = float(config.get("mlm_weight", 1.0))
        self.het_weight: float = float(config.get("het_weight", 0.0))
        self.num_workers: int = config.get("num_workers", 0)
        self.save_steps: int = config.get("save_steps", 5000)
        self.eval_steps: int = config.get("eval_steps", 2500)
        self.log_steps: int = config.get("log_steps", 100)
        self.keep_last_n: int = config.get("keep_last_n_checkpoints", 3)
        self.gradient_checkpointing: bool = config.get("gradient_checkpointing", False)
        self.fp16: bool = config.get("fp16", False)
        self.mlflow_experiment: str = config.get("mlflow_experiment", "mtdna_fm_pretraining")
        self.species_filter: str | None = (
            config.get("data", {}).get("species_filter")
            if isinstance(config.get("data"), dict)
            else None
        )

        # Paths
        data_cfg = config.get("data", {})
        self.train_parquet: str | None = (
            data_cfg.get("train_parquet") if isinstance(data_cfg, dict) else None
        )
        self.val_parquet: str | None = (
            data_cfg.get("val_parquet") if isinstance(data_cfg, dict) else None
        )
        self.resume_from: str | None = config.get("resume_from")

        # Will be populated in setup()
        self.vocabulary: KmerVocabulary | None = None
        self.model: MtDNAForMaskedModeling | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.optimizer: AdamW | None = None

        # State
        self.global_step: int = 0
        self._checkpoint_paths: list[Path] = []

    # ── Setup ──────────────────────────────────────────────────────────────────

    # ── Genome-specific derived properties ────────────────────────────────────

    @staticmethod
    def _infer_k_from_vocab_size(vocab_size: int, n_special: int = 6) -> int:
        """
        Derive k-mer size from vocabulary size.

        vocab_size = 4^k + n_special
        → k = log4(vocab_size - n_special)

        Examples:
            4102 = 4^6 + 6  → k = 6  (production)
            70   = 4^3 + 6  → k = 3  (tiny test model)
        """
        n_kmers = vocab_size - n_special
        k = round(math.log(n_kmers, 4))
        if 4**k != n_kmers:
            raise ValueError(
                f"vocab_size={vocab_size} does not correspond to a valid 4^k vocabulary. "
                f"Expected 4^k + {n_special} for integer k."
            )
        return k

    def setup(self) -> None:
        """Initialise model, vocabulary, dataloaders, optimizer. Call before train()."""
        # Build model first so we can read vocab_size and genome_length from config
        logger.info("Building model …")
        self.model = self._build_model()
        self.model = self.model.to(self.device)

        # Derive k and genome_length from model config
        vocab_size: int = self.model.config.vocab_size
        genome_length: int = self.model.config.genome_length
        kmer_size: int = self._infer_k_from_vocab_size(vocab_size)
        self._genome_length = genome_length
        self._kmer_size = kmer_size

        logger.info(
            "Model: vocab_size=%d (k=%d), genome_length=%d",
            vocab_size, kmer_size, genome_length,
        )

        logger.info("Building vocabulary (k=%d) …", kmer_size)
        self.vocabulary = KmerVocabulary.build(k=kmer_size)

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

        logger.info("Building dataloaders …")
        self.train_loader, self.val_loader = self._build_dataloaders()

        logger.info("Building optimizer …")
        self.optimizer = self._build_optimizer()

        # Phase 2: load encoder weights from Phase 1 checkpoint, fresh optimizer
        if self.resume_from is not None:
            logger.info("Phase 2: loading encoder weights from %s", self.resume_from)
            self._load_checkpoint(self.resume_from, encoder_weights_only=True)

    def _build_model(self) -> MtDNAForMaskedModeling:
        """
        Create MtDNAForMaskedModeling from model config or load from checkpoint.

        Phase 1: create fresh model from model_config dict.
        Phase 2: load base config from Phase 1 checkpoint directory.
        """
        if self.resume_from is not None:
            # Phase 2: load config from previous checkpoint
            mtdna_config = MtDNAConfig.from_pretrained(self.resume_from)
            logger.info("Loaded config from %s", self.resume_from)
        elif self.model_config_dict is not None:
            # Phase 1: create from YAML-supplied dict
            cfg_dict = {k: v for k, v in self.model_config_dict.items() if k != "model_type"}
            mtdna_config = MtDNAConfig(**cfg_dict)
        else:
            raise ValueError("Either model_config or resume_from must be provided.")

        return MtDNAForMaskedModeling(
            mtdna_config,
            mlm_weight=self.mlm_weight,
            het_weight=self.het_weight,
        )

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Load processed parquet files and build windowed DataLoaders.

        Falls back to a tiny synthetic dataset if parquet files do not exist
        (useful for unit tests and smoke tests without real data).

        window_size is capped at genome_length so the window never exceeds
        the genome (important for tiny test models with short genomes).
        """
        assert self.vocabulary is not None

        collator = MtDNAMaskingCollator(
            self.vocabulary,
            mask_prob=self.mask_prob,
        )

        # window_size is capped at genome_length for correctness
        window_size = min(512, self._genome_length)
        stride = max(1, window_size // 2)

        train_dataset = self._load_dataset(
            self.train_parquet, split="train", window_size=window_size, stride=stride
        )
        val_dataset = self._load_dataset(
            self.val_parquet, split="val", window_size=window_size, stride=stride
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collator,
            drop_last=True,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
            drop_last=False,
        )

        logger.info(
            "Datasets: %d train windows, %d val windows (window=%d, stride=%d)",
            len(train_dataset),
            len(val_dataset),
            window_size,
            stride,
        )
        return train_loader, val_loader

    def _load_dataset(
        self,
        parquet_path: str | None,
        split: str,
        window_size: int = 512,
        stride: int = 256,
    ) -> MtDNADataset:
        """Load a parquet file as an MtDNADataset, with fallback to synthetic data."""
        assert self.vocabulary is not None

        if parquet_path is not None and Path(parquet_path).exists():
            df = pd.read_parquet(parquet_path)

            # Phase 1: all species; Phase 2: homo_sapiens only
            if self.species_filter is not None and self.species_filter != "all":
                df = df[df["species"] == self.species_filter].reset_index(drop=True)

            logger.info("[%s] Loaded %d sequences from %s", split, len(df), parquet_path)

            return MtDNADataset.from_dataframe(
                df,
                vocabulary=self.vocabulary,
                sequence_col="sequence",
                het_col="het_level_vector",
                window_size=window_size,
                stride=stride,
                genome_length=self._genome_length,
                k=self._kmer_size,
            )
        else:
            # Synthetic fallback: 2 random genome_length-bp sequences
            logger.warning(
                "[%s] Parquet not found at '%s' — using synthetic fallback (2 sequences).",
                split,
                parquet_path,
            )
            rng = np.random.default_rng(42 if split == "train" else 7)
            seqs = [
                "".join(rng.choice(list("ACGT"), size=self._genome_length))
                for _ in range(2)
            ]
            return MtDNADataset(
                seqs,
                vocabulary=self.vocabulary,
                k=self._kmer_size,
                window_size=window_size,
                stride=stride,
                genome_length=self._genome_length,
            )

    def _build_optimizer(self) -> AdamW:
        """
        AdamW with weight decay applied to weights only (not biases or norms).

        Following BERT conventions: decay dense weight matrices, skip
        LayerNorm parameters and biases. This prevents over-regularization
        of the normalization parameters.
        """
        assert self.model is not None

        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight", "final_layer_norm.weight"}
        param_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(param_groups, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

    # ── Checkpoint I/O ─────────────────────────────────────────────────────────

    def _load_checkpoint(self, path: str | Path, encoder_weights_only: bool = False) -> None:
        """
        Load model weights (and optionally optimizer state) from a checkpoint.

        Parameters
        ----------
        path:
            Directory containing config.json and model.safetensors (or
            pytorch_model.bin), written by save_pretrained().
        encoder_weights_only:
            If True (Phase 2), load only the encoder portion of the model
            (weights whose keys start with "mtdna.") and skip the optimizer
            state. The prediction heads are re-initialized fresh.
            If False (standard resume), load all weights and optimizer state
            if a 'optimizer.pt' file exists alongside the checkpoint.
        """
        assert self.model is not None
        path = Path(path)

        if encoder_weights_only:
            # Phase 2: load encoder weights only, fresh optimizer
            # Load the full Phase 1 model to extract weights
            phase1_model = MtDNAForMaskedModeling.from_pretrained(str(path))
            phase1_state = phase1_model.state_dict()

            # Copy only 'mtdna.*' weights (the encoder) into the Phase 2 model
            current_state = self.model.state_dict()
            loaded_keys = []
            for key, val in phase1_state.items():
                if key.startswith("mtdna.") and key in current_state:
                    current_state[key] = val
                    loaded_keys.append(key)

            self.model.load_state_dict(current_state, strict=False)
            logger.info(
                "Phase 2 encoder load: %d/%d keys loaded from %s",
                len(loaded_keys),
                len(current_state),
                path,
            )
        else:
            # Standard resume: load full model weights
            loaded = MtDNAForMaskedModeling.from_pretrained(str(path))
            self.model.load_state_dict(loaded.state_dict())

            # Optionally restore optimizer
            opt_path = path / "optimizer.pt"
            if opt_path.exists() and self.optimizer is not None:
                opt_state = torch.load(str(opt_path), map_location=self.device)
                self.optimizer.load_state_dict(opt_state)
                logger.info("Restored optimizer state from %s", opt_path)

            # Restore step counter
            step_path = path / "trainer_state.yaml"
            if step_path.exists():
                with open(step_path) as f:
                    state = yaml.safe_load(f)
                    self.global_step = state.get("global_step", 0)
                logger.info("Resumed from step %d", self.global_step)

    def _save_checkpoint(self, step: int) -> None:
        """
        Save model + optimizer state + trainer metadata.

        Uses save_pretrained() for the model (config.json + model.safetensors)
        and torch.save() for the optimizer (not supported by HF save_pretrained).
        Keeps only the last N checkpoints to save disk space.
        """
        assert self.model is not None and self.optimizer is not None

        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(ckpt_dir))
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        with open(ckpt_dir / "trainer_state.yaml", "w") as f:
            yaml.dump({"global_step": step, "timestamp": time.time()}, f)

        logger.info("Saved checkpoint at step %d → %s", step, ckpt_dir)
        self._checkpoint_paths.append(ckpt_dir)

        # Rotate old checkpoints
        while len(self._checkpoint_paths) > self.keep_last_n:
            old = self._checkpoint_paths.pop(0)
            if old.exists() and old != self.output_dir:
                shutil.rmtree(old)
                logger.info("Removed old checkpoint %s", old)

    def _save_final(self) -> None:
        """Save the final model to output_dir (not a subdirectory)."""
        assert self.model is not None

        self.model.save_pretrained(str(self.output_dir))
        assert self.vocabulary is not None
        self.vocabulary.save_pretrained(str(self.output_dir))

        with open(self.output_dir / "training_config.yaml", "w") as f:
            yaml.dump(self.config, f)

        logger.info("Final model saved to %s", self.output_dir)

    # ── Evaluation ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, max_batches: int = 100) -> dict[str, float]:
        """
        Run validation and return metrics dict.

        Evaluates up to `max_batches` batches to keep eval time bounded.
        Returns mlm_loss, het_loss (if available), and perplexity.
        """
        assert self.model is not None and self.val_loader is not None

        self.model.eval()
        total_mlm_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            if n_batches >= max_batches:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                het_values=batch.get("het_values"),
                attention_mask=batch["attention_mask"],
                kmer_labels=batch["kmer_labels"],
                het_labels=batch.get("het_labels"),
            )

            if outputs.mlm_loss is not None:
                total_mlm_loss += outputs.mlm_loss.item()
            n_batches += 1

        self.model.train()

        if n_batches == 0:
            return {}

        avg_mlm_loss = total_mlm_loss / n_batches
        return {
            "eval/mlm_loss": avg_mlm_loss,
            "eval/perplexity": math.exp(min(avg_mlm_loss, 20.0)),
        }

    # ── Main training loop ─────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Run the pre-training loop.

        Training proceeds for max_steps gradient update steps.
        Each update uses gradient_accumulation_steps forward passes.
        LR follows cosine decay with warmup.
        Checkpoints are saved every save_steps.
        Validation runs every eval_steps.
        Metrics are logged to MLflow every log_steps.
        """
        assert self.model is not None
        assert self.train_loader is not None
        assert self.optimizer is not None

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Parameters: %d (%.1fM)", n_params, n_params / 1e6)
        logger.info(
            "Training: max_steps=%d, batch=%d, accum=%d → eff_batch=%d",
            self.max_steps,
            self.batch_size,
            self.grad_accum,
            self.batch_size * self.grad_accum,
        )

        # MLflow setup
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params({
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.grad_accum,
            "effective_batch_size": self.batch_size * self.grad_accum,
            "learning_rate": self.lr,
            "warmup_steps": self.warmup_steps,
            "mask_prob": self.mask_prob,
            "mlm_weight": self.mlm_weight,
            "het_weight": self.het_weight,
            "n_parameters": n_params,
        })

        self.model.train()

        accum_loss = 0.0
        accum_mlm_loss = 0.0
        micro_step = 0
        t_start = time.time()
        data_iter = iter(self.train_loader)

        try:
            while self.global_step < self.max_steps:
                # ── gradient accumulation micro-batch ──────────────────────────
                self.optimizer.zero_grad()
                step_loss = 0.0
                step_mlm_loss = 0.0

                for _ in range(self.grad_accum):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.train_loader)
                        batch = next(data_iter)

                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        position_ids=batch["position_ids"],
                        het_values=batch.get("het_values"),
                        attention_mask=batch["attention_mask"],
                        kmer_labels=batch["kmer_labels"],
                        het_labels=batch.get("het_labels"),
                    )

                    loss = outputs.loss / self.grad_accum
                    loss.backward()

                    step_loss += loss.item()
                    if outputs.mlm_loss is not None:
                        step_mlm_loss += outputs.mlm_loss.item() / self.grad_accum
                    micro_step += 1

                # ── optimizer step ─────────────────────────────────────────────
                # LR schedule: update per gradient step
                lr_mult = _cosine_lr_with_warmup(
                    self.global_step, self.warmup_steps, self.max_steps, self.lr
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * lr_mult

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.global_step += 1

                accum_loss += step_loss
                accum_mlm_loss += step_mlm_loss

                # ── logging ────────────────────────────────────────────────────
                if self.global_step % self.log_steps == 0:
                    elapsed = time.time() - t_start
                    avg_loss = accum_loss / self.log_steps
                    avg_mlm = accum_mlm_loss / self.log_steps
                    steps_per_sec = self.log_steps / elapsed
                    current_lr = self.lr * lr_mult

                    logger.info(
                        "step %6d | loss %.4f | mlm_loss %.4f | lr %.2e | %.2f steps/s",
                        self.global_step, avg_loss, avg_mlm, current_lr, steps_per_sec,
                    )

                    mlflow.log_metrics(
                        {
                            "train/loss": avg_loss,
                            "train/mlm_loss": avg_mlm,
                            "train/learning_rate": current_lr,
                            "train/steps_per_second": steps_per_sec,
                        },
                        step=self.global_step,
                    )

                    accum_loss = 0.0
                    accum_mlm_loss = 0.0
                    t_start = time.time()

                # ── evaluation ─────────────────────────────────────────────────
                if self.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    mlflow.log_metrics(eval_metrics, step=self.global_step)
                    logger.info(
                        "step %6d | eval_mlm_loss %.4f | perplexity %.2f",
                        self.global_step,
                        eval_metrics.get("eval/mlm_loss", float("nan")),
                        eval_metrics.get("eval/perplexity", float("nan")),
                    )

                # ── checkpoint ─────────────────────────────────────────────────
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(self.global_step)

        finally:
            mlflow.end_run()

        # Final save
        self._save_final()
        logger.info("Training complete. %d steps finished.", self.global_step)

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        model_config_path: str | Path | None = None,
        device: str = "cpu",
    ) -> MtDNATrainer:
        """
        Construct a trainer by loading config from YAML files.

        Parameters
        ----------
        config_path:
            Path to pretraining_phase*.yaml.
        model_config_path:
            Path to model_small.yaml (required for Phase 1 cold start).
            Can be omitted for Phase 2 (config loaded from Phase 1 checkpoint).
        device:
            PyTorch device string.
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_config = None
        if model_config_path is not None:
            with open(model_config_path) as f:
                model_config = yaml.safe_load(f)

        return cls(config, model_config, device=device)
