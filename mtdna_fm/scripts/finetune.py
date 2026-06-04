"""
mtdna-finetune CLI: fine-tune mtDNA-FM on downstream classification tasks.

Supports:
  haplogroup      — 26-class haplogroup classification (LoRA r=8)
  pathogenicity   — binary variant pathogenicity (LoRA r=4)  [Day 17]
  heteroplasmy    — regression on heteroplasmy level         [Day 18]

Usage:
  mtdna-finetune --task haplogroup --config configs/finetuning_haplogroup.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import typer
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = typer.Typer(
    help="Fine-tune mtDNA-FM on downstream tasks (haplogroup, pathogenicity, heteroplasmy)."
)


# ── Tiny dataset for fine-tuning ───────────────────────────────────────────────


class HaplogroupWindowDataset(Dataset):
    """
    Loads tokenised windows from a processed parquet and pairs them with
    haplogroup integer labels.

    Each genome is represented by multiple overlapping windows (window_size=512,
    stride=256). All windows from one genome share the same label. At training
    time, each window is treated as an independent example. At evaluation time,
    windows are grouped by genome and their CLS predictions are majority-voted.
    """

    # 26 major PhyloTree haplogroups used as fine-tuning labels
    HAPLOGROUPS = [
        "A", "B", "C", "D", "E", "F", "G", "H", "HV", "I",
        "J", "K", "L0", "L1", "L2", "L3", "L4", "L5", "M",
        "N", "R", "T", "U", "V", "W", "X",
    ]
    LABEL2IDX: dict[str, int] = {h: i for i, h in enumerate(HAPLOGROUPS)}

    def __init__(
        self,
        parquet_path: str | Path,
        vocabulary,
        window_size: int = 512,
        stride: int = 256,
        label_column: str = "haplogroup",
        k: int = 6,
        max_sequences: int | None = None,
    ) -> None:
        import pandas as pd

        from mtdna_fm.tokenizer.tokenize import tokenize_sequence

        df = pd.read_parquet(parquet_path)
        if max_sequences is not None:
            df = df.head(max_sequences)

        # Keep only rows with a known haplogroup label
        df = df[df[label_column].isin(self.HAPLOGROUPS)].reset_index(drop=True)
        if len(df) < 500:
            log.warning(
                "HaplogroupWindowDataset: only %d sequences matched labels in column '%s'. "
                "If using raw sub-haplogroup strings (e.g. 'H1fx', 'L5b1a'), run "
                "fix_haplogroup_labels.py first and set label_column: major_haplogroup.",
                len(df),
                label_column,
            )

        self._windows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            seq = row["sequence"]
            label_idx = self.LABEL2IDX[row[label_column]]
            tokens = tokenize_sequence(
                seq,
                vocabulary,
                k=k,
                stride=1,
                max_seq_len=len(seq),
                circular=True,
            )
            n_tokens = len(tokens["input_ids"])
            for start in range(0, n_tokens, stride):
                end = min(start + window_size, n_tokens)
                if end - start < window_size // 4:
                    continue  # skip tiny trailing windows
                pad_len = window_size - (end - start)
                ids = tokens["input_ids"][start:end] + [0] * pad_len
                pos = tokens["position_ids"][start:end] + [0] * pad_len
                mask = [1] * (end - start) + [0] * pad_len
                self._windows.append(
                    {
                        "input_ids": ids,
                        "position_ids": pos,
                        "attention_mask": mask,
                        "label": label_idx,
                    }
                )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        w = self._windows[idx]
        return {
            "input_ids": torch.tensor(w["input_ids"], dtype=torch.long),
            "position_ids": torch.tensor(w["position_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(w["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(w["label"], dtype=torch.long),
        }


# ── LR schedule helper ─────────────────────────────────────────────────────────


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


# ── Fine-tuning loop ───────────────────────────────────────────────────────────


def finetune_haplogroup(cfg: dict[str, Any]) -> None:
    """Run haplogroup classification fine-tuning."""
    from peft import LoraConfig, TaskType, get_peft_model

    from mtdna_fm.model.model import MtDNAForHaplogroupClassification, MtDNAModel
    from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

    base_model_path = cfg["base_model"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load base encoder
    if not Path(base_model_path).exists():
        typer.echo(
            f"[finetune] Base model not found at {base_model_path}. "
            "Run Phase 2 pre-training first (mtdna-train --config configs/pretraining_phase2.yaml).",
            err=True,
        )
        raise typer.Exit(code=1)

    log.info("Loading base model from %s", base_model_path)
    from mtdna_fm.model.model import MtDNAForMaskedModeling

    full_model = MtDNAForMaskedModeling.from_pretrained(base_model_path)
    base_encoder: MtDNAModel = full_model.mtdna

    vocabulary = KmerVocabulary.from_pretrained(base_model_path)

    # Build classification model
    model = MtDNAForHaplogroupClassification(
        base_encoder,
        num_labels=cfg["num_labels"],
        dropout=cfg.get("dropout", 0.1),
    )

    # Apply LoRA if requested
    if cfg.get("use_lora", True):
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            target_modules=cfg.get("lora_target_modules", ["query", "key", "value", "dense"]),
            lora_dropout=cfg.get("lora_dropout", 0.1),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model = model.to(device)

    # Load training data
    train_parquet = cfg.get("data", {}).get("train_parquet", "data/processed/train.parquet")
    val_parquet = cfg.get("data", {}).get("val_parquet", "data/processed/val.parquet")

    if not Path(train_parquet).exists():
        typer.echo(
            f"[finetune] Training data not found at {train_parquet}. "
            "Run mtdna-preprocess first.",
            err=True,
        )
        raise typer.Exit(code=1)

    log.info("Building dataset from %s", train_parquet)
    train_ds = HaplogroupWindowDataset(
        train_parquet,
        vocabulary,
        label_column=cfg.get("label_column", "haplogroup"),
    )
    log.info("Training windows: %d", len(train_ds))
    if len(train_ds) < 100:
        typer.echo(
            f"[finetune] FATAL: only {len(train_ds)} training windows after label filtering.\n"
            "Almost certainly a label_column mismatch — check config 'label_column' vs "
            "actual column values in the parquet (run fix_haplogroup_labels.py if needed).",
            err=True,
        )
        raise typer.Exit(code=1)

    val_ds = HaplogroupWindowDataset(
        val_parquet,
        vocabulary,
        label_column=cfg.get("label_column", "haplogroup"),
    ) if Path(val_parquet).exists() else None

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=0,
    )
    val_dl = (
        DataLoader(val_ds, batch_size=cfg.get("batch_size", 32), shuffle=False, num_workers=0)
        if val_ds is not None
        else None
    )

    # Compute inverse-frequency class weights to prevent collapse to majority class.
    # After windowing, larger genomes produce more windows → severe class imbalance
    # even with balanced sequences per class. Balanced weights: n / (K * count_k).
    n_labels = cfg["num_labels"]
    label_tensor = torch.tensor([w["label"] for w in train_ds._windows])
    class_counts = torch.bincount(label_tensor, minlength=n_labels).float()
    class_weights = len(train_ds) / (n_labels * (class_counts + 1e-6))
    log.info(
        "Class weights computed: min=%.3f max=%.3f (windows=%d across %d classes)",
        class_weights.min().item(), class_weights.max().item(), len(train_ds), n_labels,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # MLflow tracking
    try:
        import mlflow
        mlflow.set_experiment(cfg.get("mlflow_experiment", "mtdna_fm_haplogroup"))
        mlflow.start_run()
        mlflow.log_params(cfg)
        use_mlflow = True
    except Exception:
        use_mlflow = False

    grad_accum = cfg.get("gradient_accumulation_steps", 4)
    max_epochs = cfg.get("max_epochs", 20)
    lr = float(cfg.get("learning_rate", 1e-3))
    warmup_ratio = cfg.get("warmup_ratio", 0.1)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    total_steps = (len(train_dl) // grad_accum) * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            raw_loss = loss_fn(out.logits, labels)
            loss = raw_loss / grad_accum
            loss.backward()
            total_loss += raw_loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = total_loss / len(train_dl)
        log.info("Epoch %d/%d — train loss: %.4f", epoch + 1, max_epochs, avg_loss)

        if use_mlflow:
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Validation
        if val_dl is not None and (epoch + 1) % cfg.get("eval_every_n_epochs", 1) == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_dl:
                    input_ids = batch["input_ids"].to(device)
                    position_ids = batch["position_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    out = model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                    )
                    preds = out.logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total if total > 0 else 0.0
            log.info("  val accuracy: %.4f", val_acc)

            if use_mlflow:
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best model
                _save_model(model, output_dir / "best", cfg, vocabulary)
                log.info("  Saved best model (val_acc=%.4f)", best_val_acc)

    # Save final model
    _save_model(model, output_dir, cfg, vocabulary)

    metrics = {"best_val_accuracy": best_val_acc, "final_train_loss": avg_loss}
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    if use_mlflow:
        mlflow.log_metrics(metrics)
        mlflow.end_run()

    log.info("Fine-tuning complete. Output: %s", output_dir)
    log.info("Best validation accuracy: %.4f", best_val_acc)


class PathogenicityVariantDataset(Dataset):
    """
    Dataset for binary variant pathogenicity prediction.

    Each sample is a 512-token window centered on a variant position.
    Classification uses the hidden state at the variant token, not CLS,
    because pathogenicity is a local property of the mutation context.

    Expected parquet columns: sequence, position, label (1=pathogenic, 0=benign),
    variant_type (missense|tRNA|rRNA|D-loop) for stratified splitting.
    Falls back to synthetic data when parquet is not available, to support
    unit-testing and development without real variant data.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        vocabulary,
        window_size: int = 512,
        k: int = 6,
        max_variants: int | None = None,
    ) -> None:
        import pandas as pd

        from mtdna_fm.tokenizer.tokenize import tokenize_sequence

        path = Path(parquet_path)
        if path.exists():
            df = pd.read_parquet(path)
        else:
            # Synthetic fallback for testing: random variants on a short reference
            log.warning(
                "Variant parquet not found at %s — using 64-sample synthetic fallback",
                parquet_path,
            )
            import numpy as np

            rng = np.random.default_rng(42)
            ref = "".join(rng.choice(list("ACGT"), size=16569))
            bases = list("ACGT")
            rows = []
            for i in range(64):
                pos = int(rng.integers(0, 16569))
                ref_base = ref[pos]
                alt = rng.choice([b for b in bases if b != ref_base])
                seq = ref[:pos] + alt + ref[pos + 1 :]
                rows.append({
                    "sequence": seq,
                    "position": pos,
                    "label": int(i % 2),
                    "variant_type": rng.choice(["missense", "tRNA", "rRNA", "D-loop"]),
                })
            df = pd.DataFrame(rows)

        if max_variants is not None:
            df = df.head(max_variants)

        self._samples: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            seq = row["sequence"]
            pos = int(row["position"])
            label = float(row["label"])

            tokens = tokenize_sequence(
                seq,
                vocabulary,
                k=k,
                stride=1,
                max_seq_len=len(seq),
                circular=True,
            )
            n_tokens = len(tokens["input_ids"])

            # Center 512-token window on the variant position (clamp to boundaries)
            half = window_size // 2
            start = max(0, pos - half)
            end = min(n_tokens, start + window_size)
            start = max(0, end - window_size)  # re-align if near end

            pad_len = window_size - (end - start)
            ids = tokens["input_ids"][start:end] + [0] * pad_len
            pos_ids = tokens["position_ids"][start:end] + [0] * pad_len
            mask = [1] * (end - start) + [0] * pad_len

            # Index within the window of the token that covers the variant position
            variant_tok_idx = min(pos - start, end - start - 1)

            self._samples.append({
                "input_ids": ids,
                "position_ids": pos_ids,
                "attention_mask": mask,
                "variant_token_idx": variant_tok_idx,
                "label": label,
            })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self._samples[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "position_ids": torch.tensor(s["position_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(s["attention_mask"], dtype=torch.long),
            "variant_token_idx": torch.tensor(s["variant_token_idx"], dtype=torch.long),
            "labels": torch.tensor(s["label"], dtype=torch.float),
        }


def finetune_pathogenicity(cfg: dict[str, Any]) -> None:
    """Run variant pathogenicity binary classification fine-tuning."""
    from peft import LoraConfig, TaskType, get_peft_model

    from mtdna_fm.model.model import (
        MtDNAForMaskedModeling,
        MtDNAForVariantPathogenicity,
        MtDNAModel,
    )
    from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

    base_model_path = cfg["base_model"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    if not Path(base_model_path).exists():
        typer.echo(
            f"[finetune] Base model not found at {base_model_path}. "
            "Run Phase 2 pre-training first.",
            err=True,
        )
        raise typer.Exit(code=1)

    log.info("Loading base model from %s", base_model_path)
    full_model = MtDNAForMaskedModeling.from_pretrained(base_model_path)
    base_encoder: MtDNAModel = full_model.mtdna

    vocabulary = KmerVocabulary.from_pretrained(base_model_path)

    model = MtDNAForVariantPathogenicity(
        base_encoder,
        dropout=cfg.get("dropout", 0.1),
        pos_weight=cfg.get("pos_weight", 2.5),
    )

    if cfg.get("use_lora", True):
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.get("lora_r", 4),
            lora_alpha=cfg.get("lora_alpha", 8),
            target_modules=cfg.get("lora_target_modules", ["query", "key", "value", "dense"]),
            lora_dropout=cfg.get("lora_dropout", 0.1),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model = model.to(device)

    train_parquet = cfg.get("data", {}).get(
        "train_parquet", "data/processed/variants_pathogenicity_train.parquet"
    )
    val_parquet = cfg.get("data", {}).get(
        "val_parquet", "data/processed/variants_pathogenicity_val.parquet"
    )

    log.info("Building dataset from %s", train_parquet)
    train_ds = PathogenicityVariantDataset(
        train_parquet,
        vocabulary,
        window_size=cfg.get("window_size", 512),
    )
    val_ds = PathogenicityVariantDataset(
        val_parquet,
        vocabulary,
        window_size=cfg.get("window_size", 512),
    ) if Path(val_parquet).exists() else None

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=0,
    )
    val_dl = (
        DataLoader(val_ds, batch_size=cfg.get("batch_size", 32), shuffle=False, num_workers=0)
        if val_ds is not None
        else None
    )

    try:
        import mlflow
        mlflow.set_experiment(cfg.get("mlflow_experiment", "mtdna_fm_pathogenicity"))
        mlflow.start_run()
        mlflow.log_params(cfg)
        use_mlflow = True
    except Exception:
        use_mlflow = False

    grad_accum = cfg.get("gradient_accumulation_steps", 4)
    max_epochs = cfg.get("max_epochs", 20)
    lr = float(cfg.get("learning_rate", 1e-3))
    warmup_ratio = cfg.get("warmup_ratio", 0.1)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=cfg.get("weight_decay", 0.1),
    )

    total_steps = (len(train_dl) // grad_accum) * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_auroc = 0.0
    avg_loss = 0.0
    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            variant_token_idx = batch["variant_token_idx"].to(device)
            labels = batch["labels"].to(device)

            out = model(
                input_ids=input_ids,
                position_ids=position_ids,
                variant_token_idx=variant_token_idx,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss / grad_accum
            loss.backward()
            total_loss += out.loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = total_loss / len(train_dl)
        log.info("Epoch %d/%d — train loss: %.4f", epoch + 1, max_epochs, avg_loss)

        if use_mlflow:
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        if val_dl is not None and (epoch + 1) % cfg.get("eval_every_n_epochs", 1) == 0:
            model.eval()
            all_probs: list[float] = []
            all_labels: list[float] = []
            with torch.no_grad():
                for batch in val_dl:
                    input_ids = batch["input_ids"].to(device)
                    position_ids = batch["position_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    variant_token_idx = batch["variant_token_idx"].to(device)
                    labels_b = batch["labels"].to(device)
                    out = model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        variant_token_idx=variant_token_idx,
                        attention_mask=attention_mask,
                    )
                    all_probs.extend(out.probs.cpu().tolist())
                    all_labels.extend(labels_b.cpu().tolist())

            try:
                from sklearn.metrics import roc_auc_score
                val_auroc = float(roc_auc_score(all_labels, all_probs))
            except Exception:
                val_auroc = 0.0

            log.info("  val AUROC: %.4f", val_auroc)

            if use_mlflow:
                mlflow.log_metric("val_auroc", val_auroc, step=epoch)

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                _save_model(model, output_dir / "best", cfg, vocabulary)
                log.info("  Saved best model (val_auroc=%.4f)", best_val_auroc)

    _save_model(model, output_dir, cfg, vocabulary)

    metrics = {"best_val_auroc": best_val_auroc, "final_train_loss": avg_loss}
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    if use_mlflow:
        mlflow.log_metrics(metrics)
        mlflow.end_run()

    log.info("Fine-tuning complete. Output: %s", output_dir)
    log.info("Best validation AUROC: %.4f", best_val_auroc)


class HeteroplasmyRegressionDataset(Dataset):
    """
    Dataset for heteroplasmy level regression.

    Each sample is a 512-token window centered on a variant position.
    The regression target is the mean observed heteroplasmy level (float in
    [0, 1]) across carriers in gnomAD (>=50 heteroplasmic carriers filter).

    Expected parquet columns: sequence (str), position (int, 0-based),
    het_level (float in [0, 1]).
    Falls back to 64-sample synthetic data when parquet is not available,
    to support unit-testing and development without real variant data.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        vocabulary,
        window_size: int = 512,
        k: int = 6,
        max_variants: int | None = None,
    ) -> None:
        import pandas as pd

        from mtdna_fm.tokenizer.tokenize import tokenize_sequence

        path = Path(parquet_path)
        if path.exists():
            df = pd.read_parquet(path)
        else:
            log.warning(
                "Heteroplasmy parquet not found at %s — using 64-sample synthetic fallback",
                parquet_path,
            )
            import numpy as np

            rng = np.random.default_rng(42)
            ref = "".join(rng.choice(list("ACGT"), size=16569))
            bases = list("ACGT")
            rows = []
            for _i in range(64):
                pos = int(rng.integers(0, 16569))
                ref_base = ref[pos]
                alt = rng.choice([b for b in bases if b != ref_base])
                seq = ref[:pos] + alt + ref[pos + 1 :]
                rows.append({
                    "sequence": seq,
                    "position": pos,
                    "het_level": float(rng.uniform(0.01, 0.99)),
                })
            df = pd.DataFrame(rows)

        if max_variants is not None:
            df = df.head(max_variants)

        self._samples: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            seq = row["sequence"]
            pos = int(row["position"])
            het_level = float(row["het_level"])

            tokens = tokenize_sequence(
                seq,
                vocabulary,
                k=k,
                stride=1,
                max_seq_len=len(seq),
                circular=True,
            )
            n_tokens = len(tokens["input_ids"])

            half = window_size // 2
            start = max(0, pos - half)
            end = min(n_tokens, start + window_size)
            start = max(0, end - window_size)

            pad_len = window_size - (end - start)
            ids = tokens["input_ids"][start:end] + [0] * pad_len
            pos_ids = tokens["position_ids"][start:end] + [0] * pad_len
            mask = [1] * (end - start) + [0] * pad_len

            variant_tok_idx = min(pos - start, end - start - 1)

            self._samples.append({
                "input_ids": ids,
                "position_ids": pos_ids,
                "attention_mask": mask,
                "variant_token_idx": variant_tok_idx,
                "het_level": het_level,
            })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self._samples[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "position_ids": torch.tensor(s["position_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(s["attention_mask"], dtype=torch.long),
            "variant_token_idx": torch.tensor(s["variant_token_idx"], dtype=torch.long),
            "labels": torch.tensor(s["het_level"], dtype=torch.float),
        }


def finetune_heteroplasmy(cfg: dict[str, Any]) -> None:
    """
    Run heteroplasmy regression fine-tuning with 5-fold cross-validation.

    Reports R-squared and Spearman correlation across folds. A Spearman
    rho > 0.30 indicates the model is capturing something real about
    selective constraint on heteroplasmy levels.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    from mtdna_fm.model.model import (
        MtDNAForHeteroplasmyRegression,
        MtDNAForMaskedModeling,
        MtDNAModel,
    )
    from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

    base_model_path = cfg["base_model"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    if not Path(base_model_path).exists():
        typer.echo(
            f"[finetune] Base model not found at {base_model_path}. "
            "Run Phase 2 pre-training first.",
            err=True,
        )
        raise typer.Exit(code=1)

    log.info("Loading base model from %s", base_model_path)
    full_model = MtDNAForMaskedModeling.from_pretrained(base_model_path)
    base_encoder: MtDNAModel = full_model.mtdna

    vocabulary = KmerVocabulary.from_pretrained(base_model_path)

    parquet_path = cfg.get("data", {}).get(
        "parquet", "data/processed/variants_heteroplasmy.parquet"
    )
    log.info("Building dataset from %s", parquet_path)
    full_ds = HeteroplasmyRegressionDataset(
        parquet_path,
        vocabulary,
        window_size=cfg.get("window_size", 512),
    )

    n_folds = cfg.get("n_folds", 5)
    grad_accum = cfg.get("gradient_accumulation_steps", 4)
    max_epochs = cfg.get("max_epochs", 15)
    lr = float(cfg.get("learning_rate", 1e-3))
    warmup_ratio = cfg.get("warmup_ratio", 0.1)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    batch_size = cfg.get("batch_size", 16)

    try:
        import mlflow
        mlflow.set_experiment(cfg.get("mlflow_experiment", "mtdna_fm_heteroplasmy"))
        mlflow.start_run()
        mlflow.log_params(cfg)
        use_mlflow = True
    except Exception:
        use_mlflow = False

    fold_r2: list[float] = []
    fold_spearman: list[float] = []

    indices = list(range(len(full_ds)))
    fold_size = len(indices) // n_folds

    for fold in range(n_folds):
        log.info("Fold %d/%d", fold + 1, n_folds)
        val_indices = indices[fold * fold_size : (fold + 1) * fold_size]
        train_indices = indices[: fold * fold_size] + indices[(fold + 1) * fold_size :]

        from torch.utils.data import Subset

        train_ds = Subset(full_ds, train_indices)
        val_ds = Subset(full_ds, val_indices)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Re-initialise model from frozen base for each fold
        fold_base = MtDNAModel(base_encoder.config)
        fold_base.load_state_dict(base_encoder.state_dict())

        model = MtDNAForHeteroplasmyRegression(
            fold_base,
            dropout=cfg.get("dropout", 0.1),
            huber_delta=cfg.get("huber_delta", 0.1),
        )

        if cfg.get("use_lora", True):
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=cfg.get("lora_r", 4),
                lora_alpha=cfg.get("lora_alpha", 8),
                target_modules=cfg.get(
                    "lora_target_modules", ["query", "key", "value", "dense"]
                ),
                lora_dropout=cfg.get("lora_dropout", 0.1),
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)

        model = model.to(device)

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=cfg.get("weight_decay", 0.01),
        )
        total_steps = (len(train_dl) // grad_accum) * max_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        for epoch in range(max_epochs):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_dl):
                input_ids = batch["input_ids"].to(device)
                position_ids = batch["position_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                variant_token_idx = batch["variant_token_idx"].to(device)
                labels = batch["labels"].to(device)

                out = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    variant_token_idx=variant_token_idx,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss / grad_accum
                loss.backward()
                total_loss += out.loss.item()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = total_loss / len(train_dl)
            log.info("  Epoch %d/%d — train loss: %.4f", epoch + 1, max_epochs, avg_loss)

        # Evaluate on validation fold
        model.eval()
        all_preds: list[float] = []
        all_targets: list[float] = []
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                position_ids = batch["position_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                variant_token_idx = batch["variant_token_idx"].to(device)
                out = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    variant_token_idx=variant_token_idx,
                    attention_mask=attention_mask,
                )
                all_preds.extend(out.predictions.cpu().tolist())
                all_targets.extend(batch["labels"].tolist())

        try:
            from scipy.stats import spearmanr
            from sklearn.metrics import r2_score

            r2 = float(r2_score(all_targets, all_preds))
            spearman_rho = float(spearmanr(all_targets, all_preds).statistic)
        except Exception:
            r2 = 0.0
            spearman_rho = 0.0

        log.info("  Fold %d — R²: %.4f  Spearman ρ: %.4f", fold + 1, r2, spearman_rho)
        fold_r2.append(r2)
        fold_spearman.append(spearman_rho)

        if use_mlflow:
            mlflow.log_metric(f"fold_{fold + 1}_r2", r2, step=fold)
            mlflow.log_metric(f"fold_{fold + 1}_spearman", spearman_rho, step=fold)

    mean_r2 = sum(fold_r2) / len(fold_r2)
    mean_spearman = sum(fold_spearman) / len(fold_spearman)

    log.info("5-fold CV — mean R²: %.4f  mean Spearman ρ: %.4f", mean_r2, mean_spearman)
    if mean_spearman > 0.30:
        log.info("Spearman > 0.30: model is capturing selective constraint on heteroplasmy.")

    metrics = {
        "mean_r2": mean_r2,
        "mean_spearman": mean_spearman,
        "fold_r2": fold_r2,
        "fold_spearman": fold_spearman,
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    if use_mlflow:
        mlflow.log_metrics({"mean_r2": mean_r2, "mean_spearman": mean_spearman})
        mlflow.end_run()

    # Save final model trained on all data
    final_base = MtDNAModel(base_encoder.config)
    final_base.load_state_dict(base_encoder.state_dict())
    final_model = MtDNAForHeteroplasmyRegression(
        final_base,
        dropout=cfg.get("dropout", 0.1),
        huber_delta=cfg.get("huber_delta", 0.1),
    )
    if cfg.get("use_lora", True):
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.get("lora_r", 4),
            lora_alpha=cfg.get("lora_alpha", 8),
            target_modules=cfg.get("lora_target_modules", ["query", "key", "value", "dense"]),
            lora_dropout=cfg.get("lora_dropout", 0.1),
            bias="none",
        )
        final_model = get_peft_model(final_model, lora_cfg)
    _save_model(final_model, output_dir, cfg, vocabulary)

    log.info("Fine-tuning complete. Output: %s", output_dir)
    log.info("Mean R²: %.4f  Mean Spearman ρ: %.4f", mean_r2, mean_spearman)


def _save_model(model, output_dir: Path, cfg: dict, vocabulary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # For PEFT models, save adapter + config
        model.save_pretrained(str(output_dir))
    except Exception as e:
        log.warning("save_pretrained failed (%s), saving state_dict instead", e)
        torch.save(model.state_dict(), output_dir / "model.pt")

    vocabulary.save_pretrained(str(output_dir))
    (output_dir / "finetune_config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}, indent=2)
    )


# ── CLI entry point ────────────────────────────────────────────────────────────


@app.command()
def main(
    task: str = typer.Option(..., help="Task: haplogroup | pathogenicity | heteroplasmy"),
    config: str = typer.Option(..., help="Path to fine-tuning config YAML"),
) -> None:
    """Fine-tune mtDNA-FM on a downstream task."""
    cfg_path = Path(config)
    if not cfg_path.exists():
        typer.echo(f"[finetune] Config not found: {config}", err=True)
        raise typer.Exit(code=1)

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    typer.echo(f"[finetune] Task: {task}")
    typer.echo(f"[finetune] Config: {config}")

    if task == "haplogroup":
        finetune_haplogroup(cfg)
    elif task == "pathogenicity":
        finetune_pathogenicity(cfg)
    elif task == "heteroplasmy":
        finetune_heteroplasmy(cfg)
    else:
        typer.echo(f"[finetune] Unknown task: {task}. Choose: haplogroup | pathogenicity | heteroplasmy", err=True)
        raise typer.Exit(code=1)
