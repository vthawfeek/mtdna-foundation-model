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
    elif task in ("pathogenicity", "heteroplasmy"):
        typer.echo(f"[finetune] Task '{task}' will be implemented on Day 17/18.", err=True)
        raise typer.Exit(code=1)
    else:
        typer.echo(f"[finetune] Unknown task: {task}. Choose: haplogroup | pathogenicity | heteroplasmy", err=True)
        raise typer.Exit(code=1)
