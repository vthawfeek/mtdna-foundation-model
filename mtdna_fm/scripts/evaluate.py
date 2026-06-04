"""
mtdna-evaluate: run all mtDNA-FM evaluations and write reports/eval_summary.json.

Usage:
    mtdna-evaluate --model <checkpoint_dir> [--output-dir reports]

The command:
  1. Loads the fine-tuned haplogroup classifier from <checkpoint_dir> (PEFT adapter).
  2. Evaluates haplogroup accuracy on data/processed/test.parquet.
     Requires test.parquet to have a 'major_haplogroup' column — run
     fix_haplogroup_labels.py first if it is missing.
  3. Evaluates variant pathogenicity if a labeled test parquet exists at
     data/processed/variants_pathogenicity_test.parquet; skips with a clear
     message otherwise.  Never silently substitutes synthetic numbers.
  4. Writes metrics to <output_dir>/eval_summary.json with source="real".
  5. Saves ROC curve figure to <output_dir>/ when variant data is available.

This is the DVC-trackable metric entrypoint: dvc repro → metrics show.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from mtdna_fm.evaluation.haplogroup_eval import compute_metrics as haplogroup_metrics
from mtdna_fm.evaluation.variant_eval import compute_metrics as variant_metrics

app = typer.Typer(help="Evaluate a trained mtDNA-FM model and write metrics to reports/.")

# Must match HaplogroupWindowDataset.HAPLOGROUPS in finetune.py
_HAPLOGROUPS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "HV", "I",
    "J", "K", "L0", "L1", "L2", "L3", "L4", "L5", "M",
    "N", "R", "T", "U", "V", "W", "X",
]
_LABEL2IDX: dict[str, int] = {h: i for i, h in enumerate(_HAPLOGROUPS)}


def _map_to_major(hap: str | None) -> str | None:
    """Map a detailed sub-haplogroup label to the nearest major haplogroup."""
    if not hap or not isinstance(hap, str):
        return None
    hap = hap.strip()
    # L0–L5 are their own major groups; L6+ not in our 26
    if hap.startswith("L") and len(hap) > 1 and hap[1].isdigit():
        clade = "L" + hap[1]
        return clade if clade in _LABEL2IDX else None
    # HV must be checked before H (it starts with H)
    if hap.startswith("HV"):
        return "HV"
    first = hap[0].upper()
    return first if first in _LABEL2IDX else None


# ── Evaluation helpers ─────────────────────────────────────────────────────────


def _real_haplogroup_eval(
    model_path: str,
    test_parquet: str = "data/processed/test.parquet",
) -> dict:
    """
    Evaluate the fine-tuned haplogroup classifier on the real labeled test set.

    Expects:
      - <model_path>/adapter_config.json  pointing to a base model
      - <model_path>/adapter_model.safetensors  LoRA adapter weights
      - test.parquet to have a 'major_haplogroup' column
        (run fix_haplogroup_labels.py first)
    """
    import json as _json

    import torch
    from torch.utils.data import DataLoader

    from mtdna_fm.model.model import MtDNAForHaplogroupClassification, MtDNAForMaskedModeling
    from mtdna_fm.scripts.finetune import HaplogroupWindowDataset
    from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

    model_path = Path(model_path)
    test_parquet_path = Path(test_parquet)

    if not test_parquet_path.exists():
        raise FileNotFoundError(f"Test parquet not found: {test_parquet_path}")

    import pandas as pd

    df_check = pd.read_parquet(test_parquet_path)
    if "major_haplogroup" not in df_check.columns:
        raise ValueError(
            f"'major_haplogroup' column missing from {test_parquet_path}.\n"
            "Run: uv run python paper/experiments/evaluation/fix_haplogroup_labels.py"
        )

    adapter_cfg_path = model_path / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {model_path}. "
            "Expected a PEFT LoRA adapter checkpoint from mtdna-finetune."
        )
    with open(adapter_cfg_path) as f:
        adapter_cfg = _json.load(f)
    base_path = adapter_cfg.get("base_model_name_or_path", "models/phase1_v1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"[evaluate] Loading base model from {base_path}")
    full_model = MtDNAForMaskedModeling.from_pretrained(base_path)
    vocabulary = KmerVocabulary.from_pretrained(base_path)

    clf = MtDNAForHaplogroupClassification(full_model.mtdna, num_labels=26)

    from peft import PeftModel

    typer.echo(f"[evaluate] Loading LoRA adapter from {model_path}")
    clf = PeftModel.from_pretrained(clf, str(model_path))
    clf = clf.to(device)
    clf.eval()

    typer.echo(f"[evaluate] Building test dataset from {test_parquet_path}")
    test_ds = HaplogroupWindowDataset(
        test_parquet_path,
        vocabulary,
        window_size=512,
        stride=256,
        label_column="major_haplogroup",
    )
    if len(test_ds) == 0:
        raise ValueError(
            "Test dataset is empty after filtering to major_haplogroup labels. "
            "Ensure fix_haplogroup_labels.py has been run on test.parquet."
        )
    typer.echo(f"[evaluate] Test dataset: {len(test_ds)} windows from "
               f"{df_check['major_haplogroup'].notna().sum()} labeled sequences")

    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(loader, desc="Haplogroup inference", unit="batch"):
            out = clf(
                input_ids=batch["input_ids"].to(device),
                position_ids=batch["position_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = out.logits.argmax(dim=-1).cpu()
            y_true_all.extend(batch["labels"].tolist())
            y_pred_all.extend(preds.tolist())

    typer.echo(f"[evaluate] Haplogroup inference: {len(y_true_all)} windows evaluated.")
    return haplogroup_metrics(y_true_all, y_pred_all, _HAPLOGROUPS)


def _real_variant_eval(
    model_path: str,
    test_parquet: str = "data/processed/variants_pathogenicity_test.parquet",
) -> dict:
    """
    Evaluate the fine-tuned variant pathogenicity model on the real labeled test set.

    Expects test_parquet with columns: sequence, position, label (1=pathogenic, 0=benign).
    """
    import json as _json

    import torch
    from torch.utils.data import DataLoader

    from mtdna_fm.model.model import MtDNAForMaskedModeling, MtDNAForVariantPathogenicity
    from mtdna_fm.scripts.finetune import PathogenicityVariantDataset
    from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

    model_path = Path(model_path)
    test_parquet_path = Path(test_parquet)

    if not test_parquet_path.exists():
        raise FileNotFoundError(
            f"Variant test parquet not found: {test_parquet_path}\n"
            "Run prepare_variant_data.py to create labeled variant evaluation data."
        )

    adapter_cfg_path = model_path / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json in {model_path}. "
            "Expected a PEFT LoRA adapter checkpoint from mtdna-finetune."
        )
    with open(adapter_cfg_path) as f:
        adapter_cfg = _json.load(f)
    base_path = adapter_cfg.get("base_model_name_or_path", "models/phase1_v1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model = MtDNAForMaskedModeling.from_pretrained(base_path)
    vocabulary = KmerVocabulary.from_pretrained(base_path)

    var_model = MtDNAForVariantPathogenicity(full_model.mtdna)

    from peft import PeftModel

    var_model = PeftModel.from_pretrained(var_model, str(model_path))
    var_model = var_model.to(device)
    var_model.eval()

    test_ds = PathogenicityVariantDataset(test_parquet_path, vocabulary)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    y_true_all: list[int] = []
    y_score_all: list[float] = []
    positions_all: list[int] = []

    with torch.no_grad():
        for batch in loader:
            out = var_model(
                input_ids=batch["input_ids"].to(device),
                position_ids=batch["position_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                variant_positions=batch["variant_position"].to(device),
            )
            scores = torch.sigmoid(out.logits.squeeze(-1)).cpu().tolist()
            y_true_all.extend(batch["labels"].tolist())
            y_score_all.extend(scores if isinstance(scores, list) else [scores])
            if "position" in batch:
                positions_all.extend(batch["position"].tolist())

    return variant_metrics(y_true_all, y_score_all, positions_all or None)


# ── CLI ────────────────────────────────────────────────────────────────────────


@app.command()
def main(
    model: str = typer.Option(..., help="Path to fine-tuned model checkpoint directory"),
    output_dir: str = typer.Option("reports", help="Output directory for metrics and figures"),
    test_parquet: str = typer.Option(
        "data/processed/test.parquet",
        help="Path to labeled haplogroup test parquet",
    ),
    variant_test_parquet: str = typer.Option(
        "data/processed/variants_pathogenicity_test.parquet",
        help="Path to labeled variant pathogenicity test parquet",
    ),
) -> None:
    """Run evaluation suite and write eval_summary.json."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "model": model,
        "source": "real",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "haplogroup": {},
        "variant_pathogenicity": {},
    }

    model_path = Path(model)
    if not model_path.exists():
        typer.echo(f"[evaluate] Model path '{model}' does not exist.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[evaluate] Running haplogroup evaluation on {test_parquet} …")
    try:
        hap_metrics = _real_haplogroup_eval(model, test_parquet)
    except Exception as exc:
        typer.echo(f"[evaluate] Haplogroup evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"[evaluate] Checking for variant data at {variant_test_parquet} …")
    var_metrics_data = None
    variant_parquet_path = Path(variant_test_parquet)
    if variant_parquet_path.exists():
        try:
            var_metrics_data = _real_variant_eval(model, variant_test_parquet)
        except Exception as exc:
            typer.echo(
                f"[evaluate] Variant pathogenicity evaluation skipped: {exc}", err=True
            )
    else:
        typer.echo(
            "[evaluate] Variant pathogenicity evaluation skipped: no labeled variant "
            f"test dataset found at {variant_test_parquet}. "
            "Run prepare_variant_data.py to create one."
        )

    # Build summary
    hap_summary = {k: v for k, v in hap_metrics.items() if k != "confusion_matrix"}
    hap_summary["confusion_matrix_shape"] = [
        len(hap_metrics["confusion_matrix"]),
        len(hap_metrics["confusion_matrix"][0]) if hap_metrics["confusion_matrix"] else 0,
    ]
    summary["haplogroup"] = hap_summary

    if var_metrics_data is not None:
        summary["variant_pathogenicity"] = {
            "auroc": var_metrics_data["auroc"],
            "auprc": var_metrics_data["auprc"],
            "n_positive": var_metrics_data["n_positive"],
            "n_negative": var_metrics_data["n_negative"],
            "per_type": var_metrics_data["per_type"],
        }
    else:
        summary["variant_pathogenicity"] = {
            "auroc": None,
            "auprc": None,
            "note": (
                "No labeled variant test dataset available. "
                "Run prepare_variant_data.py to enable real variant evaluation."
            ),
        }

    # Write eval_summary.json
    summary_path = out_path / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    typer.echo(f"[evaluate] Wrote {summary_path}")

    # Write detailed metrics JSON files
    hap_detail_path = out_path / "eval_haplogroup_detail.json"
    with open(hap_detail_path, "w") as f:
        json.dump(hap_metrics, f, indent=2)

    if var_metrics_data is not None:
        var_detail_path = out_path / "eval_variant_detail.json"
        with open(var_detail_path, "w") as f:
            json.dump(var_metrics_data, f, indent=2)

    # Print summary table
    typer.echo("\n── Evaluation Summary ─────────────────────────")
    typer.echo(f"  Haplogroup accuracy : {hap_metrics['accuracy']:.4f}")
    typer.echo(f"  Haplogroup macro-F1 : {hap_metrics['macro_f1']:.4f}")
    if var_metrics_data is not None:
        typer.echo(f"  Variant AUROC       : {var_metrics_data['auroc']:.4f}")
        typer.echo(f"  Variant AUPRC       : {var_metrics_data['auprc']:.4f}")
    else:
        typer.echo("  Variant AUROC       : N/A (no labeled variant data)")
    typer.echo("───────────────────────────────────────────────\n")

    # ROC curve (only when variant data is available)
    if var_metrics_data is not None:
        try:
            from mtdna_fm.evaluation.viz import plot_roc_curve

            fig = plot_roc_curve(
                var_metrics_data["roc_curve"]["fpr"],
                var_metrics_data["roc_curve"]["tpr"],
                var_metrics_data["auroc"],
            )
            roc_path = out_path / "eval_roc_curve.png"
            fig.savefig(roc_path, dpi=120, bbox_inches="tight")
            typer.echo(f"[evaluate] Saved ROC curve → {roc_path}")
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
        except Exception as exc:
            typer.echo(f"[evaluate] ROC figure skipped: {exc}")

    typer.echo("[evaluate] Done.")
