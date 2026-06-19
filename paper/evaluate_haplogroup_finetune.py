"""
Evaluate the best fine-tuned haplogroup checkpoint on the held-out test set.

Run after mtdna-finetune completes:
    python paper/evaluate_haplogroup_finetune.py \
        --checkpoint models/finetune_haplogroup_gpu/best \
        --test-parquet data/processed/test.parquet \
        --output reports/finetune_haplogroup_gpu.json

The output JSON is read by paper/update_paper_finetune.py to fill in
Table 1 and the fine-tuning paragraph in main.tex.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

MAJOR_26 = [
    "A", "B", "C", "D", "E", "F", "G", "H", "HV", "I",
    "J", "K", "L0", "L1", "L2", "L3", "L4", "L5", "M",
    "N", "R", "T", "U", "V", "W", "X",
]


def evaluate(checkpoint_dir: str, test_parquet: str, output_path: str) -> None:
    from peft import PeftModel
    from mtdna_fm.model.model import MtDNAForHaplogroupClassification, MtDNAForMaskedModeling, MtDNAModel
    from mtdna_fm.tokenizer.vocabulary import KmerVocabulary
    from mtdna_fm.scripts.finetune import HaplogroupWindowDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = Path(checkpoint_dir)
    vocab_dir = ckpt

    print(f"Loading vocabulary from {vocab_dir}")
    vocabulary = KmerVocabulary.from_pretrained(str(vocab_dir))

    print(f"Loading model from {ckpt}")
    # Load config to get num_labels
    cfg_path = ckpt / "finetune_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    num_labels = cfg.get("num_labels", 26)

    # Re-build model architecture and load LoRA weights via PEFT
    base_mlm = MtDNAForMaskedModeling.from_pretrained(str(ckpt))
    base_encoder: MtDNAModel = base_mlm.mtdna
    classifier = MtDNAForHaplogroupClassification(base_encoder, num_labels=num_labels)
    model = PeftModel.from_pretrained(classifier, str(ckpt))
    model = model.to(device)
    model.eval()

    print(f"Building test dataset from {test_parquet}")
    test_ds = HaplogroupWindowDataset(test_parquet, vocabulary, label_column="major_haplogroup")
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    print(f"  {len(test_ds):,} test windows")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_dl:
            out = model(
                input_ids=batch["input_ids"].to(device),
                position_ids=batch["position_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    report = classification_report(
        all_labels, all_preds,
        target_names=MAJOR_26,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = report["macro avg"]["f1-score"]
    per_class_f1 = {cls: report[cls]["f1-score"] for cls in MAJOR_26 if cls in report}
    per_class_precision = {cls: report[cls]["precision"] for cls in MAJOR_26 if cls in report}
    per_class_recall = {cls: report[cls]["recall"] for cls in MAJOR_26 if cls in report}
    per_class_n = {cls: report[cls]["support"] for cls in MAJOR_26 if cls in report}

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(MAJOR_26)))).tolist()

    results = {
        "test_accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_n": per_class_n,
        "confusion_matrix": cm,
        "n_test_windows": len(test_ds),
        "n_classes": len(MAJOR_26),
        "checkpoint": str(ckpt),
        "device": str(device),
        "random_baseline": round(1 / len(MAJOR_26), 4),
    }
    results["lift"] = round(accuracy / results["random_baseline"], 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2))

    print(f"\nTest accuracy : {accuracy:.1%}")
    print(f"Macro-F1      : {macro_f1:.4f}")
    print(f"Lift over random ({results['random_baseline']:.2%}): {results['lift']}x")
    print(f"\nResults saved to {output_path}")
    print("\nPer-class F1:")
    for cls in MAJOR_26:
        n = int(per_class_n.get(cls, 0))
        f1 = per_class_f1.get(cls, 0.0)
        print(f"  {cls:<4}  F1={f1:.3f}  n={n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/finetune_haplogroup_gpu/best")
    parser.add_argument("--test-parquet", default="data/processed/test.parquet")
    parser.add_argument("--output", default="reports/finetune_haplogroup_gpu.json")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.test_parquet, args.output)
