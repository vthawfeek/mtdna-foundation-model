"""
Haplogroup classification evaluation.

Computes accuracy, macro-F1, per-haplogroup precision/recall/F1, and a
confusion matrix from model predictions on a labelled haplogroup dataset.
Results are returned as a plain dict so callers can serialise to JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# ── mtDNA haplogroup phylogenetic ordering ─────────────────────────────────────
# L0 → L1 → L2 → L3 → M (East-Asian) and N → R → H/HV (European)
HAPLOGROUP_ORDER = [
    "L0", "L1", "L2", "L3", "L4", "L5", "L6",
    "M", "C", "D", "E", "G", "Q", "Z",
    "N", "A", "I", "O", "S", "W", "X", "Y",
    "R", "B", "F", "P",
    "HV", "H", "V", "T", "J", "U", "K",
]


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """
    Compute haplogroup classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth integer class indices.
    y_pred:
        Predicted integer class indices.
    label_names:
        Human-readable label for each class index.  Used only for labelling
        the per-class breakdown in the output dict.

    Returns
    -------
    dict with keys:
        accuracy       – overall top-1 accuracy
        macro_f1       – unweighted macro-averaged F1
        per_class      – list of dicts {label, precision, recall, f1, support}
        confusion_matrix – 2-D list (num_classes × num_classes), rows=true cols=pred
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    n = len(y_true)
    num_classes = max(y_true.max(), y_pred.max()) + 1

    # Overall accuracy
    accuracy = float((y_true == y_pred).sum() / n)

    # Per-class precision, recall, F1
    per_class: list[dict] = []
    f1_scores: list[float] = []

    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = int((y_true == c).sum())

        name = label_names[c] if label_names is not None else str(c)
        per_class.append(
            {
                "label": name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        )
        if support > 0:
            f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    # Confusion matrix: rows = true class, cols = predicted class
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_predictions(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """Alias for compute_metrics for use in the evaluate CLI."""
    return compute_metrics(y_true, y_pred, label_names)
