"""
Variant pathogenicity evaluation.

Computes AUROC, AUPRC, and per-variant-type breakdowns from model probability
scores on a labelled variant dataset.  Results are returned as a plain dict
so callers can serialise to JSON.

Variant types tracked:
    missense  – protein-coding gene, single amino-acid change
    tRNA      – tRNA gene variants (MT-T*)
    rRNA      – rRNA gene variants (MT-RNR1, MT-RNR2)
    d_loop    – D-loop control region (bp 16024-576, 1-based)
    other     – anything else (non-coding, unknown annotation)
"""

from __future__ import annotations

import numpy as np

# mtDNA gene boundaries (1-based, approximate rCRS coordinates)
# tRNA genes: all MT-T* genes
TRNA_RANGES: list[tuple[int, int]] = [
    (577, 647),    # MT-TF
    (1602, 1670),  # MT-TV
    (3230, 3304),  # MT-TL1
    (4263, 4331),  # MT-TI
    (4329, 4400),  # MT-TQ  (overlaps MT-TI on opposite strand)
    (4402, 4469),  # MT-TM
    (5512, 5579),  # MT-TW
    (5587, 5655),  # MT-TA
    (5657, 5729),  # MT-TN
    (5761, 5826),  # MT-TC
    (5826, 5891),  # MT-TY
    (7518, 7585),  # MT-TS1
    (7586, 7647),  # MT-TD
    (8295, 8364),  # MT-TK
    (9991, 10058), # MT-TG
    (10405, 10469),# MT-TR
    (10470, 10534),# MT-TH
    (10651, 10725),# MT-TS2
    (10766, 10837),# MT-TL2
    (11742, 11816),# MT-TE
    (14674, 14742),# MT-TT
    (15888, 15953),# MT-TP
]

RRNA_RANGES: list[tuple[int, int]] = [
    (648, 1601),   # MT-RNR1 (12S rRNA)
    (1671, 3229),  # MT-RNR2 (16S rRNA)
]

# Protein-coding gene ranges (approximate)
PROTEIN_CODING_RANGES: list[tuple[int, int]] = [
    (3307, 4262),   # MT-ND1
    (4470, 5511),   # MT-ND2
    (5904, 7445),   # MT-CO1
    (7586, 8269),   # MT-CO2  (approximate, overlaps tRNA)
    (8366, 8572),   # MT-ATP8
    (8527, 9207),   # MT-ATP6
    (9207, 9990),   # MT-CO3
    (10059, 10404), # MT-ND3
    (10470, 10766), # MT-ND4L (approximate)
    (10760, 12137), # MT-ND4
    (12337, 14148), # MT-ND5
    (14149, 14673), # MT-ND6 (complement strand)
    (14747, 15887), # MT-CYB
]

# D-loop: 16024–16569 + 1–576 (wraps around origin)
DLOOP_RANGES: list[tuple[int, int]] = [
    (16024, 16569),
    (1, 576),
]


def _classify_variant_type(pos_1based: int) -> str:
    """Return variant type string for a 1-based genomic position."""
    p = pos_1based
    for start, end in TRNA_RANGES:
        if start <= p <= end:
            return "tRNA"
    for start, end in RRNA_RANGES:
        if start <= p <= end:
            return "rRNA"
    for start, end in DLOOP_RANGES:
        if start <= p <= end:
            return "d_loop"
    for start, end in PROTEIN_CODING_RANGES:
        if start <= p <= end:
            return "missense"
    return "other"


def _trapz_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal-rule AUC; handles unsorted input by sorting on fpr."""
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def _precision_recall_auc(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AUPRC (area under the precision-recall curve).
    Returns (auprc, precision_array, recall_array).
    """
    thresholds = np.sort(np.unique(y_score))[::-1]
    precisions = []
    recalls = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    precisions_arr = np.array(precisions)
    recalls_arr = np.array(recalls)
    # Sort by recall ascending for integration
    order = np.argsort(recalls_arr)
    auprc = float(np.trapezoid(precisions_arr[order], recalls_arr[order]))
    return auprc, precisions_arr, recalls_arr


def _roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays for the ROC curve."""
    thresholds = np.sort(np.unique(y_score))[::-1]
    fprs = []
    tprs = []
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        fprs.append(fpr)
        tprs.append(tpr)
    return np.array(fprs), np.array(tprs)


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
    positions: list[int] | None = None,
) -> dict:
    """
    Compute variant pathogenicity evaluation metrics.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (1=pathogenic, 0=benign).
    y_score:
        Model probability scores in [0, 1].
    positions:
        Optional 1-based genomic positions for per-variant-type breakdown.

    Returns
    -------
    dict with keys:
        auroc          – area under the ROC curve
        auprc          – area under the precision-recall curve
        n_positive     – number of pathogenic variants
        n_negative     – number of benign variants
        per_type       – dict mapping variant type → {auroc, auprc, n_pos, n_neg}
                         (only populated when `positions` is supplied)
        roc_curve      – {fpr: list, tpr: list} (sampled at ≤200 points)
        pr_curve       – {precision: list, recall: list} (sampled at ≤200 points)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())

    fpr, tpr = _roc_curve(y_true, y_score)
    auroc = _trapz_auc(fpr, tpr)
    auprc, prec, rec = _precision_recall_auc(y_true, y_score)

    # Downsample curves to ≤200 points for JSON storage
    def _downsample(arr: np.ndarray, n: int = 200) -> list:
        if len(arr) <= n:
            return arr.tolist()
        idx = np.linspace(0, len(arr) - 1, n, dtype=int)
        return arr[idx].tolist()

    result: dict = {
        "auroc": round(float(auroc), 4),
        "auprc": round(float(auprc), 4),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "roc_curve": {
            "fpr": _downsample(np.sort(fpr)),
            "tpr": _downsample(np.sort(tpr)),
        },
        "pr_curve": {
            "precision": _downsample(prec),
            "recall": _downsample(rec),
        },
        "per_type": {},
    }

    if positions is not None:
        variant_types = [_classify_variant_type(p) for p in positions]
        for vtype in ["missense", "tRNA", "rRNA", "d_loop", "other"]:
            mask = np.array([t == vtype for t in variant_types])
            if mask.sum() < 2:
                continue
            yt = y_true[mask]
            ys = y_score[mask]
            n_p = int(yt.sum())
            n_n = int((1 - yt).sum())
            if n_p == 0 or n_n == 0:
                result["per_type"][vtype] = {
                    "auroc": None,
                    "auprc": None,
                    "n_pos": n_p,
                    "n_neg": n_n,
                }
                continue
            fpr_t, tpr_t = _roc_curve(yt, ys)
            auroc_t = _trapz_auc(fpr_t, tpr_t)
            auprc_t, _, _ = _precision_recall_auc(yt, ys)
            result["per_type"][vtype] = {
                "auroc": round(float(auroc_t), 4),
                "auprc": round(float(auprc_t), 4),
                "n_pos": n_p,
                "n_neg": n_n,
            }

    return result


def evaluate_predictions(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
    positions: list[int] | None = None,
) -> dict:
    """Alias for compute_metrics for use in the evaluate CLI."""
    return compute_metrics(y_true, y_score, positions)
