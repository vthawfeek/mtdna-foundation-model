"""
Tests for mtdna_fm/evaluation/.

Covers haplogroup_eval, variant_eval, and viz utilities.
No real model or external data required — all tests use synthetic inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── TestHaplogroupMetrics ──────────────────────────────────────────────────────


class TestHaplogroupMetrics:
    def _preds(self, n_classes=4, n_per_class=10, accuracy=0.9, seed=0):
        rng = np.random.default_rng(seed)
        n = n_classes * n_per_class
        y_true = np.repeat(np.arange(n_classes), n_per_class)
        y_pred = y_true.copy()
        n_corrupt = int(n * (1.0 - accuracy))
        idx = rng.choice(n, size=n_corrupt, replace=False)
        y_pred[idx] = rng.integers(0, n_classes, size=n_corrupt)
        return y_true, y_pred

    def test_perfect_accuracy(self):
        from mtdna_fm.evaluation.haplogroup_eval import compute_metrics

        y = np.array([0, 1, 2, 3])
        metrics = compute_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_accuracy_range(self):
        from mtdna_fm.evaluation.haplogroup_eval import compute_metrics

        y_true, y_pred = self._preds(accuracy=0.7)
        metrics = compute_metrics(y_true, y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_macro_f1_range(self):
        from mtdna_fm.evaluation.haplogroup_eval import compute_metrics

        y_true, y_pred = self._preds()
        metrics = compute_metrics(y_true, y_pred)
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_confusion_matrix_shape(self):
        from mtdna_fm.evaluation.haplogroup_eval import compute_metrics

        n_classes = 5
        y_true, y_pred = self._preds(n_classes=n_classes)
        metrics = compute_metrics(y_true, y_pred)
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (n_classes, n_classes)

    def test_confusion_matrix_row_sums(self):
        """Each row should sum to the number of true instances for that class."""
        from mtdna_fm.evaluation.haplogroup_eval import compute_metrics

        n_classes = 4
        n_per = 10
        y_true = np.repeat(np.arange(n_classes), n_per)
        metrics = compute_metrics(y_true, y_true)  # perfect predictions
        cm = np.array(metrics["confusion_matrix"])
        for c in range(n_classes):
            assert cm[c].sum() == n_per

    def test_per_class_length(self):
        from mtdna_fm.evaluation.haplogroup_eval import compute_metrics

        n_classes = 6
        y_true, y_pred = self._preds(n_classes=n_classes)
        metrics = compute_metrics(y_true, y_pred)
        assert len(metrics["per_class"]) == n_classes

    def test_per_class_keys(self):
        from mtdna_fm.evaluation.haplogroup_eval import compute_metrics

        y_true, y_pred = self._preds()
        metrics = compute_metrics(y_true, y_pred)
        for entry in metrics["per_class"]:
            for key in ("label", "precision", "recall", "f1", "support"):
                assert key in entry

    def test_label_names_applied(self):
        from mtdna_fm.evaluation.haplogroup_eval import compute_metrics

        label_names = ["L0", "H", "M", "N"]
        y = np.array([0, 1, 2, 3])
        metrics = compute_metrics(y, y, label_names)
        labels = [e["label"] for e in metrics["per_class"]]
        assert labels == label_names

    def test_zero_predictions_graceful(self):
        """All predictions wrong should not crash."""
        from mtdna_fm.evaluation.haplogroup_eval import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_evaluate_predictions_alias(self):
        from mtdna_fm.evaluation.haplogroup_eval import evaluate_predictions

        y = np.array([0, 1, 2])
        result = evaluate_predictions(y, y)
        assert "accuracy" in result


# ── TestVariantMetrics ─────────────────────────────────────────────────────────


class TestVariantMetrics:
    def _data(self, n=200, auroc_target=0.75, seed=0):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, size=n)
        # Create scores correlated with labels to hit target AUROC roughly
        shift = (auroc_target - 0.5) * 2.0
        y_score = np.where(
            y_true == 1,
            rng.normal(0.5 + shift * 0.25, 0.2, size=n).clip(0, 1),
            rng.normal(0.5 - shift * 0.25, 0.2, size=n).clip(0, 1),
        )
        return y_true, y_score

    def test_auroc_range(self):
        from mtdna_fm.evaluation.variant_eval import compute_metrics

        y_true, y_score = self._data()
        metrics = compute_metrics(y_true, y_score)
        assert 0.0 <= metrics["auroc"] <= 1.0

    def test_auprc_range(self):
        from mtdna_fm.evaluation.variant_eval import compute_metrics

        y_true, y_score = self._data()
        metrics = compute_metrics(y_true, y_score)
        assert 0.0 <= metrics["auprc"] <= 1.0

    def test_perfect_auroc(self):
        from mtdna_fm.evaluation.variant_eval import compute_metrics

        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        metrics = compute_metrics(y_true, y_score)
        assert metrics["auroc"] == pytest.approx(1.0, abs=0.01)

    def test_n_positive_negative(self):
        from mtdna_fm.evaluation.variant_eval import compute_metrics

        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.2, 0.3, 0.6, 0.7, 0.8])
        metrics = compute_metrics(y_true, y_score)
        assert metrics["n_positive"] == 3
        assert metrics["n_negative"] == 2

    def test_roc_curve_in_output(self):
        from mtdna_fm.evaluation.variant_eval import compute_metrics

        y_true, y_score = self._data()
        metrics = compute_metrics(y_true, y_score)
        assert "roc_curve" in metrics
        assert "fpr" in metrics["roc_curve"]
        assert "tpr" in metrics["roc_curve"]

    def test_pr_curve_in_output(self):
        from mtdna_fm.evaluation.variant_eval import compute_metrics

        y_true, y_score = self._data()
        metrics = compute_metrics(y_true, y_score)
        assert "pr_curve" in metrics

    def test_per_type_populated_with_positions(self):
        from mtdna_fm.evaluation.variant_eval import compute_metrics

        rng = np.random.default_rng(7)
        n = 300
        y_true = rng.integers(0, 2, size=n)
        y_score = rng.uniform(0, 1, size=n)
        # Include positions in protein-coding range (3307–4262)
        positions = [3500 + i % 100 for i in range(n)]
        metrics = compute_metrics(y_true, y_score, positions)
        # missense type should appear since positions are in coding range
        assert "missense" in metrics["per_type"]

    def test_per_type_empty_without_positions(self):
        from mtdna_fm.evaluation.variant_eval import compute_metrics

        y_true, y_score = self._data()
        metrics = compute_metrics(y_true, y_score, positions=None)
        assert metrics["per_type"] == {}

    def test_variant_type_classifier_trna(self):
        from mtdna_fm.evaluation.variant_eval import _classify_variant_type

        assert _classify_variant_type(600) == "tRNA"

    def test_variant_type_classifier_rrna(self):
        from mtdna_fm.evaluation.variant_eval import _classify_variant_type

        assert _classify_variant_type(700) == "rRNA"

    def test_variant_type_classifier_dloop(self):
        from mtdna_fm.evaluation.variant_eval import _classify_variant_type

        assert _classify_variant_type(16100) == "d_loop"

    def test_variant_type_classifier_missense(self):
        from mtdna_fm.evaluation.variant_eval import _classify_variant_type

        assert _classify_variant_type(3500) == "missense"

    def test_evaluate_predictions_alias(self):
        from mtdna_fm.evaluation.variant_eval import evaluate_predictions

        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.2, 0.8])
        result = evaluate_predictions(y_true, y_score)
        assert "auroc" in result


# ── TestViz ────────────────────────────────────────────────────────────────────


class TestViz:
    """Tests for viz.py — check each function returns a Figure without error."""

    def test_plot_roc_curve(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from mtdna_fm.evaluation.viz import plot_roc_curve

        fpr = [0.0, 0.2, 0.5, 1.0]
        tpr = [0.0, 0.6, 0.8, 1.0]
        fig = plot_roc_curve(fpr, tpr, auroc=0.72)
        assert fig is not None
        plt.close(fig)

    def test_plot_confusion_matrix_normalised(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from mtdna_fm.evaluation.viz import plot_confusion_matrix

        cm = [[10, 2], [1, 9]]
        fig = plot_confusion_matrix(cm, normalise=True)
        assert fig is not None
        plt.close(fig)

    def test_plot_confusion_matrix_unnormalised(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from mtdna_fm.evaluation.viz import plot_confusion_matrix

        cm = [[5, 1, 0], [0, 4, 2], [1, 0, 6]]
        fig = plot_confusion_matrix(cm, normalise=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_confusion_matrix_with_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from mtdna_fm.evaluation.viz import plot_confusion_matrix

        cm = [[8, 2], [1, 7]]
        fig = plot_confusion_matrix(cm, label_names=["L0", "H"])
        assert fig is not None
        plt.close(fig)

    def test_plot_attention_heatmap_numpy(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from mtdna_fm.evaluation.viz import plot_attention_heatmap

        n_layers, n_heads, seq_len = 4, 4, 32
        rng = np.random.default_rng(1)
        attentions = [
            rng.random((1, n_heads, seq_len, seq_len)) for _ in range(n_layers)
        ]
        fig = plot_attention_heatmap(attentions, n_positions=32)
        assert fig is not None
        plt.close(fig)

    def test_plot_attention_heatmap_torch(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch

        from mtdna_fm.evaluation.viz import plot_attention_heatmap

        n_layers, n_heads, seq_len = 3, 8, 16
        rng = np.random.default_rng(2)
        attentions = [
            torch.tensor(rng.random((1, n_heads, seq_len, seq_len)), dtype=torch.float32)
            for _ in range(n_layers)
        ]
        fig = plot_attention_heatmap(attentions, n_positions=16)
        assert fig is not None
        plt.close(fig)

    def test_plot_attention_heatmap_with_position_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from mtdna_fm.evaluation.viz import plot_attention_heatmap

        n_layers = 2
        seq_len = 20
        rng = np.random.default_rng(3)
        attentions = [rng.random((n_layers, 4, seq_len, seq_len))]
        pos_labels = list(range(1000, 1000 + seq_len))
        fig = plot_attention_heatmap(attentions, position_labels=pos_labels, n_positions=seq_len)
        assert fig is not None
        plt.close(fig)


# ── TestEvaluationInit ─────────────────────────────────────────────────────────


class TestEvaluationInit:
    def test_public_imports(self):
        from mtdna_fm.evaluation import (
            haplogroup_metrics,
            plot_attention_heatmap,
            plot_confusion_matrix,
            plot_roc_curve,
            variant_metrics,
        )
        assert callable(haplogroup_metrics)
        assert callable(variant_metrics)
        assert callable(plot_roc_curve)
        assert callable(plot_confusion_matrix)
        assert callable(plot_attention_heatmap)
