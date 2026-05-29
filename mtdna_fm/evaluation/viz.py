"""
Visualisation utilities for mtDNA-FM evaluation.

All functions return matplotlib Figure objects so callers can either show
interactively or save to disk.  No display() calls here — keep side-effect-free.

Four visualisations:
    plot_umap           – 2-D UMAP of genome embeddings coloured by haplogroup
    plot_roc_curve      – ROC curve with AUROC annotation
    plot_confusion_matrix – Labelled confusion matrix heatmap
    plot_attention_heatmap – Per-layer attention weight heatmap for a sequence
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure


# ── Colour palette for haplogroup clades ──────────────────────────────────────
# Groups by major clade so related haplogroups share a hue family.
CLADE_COLOURS: dict[str, str] = {
    # L clade (African root)
    "L0": "#8B0000", "L1": "#A52A2A", "L2": "#CD5C5C",
    "L3": "#DC143C", "L4": "#FF6347", "L5": "#FF7F50", "L6": "#FFA07A",
    # M clade (East Asian)
    "M": "#006400", "C": "#228B22", "D": "#32CD32",
    "E": "#7CFC00", "G": "#90EE90", "Q": "#ADFF2F", "Z": "#98FB98",
    # N clade
    "N": "#00008B", "A": "#0000CD", "I": "#4169E1",
    "O": "#6495ED", "S": "#87CEEB", "W": "#4682B4", "X": "#1E90FF", "Y": "#00BFFF",
    # R clade (European/Asian)
    "R": "#8B008B", "B": "#9932CC", "F": "#BA55D3", "P": "#DDA0DD",
    # HV / H subclade (European)
    "HV": "#FF8C00", "H": "#FFA500", "V": "#FFD700",
    # JT clade
    "T": "#FF1493", "J": "#FF69B4",
    # U / K clade
    "U": "#2E8B57", "K": "#3CB371",
}

_DEFAULT_COLOUR = "#999999"


def _haplogroup_colour(label: str) -> str:
    """Return a colour for a haplogroup label, using CLADE_COLOURS or a default."""
    # Exact match first, then prefix match (e.g. H1 → H colour)
    if label in CLADE_COLOURS:
        return CLADE_COLOURS[label]
    for prefix in sorted(CLADE_COLOURS, key=len, reverse=True):
        if label.startswith(prefix):
            return CLADE_COLOURS[prefix]
    return _DEFAULT_COLOUR


def plot_umap(
    embeddings: np.ndarray,
    labels: list[str],
    title: str = "Genome embeddings (UMAP)",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> matplotlib.figure.Figure:
    """
    Compute 2-D UMAP of genome embeddings and colour points by haplogroup.

    Parameters
    ----------
    embeddings:
        (n_samples, hidden_size) numpy array of genome embedding vectors.
    labels:
        List of haplogroup string labels, one per sample.
    title:
        Figure title.
    n_neighbors, min_dist, random_state:
        UMAP hyperparameters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        from umap import UMAP
    except ImportError as e:
        raise ImportError("umap-learn is required for plot_umap. Install with: pip install umap-learn") from e

    import matplotlib.pyplot as plt

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)  # (n, 2)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each haplogroup as a distinct series for legend
    unique_labels = sorted(set(labels))
    for hg in unique_labels:
        mask = np.array([lbl == hg for lbl in labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=_haplogroup_colour(hg),
            label=hg,
            s=10,
            alpha=0.7,
            linewidths=0,
        )

    # Legend: only show if ≤30 distinct labels, otherwise skip to avoid clutter
    if len(unique_labels) <= 30:
        ax.legend(
            markerscale=2,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            ncol=2,
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_roc_curve(
    fpr: list[float] | np.ndarray,
    tpr: list[float] | np.ndarray,
    auroc: float,
    title: str = "ROC Curve – Variant Pathogenicity",
) -> matplotlib.figure.Figure:
    """
    Plot ROC curve with AUROC annotation and random-classifier diagonal.

    Parameters
    ----------
    fpr, tpr:
        False-positive and true-positive rate arrays (from variant_eval).
    auroc:
        Pre-computed AUROC scalar (labelled on the plot).
    """
    import matplotlib.pyplot as plt

    fpr_arr = np.asarray(fpr)
    tpr_arr = np.asarray(tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, lw=2, color="#2563EB", label=f"mtDNA-FM (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--", color="#9CA3AF", label="Random (0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    label_names: list[str] | None = None,
    title: str = "Haplogroup Confusion Matrix",
    normalise: bool = True,
) -> matplotlib.figure.Figure:
    """
    Plot a labelled confusion matrix heatmap.

    Parameters
    ----------
    cm:
        (n_classes × n_classes) confusion matrix; rows = true, cols = pred.
    label_names:
        Class name for each row/column index.
    normalise:
        If True, normalise rows to [0, 1] (shows recall per class).
    """
    import matplotlib.pyplot as plt

    cm_arr = np.asarray(cm, dtype=float)
    n = cm_arr.shape[0]

    if normalise:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm_arr / row_sums
    else:
        cm_display = cm_arr

    # Scale figure height/width with number of classes, cap for readability
    cell_size = min(0.6, 12 / max(n, 1))
    fig_size = max(6, n * cell_size)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalise else None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if label_names is not None:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=max(6, 10 - n // 4))
        ax.set_yticklabels(label_names, fontsize=max(6, 10 - n // 4))

    ax.set_xlabel("Predicted haplogroup")
    ax.set_ylabel("True haplogroup")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_attention_heatmap(
    attentions: np.ndarray | list,
    position_labels: list[int] | None = None,
    title: str = "Attention weights – pathogenic variant",
    n_positions: int = 64,
) -> matplotlib.figure.Figure:
    """
    Plot a per-layer attention weight heatmap for a single sequence.

    Averages attention across heads within each layer and shows a
    (n_layers × n_positions) heatmap so the reader can see which genomic
    positions each layer attends to most strongly.

    Parameters
    ----------
    attentions:
        Sequence of attention tensors, one per layer.  Each tensor should be
        (1, n_heads, seq_len, seq_len) or (n_heads, seq_len, seq_len).
        If a PyTorch tensor, it will be converted to numpy automatically.
    position_labels:
        Genomic position labels for the x-axis.  Defaults to 0, 1, 2, ...
    title:
        Figure title.
    n_positions:
        Number of positions to display (first n_positions tokens).
    """
    import matplotlib.pyplot as plt

    # Convert to numpy if needed
    layers_avg: list[np.ndarray] = []
    for layer_attn in attentions:
        try:
            arr = layer_attn.detach().cpu().numpy()
        except AttributeError:
            arr = np.asarray(layer_attn)

        # Remove batch dim if present
        if arr.ndim == 4:
            arr = arr[0]  # (n_heads, seq_len, seq_len)

        # Mean over heads, then mean-pool query dim → (seq_len,) = what gets attended to
        mean_over_heads = arr.mean(axis=0)         # (seq_len, seq_len)
        attended_to = mean_over_heads.mean(axis=0) # (seq_len,)
        layers_avg.append(attended_to)

    # Stack: (n_layers, seq_len)
    heatmap = np.stack(layers_avg, axis=0)[:, :n_positions]
    n_layers = heatmap.shape[0]
    n_pos = heatmap.shape[1]

    fig_w = max(10, n_pos * 0.15)
    fig_h = max(3, n_layers * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(heatmap, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i+1}" for i in range(n_layers)], fontsize=9)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Transformer layer")
    ax.set_title(title, fontsize=12)

    if position_labels is not None and len(position_labels) >= n_pos:
        tick_step = max(1, n_pos // 10)
        tick_positions = list(range(0, n_pos, tick_step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [str(position_labels[i]) for i in tick_positions],
            rotation=45,
            ha="right",
            fontsize=8,
        )

    fig.tight_layout()
    return fig
