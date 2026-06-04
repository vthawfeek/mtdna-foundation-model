"""
Generate a publication-quality architecture figure for mtDNA-FM.

Usage:
    uv run python scripts/generate_architecture_figure.py
    uv run python scripts/generate_architecture_figure.py --output-dir paper/manuscript/figures/

Outputs:
    docs/figures/architecture.png
    docs/figures/architecture.pdf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PALETTE = {
    "input_bg": "#EBF5FB",
    "input_border": "#2166AC",
    "token_bg": "#D4E6F1",
    "token_border": "#1A5276",
    "kmer_bg": "#AED6F1",
    "kmer_border": "#2166AC",
    "circ_pe_bg": "#A9DFBF",
    "circ_pe_border": "#1E8449",
    "het_bg": "#FAD7A0",
    "het_border": "#D68910",
    "sum_bg": "#D5D8DC",
    "sum_border": "#717D7E",
    "embed_outer_bg": "#F8F9FA",
    "embed_outer_border": "#566573",
    "layer_outer_bg": "#F4F6F7",
    "layer_outer_border": "#2166AC",
    "attn_bg": "#D6EAF8",
    "attn_border": "#2166AC",
    "ffn_bg": "#D5F5E3",
    "ffn_border": "#1E8449",
    "norm_bg": "#FDFEFE",
    "norm_border": "#AAB7B8",
    "residual": "#808B96",
    "haplo_bg": "#2166AC",
    "haplo_light": "#D4E6F1",
    "patho_bg": "#C0392B",
    "patho_light": "#FADBD8",
    "het_reg_bg": "#1E8449",
    "het_reg_light": "#D5F5E3",
    "lora_bg": "#FEF9E7",
    "lora_border": "#D4AC0D",
    "arrow_main": "#1A252F",
    "dim_text": "#5D6D7E",
    "white": "white",
    "dark": "#1A252F",
}

BOX_W = 0.78   # standard box width in axes [0,1] coords
BOX_H = 0.065  # standard box height


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _box(
    ax,
    cx: float,
    cy: float,
    w: float,
    h: float,
    label: str,
    sublabel: str = "",
    bg: str = "#EBF5FB",
    border: str = "#2166AC",
    text_color: str = "#1A252F",
    fontsize: float = 8.5,
    linestyle: str = "solid",
    lw: float = 1.2,
    zorder: int = 2,
    bold: bool = False,
    boxstyle: str = "round,pad=0.03",
) -> tuple[float, float]:
    """Draw a rounded box centred at (cx, cy); return centre (cx, cy)."""
    patch = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=boxstyle,
        facecolor=bg,
        edgecolor=border,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    if sublabel:
        ax.text(cx, cy + h * 0.13, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight,
                zorder=zorder + 1, clip_on=False)
        ax.text(cx, cy - h * 0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=PALETTE["dim_text"],
                style="italic", zorder=zorder + 1, clip_on=False)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight,
                zorder=zorder + 1, clip_on=False)
    return cx, cy


def _arrow(
    ax,
    posA: tuple[float, float],
    posB: tuple[float, float],
    color: str = "#1A252F",
    lw: float = 1.2,
    arrowstyle: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
    mutation_scale: float = 10,
    zorder: int = 3,
) -> None:
    ax.annotate(
        "",
        xy=posB,
        xytext=posA,
        arrowprops=dict(
            arrowstyle=arrowstyle,
            color=color,
            lw=lw,
            connectionstyle=connectionstyle,
        ),
        zorder=zorder,
        annotation_clip=False,
    )


def _dim_label(ax, x: float, y: float, text: str, fontsize: float = 7.0) -> None:
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=PALETTE["dim_text"],
            style="italic", clip_on=False, zorder=4)


def _lora_badge(ax, cx: float, cy: float, text: str) -> None:
    patch = mpatches.FancyBboxPatch(
        (cx - 0.065, cy - 0.016), 0.13, 0.032,
        boxstyle="round,pad=0.01",
        facecolor=PALETTE["lora_bg"],
        edgecolor=PALETTE["lora_border"],
        linewidth=0.9,
        zorder=5,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=6.5, color="#7D6608", fontweight="bold",
            zorder=6, clip_on=False)


def _sum_circle(ax, cx: float, cy: float, r: float = 0.025) -> None:
    circle = plt.Circle((cx, cy), r, facecolor=PALETTE["sum_bg"],
                         edgecolor=PALETTE["sum_border"], linewidth=0.9, zorder=3)
    ax.add_patch(circle)
    ax.text(cx, cy, "⊕", ha="center", va="center",
            fontsize=9, color=PALETTE["dark"], zorder=4, clip_on=False)


def _section_label(ax, cx: float, cy: float, text: str) -> None:
    ax.text(cx, cy, text, ha="left", va="center",
            fontsize=9.5, color=PALETTE["dark"], fontweight="bold",
            clip_on=False, zorder=4)


# ---------------------------------------------------------------------------
# Panel (a): Input + Embedding
# ---------------------------------------------------------------------------

def draw_input_embedding_panel(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel label
    ax.text(0.03, 0.97, "(a)", fontsize=11, fontweight="bold",
            va="top", color=PALETTE["dark"], clip_on=False)
    ax.text(0.5, 0.97, "Input & Embedding", fontsize=10, fontweight="bold",
            ha="center", va="top", color=PALETTE["dark"], clip_on=False)

    cx = 0.5

    # ── Input sequence ──────────────────────────────────────────────────
    y_seq = 0.88
    _box(ax, cx, y_seq, BOX_W, BOX_H,
         "mtDNA sequence",
         "ATGCTAGCTGCTAGCT... (circular, 16,569 bp)",
         bg=PALETTE["input_bg"], border=PALETTE["input_border"],
         bold=True)

    # Arrow + label
    _arrow(ax, (cx, y_seq - BOX_H / 2), (cx, y_seq - BOX_H / 2 - 0.055))
    _dim_label(ax, cx + 0.22, y_seq - BOX_H / 2 - 0.028, "6-mer tokenization")

    # ── Token sequence ──────────────────────────────────────────────────
    y_tok = 0.77
    _box(ax, cx, y_tok, BOX_W, BOX_H,
         "[CLS]  [ATGCTA]  [TGCTAG]  ...  [SEP]",
         "4,102-token vocab · max 514 tokens (512 + CLS/SEP)",
         bg=PALETTE["token_bg"], border=PALETTE["token_border"])

    _arrow(ax, (cx, y_tok - BOX_H / 2), (cx, y_tok - BOX_H / 2 - 0.04))

    # ── MtDNAEmbeddings outer box ───────────────────────────────────────
    embed_top = 0.72
    embed_bot = 0.21
    embed_mid = (embed_top + embed_bot) / 2
    embed_h = embed_top - embed_bot

    outer = mpatches.FancyBboxPatch(
        (cx - BOX_W / 2 - 0.02, embed_bot),
        BOX_W + 0.04, embed_h,
        boxstyle="round,pad=0.02",
        facecolor=PALETTE["embed_outer_bg"],
        edgecolor=PALETTE["embed_outer_border"],
        linewidth=1.0,
        linestyle="dashed",
        zorder=1,
        clip_on=False,
    )
    ax.add_patch(outer)
    ax.text(cx, embed_top + 0.015, "MtDNAEmbeddings", ha="center", va="bottom",
            fontsize=8, color=PALETTE["embed_outer_border"], fontweight="bold",
            clip_on=False, zorder=2)

    # K-mer embedding
    y_kmer = 0.66
    _box(ax, cx, y_kmer, BOX_W - 0.04, BOX_H,
         "K-mer Embedding",
         "nn.Embedding(4102, 256)",
         bg=PALETTE["kmer_bg"], border=PALETTE["kmer_border"])

    # sum circle
    y_sum1 = 0.575
    _sum_circle(ax, cx, y_sum1)
    _arrow(ax, (cx, y_kmer - BOX_H / 2), (cx, y_sum1 + 0.025))

    # Circular PE
    y_cpe = 0.50
    _box(ax, cx, y_cpe, BOX_W - 0.04, BOX_H,
         "Circular Positional Encoding  ★",
         "Fixed buffer · angle = 2π×pos / 16,569",
         bg=PALETTE["circ_pe_bg"], border=PALETTE["circ_pe_border"])
    _arrow(ax, (cx, y_sum1 - 0.025), (cx, y_cpe + BOX_H / 2))
    # side note
    ax.text(0.02, y_cpe, "★ novel", ha="left", va="center",
            fontsize=6.5, color=PALETTE["circ_pe_border"], style="italic",
            clip_on=False, zorder=4)

    # sum circle
    y_sum2 = 0.425
    _sum_circle(ax, cx, y_sum2)
    _arrow(ax, (cx, y_cpe - BOX_H / 2), (cx, y_sum2 + 0.025))

    # Het projection (dashed = optional)
    y_het = 0.355
    _box(ax, cx, y_het, BOX_W - 0.04, BOX_H,
         "Heteroplasmy Projection  (optional)",
         "Linear(1→256) + LayerNorm",
         bg=PALETTE["het_bg"], border=PALETTE["het_border"],
         linestyle="dashed")
    _arrow(ax, (cx, y_sum2 - 0.025), (cx, y_het + BOX_H / 2))

    # sum → LayerNorm → Dropout
    y_sum3 = 0.278
    _sum_circle(ax, cx, y_sum3)
    _arrow(ax, (cx, y_het - BOX_H / 2), (cx, y_sum3 + 0.025))

    y_ln = 0.225
    _box(ax, cx, y_ln, BOX_W - 0.04, 0.055,
         "LayerNorm  +  Dropout(0.1)",
         bg=PALETTE["norm_bg"], border=PALETTE["norm_border"])
    _arrow(ax, (cx, y_sum3 - 0.025), (cx, y_ln + 0.028))

    # Output label
    _arrow(ax, (cx, y_ln - 0.028), (cx, 0.10))
    _dim_label(ax, cx + 0.26, 0.145, "(batch, 514, 256)")

    _box(ax, cx, 0.075, BOX_W, 0.055,
         "Token embeddings → Transformer Encoder",
         bg=PALETTE["input_bg"], border=PALETTE["input_border"], bold=True)


# ---------------------------------------------------------------------------
# Panel (b): Transformer Encoder
# ---------------------------------------------------------------------------

def draw_transformer_panel(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.97, "(b)", fontsize=11, fontweight="bold",
            va="top", color=PALETTE["dark"], clip_on=False)
    ax.text(0.5, 0.97, "Transformer Encoder", fontsize=10, fontweight="bold",
            ha="center", va="top", color=PALETTE["dark"], clip_on=False)

    cx = 0.5

    # Outer "×6 layer" dashed box
    layer_top = 0.92
    layer_bot = 0.27
    layer_h = layer_top - layer_bot

    outer = mpatches.FancyBboxPatch(
        (cx - 0.44, layer_bot), 0.88, layer_h,
        boxstyle="round,pad=0.02",
        facecolor=PALETTE["layer_outer_bg"],
        edgecolor=PALETTE["layer_outer_border"],
        linewidth=1.2,
        linestyle="dashed",
        zorder=1, clip_on=False,
    )
    ax.add_patch(outer)

    # "×6" badge top-right
    badge = mpatches.FancyBboxPatch(
        (0.76, 0.89), 0.14, 0.038,
        boxstyle="round,pad=0.01",
        facecolor=PALETTE["layer_outer_border"],
        edgecolor="none",
        zorder=5, clip_on=False,
    )
    ax.add_patch(badge)
    ax.text(0.83, 0.909, "× 6 layers", ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold",
            zorder=6, clip_on=False)

    # Input hidden states
    _arrow(ax, (cx, 0.99), (cx, layer_top - 0.005))
    _dim_label(ax, cx + 0.27, 0.965, "(batch, 514, 256)")

    # Pre-LN 1
    y_ln1 = 0.875
    _box(ax, cx, y_ln1, 0.5, 0.048,
         "LayerNorm  (pre-norm)",
         bg=PALETTE["norm_bg"], border=PALETTE["norm_border"])
    _arrow(ax, (cx, layer_top - 0.005), (cx, y_ln1 + 0.024))

    # Multi-head attention block
    y_attn = 0.775
    attn_h = 0.085
    _box(ax, cx, y_attn, 0.78, attn_h,
         "Multi-Head Self-Attention",
         "8 heads · head_dim = 32 · Q / K / V / O projections",
         bg=PALETTE["attn_bg"], border=PALETTE["attn_border"],
         fontsize=8.5, bold=True)
    _arrow(ax, (cx, y_ln1 - 0.024), (cx, y_attn + attn_h / 2))

    # Q K V O sub-boxes inside attention
    sub_y = y_attn - 0.005
    sub_h = 0.028
    sub_w = 0.13
    for i, lbl in enumerate(["Q", "K", "V", "O"]):
        sub_cx = 0.18 + i * 0.165
        p = mpatches.FancyBboxPatch(
            (sub_cx - sub_w / 2, sub_y - sub_h / 2), sub_w, sub_h,
            boxstyle="round,pad=0.01",
            facecolor="white", edgecolor=PALETTE["attn_border"],
            linewidth=0.8, zorder=4, clip_on=False,
        )
        ax.add_patch(p)
        ax.text(sub_cx, sub_y, lbl, ha="center", va="center",
                fontsize=7.5, color=PALETTE["attn_border"], fontweight="bold",
                zorder=5, clip_on=False)

    # Residual 1 (curved arc around right side)
    res1_x = 0.94
    ax.annotate(
        "", xy=(cx, y_attn - attn_h / 2 - 0.015),
        xytext=(cx, layer_top - 0.005),
        arrowprops=dict(
            arrowstyle="-|>",
            color=PALETTE["residual"],
            lw=1.0,
            connectionstyle="arc3,rad=-0.55",
        ),
        zorder=3, annotation_clip=False,
    )
    ax.text(res1_x + 0.01, (layer_top + y_attn - attn_h / 2) / 2,
            "residual", ha="left", va="center",
            fontsize=6.5, color=PALETTE["residual"], style="italic",
            clip_on=False, zorder=4)

    # Add node
    y_add1 = y_attn - attn_h / 2 - 0.015
    _sum_circle(ax, cx, y_add1, r=0.022)
    _arrow(ax, (cx, y_attn - attn_h / 2), (cx, y_add1 + 0.022))

    # Pre-LN 2
    y_ln2 = y_add1 - 0.065
    _box(ax, cx, y_ln2, 0.5, 0.048,
         "LayerNorm  (pre-norm)",
         bg=PALETTE["norm_bg"], border=PALETTE["norm_border"])
    _arrow(ax, (cx, y_add1 - 0.022), (cx, y_ln2 + 0.024))

    # FFN block
    y_ffn = y_ln2 - 0.1
    ffn_h = 0.085
    _box(ax, cx, y_ffn, 0.78, ffn_h,
         "Feed-Forward Network (FFN)",
         "Linear(256→1024) → GELU → Linear(1024→256)",
         bg=PALETTE["ffn_bg"], border=PALETTE["ffn_border"],
         fontsize=8.5, bold=True)
    _arrow(ax, (cx, y_ln2 - 0.024), (cx, y_ffn + ffn_h / 2))

    # Residual 2
    ax.annotate(
        "", xy=(cx, y_ffn - ffn_h / 2 - 0.015),
        xytext=(cx, y_add1 - 0.022),
        arrowprops=dict(
            arrowstyle="-|>",
            color=PALETTE["residual"],
            lw=1.0,
            connectionstyle="arc3,rad=-0.55",
        ),
        zorder=3, annotation_clip=False,
    )

    y_add2 = y_ffn - ffn_h / 2 - 0.015
    _sum_circle(ax, cx, y_add2, r=0.022)
    _arrow(ax, (cx, y_ffn - ffn_h / 2), (cx, y_add2 + 0.022))

    # Arrow out of layer block
    _arrow(ax, (cx, y_add2 - 0.022), (cx, layer_bot - 0.005))

    # Final LayerNorm (outside loop)
    y_final_ln = 0.225
    _box(ax, cx, y_final_ln, 0.6, 0.048,
         "Final LayerNorm",
         bg=PALETTE["norm_bg"], border=PALETTE["norm_border"], bold=True)
    _arrow(ax, (cx, layer_bot - 0.005), (cx, y_final_ln + 0.024))

    _arrow(ax, (cx, y_final_ln - 0.024), (cx, 0.135))
    _dim_label(ax, cx + 0.27, 0.17, "last_hidden_state\n(batch, 514, 256)")

    # Two output branches: CLS and variant token
    y_branch = 0.13
    y_out = 0.065
    bw = 0.31
    bh = 0.055

    # CLS branch
    cx_cls = 0.28
    ax.annotate("", xy=(cx_cls, y_branch), xytext=(cx, y_branch),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["arrow_main"], lw=1.0),
                zorder=3, annotation_clip=False)
    _box(ax, cx_cls, y_out, bw, bh,
         "[CLS] → pooler_output",
         "(batch, 256)",
         bg=PALETTE["kmer_bg"], border=PALETTE["kmer_border"], fontsize=7.5)
    _arrow(ax, (cx_cls, y_branch), (cx_cls, y_out + bh / 2))

    # Variant token branch
    cx_var = 0.72
    ax.annotate("", xy=(cx_var, y_branch), xytext=(cx, y_branch),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["arrow_main"], lw=1.0),
                zorder=3, annotation_clip=False)
    _box(ax, cx_var, y_out, bw, bh,
         "variant token hidden",
         "(batch, 256)",
         bg=PALETTE["het_bg"], border=PALETTE["het_border"], fontsize=7.5)
    _arrow(ax, (cx_var, y_branch), (cx_var, y_out + bh / 2))


# ---------------------------------------------------------------------------
# Panel (c): Fine-tuning Heads
# ---------------------------------------------------------------------------

def _task_head_block(
    ax,
    y_top: float,
    y_bot: float,
    title: str,
    lora_text: str,
    input_label: str,
    ops: list[str],
    output_label: str,
    header_bg: str,
    light_bg: str,
) -> None:
    """Draw one stacked task-head sub-panel."""
    cx = 0.5
    w = 0.88
    mid = (y_top + y_bot) / 2
    total_h = y_top - y_bot

    # Outer box
    outer = mpatches.FancyBboxPatch(
        (cx - w / 2, y_bot), w, total_h,
        boxstyle="round,pad=0.01",
        facecolor=light_bg, edgecolor=header_bg,
        linewidth=1.4, zorder=1, clip_on=False,
    )
    ax.add_patch(outer)

    # Header bar
    header_h = 0.055
    header = mpatches.FancyBboxPatch(
        (cx - w / 2, y_top - header_h), w, header_h,
        boxstyle="round,pad=0.01",
        facecolor=header_bg, edgecolor="none",
        linewidth=0, zorder=3, clip_on=False,
    )
    ax.add_patch(header)
    ax.text(cx - 0.03, y_top - header_h / 2, title,
            ha="center", va="center", fontsize=8.5, color="white",
            fontweight="bold", zorder=4, clip_on=False)

    # LoRA badge in header
    _lora_badge(ax, cx + 0.32, y_top - header_h / 2, lora_text)

    # Input label
    content_top = y_top - header_h - 0.01
    ax.text(cx, content_top - 0.018, input_label,
            ha="center", va="center", fontsize=7.5, color=PALETTE["dim_text"],
            style="italic", zorder=4, clip_on=False)

    # Operation boxes
    n = len(ops)
    spacing = (content_top - 0.04 - (y_bot + 0.02)) / n
    op_h = min(spacing * 0.6, 0.045)

    for i, op in enumerate(ops):
        oy = content_top - 0.04 - spacing * i - spacing / 2
        p = mpatches.FancyBboxPatch(
            (cx - 0.36, oy - op_h / 2), 0.72, op_h,
            boxstyle="round,pad=0.01",
            facecolor="white", edgecolor=header_bg,
            linewidth=0.8, zorder=3, clip_on=False,
        )
        ax.add_patch(p)
        ax.text(cx, oy, op, ha="center", va="center",
                fontsize=7.0, color=PALETTE["dark"], zorder=4, clip_on=False)
        if i < n - 1:
            _arrow(ax, (cx, oy - op_h / 2), (cx, oy - spacing + op_h / 2),
                   lw=0.9)

    # Output label
    ax.text(cx, y_bot + 0.018, output_label,
            ha="center", va="center", fontsize=7.0, color=PALETTE["dim_text"],
            style="italic", zorder=4, clip_on=False)


def draw_heads_panel(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.97, "(c)", fontsize=11, fontweight="bold",
            va="top", color=PALETTE["dark"], clip_on=False)
    ax.text(0.5, 0.97, "Fine-tuning Heads  (LoRA adapters)", fontsize=10,
            fontweight="bold", ha="center", va="top", color=PALETTE["dark"],
            clip_on=False)

    # Haplogroup
    _task_head_block(
        ax,
        y_top=0.92, y_bot=0.65,
        title="Haplogroup Classification",
        lora_text="LoRA r=8",
        input_label="CLS pooler output  (batch, 256)",
        ops=[
            "Dropout(0.1)",
            "Linear(256 → 26)",
            "Softmax → 26-class logits",
        ],
        output_label="Output: haplogroup label  (L0–L6, M, N, R, H, J, ...)",
        header_bg=PALETTE["haplo_bg"],
        light_bg=PALETTE["haplo_light"],
    )

    # Pathogenicity
    _task_head_block(
        ax,
        y_top=0.62, y_bot=0.34,
        title="Variant Pathogenicity",
        lora_text="LoRA r=4",
        input_label="variant-token hidden state  (batch, 256)",
        ops=[
            "Linear(256 → 1)",
            "Sigmoid → P(pathogenic)",
        ],
        output_label="Training: ClinVar (+) vs gnomAD AF>0.01 (−)  ·  BCE pos_weight=2.5",
        header_bg=PALETTE["patho_bg"],
        light_bg=PALETTE["patho_light"],
    )

    # Heteroplasmy regression
    _task_head_block(
        ax,
        y_top=0.31, y_bot=0.03,
        title="Heteroplasmy Regression",
        lora_text="LoRA r=4",
        input_label="variant-token hidden state  (batch, 256)",
        ops=[
            "Linear(256 → 64)  →  GELU",
            "Linear(64 → 1)  →  Sigmoid",
        ],
        output_label="Output: mean het level ∈ [0, 1]  ·  Huber loss (δ=0.1)  ·  5-fold CV",
        header_bg=PALETTE["het_reg_bg"],
        light_bg=PALETTE["het_reg_light"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mtDNA-FM architecture figure")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"),
                        help="Output directory (default: docs/figures)")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig = plt.figure(figsize=(18, 12))

    # Overall title
    fig.suptitle(
        "mtDNA-FM: Model Architecture",
        fontsize=15, fontweight="bold", y=0.995, color=PALETTE["dark"],
    )

    # Footer note
    fig.text(
        0.5, 0.002,
        "~6M trainable parameters  ·  Circular genome topology  ·  "
        "Heteroplasmy channel  ·  Pre-LayerNorm  ·  LoRA fine-tuning",
        ha="center", fontsize=8, color=PALETTE["dim_text"],
    )

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, figure=fig,
                  left=0.02, right=0.98, bottom=0.03, top=0.96,
                  wspace=0.10,
                  width_ratios=[3, 3, 4])

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    draw_input_embedding_panel(ax_a)
    draw_transformer_panel(ax_b)
    draw_heads_panel(ax_c)

    # Cross-panel arrows on a full-figure overlay axes
    fig.canvas.draw()
    bbox_a = ax_a.get_position()
    bbox_b = ax_b.get_position()
    bbox_c = ax_c.get_position()

    fig_ax = fig.add_axes([0, 0, 1, 1], zorder=10)
    fig_ax.set_xlim(0, 1)
    fig_ax.set_ylim(0, 1)
    fig_ax.axis("off")
    fig_ax.patch.set_alpha(0)

    mid_y_ab = (bbox_a.y0 + bbox_a.y1) / 2
    mid_y_bc = (bbox_b.y0 + bbox_b.y1) / 2

    # Panel a → Panel b
    fig_ax.annotate(
        "",
        xy=(bbox_b.x0 + 0.005, mid_y_ab),
        xytext=(bbox_a.x1 - 0.005, mid_y_ab),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["arrow_main"], lw=2.0,
                        mutation_scale=14),
        annotation_clip=False,
    )

    # Panel b → Panel c
    fig_ax.annotate(
        "",
        xy=(bbox_c.x0 + 0.005, mid_y_bc),
        xytext=(bbox_b.x1 - 0.005, mid_y_bc),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["arrow_main"], lw=2.0,
                        mutation_scale=14),
        annotation_clip=False,
    )

    # Save
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = output_dir / f"architecture.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", path)

    plt.close(fig)


if __name__ == "__main__":
    main()
