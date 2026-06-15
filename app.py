"""
mtDNA-FM Gradio Demo

Two-tab interface for the mtDNA Foundation Model:
  1. Haplogroup Classification — predict which of 26 major mtDNA haplogroups a sequence belongs to
  2. Genome Embedding          — embed a sequence and place it on a reference UMAP

Models loaded from HuggingFace Hub on first use, then cached for the session lifetime.
All inference is CPU-safe; no batching is used.
"""

from __future__ import annotations

import sys
import types

# pydub (a Gradio 4.x dependency) tries to import audioop, which was removed
# in Python 3.13. Mock it here so the import doesn't crash; we use no audio
# components, so audioop is never called at runtime.
if sys.version_info >= (3, 13):
    _audioop = types.ModuleType("audioop")
    sys.modules.setdefault("audioop", _audioop)
    sys.modules.setdefault("pyaudioop", _audioop)

from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

# ── Constants ──────────────────────────────────────────────────────────────────

HAPLOGROUPS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "HV", "I",
    "J", "K", "L0", "L1", "L2", "L3", "L4", "L5", "M",
    "N", "R", "T", "U", "V", "W", "X",
]

HAPLOGROUP_INFO: dict[str, dict] = {
    "A": {
        "region": "Americas & East Asia",
        "age": "~40,000 years ago",
        "notes": "One of four founding lineages of Native Americans. Also common in Siberian and some East Asian populations. A2 is the predominant sublineage in Indigenous Americans.",
    },
    "B": {
        "region": "Americas & Southeast Asia",
        "age": "~40,000 years ago",
        "notes": "Defines one of the major founding lineages of the Americas. B2 is found across North and South America; B4 is widespread in Southeast Asia and Oceania.",
    },
    "C": {
        "region": "Americas & Northeast Asia",
        "age": "~40,000 years ago",
        "notes": "American founding lineage also found in Siberia and Northeast Asia. C1 sublineages are common in Indigenous American populations.",
    },
    "D": {
        "region": "East Asia & Americas",
        "age": "~40,000 years ago",
        "notes": "Widespread across East Asia. D4 is especially common in Japanese and Korean populations. D1 is a founding lineage of the Americas.",
    },
    "E": {
        "region": "Southeast Asia",
        "age": "~30,000 years ago",
        "notes": "Common in the Philippines and Indonesia. E1 sublineages are associated with Austronesian expansion across the Pacific.",
    },
    "F": {
        "region": "East & Southeast Asia",
        "age": "~50,000 years ago",
        "notes": "Widespread in Southeast Asia and southern China. F1 is common in Vietnam and South China; F3 in Japan.",
    },
    "G": {
        "region": "East Asia",
        "age": "~40,000 years ago",
        "notes": "Present in East Asia and Central Asia. High frequency in some Tibetan and indigenous Siberian groups. Possible association with high-altitude adaptation.",
    },
    "H": {
        "region": "Europe & Middle East",
        "age": "~20,000 years ago",
        "notes": (
            "Most common haplogroup in Europeans (~40%). Strongly associated with post-glacial "
            "re-expansion from an Iberian refugium. H subgroups appear in studies of LHON "
            "(Leber hereditary optic neuropathy) susceptibility — some H backgrounds increase risk."
        ),
    },
    "HV": {
        "region": "Middle East & Europe",
        "age": "~25,000 years ago",
        "notes": "Ancestral to H and V. Common in Iran and the Arabian peninsula. Likely the major haplogroup of Neolithic farmers who spread into Europe from the Near East.",
    },
    "I": {
        "region": "Europe & Middle East",
        "age": "~25,000 years ago",
        "notes": "Low frequency across Europe. Possibly linked to Neolithic migration from the Middle East into northern Europe.",
    },
    "J": {
        "region": "Middle East & Europe",
        "age": "~45,000 years ago",
        "notes": (
            "Overrepresented in European centenarians — a longevity association that has been "
            "replicated across cohorts, possibly through J-specific differences in ATP synthesis "
            "efficiency. J1c2 was identified in Ötzi the Iceman's maternal relatives."
        ),
    },
    "K": {
        "region": "Europe & Middle East",
        "age": "~40,000 years ago",
        "notes": "Common in Europe; technically a branch of U8. K1a sublineage was identified in Ötzi the Iceman — meaning ~5,300 years ago this haplogroup was in the Alps.",
    },
    "L0": {
        "region": "Sub-Saharan Africa",
        "age": "~150,000 years ago",
        "notes": "The oldest haplogroup — the root of the entire human mitochondrial phylogeny. Highest frequency in Khoisan (San) peoples of southern Africa. All other human haplogroups are descendants of L0 or its siblings.",
    },
    "L1": {
        "region": "Sub-Saharan Africa",
        "age": "~150,000 years ago",
        "notes": "Basal African lineage; common in Central and West Africa. L1b is found in the Sandawe people of Tanzania.",
    },
    "L2": {
        "region": "Sub-Saharan Africa",
        "age": "~60,000 years ago",
        "notes": "Most common haplogroup in sub-Saharan Africa overall. Very frequent in West Africa. L2a was carried to the Americas via the slave trade and is now the most common haplogroup in African Americans.",
    },
    "L3": {
        "region": "Africa & worldwide",
        "age": "~70,000 years ago",
        "notes": (
            "Ancestral to ALL non-African haplogroups (M and N arose from L3). "
            "This is the 'Out of Africa' lineage — a single L3 individual's descendant population "
            "left Africa roughly 60-70 thousand years ago and populated the rest of the world."
        ),
    },
    "L4": {
        "region": "East Africa",
        "age": "~70,000 years ago",
        "notes": "Rare; primarily found in East Africa, especially Ethiopia and Tanzania. Close phylogenetic neighbor of L3.",
    },
    "L5": {
        "region": "East Africa",
        "age": "~100,000 years ago",
        "notes": "Rare; primarily found in Ethiopia. One of the deepest-branching African lineages.",
    },
    "M": {
        "region": "South & East Asia",
        "age": "~60,000 years ago",
        "notes": "One of two macrohaplogroups outside Africa (the other is N). Ancestral to D, G, C, Z, and many South/East Asian lineages. Arose independently from L3 at approximately the same time as N.",
    },
    "N": {
        "region": "Worldwide (non-African)",
        "age": "~60,000 years ago",
        "notes": "The other major macrohaplogroup outside Africa. Ancestral to A, I, W, X, and R (which in turn is ancestral to H, HV, J, T, U, B, F). The most phylogenetically diverse haplogroup.",
    },
    "R": {
        "region": "Europe, Middle East, South Asia",
        "age": "~60,000 years ago",
        "notes": "A branch of N. Ancestral to B, F, HV, H, V, J, T, U, and K — collectively most Europeans and South Asians trace their mtDNA through R.",
    },
    "T": {
        "region": "Middle East & Europe",
        "age": "~45,000 years ago",
        "notes": "Common in the Middle East and eastern Mediterranean. T2 was common in Neolithic European farmers — it peaked in early Neolithic populations and has declined since.",
    },
    "U": {
        "region": "Europe, Middle East, South Asia",
        "age": "~55,000 years ago",
        "notes": "U5 is the oldest haplogroup found in ancient European hunter-gatherers (predating Neolithic farming). K is a branch of U8. U has the broadest geographic range of any haplogroup.",
    },
    "V": {
        "region": "Europe",
        "age": "~14,000 years ago",
        "notes": "High frequency in the Sámi people and Basques. Expanded rapidly across northern Europe as glaciers retreated after the Last Glacial Maximum. Young haplogroup by mtDNA standards.",
    },
    "W": {
        "region": "South Asia & Europe",
        "age": "~25,000 years ago",
        "notes": "Low frequency across South Asia, the Middle East, and Europe. Relatively rare; notable for deep phylogenetic placement within N despite geographic distribution in the West.",
    },
    "X": {
        "region": "Middle East, Europe, Americas",
        "age": "~30,000 years ago",
        "notes": (
            "Rare; found in the Middle East, Europe, and notably in Native Americans (Great Lakes region, especially Ojibwe). "
            "X2a in North America predates Columbus and has been used in debates about pre-Columbian trans-Atlantic contact, "
            "though the current consensus is that it arrived via the Beringian route."
        ),
    },
}

CLADE_COLOURS: dict[str, str] = {
    # L clade (African root)
    "L0": "#8B0000", "L1": "#A52A2A", "L2": "#CD5C5C",
    "L3": "#DC143C", "L4": "#FF6347", "L5": "#FF7F50",
    # M clade (East Asian)
    "M": "#006400", "C": "#228B22", "D": "#32CD32",
    "E": "#7CFC00", "G": "#90EE90",
    # N clade
    "N": "#00008B", "A": "#0000CD", "I": "#4169E1",
    "W": "#4682B4", "X": "#1E90FF",
    # R clade
    "R": "#8B008B", "B": "#9932CC", "F": "#BA55D3",
    # HV / H subclade (European)
    "HV": "#FF8C00", "H": "#FFA500", "V": "#FFD700",
    # JT clade
    "T": "#FF1493", "J": "#FF69B4",
    # U / K clade
    "U": "#2E8B57", "K": "#3CB371",
}

BASE_MODEL = "vthawfeek/mtdna-foundation-model"
HAPLO_ADAPTER = "vthawfeek/mtdna-fm-haplogroup"

# mtDNA gene intervals (1-based, rCRS NC_012920.1)
_TRNA_INTERVALS = [
    (577, 647), (1602, 1670), (3230, 3304), (4263, 4331), (4329, 4400),
    (4402, 4469), (5512, 5579), (5587, 5655), (5657, 5729), (5761, 5826),
    (5826, 5891), (7446, 7514), (7518, 7585), (8295, 8364), (9991, 10058),
    (10405, 10469), (12138, 12206), (12207, 12265), (12266, 12336),
    (14674, 14742), (15888, 15953), (15956, 16023),
]
_RRNA_INTERVALS = [(648, 1601), (1671, 3229)]
_CODING_INTERVALS = [
    (3307, 4262), (4470, 5511), (5904, 7445), (7586, 8269), (8366, 8572),
    (8527, 9207), (9207, 9990), (10059, 10404), (10470, 10766), (10760, 12137),
    (12337, 14148), (14149, 14673), (14747, 15887),
]

_VARIANT_TYPE_NOTES: dict[str, tuple[str, str]] = {
    "protein_coding": (
        "Protein-coding gene",
        "AUROC 0.727 (n=56 pathogenic in reference set). Reliable.",
    ),
    "tRNA": (
        "tRNA gene",
        "AUROC 0.718 (n=44 pathogenic), AUPRC 0.773. Reliable.",
    ),
    "rRNA": (
        "rRNA gene (12S or 16S)",
        "AUROC 0.639 (n=5 pathogenic). Low statistical power — treat as indicative only.",
    ),
    "d_loop": (
        "D-loop / control region",
        "Insufficient pathogenic variants in reference set (n=1). Result unreliable for this region.",
    ),
    "other": (
        "Intergenic / other",
        "Unreliable — reference set has inverted class balance for this category.",
    ),
}


def _get_variant_type(pos_1based: int) -> str:
    """Classify a 1-based rCRS position into a functional category."""
    if pos_1based >= 16024 or pos_1based <= 576:
        return "d_loop"
    for s, e in _TRNA_INTERVALS:
        if s <= pos_1based <= e:
            return "tRNA"
    for s, e in _RRNA_INTERVALS:
        if s <= pos_1based <= e:
            return "rRNA"
    for s, e in _CODING_INTERVALS:
        if s <= pos_1based <= e:
            return "protein_coding"
    return "other"

# rCRS reference snippet (first 240 bp of human mtDNA, NC_012920.1)
EXAMPLE_SEQUENCE = (
    "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTT"
    "CGTCTGGGGGGTGTGCACGCGATAGCATTGCGAAAACTTGCTTCTCAAATACTTGGCATT"
    "ATCCCTGGCCTATGCTAGCCCTCCATCAACCCCAGCCTGGCCTCTTACTTCAAAGCACAC"
    "AACAAGACTTCCACTTCAAAATACAGCCCCAATCAACCCAAGTTTTTCAAACATTAAACAA"
    * 70  # repeat to get closer to a full genome length for demo
)[:16569]  # truncate to genome length

# ── Global model cache ─────────────────────────────────────────────────────────

_models: dict[str, Any] = {}
_reference_data: dict[str, Any] | None = None
_patho_reference_data: dict[str, Any] | None = None


def _haplogroup_colour(label: str) -> str:
    for prefix in sorted(CLADE_COLOURS, key=len, reverse=True):
        if label.upper().startswith(prefix.upper()):
            return CLADE_COLOURS[prefix]
    return "#999999"


def _parse_sequence(text: str) -> str:
    """Parse raw sequence or FASTA-format input into an uppercase DNA string."""
    text = text.strip()
    if text.startswith(">"):
        lines = text.split("\n")
        seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    else:
        seq = "".join(text.split())
    return seq.upper()


def _load_models() -> None:
    """Lazy-load model variants into the global cache."""
    if "embedder" in _models:
        return

    from peft import PeftModel

    from mtdna_fm.inference.api import MtDNAEmbedder
    from mtdna_fm.model.model import (
        MtDNAForHaplogroupClassification,
        MtDNAForMaskedModeling,
        MtDNAModel,
    )

    # PEFT compat: newer PEFT injects inputs_embeds into forward kwargs
    import functools as _ft
    def _peft_patch(cls):
        _orig = cls.forward
        @_ft.wraps(_orig)
        def forward(self, *args, inputs_embeds=None, **kwargs):
            return _orig(self, *args, **kwargs)
        cls.forward = forward
    _peft_patch(MtDNAForHaplogroupClassification)

    from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

    print("[mtdna-fm] Loading base model from HuggingFace Hub...")
    full_model = MtDNAForMaskedModeling.from_pretrained(BASE_MODEL)
    base_for_embedder: MtDNAModel = full_model.mtdna
    from huggingface_hub import hf_hub_download as _hf_download

    _vocab_file = _hf_download(repo_id=BASE_MODEL, filename="vocab.json")
    vocab = KmerVocabulary.from_pretrained(Path(_vocab_file).parent)

    _models["embedder"] = MtDNAEmbedder(base_for_embedder, vocab)
    _models["vocab"] = vocab

    print("[mtdna-fm] Loading haplogroup adapter...")
    haplo_base = MtDNAModel.from_pretrained(BASE_MODEL)
    haplo_model = MtDNAForHaplogroupClassification(haplo_base, num_labels=26)
    haplo_model = PeftModel.from_pretrained(haplo_model, HAPLO_ADAPTER)
    haplo_model.eval()
    _models["haplogroup"] = haplo_model

    print("[mtdna-fm] All models ready.")


def _load_reference() -> dict:
    """Load pre-computed reference UMAP embeddings and 2D coordinates."""
    global _reference_data
    if _reference_data is not None:
        return _reference_data

    ref_path = Path("app_reference.npz")
    if ref_path.exists():
        data = np.load(ref_path, allow_pickle=True)
        _reference_data = {
            "embeddings": data["embeddings"].astype(np.float32),
            "labels": data["labels"].tolist(),
            "umap_2d": data["umap_2d"].astype(np.float32),
        }
    else:
        _reference_data = {"embeddings": None, "labels": [], "umap_2d": None}
    return _reference_data


def _load_patho_reference() -> dict:
    """Load pre-computed pathogenicity k-NN reference embeddings and rCRS."""
    global _patho_reference_data
    if _patho_reference_data is not None:
        return _patho_reference_data

    ref_path = Path("app_patho_reference.npz")
    if ref_path.exists():
        data = np.load(ref_path, allow_pickle=False)
        _patho_reference_data = {
            "X": data["X"].astype(np.float32),
            "y": data["y"].astype(np.int32),
            "positions": data["positions"].astype(np.int32),
            "rcrs": data["rcrs"].tobytes().decode("ascii"),
        }
    else:
        _patho_reference_data = {"X": None, "y": None, "positions": None, "rcrs": None}
    return _patho_reference_data


def _project_query(
    query_emb: np.ndarray,
    ref_embs: np.ndarray,
    ref_2d: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    Project a query embedding into pre-computed 2D reference space using
    inverse-distance-weighted interpolation from the k nearest neighbours.
    """
    dists = np.linalg.norm(ref_embs - query_emb[None, :], axis=1)
    top_k = np.argsort(dists)[:k]
    top_dists = dists[top_k]
    if top_dists[0] < 1e-8:
        return ref_2d[top_k[0]]
    weights = 1.0 / top_dists
    weights /= weights.sum()
    return (ref_2d[top_k] * weights[:, None]).sum(axis=0)


# ── Tab 1: Haplogroup Classification ──────────────────────────────────────────

def _tokenize_for_haplogroup(seq: str, vocab, genome_length: int):
    """Tokenize a sequence into overlapping windows for haplogroup classification."""
    from mtdna_fm.tokenizer.tokenize import tokenize_sequence

    if len(seq) > genome_length:
        seq = seq[:genome_length]
    elif len(seq) < genome_length:
        seq = seq + "N" * (genome_length - len(seq))

    tokens = tokenize_sequence(
        seq, vocab, k=6, stride=1, max_seq_len=len(seq), circular=True
    )
    n_tokens = len(tokens["input_ids"])
    window_size, stride = 512, 256

    all_ids, all_pos = [], []
    for start in range(0, n_tokens, stride):
        widx = [(start + i) % n_tokens for i in range(window_size)]
        all_ids.append([tokens["input_ids"][j] for j in widx])
        all_pos.append([tokens["position_ids"][j] for j in widx])

    return all_ids, all_pos


def predict_haplogroup(sequence_input: str) -> tuple[Any, str]:
    """Classify a sequence into one of 26 major mtDNA haplogroups."""
    if not sequence_input or not sequence_input.strip():
        return None, "**Error:** Please enter a sequence or FASTA."

    seq = _parse_sequence(sequence_input)
    if len(seq) < 50:
        return None, "**Error:** Sequence too short — enter at least 50 nucleotides."

    try:
        _load_models()
        embedder = _models["embedder"]
        haplo_model = _models["haplogroup"]
        vocab = _models["vocab"]

        genome_length = embedder.model.config.genome_length
        all_ids, all_pos = _tokenize_for_haplogroup(seq, vocab, genome_length)

        # Shape: (1, n_windows, window_size)
        input_ids = torch.tensor([all_ids], dtype=torch.long)
        position_ids = torch.tensor([all_pos], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = haplo_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        probs = torch.softmax(out.logits.squeeze(0), dim=-1).numpy()
        top_idx = int(np.argmax(probs))
        predicted = HAPLOGROUPS[top_idx]
        confidence = float(probs[top_idx])

        # Confidence bar chart (top 8)
        top8 = np.argsort(probs)[::-1][:8]
        fig, ax = plt.subplots(figsize=(8, 4))
        hgroups = [HAPLOGROUPS[i] for i in top8]
        vals = [probs[i] * 100 for i in top8]
        colours = [_haplogroup_colour(h) for h in hgroups]
        ax.barh(hgroups[::-1], vals[::-1], color=colours[::-1], edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Confidence (%)", fontsize=11)
        ax.set_title(f"Predicted: {predicted}  ({confidence * 100:.1f}% confidence)", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.axvline(confidence * 100, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        info = HAPLOGROUP_INFO.get(predicted, {})
        description = (
            f"### Haplogroup {predicted}\n\n"
            f"**Confidence:** {confidence * 100:.1f}%  \n"
            f"**Geographic origin:** {info.get('region', 'Unknown')}  \n"
            f"**Estimated age:** {info.get('age', 'Unknown')}  \n\n"
            f"{info.get('notes', 'No description available.')}"
        )

        return fig, description
    except Exception as exc:
        import traceback as _tb
        return None, f"**Error:** {exc}\n\n```\n{_tb.format_exc()}```"


# ── Tab 2: Genome Embedding ────────────────────────────────────────────────────

def embed_genome_tab(sequence_input: str) -> tuple[Any, str, str]:
    """
    Embed a sequence and place it on the reference UMAP.
    Returns: (figure, description_markdown, csv_string)
    """
    if not sequence_input or not sequence_input.strip():
        return None, "**Error:** Please enter a sequence.", ""

    seq = _parse_sequence(sequence_input)
    if len(seq) < 50:
        return None, "**Error:** Sequence too short — enter at least 50 nucleotides.", ""

    try:
        _load_models()
        embedder = _models["embedder"]

        embedding = embedder.embed_genome(seq)  # (256,)

        # CSV string
        csv_rows = [f"dim_{i},{embedding[i]:.8f}" for i in range(len(embedding))]
        csv_string = "dimension,value\n" + "\n".join(csv_rows)

        # Load reference data and project
        ref = _load_reference()
        fig, ax = plt.subplots(figsize=(8, 6))

        if ref["embeddings"] is not None and ref["umap_2d"] is not None:
            ref_embs = ref["embeddings"]
            ref_2d = ref["umap_2d"]
            ref_labels = ref["labels"]

            # Plot reference points coloured by haplogroup
            unique_labels = sorted(set(ref_labels))
            for lbl in unique_labels:
                mask = [label == lbl for label in ref_labels]
                pts = ref_2d[mask]
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    c=_haplogroup_colour(lbl),
                    s=40, alpha=0.6, label=lbl, edgecolors="none",
                )

            # Project query sequence
            query_2d = _project_query(embedding, ref_embs, ref_2d)
            ax.scatter(
                query_2d[0], query_2d[1],
                c="black", s=200, marker="*", zorder=5,
                label="Your sequence", edgecolors="white", linewidths=1.5,
            )
            ax.annotate(
                "Your\nsequence",
                xy=(query_2d[0], query_2d[1]),
                xytext=(query_2d[0] + 0.5, query_2d[1] + 0.5),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            )

            # Compact legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles, labels, loc="upper right", fontsize=7,
                ncol=3, framealpha=0.8, markerscale=0.8,
            )

            # Find nearest reference neighbours for description
            dists = np.linalg.norm(ref_embs - embedding[None, :], axis=1)
            top3_idx = np.argsort(dists)[:3]
            neighbours = [f"{ref_labels[i]} (d={dists[i]:.2f})" for i in top3_idx]
            neighbour_str = ", ".join(neighbours)
        else:
            neighbour_str = "reference data unavailable"

        ax.set_xlabel("UMAP dimension 1", fontsize=11)
        ax.set_ylabel("UMAP dimension 2", fontsize=11)
        ax.set_title("Genome Embedding — Reference UMAP (100 mtDNA genomes)", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)
        fig.tight_layout()

        l2_norm = float(np.linalg.norm(embedding))
        description = (
            f"### Embedding computed\n\n"
            f"**Dimension:** 256  \n"
            f"**L2 norm:** {l2_norm:.4f}  \n"
            f"**Nearest reference haplogroups:** {neighbour_str}  \n\n"
            f"The star (★) on the UMAP shows where your sequence sits relative to "
            f"100 reference human mtDNA genomes spanning 26 haplogroups. "
            f"Proximity in the embedding space reflects genomic similarity."
        )

        return fig, description, csv_string
    except Exception as exc:
        import traceback as _tb
        return None, f"**Error:** {exc}\n\n```\n{_tb.format_exc()}```", ""


# ── Tab 3: Variant Pathogenicity ──────────────────────────────────────────────

def predict_pathogenicity(
    position_input: float | int | None,
    ref_allele: str,
    alt_allele: str,
) -> tuple[str, Any, str]:
    """Zero-shot pathogenicity prediction via cosine k-NN on pre-trained encoder embeddings."""
    try:
        position = int(position_input)
    except (TypeError, ValueError):
        return "**Error:** Position must be a number between 1 and 16569.", None, ""
    if not (1 <= position <= 16569):
        return "**Error:** Position must be between 1 and 16569.", None, ""
    if ref_allele not in ("A", "T", "G", "C"):
        return "**Error:** Ref allele must be A, T, G, or C.", None, ""
    if alt_allele not in ("A", "T", "G", "C"):
        return "**Error:** Alt allele must be A, T, G, or C.", None, ""
    if ref_allele == alt_allele:
        return "**Error:** Ref and alt alleles must be different.", None, ""

    try:
        _load_models()
        ref_data = _load_patho_reference()

        if ref_data["X"] is None:
            return "**Error:** app_patho_reference.npz not found.", None, ""

        rcrs = ref_data["rcrs"]
        rcrs_ref = rcrs[position - 1]
        ref_warning = (
            f"\n\n> ⚠️ rCRS has '{rcrs_ref}' at position {position}; "
            f"proceeding with specified ref '{ref_allele}'."
            if rcrs_ref != ref_allele
            else ""
        )

        mutated_seq = rcrs[: position - 1] + alt_allele + rcrs[position:]
        embedder = _models["embedder"]
        query_emb = embedder.embed_variant(mutated_seq, pos_0=position - 1, pooling="token")

        # Cosine k-NN (k=5)
        X = ref_data["X"]
        y = ref_data["y"]
        ref_positions = ref_data["positions"]
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        X_norms = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        cosine_dists = 1.0 - (X_norms @ query_norm)
        k = 5
        top_k = np.argsort(cosine_dists)[:k]
        top_labels = y[top_k]
        top_dists = cosine_dists[top_k]
        top_positions = ref_positions[top_k]

        patho_score = float(top_labels.sum()) / k
        is_pathogenic = patho_score >= 0.5
        label = "LIKELY PATHOGENIC" if is_pathogenic else "LIKELY BENIGN"
        emoji = "🔴" if is_pathogenic else "🟢"

        var_type = _get_variant_type(position)
        type_name, type_note = _VARIANT_TYPE_NOTES[var_type]

        result_md = (
            f"## {emoji} {label}\n\n"
            f"**Pathogenicity score:** {patho_score:.2f} "
            f"({int(top_labels.sum())}/{k} nearest neighbors are pathogenic)\n\n"
            f"**Variant:** m.{position}{ref_allele}>{alt_allele}  \n"
            f"**Region:** {type_name}"
            + ref_warning
        )

        neighbors_df = pd.DataFrame({
            "Rank": list(range(1, k + 1)),
            "Position": [int(p) for p in top_positions],
            "Label": ["Pathogenic" if lbl == 1 else "Benign" for lbl in top_labels],
            "Cosine distance": [round(float(d), 4) for d in top_dists],
        })

        reliability_md = (
            f"**Variant type:** {type_name}  \n"
            f"**Reliability:** {type_note}  \n\n"
            "> Zero-shot 5-fold cosine k-NN. AUROC=0.777 (95% CI 0.731–0.821) on "
            "118 ClinVar pathogenic + 419 gnomAD AF≥1% benign mtDNA SNPs.  \n"
            "> **Research use only. Not for clinical diagnosis.**"
        )

        return result_md, neighbors_df, reliability_md

    except Exception as exc:
        import traceback as _tb
        return f"**Error:** {exc}\n\n```\n{_tb.format_exc()}```", None, ""


# ── Gradio interface ───────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

with gr.Blocks(
    title="mtDNA-FM Demo",
    theme=THEME,
    css="""
    .gradio-container { max-width: 960px; margin: auto; }
    h1 { font-size: 1.8rem !important; }
    """,
) as demo:

    gr.Markdown(
        """
# mtDNA-FM: Mitochondrial DNA Foundation Model

A pre-trained BERT encoder for the human mitochondrial genome (16,569 bp).
Trained on cross-species vertebrate mtDNA (Phase 1) then fine-tuned on 47,000 human HmtDB sequences (Phase 2).
Novel architecture: **circular positional encoding** + **heteroplasmy projection channel**.

**[GitHub](https://github.com/vthawfeek/mtdna-foundation-model)**  ·
**[HuggingFace Hub](https://huggingface.co/vthawfeek/mtdna-foundation-model)**

> Models are loaded on first use — initial inference may take 30–60 s on CPU.
"""
    )

    with gr.Tabs():

        # ── Tab 1: Haplogroup Classification ──────────────────────────────────
        with gr.Tab("🧬 Haplogroup Classification"):
            gr.Markdown(
                """
Paste an mtDNA sequence (raw DNA or FASTA format) to classify it into one of **26 major haplogroups**.
The model embeds all overlapping 512-bp windows of the genome and classifies based on mean-pooled CLS embeddings.
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    haplo_seq_input = gr.Textbox(
                        label="mtDNA sequence (FASTA or raw DNA)",
                        placeholder=">my_sequence\nGATCACAGGTCTATCACCCTATTAACCACTCACGGGAGC...",
                        lines=8,
                        value="",
                    )
                    haplo_btn = gr.Button("Classify Haplogroup", variant="primary")
                    haplo_example_btn = gr.Button(
                        "Run example (rCRS-based sequence)", size="sm", variant="secondary"
                    )
            with gr.Row():
                haplo_chart = gr.Plot(label="Confidence scores")
                haplo_text = gr.Markdown(label="Haplogroup description")

            haplo_btn.click(
                fn=predict_haplogroup,
                inputs=[haplo_seq_input],
                outputs=[haplo_chart, haplo_text],
            )
            haplo_example_btn.click(
                fn=lambda: (EXAMPLE_SEQUENCE[:2000],) + predict_haplogroup(EXAMPLE_SEQUENCE[:2000]),
                inputs=[],
                outputs=[haplo_seq_input, haplo_chart, haplo_text],
            )

        # ── Tab 2: Genome Embedding ────────────────────────────────────────────
        with gr.Tab("📊 Genome Embedding"):
            gr.Markdown(
                """
Embed your mtDNA sequence into a **256-dimensional vector** and see where it falls on a reference UMAP
of 100 human mtDNA genomes spanning 26 haplogroups. Download the embedding for downstream analysis.
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    emb_seq_input = gr.Textbox(
                        label="mtDNA sequence (FASTA or raw DNA)",
                        placeholder=">my_sequence\nGATCACAGGTCTATCACCCTATT...",
                        lines=6,
                    )
                    emb_btn = gr.Button("Embed Sequence", variant="primary")
                    emb_example_btn = gr.Button(
                        "Run example (rCRS-based sequence)", size="sm", variant="secondary"
                    )
            with gr.Row():
                emb_plot = gr.Plot(label="UMAP placement")
                with gr.Column():
                    emb_text = gr.Markdown(label="Embedding info")
                    emb_csv = gr.Textbox(
                        label="Embedding CSV (256 dimensions)",
                        lines=6,
                        max_lines=8,
                        show_copy_button=True,
                    )

            emb_btn.click(
                fn=embed_genome_tab,
                inputs=[emb_seq_input],
                outputs=[emb_plot, emb_text, emb_csv],
            )
            emb_example_btn.click(
                fn=lambda: (EXAMPLE_SEQUENCE[:3000],) + embed_genome_tab(EXAMPLE_SEQUENCE[:3000]),
                inputs=[],
                outputs=[emb_seq_input, emb_plot, emb_text, emb_csv],
            )

        # ── Tab 3: Variant Pathogenicity ──────────────────────────────────────
        with gr.Tab("🔬 Variant Pathogenicity"):
            gr.Markdown(
                """
Predict whether a mitochondrial DNA variant is **pathogenic or benign** using zero-shot k-NN
on pre-trained encoder embeddings. The encoder learned evolutionary conservation through masked
language modeling on 117k cross-species vertebrate sequences — **no pathogenicity labels were
used in training**.

**AUROC = 0.777** (95% CI: 0.731–0.821) on 118 ClinVar pathogenic vs 419 gnomAD benign variants.
Reliable for missense and tRNA variants. Unreliable for D-loop and intergenic positions.

> **Research use only. Not for clinical diagnosis.**
                """
            )
            with gr.Row():
                patho_pos_input = gr.Number(
                    label="Variant position (1–16569, rCRS coordinates)",
                    precision=0,
                    value=3243,
                )
                patho_ref_input = gr.Dropdown(
                    choices=["A", "T", "G", "C"],
                    label="Ref allele",
                    value="A",
                )
                patho_alt_input = gr.Dropdown(
                    choices=["A", "T", "G", "C"],
                    label="Alt allele",
                    value="G",
                )
            with gr.Row():
                patho_btn = gr.Button("Predict Pathogenicity", variant="primary")
                patho_example_btn = gr.Button(
                    "Run example (m.3243A>G — MELAS, tRNA-Leu)",
                    size="sm",
                    variant="secondary",
                )
            with gr.Row():
                with gr.Column(scale=1):
                    patho_result = gr.Markdown(label="Prediction")
                    patho_reliability = gr.Markdown(label="Reliability notes")
                with gr.Column(scale=1):
                    patho_neighbors = gr.DataFrame(
                        label="5 nearest reference variants (ClinVar / gnomAD)",
                        headers=["Rank", "Position", "Label", "Cosine distance"],
                    )

            patho_btn.click(
                fn=predict_pathogenicity,
                inputs=[patho_pos_input, patho_ref_input, patho_alt_input],
                outputs=[patho_result, patho_neighbors, patho_reliability],
            )
            patho_example_btn.click(
                fn=lambda: (3243, "A", "G") + predict_pathogenicity(3243, "A", "G"),
                inputs=[],
                outputs=[
                    patho_pos_input, patho_ref_input, patho_alt_input,
                    patho_result, patho_neighbors, patho_reliability,
                ],
            )

    gr.Markdown(
        """
---
**Model details:** 6-layer BERT encoder · 8 attention heads · 256 hidden dims · ~6M parameters
**Vocabulary:** 4,096 6-mers + 6 special tokens = 4,102 tokens
**Fine-tuning:** LoRA r=8 haplogroup classification (26 classes)
**Zero-shot haplogroup k-NN:** ~50% accuracy without any labels (vs 3.85% random)
**Zero-shot pathogenicity AUROC:** 0.777 (95% CI: 0.731–0.821) on ClinVar pathogenic vs gnomAD benign variants — Tab 3
**Limitations:** Trained on HmtDB (European population bias). Pathogenicity tab is for research use only, not clinical diagnosis.
"""
    )


if __name__ == "__main__":
    demo.launch(share=False)
