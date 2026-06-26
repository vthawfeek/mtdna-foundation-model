---
license: apache-2.0
language:
- dna
tags:
- biology
- genomics
- mitochondrial-dna
- bert
- masked-language-model
- bioinformatics
- foundation-model
- haplogroup
- variant-effect-prediction
pipeline_tag: feature-extraction
widget:
- text: "ATGGTGAGCAAGGGCGAGGAG"
  example_title: "mtDNA fragment"
---

# mtDNA-FM: Mitochondrial DNA Foundation Model

A pre-trained BERT encoder for the human mitochondrial genome (16,569 bp circular genome).

**mtDNA-FM** is the first dedicated foundation model for mitochondrial DNA. It encodes the full circular genome topology via novel circular positional encoding, includes a heteroplasmy projection channel, and was pre-trained on 117,615 vertebrate mtDNA sequences in a two-phase cross-species curriculum.

## Quick Start

```python
from mtdna_fm.inference.api import MtDNAEmbedder

embedder = MtDNAEmbedder.from_pretrained("vthawfeek/mtdna-foundation-model")
embedding = embedder.embed_genome(my_sequence)   # shape: (256,)
```

## Architecture Novelties

### Circular Positional Encoding

Standard sinusoidal PE treats positions 1 and 16,569 as maximally distant — but mtDNA is circular, and position 16,569 is genomically adjacent to position 1. mtDNA-FM uses:

```
PE[pos, 2i]   = sin(2π × pos / L × 1/10000^(2i/d))
PE[pos, 2i+1] = cos(2π × pos / L × 1/10000^(2i/d))
```

where `L = 16569` (genome length) and `d = 256` (hidden size). This is a fixed, non-learnable buffer — the circular topology is a biological fact, not a parameter.

### Heteroplasmy Projection Channel

Heteroplasmy (the co-existence of wild-type and mutant mtDNA within a single cell) is biologically unusual and clinically significant. mtDNA-FM accepts a continuous per-base heteroplasmy level `h ∈ [0.0, 1.0]` alongside each token:

```
embedding = kmer_embedding + circular_pe + LayerNorm(Linear(1 → d)(het_values))
```

When `het_values` are all zero (Phase 1 pre-training), the channel contributes a small constant
offset, `LayerNorm(b_het)`, to the embedding rather than zeroing out.

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | BERT encoder (pre-LN) |
| Vocabulary | 4,096 6-mers + 6 special tokens = 4,102 |
| Hidden size | 256 |
| Layers | 6 |
| Attention heads | 8 |
| Intermediate size | 1,024 |
| Max sequence length | 514 (512 + CLS + SEP) |
| Parameters | ~5.8M |
| Genome length | 16,569 bp |

## Training

**Tokenisation:** 6-mer overlapping sliding window (stride=1), circular wrapping at the genome junction. Each full genome produces 16,569 tokens.

**Windowing:** 512-token overlapping windows (stride=256) over the token stream, ~65 windows per genome.

**Phase 1 pre-training (cross-species):**
- 117,615 vertebrate mtDNA sequences from NCBI
- Standard BERT masking (15% MLM, 80/10/10 mask/random/keep)
- D-loop homopolymeric C-tract (positions 303–315) blacklisted from masking (sequencing noise)
- Cosine LR schedule: peak 1×10⁻⁴, 2k warmup steps
- Effective batch size 128 (16 × 8 gradient accumulation steps)
- MLM loss: random baseline 8.32 → converged 4.25 (perplexity 70)

**Phase 2 pre-training (human-specific):**
- 34,975 human HmtDB sequences (47,000 total in database; filtered to ≤10% ambiguous bases)
- het_weight=0.3 (heteroplasmy prediction enabled)
- Learning rate 3×10⁻⁵, 25k steps
- Loaded Phase 1 encoder weights, fresh optimizer

## Performance

| Task | Metric | Majority class | k-mer + LR | DNABERT-2 (zero-shot) | mtDNA-FM (zero-shot) |
|------|--------|---------------|------------|----------------------|---------------------|
| Haplogroup classification | Accuracy | 3.85% | 78.7% (26-class, same 1,509-seq library) | 66.3%* (26-class) | **37.9%** (26-class) |
| Pathogenic variant prediction | AUROC | 0.50 | not evaluated | not evaluated | 0.777 (95% CI 0.731-0.821) |
| Ancient DNA placement | L2 ratio vs modern | — | — | not evaluated | 1.19x–1.25x |

\* DNABERT-2 (117M params, nuclear-genome pre-training) zero-shot 5-NN on same train/test split; each 16,569 bp genome truncated to ~3,000 nt (18%) due to 512-token BPE context limit. Results in `reports/dnabert2_haplogroup_knn.json`.

Zero-shot 5-NN (cosine) on Phase 1 checkpoint embeddings, full 26-class haplogroup evaluation (13,884 NCBI-labeled sequences; 3.85% random baseline; **9.9x lift**; 95% CI 34.4-41.2%). Per-class results in `reports/zeroshot_haplogroup_knn.json`.

Pathogenicity: Zero-shot 5-fold stratified k-NN (k=5, cosine). 118 ClinVar pathogenic + 419 gnomAD AF>=1% benign mitochondrial SNPs. No pathogenicity labels used during pre-training. Per-type: missense 0.727 (n=56), tRNA 0.718 (n=44).

**Ancient DNA zero-shot:** Neanderthal (NC_011137.1, Vindija Cave) and Denisovan (FR695060.1, Altai Cave)
embedded without any task-specific supervision. L2 distance from modern humans: Neanderthal 0.094
(1.25x the modern pairwise baseline of 0.075) and Denisovan 0.089 (1.19x baseline), consistent
with paleoanthropological expectations.

## Usage

```python
from mtdna_fm.inference.api import MtDNAEmbedder

# Load the pre-trained model
embedder = MtDNAEmbedder.from_pretrained("vthawfeek/mtdna-foundation-model")

# Embed a full mtDNA genome → (256,) vector
embedding = embedder.embed_genome(my_sequence)

# Embed the context around a variant position
variant_embedding = embedder.embed_variant(my_sequence, position=3243)

# Batch embedding for a DataFrame of sequences
import pandas as pd
df = pd.DataFrame({"sequence": [seq1, seq2, seq3]})
embeddings = embedder.embed_dataset(df)  # shape: (3, 256)
```

Supervised fine-tuning experiments (LoRA) and per-class benchmarks against external tools are planned for the extended journal paper.

## Known Limitations

- **Population bias:** HmtDB has strong European bias (haplogroup H is overrepresented). Performance on underrepresented haplogroups (especially African L sub-haplogroups) may be lower.
- **Heteroplasmy channel:** The het projection is architecturally present; Phase 2 training used het_weight=0.3, but real per-base heteroplasmy labels were limited to gnomAD variant-level data rather than full-genome measurements.
- **Zero-shot haplogroup:** Phase 1 checkpoint zero-shot 5-NN achieves 37.9% on the full 26-class evaluation (3.85% random baseline; 9.9× lift; 95% CI 34.4–41.2%).
- **Cosine similarity:** Mean-pooled CLS embeddings exhibit high cosine similarity at the population level. Use L2 distance for zero-shot comparisons.

## Citation

If you use mtDNA-FM in your research, please cite:

```bibtex
@misc{varusai2026mtdnafm,
  author = {Varusai, Thawfeek},
  title  = {mtDNA-FM: A Foundation Model for Mitochondrial DNA},
  year   = {2026},
  url    = {https://huggingface.co/vthawfeek/mtdna-foundation-model},
  note   = {GitHub: https://github.com/vthawfeek/mtdna-foundation-model}
}
```

## Project

- GitHub: [vthawfeek/mtdna-foundation-model](https://github.com/vthawfeek/mtdna-foundation-model)
- Blog: [rokpayprsizors.wordpress.com](https://rokpayprsizors.wordpress.com/)
- X: [@vthawfeek](https://x.com/vthawfeek)

Built as a 4-week portfolio project demonstrating production-quality foundation model engineering
on a clinically relevant niche dataset. The full mtDNA genome is 16,569 bp — the only human genome
small enough to train a BERT encoder on a laptop, but biologically rich enough to matter.
