# mtDNA-FM: Paper Reproduction Guide

This directory contains everything needed to reproduce all experiments and figures in
the mtDNA-FM bioRxiv paper.

## Directory Structure

```
paper/
├── manuscript/
│   ├── main.tex              # LaTeX source (main paper)
│   ├── supplementary.tex     # Supplementary materials
│   ├── references.bib        # BibTeX bibliography
│   └── figures/
│       └── generate_figures.py  # Reproduces all 6 paper figures
├── experiments/
│   ├── ablations/
│   │   ├── ablate_circular_pe.py     # G1-A1: Circular PE vs linear/learnable
│   │   ├── ablate_curriculum.py      # G1-A2: Two-phase vs single-phase
│   │   └── ablate_het_channel.py     # G1-A3 + G4: Het channel ablation
│   ├── baselines/
│   │   ├── dnabert2_baseline.py      # G3: DNABERT2 comparison
│   │   └── kmer_frequency_baseline.py # G3: k-mer + LR/SVM baseline
│   └── evaluation/
│       ├── create_eval_splits.py           # G2: Proper held-out test set
│       ├── compute_confidence_intervals.py  # G5: Bootstrap CIs
│       ├── external_pathogenicity_eval.py   # G8: MITOMAP/HelixMTdb validation
│       └── ancient_dna_extended.py          # G7: 10+ ancient samples
└── review/
    ├── gap_analysis.md       # Scientific gaps and resolution tracker
    ├── related_work.md       # Literature review narrative
    └── reviewer_checklist.md # Anticipated reviewer objections + responses
```

## Prerequisites

```bash
# Install project dependencies
uv sync

# Ensure pre-training checkpoints exist
ls models/phase1_v1/  # Phase 1 checkpoint
ls models/phase2_v1/  # Phase 2 checkpoint (run pre-training if absent)
ls models/vocabulary/ # k-mer vocabulary
```

## One-Command Reproduction

```bash
bash paper/reproduce_all.sh
```

Or run each step individually (see below).

## Step-by-Step Reproduction

### Step 1: Pre-training (if checkpoints not available)

```bash
# Phase 1: cross-species vertebrate mtDNA (50k steps, ~8h CPU)
uv run mtdna-pretrain --phase 1 --output models/phase1_v1

# Phase 2: human HmtDB with heteroplasmy channel (25k steps, ~4h CPU)
uv run mtdna-pretrain --phase 2 --output models/phase2_v1
```

### Step 2: Create proper held-out evaluation splits (G2)

```bash
uv run python paper/experiments/evaluation/create_eval_splits.py
# Output: paper/experiments/evaluation/held_out_test.parquet
```

### Step 3: Run ablation studies (G1)

```bash
# Circular PE ablation (requires 3 trained models — ~24h total)
uv run python paper/experiments/ablations/ablate_circular_pe.py

# Curriculum ablation (requires single-phase trained model)
uv run python paper/experiments/ablations/ablate_curriculum.py

# Heteroplasmy channel ablation (completes Phase 2 training, G4)
uv run python paper/experiments/ablations/ablate_het_channel.py --train
```

### Step 4: Run baseline comparisons (G3)

```bash
# DNABERT2 baseline (requires ~4GB GPU or patience on CPU)
uv run python paper/experiments/baselines/dnabert2_baseline.py

# k-mer frequency baseline (fast, CPU-only)
uv run python paper/experiments/baselines/kmer_frequency_baseline.py
```

### Step 5: Compute confidence intervals (G5)

```bash
# Requires fine-tuned models and prediction files from evaluation
uv run mtdna-evaluate --output reports/eval_summary.json
uv run python paper/experiments/evaluation/compute_confidence_intervals.py
```

### Step 6: External validation (G7, G8)

```bash
# Ancient DNA extended evaluation
uv run python paper/experiments/evaluation/ancient_dna_extended.py

# External pathogenicity validation (requires MITOMAP and HelixMTdb downloads)
# See paper/experiments/evaluation/external_pathogenicity_eval.py for download instructions
uv run python paper/experiments/evaluation/external_pathogenicity_eval.py
```

### Step 7: Generate all paper figures

```bash
uv run python paper/manuscript/figures/generate_figures.py
# Output: paper/manuscript/figures/fig{1..6}.{pdf,png}
```

### Step 8: Compile the manuscript

```bash
cd paper/manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
# Output: main.pdf
```

## Data Downloads (External Databases)

Some external datasets require manual download due to licensing/terms of use:

| Dataset | URL | Save to |
|---------|-----|---------|
| MITOMAP pathogenic variants | https://www.mitomap.org | `data/raw/mitomap/mitomap_confirmed_pathogenic.tsv` |
| HelixMTdb variants | https://helix.com/helixmtdb | `data/raw/helixmtdb/helixmtdb_variants.tsv` |

All other datasets (HmtDB, NCBI, gnomAD, ClinVar, PhyloTree) are downloaded automatically
by the DVC pipeline (`dvc repro`).

## Model Weights

Pre-trained model weights and LoRA adapters are available on HuggingFace Hub:
- Base model: https://huggingface.co/vthawfeek/mtdna-foundation-model
- Haplogroup adapter: https://huggingface.co/vthawfeek/mtdna-fm-haplogroup
- Pathogenicity adapter: https://huggingface.co/vthawfeek/mtdna-fm-pathogenicity

## Expected Runtimes (CPU-only, 8-core laptop)

| Step | Estimated time |
|------|---------------|
| Phase 1 pre-training (50k steps) | ~8h |
| Phase 2 pre-training (25k steps) | ~4h |
| Circular PE ablation (3 models) | ~24h |
| k-mer baseline | ~10 min |
| DNABERT2 baseline (CPU) | ~2h |
| Confidence intervals | ~5 min |
| Ancient DNA evaluation | ~30 min |
| Figure generation | ~10 min |
| LaTeX compilation | ~1 min |

## Paper Gap Resolution Status

See [gap_analysis.md](review/gap_analysis.md) for the full gap tracker.

| Gap | Script | Status |
|-----|--------|--------|
| G1 Ablation studies | `experiments/ablations/` | Scripts ready; needs training runs |
| G2 Proper eval splits | `evaluation/create_eval_splits.py` | Script ready; run after DVC pipeline |
| G3 DNA LLM baselines | `baselines/` | Scripts ready; DNABERT2 needs GPU |
| G4 Phase 2 training | `uv run mtdna-pretrain --phase 2` | Run manually |
| G5 Confidence intervals | `evaluation/compute_confidence_intervals.py` | Script ready |
| G6 Literature review | `review/related_work.md` | Written; verify DOIs |
| G7 Ancient DNA extended | `evaluation/ancient_dna_extended.py` | Script ready |
| G8 External pathogenicity | `evaluation/external_pathogenicity_eval.py` | Needs MITOMAP download |
| G9 Paper figures | `manuscript/figures/generate_figures.py` | Script ready |
| G10 Model card | `huggingface_hub/push_to_hub.py` | Update after paper draft |

## Citation

```bibtex
@article{varusai2024mtdnafm,
  title   = {mtDNA-FM: A domain-specialized foundation model for mitochondrial DNA
             with circular positional encoding and heteroplasmy modeling},
  author  = {Varusai, Thawfeek},
  journal = {bioRxiv},
  year    = {2024},
  doi     = {TODO: add after bioRxiv submission}
}
```
