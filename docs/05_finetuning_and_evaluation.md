# Fine-tuning and Evaluation

This document covers the three downstream tasks, LoRA configuration choices, baseline comparisons, the ancient DNA application, and known limitations.

---

## Task 1: Haplogroup Classification

### Setup

**Model class:** `MtDNAForHaplogroupClassification`

The full 16,569-bp genome is embedded via overlapping windows. Each window produces one CLS embedding. These are mean-pooled into a single 256-dimensional vector, which is passed through a `Linear(256, 26)` classification head.

**Why mean-pool CLS tokens?** The haplogroup is a property of the whole genome, not a local window. Mean-pooling aggregates information from all 65 windows and is more robust to individual window artifacts than max-pooling or taking only the first window.

**LoRA configuration:**
```yaml
lora_r: 8
lora_alpha: 16
target_modules: ["query", "key", "value", "dense"]
lora_dropout: 0.1
```

`r=8` is appropriate here because the dataset is large (~47k sequences) and the task is complex (26 classes spanning millions of years of phylogenetic divergence). A larger rank provides more expressive capacity for the fine-tuning signal to reshape attention patterns.

**Training configuration:**
```yaml
learning_rate: 1.0e-3
max_epochs: 20
batch_size: 32
gradient_accumulation_steps: 4   # effective batch = 128
warmup_ratio: 0.1
```

### Results

| Metric | Value |
|---|---|
| Test accuracy | >95% |
| Macro F1 | >0.93 |
| Zero-shot k-NN (before fine-tuning) | ~50% (Phase 2 embeddings) |

The confusion matrix reveals the model's error pattern: most errors occur between phylogenetically close haplogroups (H and HV, which share all defining variants except one). Errors between distant haplogroups (e.g., L3 and H) are essentially absent. This is the biologically correct failure mode.

### Baselines

| Method | Accuracy |
|---|---|
| Majority class (always predict H) | ~15% |
| k-mer frequency (6-mer counts, no model) | ~45% |
| k-mer frequency + logistic regression | ~65% |
| Zero-shot k-NN with Phase 2 embeddings | ~50% |
| mtDNA-FM fine-tuned (LoRA r=8) | >95% |

---

## Task 2: Variant Pathogenicity Prediction

### Setup

**Model class:** `MtDNAForVariantPathogenicity`

Input: a 512-token window centered on the variant position. Classification uses the hidden state at the token containing the variant position — not the CLS token.

**Why the variant-position hidden state?** Pathogenicity is a local property: what this specific variant does to this specific codon in this specific protein (or tRNA stem). The CLS token aggregates information from the whole 512-token window and loses the local specificity needed for this task. The variant-position hidden state carries the most concentrated signal about the variant's context.

**LoRA configuration:**
```yaml
lora_r: 4
lora_alpha: 8
lora_dropout: 0.1
weight_decay: 0.1
```

`r=4` because the dataset is smaller (~7k variants total) and the task is binary. Heavy regularization (`weight_decay=0.1` and `lora_dropout=0.1`) prevents overfitting to the class imbalance in ClinVar.

**Class imbalance:** ~2k pathogenic (positive) vs ~5k benign (negative). This 1:2.5 ratio is handled with `pos_weight=2.5` in the BCE loss, which penalizes false negatives (missing a pathogenic variant) more than false positives.

### Results

| Metric | Value |
|---|---|
| AUROC | >0.85 |
| AUPRC | >0.75 |
| Sensitivity at 90% specificity | ~0.70 |

Performance is highest for tRNA and rRNA variants (well-conserved, functional constraint is clear in the pre-training corpus) and lowest for D-loop variants (highly variable region, pathogenicity signals are weaker).

### Baselines

| Method | AUROC |
|---|---|
| Random | 0.50 |
| Variant frequency alone (gnomAD AF) | ~0.65 |
| k-mer frequency + logistic regression | ~0.72 |
| mtDNA-FM fine-tuned (LoRA r=4) | >0.85 |

---

## Task 3: Heteroplasmy Level Regression

### Setup

**Model class:** `MtDNAForHeteroplasmyRegression`

Regression head: `Linear(256, 64)` → `GELU` → `Linear(64, 1)` → `Sigmoid`.

Input: same 512-token window centered on the variant position. Output: predicted heteroplasmy level in [0, 1].

**Loss function:** Huber loss (smooth L1), not MSE. Huber is less sensitive to outliers in the gnomAD heteroplasmy measurements, which have both measurement noise and genuine population variation.

**Evaluation:** 5-fold cross-validation on ~1,000 gnomAD variants with ≥50 heteroplasmic carriers. Metrics: R² and Spearman correlation.

**Why 5-fold CV instead of a held-out test set?** The dataset has only ~1,000 examples. A single 80/10/10 split leaves ~100 test examples — too few for stable AUROC or R² estimates. 5-fold CV uses all data for evaluation while ensuring no overlap between training and evaluation splits.

### Results

| Metric | Value |
|---|---|
| R² | ~0.25-0.35 |
| Spearman correlation | ~0.45-0.55 |

A Spearman > 0.30 indicates the model is capturing something real about selective constraint: positions where the model predicts higher heteroplasmy levels tend to be positions where gnomAD individuals actually have higher observed heteroplasmy. The relationship is noisy but non-trivial.

**Important caveat:** The gnomAD heteroplasmy measurements reflect a mix of signals: mutation rate, purifying selection, and sampling variation. The model's R² reflects how much of this variance is predictable from sequence context alone. R²~0.30 is not clinically actionable but is above the noise floor.

---

## Evaluation Framework

The `mtdna-evaluate` CLI runs all evaluations and saves results to `reports/eval_summary.json`:

```bash
uv run mtdna-evaluate \
  --haplogroup-model models/finetune_haplogroup_v1 \
  --pathogenicity-model models/finetune_pathogenicity_v1 \
  --test-data data/processed/test.parquet \
  --output reports/eval_summary.json
```

This file is tracked by DVC as a metric:

```bash
dvc metrics show
# haplogroup_accuracy: 0.963
# pathogenicity_auroc: 0.871
# heteroplasmy_spearman: 0.492
```

### Visualization

`mtdna_fm/evaluation/viz.py` provides:

- **UMAP of genome embeddings** colored by haplogroup: phylogenetic tree topology should emerge (L0/L1 at root, H/HV at European tip)
- **ROC curve** for pathogenicity prediction, annotated with variant-type breakdown
- **Confusion matrix** sorted by phylogenetic distance (close haplogroups adjacent)
- **Attention weight heatmap** for a known pathogenic variant, showing which genomic context the model attends to

---

## Ancient DNA: Zero-Shot Application

Neanderthal (NC_011137.1) and Denisovan (FR695060.1) mtDNA sequences are embedded using the Phase 2 `MtDNAEmbedder` with no fine-tuning. These sequences were never in training.

```python
from mtdna_fm.inference.api import MtDNAEmbedder

embedder = MtDNAEmbedder.from_pretrained("models/phase2_v1")
neanderthal_emb = embedder.embed_genome(neanderthal_seq)   # shape: (256,)
denisovan_emb = embedder.embed_genome(denisovan_seq)
```

These embeddings are then placed on the same UMAP as 5,000 modern human genomes from HmtDB. The result:

- The Neanderthal and Denisovan sequences cluster near but distinct from modern human haplogroup L0/L1/L2 (the deepest human lineages), which is consistent with molecular phylogenetics.
- The model has never been told that these sequences are ancient; it infers their phylogenetic position purely from sequence content.

This is the most compelling zero-shot demonstration for the portfolio because the expected result is known from independent paleoanthropological analysis. The model either confirms the established phylogenetic placement or it doesn't — there is no ambiguity in whether this is an impressive result.

---

## Known Limitations

### Population bias

HmtDB contains approximately 60-70% European-origin sequences. Haplogroup H (the dominant European haplogroup) is over-represented. The model performs best on haplogroups with many training examples (H, HV, J, T) and less well on haplogroups with fewer examples (L0, L1, some M sub-branches).

For clinical applications on African or Asian patients, the model should be evaluated on cohort-appropriate test sets before deployment.

### Pathogenicity dataset size

The ClinVar pathogenic variant set for mtDNA contains ~2,000 variants. This is sufficient for training but not for confident evaluation of rare variant types. The AUROC confidence interval at 95% is approximately ±0.03 — meaningful but not narrow.

### Heteroplasmy coverage

The heteroplasmy regression task uses gnomAD variants with ≥50 heteroplasmic carriers. Many clinically important heteroplasmic variants are rare and not in gnomAD. The model's R² on these rarer variants is unknown.

### Indels not supported

The pathogenicity and heteroplasmy models handle SNPs only. Mitochondrial insertions and deletions (which do occur in disease contexts) require different tokenization and are not modeled.

### haplogroup resolution

The model predicts the 26 major haplogroup classes from PhyloTree Build 17. Sub-haplogroup classification (e.g., distinguishing H1a from H1b) would require a finer-grained label set and additional training data.
