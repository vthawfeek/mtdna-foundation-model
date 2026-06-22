# Gap Analysis: mtDNA-FM → bioRxiv

This document identifies what is required before the project can be submitted as a preprint.
Gaps are rated **Critical** (will cause rejection), **Important** (substantially weakens the paper),
or **Minor** (polish).

---

## Critical Gaps

### G1 — Ablation studies absent

The paper's three novelty claims are each unablated:

| Claim | Current state | Required |
|-------|--------------|---------|
| Circular PE improves over standard sinusoidal/linear PE | Stated, not quantified | Train ablation model with standard PE; compare zero-shot k-NN + fine-tuned haplogroup accuracy |
| Two-phase curriculum outperforms single-phase | Stated, not quantified | Train single-phase baseline on combined data; compare MLM loss + downstream accuracy |
| Heteroplasmy channel captures useful signal | Phase 2 not trained | Complete Phase 2 with and without het_weight; compare regression R² / Spearman ρ |

Scripts: `experiments/ablations/ablate_circular_pe.py`, `ablate_curriculum.py`, `ablate_het_channel.py`

### G2 — Synthetic evaluation data

The haplogroup confusion matrix and all per-class F1 scores were computed on **260 programmatically
generated sequences** (10 per class, synthetic). This will not survive peer review.

**Required:**
- Carve out a held-out test split from HmtDB *before* pre-training; sequences in test must not
  appear in pre-training corpus.
- Minimum 50 sequences per represented haplogroup (≥20 per underrepresented clade).
- Store in `paper/experiments/evaluation/held_out_test.parquet`.

Script: `experiments/evaluation/create_eval_splits.py`

### G3 — No direct comparison against other DNA LLMs

DNABERT2 and HyenaDNA are mentioned by name but their embeddings are never evaluated on the
same mtDNA task. Reviewers will ask: "how do you know your model is better?"

**Required:**
- Extract DNABERT2 CLS embeddings for haplogroup test set → 5-fold k-NN accuracy
- Extract DNABERT2 per-position embeddings for variant position → logistic regression AUROC
- Same for HyenaDNA if feasible (DNA character-level, may require sequence truncation)
- Classical k-mer frequency + LR/SVM as quantified (not just qualitative) baseline

Script: `experiments/baselines/dnabert2_baseline.py`, `kmer_frequency_baseline.py`

### G4 — Phase 2 training not completed

`models/phase2_v1/` is noted as empty in the reports. Phase 2 (human-only sequences,
heteroplasmy loss enabled, het_weight=0.3) is central to two claims:
1. The heteroplasmy channel captures signal
2. The Phase 2 zero-shot k-NN accuracy (~50%) is valid

**Required:**
- Run Phase 2 training to convergence
- Re-evaluate zero-shot k-NN on Phase 2 checkpoint
- Re-evaluate heteroplasmy regression on Phase 2 checkpoint

---

## Important Gaps

### G5 — No confidence intervals

All metrics are point estimates. Journals require uncertainty quantification.

**Required:**
- Bootstrap AUROC (1,000 resamples) → 95% CI for pathogenicity AUROC
- Stratified 5-fold CV for haplogroup accuracy → mean ± std
- Run 3 fine-tuning seeds per task → report mean ± std across seeds

Script: `experiments/evaluation/compute_confidence_intervals.py`

### G6 — Literature review incomplete

No formal citations exist for:
- Foundational ML papers (Vaswani et al. 2017, Devlin et al. 2018)
- DNABERT2, HyenaDNA, Nucleotide Transformer, Evo papers
- Domain-specific mtDNA tools: HaploGrep2, Haplocheck, MITOMAP, HelixMTdb, MitImpact
- Variant effect prediction: SIFT, PolyPhen-2, EVE, ESM1v
- Positional encoding variants: RoPE, ALiBi

**Required:**
- `paper/review/related_work.md` — narrative lit review
- `paper/manuscript/references.bib` — BibTeX entries with verified DOIs

### G7 — No independent external validation for pathogenicity

All pathogenicity evaluation uses ClinVar (training positives) and gnomAD (training negatives).
There is potential data leakage: the model may have overfit to ClinVar curation artefacts.

**Required:**
- Download MITOMAP confirmed pathogenic variants (independent curation)
- Download HelixMTdb common variants (independent population data)
- Evaluate pathogenicity model AUROC on this non-overlapping set

Script: `experiments/evaluation/external_pathogenicity_eval.py`

---

## Minor Gaps

### G8 — Figure quality

Existing notebook figures are exploratory (matplotlib defaults, no consistent style).
Paper requires 300 DPI, consistent color palette, proper axis labels and fonts.

**Required:**
- `paper/manuscript/figures/generate_figures.py` — reproducible script for all 4 paper figures
- Figures 1–4 as described in the manuscript (architecture, PE comparison, haplogroup, pathogenicity)

### G10 — Model card and data availability statement

bioRxiv requires a data availability section; journals require code availability.

**Required:**
- Update HuggingFace Hub model card to include paper citation
- `paper/manuscript/main.tex` data availability section: GitHub + HF Hub links
- DVC pipeline as reproducibility statement

---

## Gap Resolution Tracker

| Gap | Priority | Script/File | Status |
|-----|----------|-------------|--------|
| G1-A1 Circular PE ablation | Critical | `experiments/ablations/ablate_circular_pe.py` | TODO |
| G1-A2 Curriculum ablation | Critical | `experiments/ablations/ablate_curriculum.py` | TODO |
| G1-A3 Het channel ablation | Critical | `experiments/ablations/ablate_het_channel.py` | TODO |
| G2 Proper eval splits | Critical | `experiments/evaluation/create_eval_splits.py` | TODO |
| G3 DNABERT2 baseline | Critical | `experiments/baselines/dnabert2_baseline.py` | TODO |
| G3 k-mer baseline (quantified) | Critical | `experiments/baselines/kmer_frequency_baseline.py` | TODO |
| G4 Phase 2 training | Critical | CLI: `uv run mtdna-pretrain --phase 2` | TODO |
| G5 CIs + seeds | Important | `experiments/evaluation/compute_confidence_intervals.py` | DONE (bootstrap CIs in manuscript) |
| G6 Lit review | Important | `review/related_work.md` + `manuscript/references.bib` | DONE |
| G7 External pathogenicity | Important | `experiments/evaluation/external_pathogenicity_eval.py` | TODO (deferred to extended paper) |
| G8 Paper figures (4) | Minor | `manuscript/figures/generate_figures.py` | DONE |
| G9 Model card update | Minor | `huggingface_hub/push_to_hub.py` | DONE (HF Hub live) |
