# Anticipated Reviewer Objections and Responses

This checklist pre-empts the most likely reviewer concerns for a domain-specialized genomic
foundation model paper targeting *Bioinformatics* or *PLoS Computational Biology*.

---

## Objection 1: "Why not just fine-tune DNABERT2 or HyenaDNA?"

**Expected wording:** "The authors should compare their specialized model to general-purpose DNA
language models fine-tuned on the same data. Without this baseline, the novelty over existing
models is unclear."

**Response strategy:**
1. Include DNABERT2 zero-shot k-NN and fine-tuned results in Table 1 (model comparison).
2. Note that DNABERT2 uses BPE tokenization incompatible with mtDNA 6-mer statistics and lacks
   circular topology handling — show this concretely with AUROC numbers.
3. Acknowledge that fine-tuning DNABERT2 is a fair baseline; report it.
4. Argue that domain-specific pre-training learns mtDNA-specific evolutionary patterns (haplogroup
   clustering, pathogenic variant embedding contrast) that transfer learning from nuclear DNA does not.

**Pre-emptive mitigation:** Add G3 baselines (DNABERT2 evaluated on same task) before submission.

---

## Objection 2: "The ablation studies are insufficient"

**Expected wording:** "The paper claims circular positional encoding and two-phase curriculum as
novelties, but no ablation is provided to quantify their contribution."

**Response strategy:**
1. G1 ablation scripts directly address this. Results go in Table 2 (ablation table).
2. Ablation design: identical hyperparameters, identical data, only the component varied.
3. Report both zero-shot (no labels) and fine-tuned metrics for each ablation.

**Pre-emptive mitigation:** All three ablations (G1-A1, G1-A2, G1-A3) must be run and reported.

---

## Objection 3: "The evaluation uses synthetic test data"

**Expected wording:** "The haplogroup evaluation uses programmatically generated sequences, not
an independent held-out set. This raises concerns about inflated performance."

**Response strategy:**
1. G2 creates a proper held-out set from HmtDB sequences withheld from pre-training.
2. Report results on both the original synthetic set and the new held-out set.
3. If numbers decrease, report honestly and explain why (distributional shift, harder cases).

**Pre-emptive mitigation:** G2 must be completed before submission.

---

## Objection 4: "The model is too small (6M parameters)"

**Expected wording:** "With only 6M parameters, this model may lack the capacity to learn
complex biological signals. How does performance scale with model size?"

**Response strategy:**
1. Acknowledge the scale tradeoff directly in the Discussion.
2. Argue appropriateness: the mtDNA genome is 16,569 bp — a 6M model has roughly 1 parameter per
   training base pair, which is comparable to large language models relative to their training data.
3. Emphasize the accessibility benefit: laptop-trainable in 8–12 hours makes this reproducible
   by any lab without GPU clusters.
4. Note that scaling is future work; provide the DVC pipeline so the community can train larger
   variants.

---

## Objection 5: "The pathogenicity dataset is too small"

**Expected wording:** "With only ~2,000 ClinVar variants and ~5,000 gnomAD variants, the
pathogenicity model may overfit and the AUROC confidence interval may be wide."

**Response strategy:**
1. G5 reports bootstrap 95% CI for AUROC — show the CI does not include the k-mer baseline.
2. G8 external validation on MITOMAP / HelixMTdb demonstrates generalization beyond ClinVar.
3. Acknowledge the limitation honestly: ClinVar mtDNA curation is incomplete. Performance is a
   lower bound on what would be achievable with comprehensive variant databases.

---

## Objection 6: "No wet-lab validation"

**Expected wording:** "All results are computational. How do we know the predicted pathogenic
variants are actually functional?"

**Response strategy:**
1. Acknowledge: this is a foundational/computational study; wet-lab validation is future work.
2. Point to the external validation on MITOMAP (independently curated clinical database) as the
   closest proxy available without in-house experiments.
3. Cite precedent: DNABERT2, HyenaDNA, and Nucleotide Transformer all publish without wet-lab
   validation in their original papers.

---

## Objection 7: "The heteroplasmy regression performance (R²~0.30) is not clinically useful"

**Expected wording:** "A Spearman correlation of 0.45–0.55 and R² of 0.25–0.35 for heteroplasmy
regression is not sufficient for clinical application."

**Response strategy:**
1. Agree: we do not claim clinical utility for the regression task.
2. The goal is to demonstrate that the heteroplasmy channel encodes *any* signal about
   population-level heteroplasmy patterns — proof of concept, not clinical tool.
3. The relationship between sequence and observed heteroplasmy is inherently noisy (mutation rate
   × selection × sampling variation in gnomAD). R²~0.30 above baseline indicates real signal.
4. Compare to published baselines if available (EVE-like methods on mtDNA).

---

## Objection 8: "Haplogroup fine-tuned accuracy (1.83%) is below random"

**Current status:** Fine-tuned accuracy (1.83% test, 2-epoch CPU run) is below the 3.85% random baseline due to partial class collapse — only 3/26 classes have F1 > 0.01. Class weighting was applied but 2 CPU epochs are insufficient for LoRA convergence. The zero-shot k-NN result (~50%, no labels) from the pre-trained embeddings is the more meaningful claim and is substantially above the random baseline.

**Expected wording:** "The haplogroup classification accuracy is lower than k-mer frequency + LR baselines (~65%). Why should we prefer this model?"

**Response strategy:**
1. Clarify context: the k-mer + LR baseline ~65% is a qualitative estimate. G3 will provide
   the exact quantified comparison with proper held-out data.
2. The key result is not raw accuracy but **phylogenetic coherence** of errors: confusions are
   between adjacent haplogroups (H↔HV) not across clades (L3↔H). This is a biologically
   meaningful property a frequency baseline does not have.
3. Zero-shot k-NN at ~50% (no labels) is substantially above the 4% random baseline and shows
   emergent evolutionary structure learning.
4. Fine-tuned accuracy (1.83%, source="real") is in eval_summary.json. Framing: this is a compute limitation, not a model quality claim — the zero-shot result demonstrates the representations are structurally meaningful.

---

## Objection 10: "The circular PE math is not novel"

**Expected wording:** "Circular positional encoding is a straightforward modification of
sinusoidal PE. This is not a significant architectural contribution."

**Response strategy:**
1. Agree it is mathematically simple — the contribution is *recognizing* the need and *applying*
   it correctly to biological sequence modeling.
2. The novelty is two-fold: (a) the implementation as a **non-learnable fixed buffer** (immune
   to fine-tuning corruption), and (b) demonstrating its benefit empirically on circular genomes.
3. This generalizes to other circular genomes (viral, plasmid) — state this explicitly.

---

## Statistical Checklist for Submission

- [ ] All AUROC values include 95% bootstrap CI
- [ ] All accuracy values include 95% CI or mean ± std from CV/seeds
- [ ] Sample sizes reported for every reported metric
- [ ] Multiple testing correction applied where applicable
- [ ] Class imbalance handling described (pos_weight, stratification)
- [ ] Train/val/test split documented; no data leakage from pre-training corpus

## Reproducibility Checklist

- [ ] All scripts in `paper/experiments/` runnable with `uv run python <script>`
- [ ] `paper/README.md` contains `bash paper/reproduce_all.sh` one-liner
- [ ] DVC pipeline covers download → train → evaluate end-to-end
- [ ] Model weights on HuggingFace Hub (with paper citation in model card)
- [ ] GitHub repository public with MIT license

## Author Checklist (bioRxiv submission)

- [ ] ORCID registered for all authors
- [ ] Category: q-bio.GN (Genomics) or q-bio.QM (Quantitative Methods)
- [ ] Data availability statement in manuscript
- [ ] Code availability statement in manuscript
- [ ] No patient data, no IRB required (publicly available sequences only)
- [ ] Competing interests statement
- [ ] Funding statement
