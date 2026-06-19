# Cover Letter — mtDNA-FM Preprint Submission

---

**To:** bioRxiv Editorial Team

**From:** Thawfeek Varusai (vthawfeek@gmail.com)

**Submission type:** New Preprint

**Subject area:** Bioinformatics

---

## Cover Letter Text

Dear bioRxiv Team,

I am submitting a new preprint entitled **"mtDNA-FM: A Domain-Specialized Foundation Model for Mitochondrial DNA with Circular Positional Encoding and Heteroplasmy Modeling"** for posting on bioRxiv in the Bioinformatics subject area.

This manuscript presents the first foundation model dedicated to mitochondrial DNA (mtDNA). Existing genomic foundation models — DNABERT-2, HyenaDNA, and Nucleotide Transformer — are pre-trained exclusively on nuclear genomes and employ linear positional encodings that structurally misrepresent the circular topology of mtDNA. Our model addresses two specific architectural problems: (1) a circular positional encoding that enforces continuity at the genome junction, eliminating the artificial discontinuity at the D-loop boundary; and (2) a heteroplasmy projection channel that accepts continuous per-position variant allele fractions, enabling the model to distinguish heteroplasmic variant states.

The key results reported in this preprint are entirely zero-shot (no task-specific labels in training):

- **37.9% zero-shot haplogroup identification** (26-class cosine 5-NN; 3.85% random baseline, 9.8× lift; 95% CI 34.4–41.2%)
- **AUROC 0.777** (95% CI 0.731–0.821) on pathogenic variant discrimination (118 ClinVar pathogenic vs. 419 gnomAD benign SNPs)
- **Correct phylogenetic placement** of Neanderthal and Denisovan sequences without any archaic DNA in training
- **Unsupervised recovery of gene-type structure** (protein-coding, tRNA, rRNA) in embedding space

All code, pre-trained weights, LoRA fine-tuning adapters, a Gradio demonstration interface, and a DVC reproducibility pipeline are publicly available at github.com/vthawfeek/mtdna-foundation-model and huggingface.co/vthawfeek/mtdna-foundation-model.

The preprint reports fine-tuning results honestly: LoRA fine-tuning under CPU compute constraints produced class collapse (1.83% accuracy, documented as a compute limitation requiring GPU resources for convergence). This failure mode is scientifically informative and is reported in full to assist practitioners building similar systems.

This work will be of interest to researchers in genomic foundation models, mitochondrial disease, population genetics, and computational biology. The circular positional encoding design is generalizable to any circular genome (bacterial chromosomes, plasmids, viral genomes, chloroplast DNA).

There are no conflicts of interest to declare. The manuscript has not been submitted to any journal.

Sincerely,

**Thawfeek Varusai**
Independent Researcher
vthawfeek@gmail.com
ORCID: [add your ORCID here]

---

## Submission Checklist

### bioRxiv.org Steps

1. Go to: https://www.biorxiv.org/submit-a-manuscript
2. Sign in / create account (ORCID recommended)
3. Click **"New Submission"**
4. Fill in:
   - **Title:** mtDNA-FM: A Domain-Specialized Foundation Model for Mitochondrial DNA with Circular Positional Encoding and Heteroplasmy Modeling
   - **Authors:** Thawfeek Varusai
   - **Author email:** vthawfeek@gmail.com
   - **Category:** Bioinformatics
   - **Subject area:** Genomics and Bioinformatics
5. Upload:
   - Main PDF (compiled from `paper/manuscript/main_filled.tex`)
   - Supplementary PDF (compiled from `paper/manuscript/supplementary.tex`)
6. **License:** CC-BY 4.0 (recommended for maximum reuse)
7. **Data Availability Statement:** Paste from paper Data Availability section
8. **Code Availability Statement:** https://github.com/vthawfeek/mtdna-foundation-model
9. Check "Yes, this manuscript is already available as a preprint" — NO (it isn't yet)
10. Submit → expect 1–3 business day review
11. After DOI issued:
    - Add DOI to HuggingFace Hub model card
    - Post on LinkedIn and X/Twitter with model link

### Post-bioRxiv Journal Submission Options

| Journal | Scope | IF | Notes |
|---------|-------|----|-------|
| Bioinformatics Advances (OUP) | Software & methods | ~4 | Short notes accepted; fast review |
| NAR Genomics & Bioinformatics | Genomics methods | ~5 | Open access; software papers welcome |
| Briefings in Bioinformatics | Methods reviews | ~9 | Good for domain-first models |
| Nucleic Acids Research (full paper) | Methods + resources | ~14 | Requires benchmarking vs. baselines |
| Genome Biology | Major methods | ~17 | Requires ablations + baselines |

**Recommended for bioRxiv → journal path:**
1. Submit bioRxiv now (2–3 weeks to prepare)
2. Run GPU fine-tuning + ablations (4–6 weeks; see `paper/run_paper.sh`)
3. Submit extended version to NAR or Genome Biology (6–8 weeks from now)
