# Related Work: mtDNA-FM Literature Review

This document covers prior work in four areas: (1) genomic foundation models, (2) mtDNA-specific
bioinformatics tools, (3) variant effect prediction methods, and (4) positional encoding variants.
It is the source for the Related Work section of the paper.

---

## 1. Genomic Foundation Models

### 1.1 BERT-derived models for DNA

**DNABERT** (Ji et al., 2021; *Bioinformatics*)
- First BERT-based model for DNA sequences
- 6-mer tokenization on human reference genome (GRCh38)
- Evaluated on transcription factor binding, promoter prediction, splice sites
- Limitation: human nuclear DNA only; linear positional encoding; no circular support
- Citation key: `ji2021dnabert`

**DNABERT-2** (Zhou et al., 2023; ICLR 2024)
- BPE tokenization (corpus-dependent vocabulary)
- Multi-species (32 reference genomes): human, mouse, fly, zebrafish, etc.
- 117M parameters; substantially larger than mtDNA-FM
- No heteroplasmy support; linear PE; nuclear DNA only
- State-of-the-art on benchmark suite (GUE — Genome Understanding Evaluation)
- The closest existing model in spirit; our primary baseline
- Citation key: `zhou2024dnabert2`

**Nucleotide Transformer** (Dalla-Torre et al., 2023; *Nature Methods*)
- Character-level and k-mer tokenization variants (3-mer, 6-mer)
- Multiple model sizes (50M – 2.5B parameters)
- Multi-species (850 human + 1000 other species genomes)
- Evaluated on 18 downstream genomic tasks
- Linear positional encoding; no circular/heteroplasmy support
- Citation key: `dallatorre2023nucleotidetransformer`

### 1.2 Long-range and alternative architectures

**HyenaDNA** (Nguyen et al., 2023; NeurIPS 2023)
- Hyena (implicit) operator; not Transformer; ultra-long context (up to 1M tokens)
- Single-character tokenization
- Handles very long sequences better than standard attention
- Pre-trained on human reference genome; nuclear DNA only
- No circular topology or heteroplasmy handling
- Citation key: `nguyen2023hyenadna`

**Evo** (Nguyen et al., 2024; *Science*)
- Striped Hyena architecture; 7B parameters
- Pre-trained on 300M nucleotides across prokaryotes and eukaryotes
- Includes some bacterial plasmid (circular) sequences but no explicit circular PE
- Generation, prediction, and design tasks
- Focuses on protein-coding and regulatory evolution, not mtDNA-specific tasks
- Citation key: `nguyen2024evo`

**GenomicBERT / GROVER** (various, 2022–2024)
- Several BERT variants pre-trained on plant, microbial, or pan-genome DNA
- Less directly relevant; no circular genome support

### 1.3 Why existing models fail on mtDNA

| Issue | DNABERT2 | HyenaDNA | Evo | mtDNA-FM |
|-------|----------|----------|-----|----------|
| Circular topology modeled | ✗ | ✗ | ✗ (partially) | ✓ (fixed buffer) |
| Heteroplasmy channel | ✗ | ✗ | ✗ | ✓ |
| Domain: mtDNA pre-training | ✗ | ✗ | ✗ | ✓ |
| Haplogroup structure learned | No | No | No | Yes (emergent) |
| Model size (params) | 117M | 6.5M–650M | 7B | 6M |
| Laptop trainable? | No | Small only | No | Yes |

---

## 2. mtDNA-Specific Bioinformatics Tools

### 2.1 Haplogroup classification tools

**HaploGrep 2** (Weissensteiner et al., 2016; *Nucleic Acids Research*)
- Rule-based haplogroup classifier using PhyloTree build hierarchy
- Takes variant calls (VCF) as input; not raw sequence embeddings
- Deterministic: assigns haplogroup based on variant presence/absence
- Gold standard for haplogroup annotation; used to generate HmtDB labels
- Citation key: `weissensteiner2016haplogrep`

**Haplocheck** (Weissensteiner et al., 2021; *Genome Research*)
- Detects haplogroup contamination and heteroplasmy from sequencing data
- Uses HaploGrep2 internally; adds mixture model for contamination
- Relevant: demonstrates that heteroplasmy detection has clinical utility
- Citation key: `weissensteiner2021haplocheck`

**MitoTool** (Fan & Yao, 2011; *Mitochondrion*)
- Python tool for haplogroup assignment and mtDNA analysis pipeline
- Simpler rule-based approach than HaploGrep2
- Citation key: `fan2011mitotool`

### 2.2 Pathogenicity and variant databases

**MITOMAP** (Lott et al., 2013; *Nucleic Acids Research*)
- Curated database of human mitochondrial DNA variants
- Includes disease associations, confirmed pathogenic variants, population frequencies
- Primary independent validation source (G8)
- URL: https://www.mitomap.org
- Citation key: `lott2013mitomap`

**HelixMTdb** (Bolze et al., 2019; *bioRxiv*)
- Allele frequencies from 196,554 genomes (Helix population cohort)
- Large-scale unbiased population frequency reference for mtDNA variants
- Complements gnomAD; used for G8 external validation
- Citation key: `bolze2019helixmtdb`

**MitImpact** (Castellana & Mazza, 2013; *Bioinformatics*)
- Pre-computed pathogenicity scores for all mtDNA non-synonymous variants
- Aggregates multiple computational predictors (PolyPhen-2, SIFT, MutationAssessor)
- Comparison against MitImpact scores would strengthen pathogenicity results
- Citation key: `castellana2013mitimpact`

### 2.3 Reference databases used in this work

**HmtDB** (Clima et al., 2017; *Nucleic Acids Research*)
- 47,000+ human complete mitochondrial genomes with metadata
- Haplogroup labels, geographic origin, disease status
- Primary source for Phase 2 pre-training and fine-tuning
- Citation key: `clima2017hmtdb`

**PhyloTree** (van Oven & Kayser, 2009; *Human Mutation*)
- Phylogenetic classification of human mitochondrial haplogroups
- Build 17 used in this work (2016); defines the 26 major haplogroup labels
- Citation key: `vanoven2009phylotree`

**gnomAD** (Karczewski et al., 2020; *Nature*)
- Population allele frequencies for nuclear and mitochondrial DNA
- Version 3.1 chrM variants used as benign examples for pathogenicity training
- Citation key: `karczewski2020gnomad`

**ClinVar** (Landrum et al., 2018; *Nucleic Acids Research*)
- Clinically interpreted variants with disease associations
- Pathogenic mtDNA variants used as positive training examples
- Citation key: `landrum2018clinvar`

---

## 3. Variant Effect Prediction Methods

### 3.1 Classical methods

**SIFT** (Ng & Henikoff, 2003; *Nucleic Acids Research*)
- Evolutionary conservation-based variant scoring
- Uses alignment of homologous sequences; applicable to protein-coding mtDNA
- Citation key: `ng2003sift`

**PolyPhen-2** (Adzhubei et al., 2010; *Nature Methods*)
- Structure- and conservation-based pathogenicity prediction for missense variants
- Only applicable to protein-coding variants (not tRNA, rRNA, D-loop)
- Citation key: `adzhubei2010polyphen`

### 3.2 Deep learning methods

**EVE** (Fraternali et al. / Marks lab, 2021; *Nature*)
- Unsupervised variational autoencoder on protein multiple sequence alignments
- Predicts variant effect from evolutionary distribution
- Inspired the use of pre-training for variant pathogenicity; not applicable to non-coding DNA
- Citation key: `fraternali2021eve`

**ESM-1v** (Meier et al., 2021; *NeurIPS*)
- Protein language model for variant effect prediction (log-likelihood ratio)
- Zero-shot performance competitive with supervised methods
- Directly analogous to what mtDNA-FM does for DNA; cited as prior art
- Citation key: `meier2021esm1v`

**AlphaMissense** (Cheng et al., 2023; *Science*)
- Pathogenicity prediction for human missense variants via protein structure context
- Nuclear DNA focused; no mtDNA application
- Shows the field is moving toward structure-informed variant effect prediction
- Citation key: `cheng2023alphamissense`

### 3.3 Gap

No published method uses a pre-trained **DNA language model** specifically for mtDNA variant
pathogenicity prediction. MitImpact aggregates existing scores; no model learns from raw
sequence context at nucleotide resolution for all variant classes.

---

## 4. Positional Encoding Variants

### Standard sinusoidal PE (Vaswani et al., 2017)

- `PE[pos, 2i] = sin(pos / 10000^{2i/d})`
- Assigns unique, fixed encoding to each absolute position
- Position 16568 and position 0 receive maximally different encodings
- Breaks the circular mtDNA junction (D-loop spans positions 303–16569 on circle)
- Citation key: `vaswani2017attention`

### Learnable absolute PE

- Replaces sinusoidal with a learnable embedding table
- Gradient-updatable during pre-training and fine-tuning
- Suffers the same circularity problem as sinusoidal; additionally, can be "unlearned"
  during domain fine-tuning on tasks with shorter sequence contexts

### Relative PE (Shaw et al., 2018; RoPE — Su et al., 2021)

- Encodes relative distance between token pairs, not absolute positions
- RoPE: multiplies Q, K by a rotation matrix depending on relative position
- Does not explicitly model circularity — wrapping from 16568 back to 0 is not handled
- These methods improve extrapolation to longer sequences; not designed for circular topologies
- Citation keys: `shaw2018relative`, `su2021rope`

### ALiBi (Press et al., 2022)

- Biases attention scores by relative distance (linear penalty)
- Length generalization without explicit PE; also does not model circularity
- Citation key: `press2022alibi`

### Circular PE (this work)

- Substitutes `pos / genome_length × 2π` for `pos` in the sinusoidal formula
- Positions 0 and 16,569 receive identical encodings — the junction is smooth
- Implemented as a non-learnable `nn.Buffer` to prevent fine-tuning corruption
- Generalizes to any circular genome (plasmid, virus, organelle)
- To our knowledge, no prior published work applies circular positional encoding to
  sequence foundation models (though the mathematical extension is straightforward)

---

## BibTeX Keys to Add to references.bib

```
ji2021dnabert
zhou2024dnabert2
dallatorre2023nucleotidetransformer
nguyen2023hyenadna
nguyen2024evo
weissensteiner2016haplogrep
weissensteiner2021haplocheck
fan2011mitotool
lott2013mitomap
bolze2019helixmtdb
castellana2013mitimpact
clima2017hmtdb
vanoven2009phylotree
karczewski2020gnomad
landrum2018clinvar
ng2003sift
adzhubei2010polyphen
fraternali2021eve
meier2021esm1v
cheng2023alphamissense
vaswani2017attention
shaw2018relative
su2021rope
press2022alibi
devlin2018bert
```

All entries need verified DOIs before inclusion in the manuscript. Cross-check with
Semantic Scholar / CrossRef before adding to `references.bib`.
