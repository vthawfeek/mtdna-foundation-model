# Data Pipeline

This document covers every dataset used in mtDNA-FM, how to download it, what the preprocessing steps do and why, and how the DVC pipeline connects the steps together.

---

## Datasets

### HmtDB (Human mtDNA)

| Property | Value |
|---|---|
| URL | https://www.hmtdb.uniba.it/ |
| Size | ~50 MB zip, ~47,000 sequences |
| Format | FASTA + metadata CSV |
| Used for | Phase 2 pre-training (human-only), haplogroup fine-tuning labels |

HmtDB is the primary human mtDNA reference database. Each sequence comes with a haplogroup assignment, geographic origin, and sample metadata. The database has a European cohort bias (haplogroup H is over-represented), which is a known limitation documented in the model card.

If HmtDB bulk download is unavailable, the same sequences can be retrieved from NCBI:
```
human[Organism] AND mitochondrion[Filter] AND complete genome[Title]
```

### NCBI Vertebrate mtDNA

| Property | Value |
|---|---|
| Query | `vertebrate[Organism] AND complete genome[Title] AND mitochondrion[Filter]` |
| Size | ~200 MB, ~30,000 sequences |
| Format | GenBank or FASTA via Entrez |
| Used for | Phase 1 cross-species pre-training |

This corpus provides broad evolutionary diversity — mammals, fish, reptiles, birds. The goal is to teach the model conserved structural features before specializing on human variation. No haplogroup labels are needed for Phase 1 (MLM objective only).

The `ncbi_client.py` uses Entrez `esearch` + `efetch` with `usehistory=True` (WebEnv) to batch-fetch records in chunks of 500. This avoids time-outs on large result sets. Set `NCBI_API_KEY` in your environment for 10 req/s instead of 3 req/s.

### gnomAD v3.1 chrM

| Property | Value |
|---|---|
| URL | `storage.googleapis.com/gcp-public-data--gnomad/release/3.1/vcf/genomes/gnomad.genomes.v3.1.sites.chrM.vcf.bgz` |
| Size | ~50 MB (chrM only) |
| Format | bgzipped VCF |
| Used for | Heteroplasmy labels, benign variant set for pathogenicity task |

**Always extract chrM first with tabix** before loading. The full gnomAD VCF is 5 GB. The chrM-only file is ~50 MB.

```bash
tabix -h gnomad.genomes.v3.1.sites.chrM.vcf.bgz chrM > chrM_only.vcf
```

gnomAD provides per-variant heteroplasmy levels (fraction of mtDNA copies carrying the alternate allele per individual). Variants with AF > 0.01 are used as the benign negative class for pathogenicity training.

### ClinVar Pathogenic mtDNA

| Property | Value |
|---|---|
| FTP | `ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz` |
| Filtered size | ~2 MB (chrM only) |
| Used for | Positive class in pathogenicity fine-tuning |

After downloading, filter to chrM and keep only `CLNSIG` records containing `Pathogenic` or `Likely_pathogenic`. This yields approximately 2,000 labeled pathogenic variants.

### PhyloTree Build 17

| Property | Value |
|---|---|
| URL | phylotree.org |
| Size | ~1 MB |
| Used for | Haplogroup-defining variant markers |

PhyloTree defines 5,400+ haplogroup-defining variants. This is used to construct the `haplogroup_markers.parquet` file that maps each variant position to a haplogroup label. The haplogroup assignment in HmtDB is already derived from PhyloTree, so this dataset is used for validation and interpretation rather than direct training signal.

### Ancient DNA (Zero-Shot)

| Sequence | NCBI Accession | Source |
|---|---|---|
| Neanderthal | NC_011137.1 | Vindija Cave, Croatia |
| Denisovan | FR695060.1 | Altai Cave, Russia |

These sequences are never included in training. They are used in Day 20 to demonstrate that the model learned evolutionary structure: zero-shot embeddings place ancient hominins near but distinct from modern human root haplogroups.

---

## Preprocessing Steps

The preprocessor (`mtdna_fm/data/preprocessor.py`) runs four functions in sequence. Each function is independent and testable.

### Step 1: `clean_sequence`

```python
def clean_sequence(seq: str) -> str:
    seq = seq.upper()
    seq = re.sub(r"[^ACGTN]", "N", seq)
    # Remove trailing junction duplicate if present
    n = 200
    if len(seq) > 2 * n and seq[:n] == seq[-n:]:
        seq = seq[:-n]
    return seq
```

**What it does:** Uppercases all bases, replaces non-ACGTN characters with N (maps ambiguity codes like R, Y, S, W to N), and removes a trailing junction duplicate.

**Why the junction duplicate check:** HmtDB (and some other databases) append the first 200 bases to the end of each sequence so that analysis tools using sliding windows always get complete k-mers at the circular junction. The model handles circularity differently (via circular PE and explicit circular tokenization), so this suffix is redundant and should be removed before length normalization.

### Step 2: `normalize_length`

```python
def normalize_length(seq: str, target_length: int = 16569, pad_position: int = 576) -> str:
    if current < target_length:
        seq = seq[:pad_position] + "N" * n_pad + seq[pad_position:]
    else:
        seq = seq[:target_length]
    return seq
```

**What it does:** Pads sequences shorter than 16,569 bp with N characters, or trims sequences longer than 16,569 bp. Padding is inserted at position 576 (the start of the D-loop), not appended to the end.

**Why insert at the D-loop:** The protein-coding genes in mtDNA have canonical genomic coordinates. If padding is appended to the 3' end, every downstream coordinate calculation remains correct. However, if padding is inserted at the D-loop, all coding gene coordinates stay at their expected positions. The D-loop is the most variable region anyway (7× higher Shannon entropy than coding regions), so inserting N characters there causes less downstream confusion than inserting them into a conserved gene.

### Step 3: `stratified_split`

Adds a `split` column (`train`/`val`/`test`) to the DataFrame using `sklearn.model_selection.StratifiedShuffleSplit` with an 80/10/10 ratio, stratified by haplogroup.

**Why stratify by haplogroup:** The HmtDB dataset has extreme haplogroup imbalance (haplogroup H comprises ~40% of European-origin samples). Without stratification, the test set might contain only a subset of haplogroups, making accuracy estimates unreliable.

Cross-species sequences (Phase 1 corpus) have no haplogroup label and always go to `train`.

### Step 4: Parquet output

Each split is saved as a Parquet file with these columns:

| Column | Type | Description |
|---|---|---|
| `accession` | str | Source database accession ID |
| `sequence` | str | Cleaned, length-normalized sequence (always 16,569 bp) |
| `haplogroup` | str or null | Major haplogroup (26 classes), null for cross-species |
| `species` | str | Taxonomy label |
| `geographic_origin` | str or null | Country/region from HmtDB metadata |
| `het_level_vector` | list[float] or null | Per-position heteroplasmy from gnomAD, when available |

Parquet is preferred over CSV for this data because: it preserves dtypes (avoids str/float ambiguity on heteroplasmy floats), it is ~5× smaller due to column compression, and Pandas reads it ~10× faster than CSV for 47k rows.

---

## DVC Pipeline

`dvc.yaml` defines the full pipeline as a directed acyclic graph. Each stage has explicit `deps` (inputs), `outs` (outputs), and `params` so DVC can determine which stages are stale and need re-running.

```yaml
stages:
  download_hmtdb:
    cmd: uv run mtdna-download --source hmtdb --output data/raw/hmtdb
    outs:
      - data/raw/hmtdb/sequences.fasta:
          persist: true
      - data/raw/hmtdb/metadata.parquet:
          persist: true

  download_ncbi:
    cmd: uv run mtdna-download --source ncbi-refseq --output data/raw/ncbi
    outs:
      - data/raw/ncbi/vertebrate_mtdna.fasta:
          persist: true

  download_variants:
    cmd: uv run mtdna-download --source gnomad --source clinvar --source phylotree --output data/raw/variants
    outs:
      - data/raw/variants/:
          persist: true

  preprocess:
    cmd: uv run mtdna-preprocess --raw-dir data/raw --output-dir data/processed
    deps:
      - data/raw/hmtdb/sequences.fasta
      - data/raw/ncbi/vertebrate_mtdna.fasta
    outs:
      - data/processed/train.parquet
      - data/processed/val.parquet
      - data/processed/test.parquet

  build_vocabulary:
    cmd: uv run python -c "from mtdna_fm.tokenizer.vocabulary import KmerVocabulary; KmerVocabulary.build(k=6).save_pretrained('models/vocabulary')"
    outs:
      - models/vocabulary/

  pretrain_phase1:
    cmd: uv run mtdna-train --config configs/pretraining_phase1.yaml --model-config configs/model_small.yaml
    deps:
      - data/processed/train.parquet
      - configs/pretraining_phase1.yaml
    outs:
      - models/phase1_v1/:
          persist: true
          cache: false

  pretrain_phase2:
    cmd: uv run mtdna-train --config configs/pretraining_phase2.yaml --model-config configs/model_small.yaml
    deps:
      - models/phase1_v1/
      - data/processed/train.parquet
    outs:
      - models/phase2_v1/:
          persist: true
          cache: false

  finetune_haplogroup:
    cmd: uv run mtdna-finetune --config configs/finetuning_haplogroup.yaml
    deps:
      - models/phase2_v1/
      - data/processed/train.parquet
    outs:
      - models/finetune_haplogroup_v1/:
          persist: true

  evaluate:
    cmd: uv run mtdna-evaluate --model models/finetune_haplogroup_v1 --test-data data/processed/test.parquet --output reports/eval_summary.json
    deps:
      - models/finetune_haplogroup_v1/
      - data/processed/test.parquet
    metrics:
      - reports/eval_summary.json:
          cache: false
```

`persist: true` prevents `dvc gc` from deleting model checkpoints. `cache: false` on metrics files lets DVC track them as metrics without caching them like regular outputs.

### Running the full pipeline

```bash
# Run everything from scratch (download data, preprocess, train, evaluate)
dvc repro

# Check which stages are stale
dvc status

# Show final evaluation metrics
dvc metrics show

# Reproduce in a fresh clone
git clone https://github.com/vthawfeek/mtdna-foundation-model /tmp/repro
cd /tmp/repro && uv sync && dvc repro
```
