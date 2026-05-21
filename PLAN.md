# mtDNA Foundation Model: 4-Week Portfolio Plan

---

## How to Use This Plan

### Where the plan lives

The plan is stored in two places:

1. **In the repo** at `PLAN.md` — this is the primary copy, tracked in git, visible on GitHub, and what the `/day` command reads.
2. **System copy** at `/home/user/.claude/plans/i-want-to-build-idempotent-starlight.md` — used only by Claude Code's plan mode. Kept in sync with `PLAN.md` manually when the plan changes.

The plan is also pinned via `CLAUDE.md` in the project root. Claude Code always loads `CLAUDE.md` into context, so the plan reference and workflow are always visible without having to find any file path.

When the plan is updated (e.g., a day is marked complete), update `PLAN.md` in the repo, commit, and push. The system copy can be updated separately if plan mode needs to reflect the change.

### Triggering a Day's Work

Each day's work is triggered by typing `/day N` in Claude Code (where N is the day number, 1-28).

This works via a project slash command at `.claude/commands/day.md`. The command reads this plan, executes all tasks for the requested day, creates a daily report, commits all changes, and pushes to GitHub.

**To run Day 2:** type `/day 2`
**To run Day 7:** type `/day 7`

The command will:
1. Execute all tasks listed under that day
2. Run `uv run ruff check` and `uv run pytest` to confirm nothing is broken
3. Write `reports/day-N-<topic>.md` (the daily report)
4. Commit everything with a descriptive message
5. Push to `origin main`

### GitHub Setup (one-time, before Day 2)

Day 1 is already committed locally. Before starting Day 2, connect the local repo to GitHub:

```bash
# 1. Create the repo on GitHub (do this at github.com/vthawfeek manually, or via API)
#    Repo name: mtdna-foundation-model
#    Visibility: Public
#    Do NOT initialise with README or .gitignore (the local repo already has these)

# 2. Add the remote
git remote add origin https://github.com/vthawfeek/mtdna-foundation-model.git

# 3. Push the existing commit
git push -u origin main
```

For authentication, use a GitHub Personal Access Token (PAT):
- Go to github.com -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
- Generate a token with `repo` scope
- When git prompts for a password, paste the token (or store it with `gh auth login` once `gh` is installed)

To avoid repeated prompts:
```bash
git config credential.helper store
# then git push once — git will cache the token permanently
```

### Daily Report Format

At the end of each day, a file is written to `reports/day-N-<topic>.md` with this structure:

```markdown
# Day N: <Topic>

## What was built
Short bullets — the actual files and components created or modified.

## What was learned
The concepts this day covered and why they matter for the project.

## Key decisions
Each significant technical choice and the reasoning behind it.

## Verification
What was run to confirm everything works.

## Next up
One sentence on what Day N+1 will tackle.
```

These reports accumulate into a readable build diary that feeds directly into the blog posts.

---

## Why This Project

Mitochondrial DNA is the right niche for this portfolio. The full human mtDNA genome is 16,569 base pairs, which makes it laptop-trainable, but it is clinically significant enough to matter: mutations cause 1 in 5,000 rare diseases, copy number changes are biomarkers for cancer and aging, and heteroplasmy (mixed wild-type and mutant populations within a single cell) makes it biologically unusual. No dedicated foundation model for mtDNA exists. DNABERT2, HyenaDNA, and Nucleotide Transformer are all trained on nuclear DNA and do not model the circular genome topology, heteroplasmy, or haplogroup structure.

This project produces a pre-trained encoder, three fine-tuned models, a reusable Python package, a reproducible DVC pipeline, a live HuggingFace Spaces demo, and a body of writing that shows how the model was built and why each decision was made. The writing is the portfolio, not the marketing.

The working directory is `/home/user/Documents/Personal/ai_lab/mtdna_foundation_model`. Two sibling projects provide patterns to follow: `sc_foundation_model` (single-cell BERT, PyTorch + Transformers + MLflow + DVC + uv) and `foundational_kg_model` (Snakemake pipeline). Nearly every file in this project has a direct analogue in `sc_foundation_model`.

**Compute:** Running on Linux. Use CUDA if a GPU is available. Otherwise, `hidden_size=256, 6 layers` is about 6M parameters and trains in 8-12 hours on a modern CPU at 50k steps.

---

## Datasets

All of these are freely accessible with no registration required.

| Dataset | Access | Size | Use |
|---------|--------|------|-----|
| HmtDB human mtDNA | https://www.hmtdb.uniba.it/ bulk FASTA + metadata CSV | ~50 MB, ~47k sequences | Phase 2 pre-training and fine-tuning labels |
| NCBI vertebrate complete mtDNA | Entrez query: `vertebrate[Organism] AND complete genome[Title] AND mitochondrion[Filter]` | ~200 MB, ~30k sequences | Phase 1 cross-species curriculum |
| gnomAD v3.1 chrM VCF | `storage.googleapis.com/gcp-public-data--gnomad/release/3.1/vcf/genomes/gnomad.genomes.v3.1.sites.chrM.vcf.bgz` | ~50 MB (chrM only via tabix) | Heteroplasmy labels, benign variant set |
| ClinVar pathogenic mtDNA | `ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz` then filter chrM | ~2 MB | Pathogenicity fine-tuning |
| PhyloTree Build 17 | phylotree.org, 5,400+ haplogroup-defining variants | ~1 MB | Haplogroup fine-tuning labels |
| Neanderthal + Denisovan mtDNA | NCBI accessions NC_011137.1 (Neanderthal) and FR695060.1 (Denisovan) | <1 MB | Zero-shot ancient DNA demo |

**Risk note:** If HmtDB bulk download is unavailable, fall back to NCBI: `human[Organism] AND mitochondrion[Filter] AND complete genome[Title]` produces the same ~47k sequences. For gnomAD, always run `tabix -h gnomad.vcf.bgz chrM > chrM_only.vcf` before loading — the full file is 5 GB.

---

## Architecture

**Tokenization:** 6-mer overlapping sliding window over the DNA sequence. Vocabulary is all 4^6 = 4,096 possible 6-mers plus 6 special tokens (`[PAD]`, `[CLS]`, `[MASK]`, `[UNK]`, `[SEP]`, `[HET]`), for a total of 4,102 tokens. This vocabulary is deterministic and built at initialization, not learned from data, which means it is interpretable and reproducible.

**Circular positional encoding:** mtDNA is circular, so position 16,569 is genomically adjacent to position 1. Standard sinusoidal positional encoding treats these as maximally distant. The novel encoding uses:

```
PE[pos, 2i]   = sin(2pi * pos / genome_length * 1/10000^(2i/d))
PE[pos, 2i+1] = cos(2pi * pos / genome_length * 1/10000^(2i/d))
```

This is a fixed, non-learnable buffer registered on the model. The circular topology is a biological fact, not a parameter.

**Heteroplasmy projection:** A continuous float channel (per-position heteroplasmy level, 0.0 to 1.0) is projected into the embedding space alongside the k-mer token IDs, following the same pattern as scFM's `expression_values` channel. This is the direct analogue of `value_projection` in `scfm/model/embeddings.py`.

**Two-phase pre-training:**
- Phase 1: cross-species vertebrate mtDNA (30k sequences, broader diversity, `het_weight=0`)
- Phase 2: load Phase 1 checkpoint, train on human HmtDB sequences, enable heteroplasmy loss (`het_weight=0.3`)

**Model size:** `hidden_size=256, 6 layers, 8 heads, intermediate_size=1024` gives approximately 6M parameters.

**Fine-tuning tasks:**
1. Haplogroup classification (26 major haplogroups, LoRA r=8)
2. Pathogenic variant prediction (binary, LoRA r=4, ClinVar vs gnomAD common variants)
3. Heteroplasmy level regression (gnomAD variants with heteroplasmic carriers, Huber loss)

---

## Project Structure

```
mtdna_foundation_model/
├── mtdna_fm/
│   ├── model/
│   │   ├── config.py               # MtDNAConfig(PretrainedConfig)
│   │   ├── embeddings.py           # KmerEmbedding + CircularPE + HetProjection
│   │   ├── transformer.py          # Standard pre-LN transformer (copy from scfm)
│   │   └── model.py                # MtDNAModel + MtDNAForMaskedModeling
│   ├── tokenizer/
│   │   ├── vocabulary.py           # KmerVocabulary (reusable for any small genome)
│   │   └── tokenize.py             # tokenize_sequence (reusable tool)
│   ├── data/
│   │   ├── hmtdb_client.py         # idempotent HmtDB downloader
│   │   ├── ncbi_client.py          # idempotent NCBI Entrez downloader
│   │   ├── variant_downloader.py   # gnomAD/ClinVar/PhyloTree downloader
│   │   ├── preprocessor.py         # clean, normalize, split
│   │   ├── dataset.py              # MtDNADataset (windowed, circular-aware)
│   │   └── variant_dataset.py      # VariantDataset for pathogenicity task
│   ├── training/
│   │   ├── masking.py              # MtDNAMaskingCollator (with blacklist)
│   │   ├── losses.py               # MLM + het MSE combined loss
│   │   └── trainer.py              # two-phase aware trainer
│   ├── inference/
│   │   └── api.py                  # MtDNAEmbedder (reusable public API)
│   ├── evaluation/
│   │   ├── haplogroup_eval.py
│   │   ├── variant_eval.py
│   │   └── viz.py
│   └── scripts/
│       ├── download.py             # mtdna-download CLI
│       ├── preprocess.py           # mtdna-preprocess CLI
│       ├── train.py                # mtdna-train CLI
│       ├── finetune.py             # mtdna-finetune CLI
│       └── evaluate.py             # mtdna-evaluate CLI
├── tests/
│   ├── conftest.py                 # tiny_config, tiny_vocabulary, synthetic_sequence
│   ├── test_tokenizer.py
│   ├── test_model.py
│   └── test_data.py
├── configs/
│   ├── model_small.yaml
│   ├── data.yaml
│   ├── pretraining_phase1.yaml
│   ├── pretraining_phase2.yaml
│   ├── finetuning_haplogroup.yaml
│   └── finetuning_pathogenicity.yaml
├── docs/
│   ├── 01_data_pipeline.md
│   ├── 02_tokenization.md
│   ├── 03_architecture.md
│   ├── 04_pretraining.md
│   └── 05_finetuning_and_evaluation.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pretraining_analysis.ipynb
│   ├── 03_finetuning_results.ipynb
│   └── 04_showcase.ipynb
├── data/raw/, data/interim/, data/processed/
├── models/
├── reports/
│   ├── day-01-scaffold.md
│   ├── day-02-tokenizer.md
│   └── ...
├── .claude/
│   └── commands/
│       └── day.md              # /day N slash command
├── pyproject.toml
├── dvc.yaml
└── README.md
```

---

## End-of-Day Standard Procedure (applies to every day)

Every day ends with the same four steps, executed automatically by `/day N`:

1. `uv run ruff check mtdna_fm/ tests/` — must pass with no errors
2. `uv run pytest tests/ -m "not slow and not integration"` — must pass
3. Write `reports/day-N-<topic>.md` using the format above
4. `git add -p` (stage relevant files), commit with a descriptive message, `git push origin main`

If tests or lint fail, the day is not complete. Fix before committing.

---

## Week 1 (Days 1-7): Infrastructure and Data

The first week is about building a solid foundation that the rest of the project depends on. Every component built this week is a reusable tool. The tokenizer can be used with any small circular genome. The download clients handle rate limiting, retries, and idempotency. The dataset class is the single interface between raw data and the model.

### Day 1: Scaffold and dependencies [COMPLETE — commit 12d614c]

Day 1 is done. The scaffold, pyproject.toml, CLI stubs, CI workflow, and first git commit exist. Before Day 2, complete the GitHub setup steps in the "How to Use This Plan" section above, and create the `/day` slash command file below.

**One-time setup tasks to complete before Day 2:**

1. Create the GitHub repo at github.com/vthawfeek/mtdna-foundation-model (public, no README)
2. `git remote add origin https://github.com/vthawfeek/mtdna-foundation-model.git`
3. `git push -u origin main`
4. Copy this plan into the repo as `PLAN.md` (so it is tracked in git and visible on GitHub)
5. Create `CLAUDE.md` in the project root (pins the plan, shows day status and how to trigger work)
6. Create `.claude/commands/day.md` with the content shown in the Slash Command section below
7. Write `reports/day-01-scaffold.md` (the Day 1 report — what was built, learned, decided)
8. Commit and push: `PLAN.md`, `CLAUDE.md`, `reports/day-01-scaffold.md`, `.claude/commands/day.md`

**CLAUDE.md to create at the project root:**

```markdown
# mtDNA Foundation Model

## Plan

Full 4-week plan: [PLAN.md](PLAN.md) (in this repo — also at `/home/user/.claude/plans/i-want-to-build-idempotent-starlight.md` for plan mode)

## How to trigger a day's work

Type `/day N` where N is the day number (1-28).

Each invocation executes all tasks for that day, runs lint and tests, writes
`reports/day-N-<topic>.md`, commits, and pushes to GitHub.

## Current status

- Day 1: COMPLETE (commit 12d614c) — scaffold, pyproject.toml, CLI stubs, CI
- Day 2: tokenizer (KmerVocabulary + tokenize_sequence)
- Day 3: data download clients (HmtDB, NCBI Entrez)
- Day 4: preprocessing pipeline + EDA notebook
- ...see plan for full schedule

## Project

- GitHub: https://github.com/vthawfeek/mtdna-foundation-model
- HuggingFace: https://huggingface.co/vthawfeek/mtdna-foundation-model (created Day 21)
- Working directory: /home/user/Documents/Personal/ai_lab/mtdna_foundation_model

## Architecture summary

Pre-trained BERT encoder for mitochondrial DNA (16,569 bp circular genome).
Novel: circular positional encoding + heteroplasmy projection channel.
Vocabulary: 4,096 6-mers + 6 special tokens = 4,102 tokens.
Model: 6 layers, 8 heads, 256 hidden dim, ~6M parameters.
```

**Slash command to create at `.claude/commands/day.md`:**

```markdown
Execute Day $ARGUMENTS of the mtDNA Foundation Model project.

Steps:
1. Read the plan at PLAN.md (in the project root — this is the in-repo copy of the full plan)
2. Find the section for Day $ARGUMENTS and execute every task listed there
3. Run: uv run ruff check mtdna_fm/ tests/ — fix any errors
4. Run: uv run pytest tests/ -m "not slow and not integration" — fix any failures
5. Write reports/day-$ARGUMENTS-<topic>.md using the standard daily report format
6. Stage all new and modified files (do not stage data/, models/, or mlruns/)
7. Commit with message: "day $ARGUMENTS: <short description of what was built>"
8. Push to origin main

The working directory is /home/user/Documents/Personal/ai_lab/mtdna_foundation_model.
The plan file contains the full task list for every day. Follow it precisely.
```

**Day 1 deliverables (for the report):**
- `pyproject.toml` with all dependencies and 5 CLI entry points
- Package directory structure with `__init__.py` stubs
- Placeholder CLI scripts in `mtdna_fm/scripts/` (download, preprocess, train, finetune, evaluate)
- `tests/conftest.py` with synthetic sequence fixtures
- `.github/workflows/ci.yml` (lint + test on push/PR)
- `.gitignore` with data/, models/, mlruns/ excluded
- `README.md` skeleton
- First git commit `12d614c`

Initialize the project as a proper Python package from the start, not as a collection of scripts.

```bash
cd /home/user/Documents/Personal/ai_lab/mtdna_foundation_model
git init
uv init --name mtdna-fm --python 3.11
mkdir -p mtdna_fm/{model,tokenizer,data,training,inference,evaluation,scripts}
mkdir -p tests configs docs notebooks data/{raw,interim,processed} models reports
touch mtdna_fm/__init__.py
touch mtdna_fm/{model,tokenizer,data,training,inference,evaluation,scripts}/__init__.py
touch tests/__init__.py tests/conftest.py
```

`pyproject.toml` dependencies (mirror scFM versions):
```toml
[project]
name = "mtdna-fm"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "dvc>=3.50.0",
    "mlflow>=2.12.0",
    "numpy>=1.26.0",
    "scipy>=1.13.0",
    "scikit-learn>=1.4.0",
    "biopython>=1.83",
    "pandas>=2.2.0",
    "pyarrow>=15.0",
    "pyyaml>=6.0",
    "rich>=13.0",
    "typer>=0.12.0",
    "safetensors>=0.4.0",
    "huggingface-hub>=0.22.0",
    "requests>=2.31.0",
    "tqdm>=4.66.0",
]
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.4.0", "ipykernel>=6.0", "gradio>=4.0"]

[project.scripts]
mtdna-download = "mtdna_fm.scripts.download:app"
mtdna-preprocess = "mtdna_fm.scripts.preprocess:app"
mtdna-train = "mtdna_fm.scripts.train:app"
mtdna-finetune = "mtdna_fm.scripts.finetune:app"
mtdna-evaluate = "mtdna_fm.scripts.evaluate:app"
```

Verification: `uv sync --extra dev && uv run pytest` should show 0 collected, 0 errors.

**What you learn:** Why packaging matters from day one (importable module vs script soup), how uv lockfiles differ from pip requirements.txt, why CLI entry points are better than `python scripts/train.py`.

**GitHub:** Push initial scaffold commit. Keep the commit history clean throughout — one commit per logical unit of work, no "fix typo" commits.

**X/Twitter:** Brief post about the project start. Not promotional. Just: "Starting an mtDNA foundation model. The full mitochondrial genome is 16,569 bp — the entire thing fits in a text file. No dedicated FM exists for it. Building one over the next 4 weeks. [repo link]"

### Day 2: Tokenizer as a reusable tool

The tokenizer is designed to be independently usable. Anyone should be able to `from mtdna_fm.tokenizer import KmerVocabulary, tokenize_sequence` and tokenize any short circular genome, not just human mtDNA.

`mtdna_fm/tokenizer/vocabulary.py`: `KmerVocabulary.build(k=6)` generates all 4,096 possible 6-mers from the ACGT alphabet and assigns each a stable index. Special tokens at indices 0-5: `[PAD]=0, [CLS]=1, [MASK]=2, [UNK]=3, [SEP]=4, [HET]=5`. N characters map to `[UNK]`. The vocabulary is deterministic and reproducible across machines. Save and load via `save_pretrained()` and `from_pretrained()` to follow HuggingFace conventions.

`mtdna_fm/tokenizer/tokenize.py`: `tokenize_sequence(seq, vocabulary, k=6, stride=1, max_seq_len=512, circular=True, het_levels=None)` returns a dict with `input_ids`, `attention_mask`, `position_ids`, and `het_values`. When `circular=True`, the last k-1 bases are appended to the front before k-merizing, so the genome junction at position 16568/0 gets its own set of tokens.

`tests/test_tokenizer.py`:
```python
def test_vocabulary_size():
    vocab = KmerVocabulary.build(k=6)
    assert len(vocab) == 4102  # 4096 + 6 special tokens

def test_circular_junction_covered():
    # With circular=True, every base pair is covered by at least one k-mer token
    seq = "ACGT" * 25  # 100 bp synthetic circular genome
    tokens = tokenize_sequence(seq, vocabulary=vocab, circular=True)
    # The junction token (wrapping around position 99 to position 0) must exist
    assert len(tokens["input_ids"]) == len(seq)

def test_encode_decode_roundtrip():
    vocab = KmerVocabulary.build(k=6)
    kmer = "ATGCAT"
    assert vocab.decode(vocab.encode(kmer)) == kmer

def test_vocabulary_save_load(tmp_path):
    vocab = KmerVocabulary.build(k=6)
    vocab.save_pretrained(tmp_path)
    loaded = KmerVocabulary.from_pretrained(tmp_path)
    assert len(loaded) == len(vocab)
```

**What you learn:** The difference between k-mer tokenization and BPE. Why k-mers have a deterministic vocabulary (4^k) while BPE vocabulary depends on training corpus statistics. Why determinism matters for reproducibility.

### Day 3: Data download clients

Both download scripts are idempotent: running them a second time checks what already exists and skips completed work. This is a production practice worth documenting explicitly.

`mtdna_fm/data/hmtdb_client.py`: Downloads the HmtDB bulk FASTA and metadata CSV. Verifies SHA256 of the zip before extraction. Falls back gracefully to NCBI if HmtDB is unavailable. Uses BioPython `SeqIO.parse` for FASTA parsing. Writes `data/raw/hmtdb/sequences.fasta` and `data/raw/hmtdb/metadata.parquet`.

`mtdna_fm/data/ncbi_client.py`: Uses Entrez `esearch` + `efetch` with `usehistory=True` (WebEnv) to batch-fetch records in chunks of 500. Reads NCBI_API_KEY from environment for higher rate limits (10 req/s vs 3 req/s). The script is resumable: it tracks which batches have been downloaded to a progress file and skips completed ones.

`mtdna_fm/scripts/download.py`: Typer CLI that wraps both clients with `--source` flag accepting `hmtdb`, `ncbi-refseq`, `gnomad`, `clinvar`, `phylotree`.

DVC stages for both downloads go in `dvc.yaml` with `persist: true` so DVC does not delete them on `dvc gc`.

Verification:
```bash
uv run mtdna-download --source ncbi-refseq --output data/raw/ncbi
python -c "
from Bio import SeqIO
seqs = list(SeqIO.parse('data/raw/ncbi/vertebrate_mtdna.fasta', 'fasta'))
print(f'{len(seqs)} sequences downloaded')
"
```

### Day 4: Preprocessing pipeline and EDA notebook

`mtdna_fm/data/preprocessor.py`: Each step is a separate function, not a monolithic class. This makes individual steps testable and reusable.

Steps:
1. Clean sequences: uppercase, replace non-ACGTN with N, detect and remove trailing duplicated junction region (some databases append the first 200 bp at the end for completeness)
2. Length normalization: pad shorter sequences to 16,569 with N at the D-loop start (position 576), trim longer ones. All sequences must be exactly 16,569 bp for batching.
3. Stratified 80/10/10 train/val/test split by haplogroup using `sklearn.model_selection.StratifiedShuffleSplit`
4. Cross-species sequences go to pre-training only (no labels needed for Phase 1 MLM)
5. Save as parquet with columns: `accession`, `sequence`, `haplogroup`, `species`, `geographic_origin`, `het_level_vector` (nullable)

`notebooks/01_data_exploration.ipynb`: This notebook is documentation, not exploration throwaway. Include:
- Haplogroup distribution (H dominates in European cohorts, L3 at phylogenetic root)
- Shannon entropy per genomic position (D-loop is 7x more variable than coding regions)
- N-content distribution per sequence (quality filter)
- Sequence length histogram before normalization
- Geographic distribution of HmtDB samples

**X post:** Post one of the figures with context. Not "look at my visualization" but "The D-loop (positions 576-16024) is 7x more variable than the coding region. This is why every haplogroup assignment system focuses on it. The model needs to encode this boundary cleanly — that's one reason circular PE matters."

### Day 5: Variant datasets

Download gnomAD chrM, ClinVar mtDNA, PhyloTree Build 17. Always extract chrM first with tabix.

`mtdna_fm/data/variant_processor.py`: Creates three clean parquet files:
- `variants_gnomad.parquet`: one row per variant-individual combination with `pos`, `ref`, `alt`, `af`, `het_level`, `n_het`, `n_hom`
- `variants_clinvar.parquet`: pathogenic mtDNA variants with `pos`, `ref`, `alt`, `label` (1=pathogenic, 0=benign proxy from gnomAD common variants)
- `haplogroup_markers.parquet`: PhyloTree variant to haplogroup mapping

### Day 6: PyTorch Dataset class

`mtdna_fm/data/dataset.py`: `MtDNADataset(Dataset)` uses overlapping windows (size=512, stride=256) over each 16,569-bp genome, producing about 63 windows per genome and roughly 3M training examples from 47k sequences. The `position_ids` in each window are absolute genomic coordinates (not window-relative) so the circular positional encoding maps them correctly to the genome coordinate system.

Key design decision worth documenting: sequences are short enough that all 47k fit in memory (~780 MB as raw strings). Use `np.memmap` for the tokenized array only if memory is tight.

`mtdna_fm/data/variant_dataset.py`: `VariantDataset` applies a SNP to the rCRS reference sequence, extracts a 512-token window centered on the variant position, and returns the label. Only SNPs for now, not indels.

`tests/test_data.py`:
```python
def test_all_positions_covered():
    # Every genomic position should appear in at least one window
    
def test_circular_junction_window():
    # One window must span the 16568/0 boundary
    
def test_het_values_range():
    # All het values must be in [0.0, 1.0]

def test_dataset_length():
    # Expected window count per genome given stride
```

### Day 7: CI and Blog Post 1

`.github/workflows/ci.yml`: lint + test on every push. Mirror the scFM CI structure. Two jobs: `lint` (ruff check + ruff format --check) and `test` (pytest, skipping `@pytest.mark.slow` and `@pytest.mark.integration`).

**Blog Post 1 (dev.to + LinkedIn):** "Why Mitochondrial DNA Needs Its Own Foundation Model"

Write this honestly. What does circular topology mean for positional encoding? What does heteroplasmy mean for a sequence model that expects a single definitive base at each position? Why do existing DNA models miss these properties? Include the D-loop entropy figure. This post should read like it was written by someone who actually thought about the biology, not a model capabilities summary.

**Week 1 exit criteria:**
- `uv run pytest tests/ -v` all passing
- `uv run ruff check mtdna_fm/` no errors
- `data/processed/train.parquet` exists with >40k rows, all sequences exactly 16,569 bp
- `KmerVocabulary.build(k=6)` produces 4,102 tokens, roundtrip encode/decode works
- GitHub CI badge green

---

## Week 2 (Days 8-14): Model Architecture and Pre-training

This is the hardest week technically. The model is built file by file, following scFM as a template but replacing every scFM-specific component with the mtDNA analogue. Pre-training is launched by Day 10 and runs in the background while the test suite is completed.

### Day 8: Model architecture

`configs/model_small.yaml`:
```yaml
model_type: "mtdna_fm"
vocab_size: 4102
hidden_size: 256
num_hidden_layers: 6
num_attention_heads: 8
intermediate_size: 1024
max_seq_len: 514          # 512 k-mer tokens + CLS + SEP
genome_length: 16569
use_circular_encoding: true
use_het_projection: true
dropout_prob: 0.1
attention_dropout_prob: 0.1
layer_norm_eps: 1.0e-12
pad_token_id: 0
cls_token_id: 1
mask_token_id: 2
```

`mtdna_fm/model/config.py`: `MtDNAConfig(PretrainedConfig)` with `genome_length` and `use_circular_encoding` as new fields. Mirror `scfm/model/config.py`.

`mtdna_fm/model/embeddings.py`: `MtDNACircularPositionalEncoding` is the key novel component. Pre-compute all 16,569 position encodings as a non-learnable buffer at init time. The forward pass indexes into this buffer using absolute position IDs.

```python
class MtDNACircularPositionalEncoding(nn.Module):
    def __init__(self, genome_length: int, hidden_size: int):
        super().__init__()
        pe = torch.zeros(genome_length, hidden_size)
        position = torch.arange(genome_length).float()
        # Circular angle: 2*pi * pos / genome_length
        angle = 2 * torch.pi * position / genome_length
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * 
            (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(angle.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(angle.unsqueeze(1) * div_term)
        self.register_buffer("pe", pe)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pe[position_ids]  # (batch, seq_len, hidden_size)
```

`MtDNAEmbeddings`: combines kmer_embedding + circular_pe + LayerNorm(Linear(1, hidden_size)(het_values)). The het projection is optional and zeros out when het_values are not provided.

`mtdna_fm/model/transformer.py`: copy directly from `scfm/model/transformer.py`. The pre-LayerNorm transformer block is domain-agnostic.

`mtdna_fm/model/model.py`:
- `MtDNAModel(PreTrainedModel)`: base encoder, returns `last_hidden_state` and `pooler_output` (CLS token)
- `MtDNAForMaskedModeling(PreTrainedModel)`: adds `kmer_prediction_head` (Linear: hidden_size -> 4102) and `het_prediction_head` (Linear: hidden_size -> 1, sigmoid output)

### Day 9: Masking and loss

`mtdna_fm/training/masking.py`: Standard BERT masking (80/10/10: mask token, random token, unchanged) with one addition: positions 303-315 (the homopolymeric C-tract in the D-loop) are blacklisted from masking because they are sequencing noise, not biological signal, and teaching the model to predict them is counterproductive.

`mtdna_fm/training/losses.py`:
```python
def mtdna_mlm_loss(
    kmer_logits, kmer_labels, het_preds, het_labels,
    mlm_weight: float = 1.0, het_weight: float = 0.0
) -> torch.Tensor:
    mlm_loss = F.cross_entropy(
        kmer_logits.view(-1, kmer_logits.size(-1)),
        kmer_labels.view(-1),
        ignore_index=-100
    )
    if het_weight > 0 and het_labels is not None:
        het_loss = F.mse_loss(
            het_preds.squeeze(-1)[het_labels != -1],
            het_labels[het_labels != -1]
        )
        return mlm_weight * mlm_loss + het_weight * het_loss
    return mlm_weight * mlm_loss
```

### Day 10: Pre-training launch (Phase 1)

`mtdna_fm/training/trainer.py`: Mirror `scfm/training/trainer.py`. Key difference for Phase 2: load encoder weights from checkpoint but initialize a fresh optimizer state (different learning rate, different schedule).

`configs/pretraining_phase1.yaml`:
```yaml
data:
  train_parquet: "data/processed/train.parquet"
  val_parquet: "data/processed/val.parquet"
  species_filter: "all"
batch_size: 16
gradient_accumulation_steps: 8     # effective batch = 128
num_workers: 4
learning_rate: 1.0e-4
weight_decay: 0.01
warmup_steps: 2000
max_steps: 50000
max_grad_norm: 1.0
fp16: false                         # CPU-safe; set to true for CUDA
gradient_checkpointing: true
mask_prob: 0.15
mlm_weight: 1.0
het_weight: 0.0                     # No het data in cross-species corpus
save_steps: 5000
eval_steps: 2500
log_steps: 100
keep_last_n_checkpoints: 3
mlflow_experiment: "mtdna_fm_pretraining_phase1"
output_dir: "models/phase1_v1"
```

Launch and monitor:
```bash
uv run mtdna-train --config configs/pretraining_phase1.yaml --model-config configs/model_small.yaml
mlflow ui --backend-store-uri mlruns &
```

Expected loss curve: step 0 at about 8.3 (log 4,102, random baseline), step 5k around 5.5-6.0, step 20k around 3.5-4.0, step 50k around 2.5-3.0.

**What you learn:** Why gradient accumulation makes a small-batch model behave like a large-batch model. How cosine LR decay interacts with warmup. How MLflow lets you compare runs without rewriting anything. Why `gradient_checkpointing=True` trades compute for memory on a laptop.

**X post:** "Phase 1 cross-species pre-training launched. 30k vertebrate mtDNA genomes. Random baseline MLM loss is 8.3 (log of 4,096-token vocabulary). Watching it converge."

### Days 11-12: Test suite

Write tests to >80% coverage on model and tokenizer modules. This is not optional polish — it is what makes the codebase production-grade. Every model component that can be tested with a tiny synthetic config should be.

`tests/conftest.py`:
```python
@pytest.fixture()
def tiny_config():
    return MtDNAConfig(
        vocab_size=70,           # 64 3-mers + 6 special tokens
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_seq_len=12,
        genome_length=100,
        use_circular_encoding=True,
    )

@pytest.fixture()
def tiny_vocabulary():
    return KmerVocabulary.build(k=3)  # 64 3-mers, fast for tests

@pytest.fixture()
def synthetic_sequence():
    rng = np.random.default_rng(42)
    return "".join(rng.choice(list("ACGT"), size=100))
```

Test classes to implement (each mirrors the scFM test module):
- `TestKmerVocabulary`: size, special token IDs, encode/decode roundtrip, save/load
- `TestTokenizeSequence`: window count, circular junction, het_values shape, attention_mask sum
- `TestMtDNAConfig`: JSON roundtrip
- `TestMtDNAModel`: forward shapes, pooler_output is CLS, save/load with safetensors, gradient flow end to end, PEFT LoRA compatibility
- `TestMtDNAForMaskedModeling`: loss is scalar tensor, loss decreases over 5 steps on a fixed synthetic batch
- `TestMtDNAMaskingCollator`: masking rate is approximately 15%, blacklisted positions are never masked, 80/10/10 split of masked positions holds
- `TestMtDNAMLMLoss`: CE computed only on masked positions (ignore_index=-100), het MSE computed only where het_labels != -1

### Day 13: Training analysis notebook

`notebooks/02_pretraining_analysis.ipynb`:
- MLM loss and learning rate curves from MLflow
- Token prediction accuracy broken down by genomic region (D-loop vs tRNA genes vs protein-coding)
- Attention weight heatmaps at step 0 vs step 25k: do structured patterns emerge?
- Zero-shot k-NN haplogroup classification using Phase 1 CLS embeddings: extract embeddings for 1,000 test sequences, run 5-fold cross-validated k-NN, report accuracy. Random baseline is about 3.8% (1/26 classes). Expect 30-40% after Phase 1 pre-training.

**X post with one figure from the attention heatmaps.** Explain what you see (or don't see yet) honestly.

### Day 14: Phase 2 launch and Blog Post 2

Verify Phase 1 checkpoint: `models/phase1_v1/` should have `config.json`, `model.safetensors`, `tokenizer_config.json`. Run the zero-shot k-NN check.

`configs/pretraining_phase2.yaml`:
```yaml
resume_from: "models/phase1_v1"
data:
  species_filter: "homo_sapiens"
batch_size: 16
gradient_accumulation_steps: 8
learning_rate: 3.0e-5           # lower than Phase 1
max_steps: 25000
warmup_steps: 500
het_weight: 0.3                  # enable heteroplasmy prediction
mlflow_experiment: "mtdna_fm_pretraining_phase2"
output_dir: "models/phase2_v1"
```

Phase 2 loads encoder weights only, fresh optimizer (modify `_load_checkpoint` in trainer to skip `optimizer.pt`).

Launch Phase 2 and let it run.

**Blog Post 2:** "Building the mtDNA-FM Tokenizer and Circular Positional Encoding"

Cover the actual engineering: how 6-mer tokenization works, what the vocabulary looks like, why circular PE is the mathematically correct choice and not just a novelty, the decision to make het values a continuous channel instead of discretizing them. Include actual code. This should be the kind of blog post someone could use to implement a similar tokenizer themselves.

**Week 2 exit criteria:**
- Phase 1 complete, MLM loss < 3.5 at step 50k
- `pytest --cov=mtdna_fm` shows >80% coverage
- Phase 2 training running
- Notebook 2 figures saved to `docs/figures/`

---

## Week 3 (Days 15-21): Fine-tuning and Evaluation

### Day 15: Genome embedding API

`mtdna_fm/inference/api.py`: `MtDNAEmbedder` is the reusable public interface for the pre-trained model. Anyone who installs `mtdna-fm` should be able to use this class to embed sequences without needing to understand the internal windowing logic.

```python
class MtDNAEmbedder:
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "MtDNAEmbedder": ...
    
    def embed_genome(
        self, sequence: str, het_levels: np.ndarray | None = None,
        pooling: str = "cls_mean"
    ) -> np.ndarray:
        """
        Embed a full 16,569-bp mtDNA genome.
        Strategy: overlapping windows, extract CLS token per window, mean-pool.
        Returns a single vector of shape (hidden_size,).
        """
    
    def embed_variant(
        self, sequence: str, position: int, pooling: str = "token"
    ) -> np.ndarray:
        """
        Embed the context around a specific genomic position.
        Returns the hidden state at the token containing that position.
        """
    
    def embed_dataset(
        self, df: pd.DataFrame, sequence_col: str = "sequence",
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Batch embedding for a DataFrame of sequences.
        Returns np.ndarray of shape (n_sequences, hidden_size).
        """
```

Verify Phase 2 completion. Run zero-shot k-NN: expect >40% haplogroup accuracy from Phase 2 embeddings.

### Day 16: Fine-tuning Task 1: Haplogroup classification

`MtDNAForHaplogroupClassification(PreTrainedModel)`: embeds the full genome via mean-pooled CLS tokens, applies a Linear(256, 26) classification head. Mirror `ScFMForCellTypeClassification` from scFM.

LoRA configuration: `r=8, lora_alpha=16, target_modules=["query", "key", "value", "dense"], lora_dropout=0.1`. The haplogroup dataset is large enough (47k sequences) that r=8 is appropriate.

`configs/finetuning_haplogroup.yaml`:
```yaml
task: haplogroup_classification
num_labels: 26
label_column: haplogroup
base_model: "models/phase2_v1"
use_lora: true
lora_r: 8
batch_size: 32
gradient_accumulation_steps: 4
learning_rate: 1.0e-3
max_epochs: 20
warmup_ratio: 0.1
mlflow_experiment: "mtdna_fm_haplogroup"
output_dir: "models/finetune_haplogroup_v1"
```

Expected accuracy: >95%. The phylogeny is well-defined and the haplogroup-defining variants from PhyloTree are highly specific. Errors should occur between phylogenetically close haplogroups (e.g., H and HV), not between distant ones (e.g., L3 and H).

**LinkedIn post:** Not about the model. About the production engineering: "Fine-tuning a 6M-parameter model with LoRA on a laptop. The LoRA adapter for this task is 500KB. The base model is 24MB. You can share the adapter without sharing the base model."

### Day 17: Fine-tuning Task 2: Pathogenic variant prediction

`MtDNAForVariantPathogenicity(PreTrainedModel)`: binary classifier. Input is a 512-token window centered on the variant position. Classification uses the hidden state at the token containing the variant position, not the CLS token. The rationale: pathogenicity is a local property (what this variant does to this protein or tRNA), not a global genome property.

Training data construction: 2,000 ClinVar pathogenic variants (positive class) + 5,000 gnomAD variants with AF > 0.01 (negative class). Stratify by variant type at split time: missense, tRNA, rRNA, D-loop.

LoRA: `r=4` (small dataset, smaller rank), `weight_decay=0.1` (heavier regularization), `pos_weight=2.5` in the BCE loss.

Expected AUROC: >0.85. The pre-trained k-mer context captures functional constraint — conserved positions have low k-mer entropy in the cross-species corpus, and the model's representations reflect this.

### Day 18: Fine-tuning Task 3: Heteroplasmy regression

`MtDNAForHeteroplasmyRegression(PreTrainedModel)`: regression head on variant token hidden state. Head: Linear(256, 64) -> GELU -> Linear(64, 1) -> Sigmoid. Loss: Huber (more robust to gnomAD noise than MSE).

Training data: gnomAD variants with at least 50 heteroplasmic carriers, mean heteroplasmy level as target. About 1,000 data points. Use 5-fold cross-validation and report R-squared and Spearman correlation. If Spearman > 0.30, the model is capturing something real about selective constraint.

**What you learn:** When to use Huber loss vs MSE, when to use cross-validation vs a held-out test set, how to evaluate a regression model on a small dataset without overfitting the evaluation procedure itself.

### Day 19: Evaluation framework

`mtdna_fm/evaluation/haplogroup_eval.py`: accuracy, macro-F1, per-haplogroup breakdown, confusion matrix. Save as JSON metrics.

`mtdna_fm/evaluation/variant_eval.py`: AUROC, AUPRC, per-variant-type breakdown (missense, tRNA, rRNA, D-loop variants may have different accuracy profiles).

`mtdna_fm/evaluation/viz.py`: UMAP of genome embeddings colored by haplogroup (should show phylogenetic tree topology), ROC curves, confusion matrix, attention weight heatmap for a pathogenic variant.

`mtdna_fm/scripts/evaluate.py`: `mtdna-evaluate` CLI that runs all evaluations and saves results to `reports/eval_summary.json`. This is what DVC tracks as a metric.

`notebooks/03_finetuning_results.ipynb`: The key figure here is the UMAP. If the model has learned useful representations, haplogroup clusters will form the correct phylogenetic topology: L0/L1/L2 at root, L3 branching to M and N, R emerging from N, H and HV at the European tip. This is not something you can fake — it either falls out of the embeddings or it doesn't.

Baseline comparisons for the results section:
- Majority class: ~15% haplogroup accuracy (H dominates in HmtDB which has European bias), AUROC 0.50 pathogenicity
- k-mer frequency PCA + logistic regression: ~65% haplogroup, AUROC ~0.72 pathogenicity
- mtDNA-FM fine-tuned: >95% haplogroup, AUROC >0.85 pathogenicity

### Day 20: Ancient DNA demonstration

Download Neanderthal (NC_011137.1, Vindija Cave, Croatia) and Denisovan (FR695060.1, Altai Cave, Russia) mtDNA via Entrez. These sequences were never in the training data.

Embed with Phase 2 `MtDNAEmbedder`, no fine-tuning. Place them on the same UMAP as 5,000 modern humans. If pre-training worked, the ancient sequences should cluster near but distinct from modern human root haplogroups, consistent with what molecular anthropologists have established from phylogenetic analysis.

This is the most compelling zero-shot demonstration for the portfolio because it is verifiable against existing scientific knowledge. The model either agrees with paleoanthropology or it doesn't.

**X post with the figure:** "Zero-shot ancient DNA: fed Neanderthal and Denisovan mtDNA sequences to mtDNA-FM without any fine-tuning. The model places them correctly relative to the modern human phylogeny. It learned deep evolutionary structure from sequence alone."

### Day 21: HuggingFace Hub and Blog Post 3

Push base model, tokenizer, and both LoRA adapters to HuggingFace Hub.

Model card should include: architecture novelties with equations, training corpus, benchmark table with baselines, 5-line usage example, known limitations (model is trained on HmtDB which has European population bias, performance may be lower on underrepresented haplogroups), how to cite.

5-line usage example:
```python
from mtdna_fm.inference.api import MtDNAEmbedder
embedder = MtDNAEmbedder.from_pretrained("vthawfeek/mtdna-foundation-model")
embedding = embedder.embed_genome(my_sequence)   # shape: (256,)
```

**Blog Post 3:** "What I Learned Fine-Tuning an mtDNA Foundation Model"

Cover the actual decisions and tradeoffs. Why `r=4` for pathogenicity but `r=8` for haplogroup. Why the variant classifier uses the variant-position hidden state instead of CLS. What the ancient DNA zero-shot result actually means and what it doesn't. Honest about limitations: the heteroplasmy regression R-squared is probably not impressive, but it's non-trivial and shows the model captures selective constraint.

**Week 3 exit criteria:**
- Haplogroup fine-tuned accuracy >90% on test set
- Pathogenicity AUROC >0.75 on ClinVar/gnomAD split
- Model on HuggingFace Hub, loads correctly with `from_pretrained`
- Notebook 3 figures saved

---

## Week 4 (Days 22-28): Production Polish and Publishing

### Day 22: Gradio demo on HuggingFace Spaces

`app.py` with three tabs:

1. **Haplogroup Classification**: paste or upload FASTA, return predicted haplogroup, confidence bar chart, brief explanation of what the haplogroup means (geographic origin, approximate age, notable clinical associations if any)

2. **Variant Pathogenicity Check**: sequence + position + alternate base, return pathogenicity probability + nearest ClinVar entry if any + attention weight heatmap showing which genomic context the model focused on

3. **Genome Embedding**: paste sequence, get 256-dimensional embedding as downloadable CSV and placement on a reference UMAP of 500 representative human mtDNA genomes

Keep the demo CPU-safe (no batch ops). HuggingFace Spaces free tier is slow but functional for single-sequence inputs.

Push to `huggingface-cli repo create mtdna-fm-demo --type space --space_sdk gradio`.

**X post with a GIF or screenshot of the demo.** "Live: classify your mtDNA haplogroup, check variant pathogenicity, visualize where your genome sits in human phylogenetic space. [link]"

### Day 23: Documentation

Each doc covers one topic completely. No duplication across docs.

- `docs/01_data_pipeline.md`: what each dataset contains, how to download it, what the preprocessing steps do and why, DVC stage structure
- `docs/02_tokenization.md`: 6-mer vocabulary construction, circular windowing mechanics, heteroplasmy channel, vocabulary statistics
- `docs/03_architecture.md`: circular PE derivation step by step (someone should be able to re-implement it from this), het projection, parameter count, comparison of design choices against DNABERT2 and HyenaDNA on the specific axes that matter for mtDNA
- `docs/04_pretraining.md`: two-phase curriculum rationale, expected MLM loss dynamics, how to run it on a laptop (timing estimates for CPU vs GPU), how to monitor with MLflow
- `docs/05_finetuning_and_evaluation.md`: three downstream tasks, LoRA configuration choices, baseline comparisons, ancient DNA application, known limitations

### Day 24: DVC pipeline and reproducibility

Complete `dvc.yaml` with all stages:
```yaml
stages:
  download_hmtdb:
  download_ncbi:
  download_variants:
  preprocess:
  build_vocabulary:
  pretrain_phase1:
  pretrain_phase2:
  finetune_haplogroup:
  evaluate:
```

Each stage has `deps`, `outs`, `params`, and `metrics` defined correctly. The `evaluate` stage writes `reports/eval_summary.json` as a DVC metric so `dvc metrics show` gives the final numbers.

Reproducibility test:
```bash
git clone [repo] /tmp/mtdna_repro && cd /tmp/mtdna_repro && uv sync && dvc repro
dvc metrics show   # should match saved values
```

This is a strong portfolio signal: the entire pipeline from raw data download to trained model to evaluation metrics runs with two commands.

### Day 25: Showcase notebook

`notebooks/04_mtdna_fm_showcase.ipynb`: self-contained notebook that tells the full story. This is the README hero artifact.

Sections:
1. Load the model in 3 lines, print parameter count, show config
2. TSNE of 5,000 human genomes colored by haplogroup — the phylogenetic tree should emerge from the sequence embeddings
3. Haplogroup confusion matrix, sorted by phylogenetic distance. Errors between H and HV are acceptable (they're closely related). Errors between L3 and H are not.
4. Variant pathogenicity ROC curve, attention weight heatmap for one correctly-predicted ClinVar pathogenic variant showing the model attending to nearby conserved functional elements
5. Ancient DNA: Neanderthal and Denisovan placed on the UMAP, consistent with paleoanthropology
6. Gene-type recovery without labels: embed each of the 37 mtDNA genes (13 protein-coding, 22 tRNA, 2 rRNA) and cluster. Do they separate by gene type? This tests whether the model learned function implicitly from sequence.

### Day 26: Blog Post 4 and LinkedIn article

**Blog Post 4 (personal blog or Medium, 2,000+ words):** "Building a Production-Quality Foundation Model in 4 Weeks: What Actually Took Time"

This post should be honest about the difficulty. What took longer than expected? What would you do differently? Which design decisions turned out to matter (circular PE, two-phase curriculum) and which were less impactful than expected? What did you learn about training small models on domain-specific data that isn't obvious from the literature?

Include the actual numbers: MLM loss curves, fine-tuning accuracy across epochs, how Phase 2 improved zero-shot haplogroup k-NN from X% to Y%. The post has value because it shows real results from a real experiment, not a summary of what the model can do.

**LinkedIn article (shorter, 500 words):** focus on the production engineering angle. DVC reproducibility, LoRA adapters as 500KB artifacts, HuggingFace Hub as a model registry, the complete pipeline in two commands. This resonates with people hiring for production ML roles.

### Day 27: Community engagement

Post to:
- **Biostars** (biostars.org): introduce the model and invite testing on diverse mtDNA datasets. Ask specifically about haplogroups that are underrepresented in HmtDB.
- **HuggingFace community forum** (Models section): benchmark comparison against DNABERT2 zero-shot on haplogroup classification (DNABERT2 was trained on nuclear DNA and will score around 35%; mtDNA-FM scores >95% fine-tuned and >40% zero-shot).
- **GitHub issues**: open issues for known limitations to show the project is active and honest about what it doesn't do yet.

### Day 28: Release v0.1.0

```bash
# Final quality gates
uv run ruff check mtdna_fm/
uv run pytest tests/ -m "not integration" --cov=mtdna_fm
dvc repro --dry
dvc status

# Tag release
git tag v0.1.0
git push origin v0.1.0

# Optional: publish to PyPI
uv build
uv publish
```

`CHANGELOG.md` entry for 0.1.0 covering what was added.

GitHub Release with the showcase notebook attached and a brief description of what changed.

**Final X thread (10 tweets):** The 4-week retrospective. One tweet per key technical decision: tokenization choice, circular PE, two-phase curriculum, LoRA rank choices, what the ancient DNA demo shows. Link to the blog post and HuggingFace Hub. This thread is the portfolio artifact that shows you can communicate technical work concisely and accurately, which is harder than building the model.

---

## Social Media Cadence

Posts should be about what you actually found, not announcements. The target audience is other ML practitioners and biomedical scientists, not recruiters. If the post would not be interesting to someone who already knows what a foundation model is, rewrite it.

| Day | Platform | What to post |
|-----|----------|--------------|
| 1 | X | Project start: why mtDNA, what problem, what novelty. Keep it short. |
| 4 | X | D-loop entropy figure with explanation of why it matters for the model design |
| 7 | LinkedIn | Blog Post 1 with genuine discussion of the biology |
| 10 | X | Pre-training loss curve screenshot (even early, to show the convergence trajectory) |
| 13 | X | Attention heatmap at step 25k showing structure emerging |
| 14 | LinkedIn | Blog Post 2 on circular PE with the equations |
| 17 | X | Haplogroup k-NN accuracy before and after fine-tuning (concrete numbers) |
| 20 | X | Ancient DNA TSNE figure with paleoanthropology context |
| 20 | LinkedIn | Same figure, longer explanation of what it means scientifically |
| 21 | X | Model on HuggingFace Hub, 5-line usage example |
| 22 | X | Gradio demo screenshot or GIF |
| 25 | X | Showcase notebook phylogenetic TSNE (the best figure in the project) |
| 26 | LinkedIn | Blog Post 4 (the honest retrospective) |
| 28 | X | 10-tweet retrospective thread |

---

## Reusable Tools Built During This Project

Each of these components is independently reusable beyond this project.

**`KmerVocabulary`**: Works for any short-genome project with an ACGT alphabet and a fixed k. Save/load follows HuggingFace convention.

**`tokenize_sequence`**: Works for any linear or circular DNA sequence. The `circular` flag and `genome_length` parameter generalize to any small genome.

**`MtDNACircularPositionalEncoding`**: Generalizes to any circular sequence by passing a different `genome_length`. Could apply to plasmid sequences, viral genomes, or other circular nucleic acids.

**`MtDNAEmbedder` API**: Designed as a stable public interface. Anyone who installs `mtdna-fm` can use it without understanding the windowing internals. The `from_pretrained` constructor follows HuggingFace conventions so it works with the Hub.

**The DVC pipeline**: The stage structure (idempotent download -> preprocess -> vocabulary -> pretrain -> finetune -> evaluate) is a reusable template for any genomics fine-tuning project.

**The two-phase curriculum trainer**: The pattern of loading encoder weights from Phase 1 and discarding the optimizer state for Phase 2 is reusable for any domain-adaptive pre-training setup.

---

## Critical Files to Reference from Sibling Projects

These are the direct analogues in `sc_foundation_model` that each mtDNA-FM file should mirror:

- `scfm/tokenizer/tokenize.py`: `tokenize_cell()` is the template for `tokenize_sequence()`. Replace rank-sorting with k-mer sliding window, replace `expression_values` with `het_values`.
- `scfm/model/embeddings.py`: `ScFMEmbeddings` is the template for `MtDNAEmbeddings`. Replace `rank_embeddings` with `MtDNACircularPositionalEncoding`.
- `scfm/model/model.py`: `ScFMModel` and `ScFMForMaskedModeling` define the class hierarchy to mirror.
- `scfm/training/trainer.py`: `ScFMTrainer` is the template. Modify `_load_checkpoint` for Phase 2 (encoder weights only, fresh optimizer).
- `scfm/data/dataset.py`: `SingleCellDataset` is the template. Replace lazy HDF5 loading with windowed parquet loading.
- `scfm/dvc.yaml`: Stage structure template (deps/outs/metrics/params pattern).

---

## Risks

| Risk | Mitigation |
|------|-----------|
| HmtDB requires registration | Use NCBI fallback: `human[Organism] AND mitochondrion[Filter] AND complete genome[Title]`. Same 47k sequences, no registration. |
| gnomAD full VCF is 5 GB | Always extract chrM first with tabix. chrM-only file is about 50 MB. |
| Phase 1 training takes over 12 hours on CPU | Reduce to 25k steps. The loss curve flattens before 50k steps. Consider Google Colab for Phase 1 if needed. |
| Circular PE shows no benefit over linear PE | Include both as a DVC parameter experiment. Document the ablation. A negative result that is well-documented is still a portfolio contribution. |
| Pathogenicity dataset is small (7k variants) | Augment with synonymous variants as additional negatives. Heavier regularization (r=4, weight_decay=0.2). Report confidence intervals. |
| HuggingFace Spaces free tier is slow | Limit demo to single-sequence inputs. No batch processing. |
| NCBI rate limiting | Use NCBI API key (free at ncbi.nlm.nih.gov/account). Set time.sleep(0.1) between requests. Use WebEnv batching. |

---

## Verification Checkpoints

**End of Week 1:**
```bash
uv run pytest tests/ -v
uv run ruff check mtdna_fm/
python -c "from mtdna_fm.tokenizer.vocabulary import KmerVocabulary; v=KmerVocabulary.build(k=6); assert len(v)==4102"
python -c "import pandas as pd; df=pd.read_parquet('data/processed/train.parquet'); assert (df.sequence.str.len()==16569).all()"
```

**End of Week 2:**
```bash
uv run pytest tests/ --cov=mtdna_fm     # expect >80% coverage
# In MLflow: train/mlm_loss < 5.0 at step 10k, < 3.5 at step 50k
python -c "from mtdna_fm.model.model import MtDNAModel; m=MtDNAModel.from_pretrained('models/phase1_v1'); print('Phase 1 loads OK')"
```

**End of Week 3:**
```bash
python -c "
import json
h = json.load(open('models/finetune_haplogroup_v1/eval_metrics.json'))
p = json.load(open('models/finetune_pathogenicity_v1/eval_metrics.json'))
print(f'Haplogroup accuracy: {h[\"accuracy\"]:.3f}')   # expect > 0.90
print(f'Pathogenicity AUROC: {p[\"auroc\"]:.3f}')        # expect > 0.75
"
python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('vthawfeek/mtdna-foundation-model'); print('HF Hub OK')"
```

**End of Week 4:**
```bash
dvc repro --dry    # no errors
dvc status         # all stages up to date
uv run pytest tests/ -v
dvc metrics show   # print final eval summary
```
