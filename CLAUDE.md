# mtDNA Foundation Model

## Plan

Full 4-week plan: [PLAN.md](PLAN.md) (in this repo — also at `/home/user/.claude/plans/i-want-to-build-idempotent-starlight.md` for plan mode)

## How to trigger a day's work

Type `/day N` where N is the day number (1-28).

Each invocation executes all tasks for that day, runs lint and tests, writes
`reports/day-N-<topic>.md`, commits, and pushes to GitHub.

## Current status

- Day 1: COMPLETE (commit 12d614c) — scaffold, pyproject.toml, CLI stubs, CI
- Day 2: COMPLETE (commit 3e1f9d9) — tokenizer (`KmerVocabulary` + `tokenize_sequence`)
- Day 3: COMPLETE (commit dabd8c5) — data download clients (HmtDB, NCBI Entrez)
- Day 4: COMPLETE (commit TBD) — preprocessing pipeline + EDA notebook
- Day 5: variant datasets (gnomAD, ClinVar, PhyloTree)
- Day 6: PyTorch Dataset class
- Day 7: GitHub CI + Blog Post 1
- Day 8-14: model architecture + pre-training
- Day 15-21: fine-tuning + evaluation + HuggingFace Hub
- Day 22-28: demo, docs, release

## Project

- GitHub: https://github.com/vthawfeek/mtdna-foundation-model
- HuggingFace: https://huggingface.co/vthawfeek/mtdna-foundation-model (created Day 21)
- Working directory: /home/user/Documents/Personal/ai_lab/mtdna_foundation_model

## Architecture summary

Pre-trained BERT encoder for mitochondrial DNA (16,569 bp circular genome).
Novel: circular positional encoding + heteroplasmy projection channel.
Vocabulary: 4,096 6-mers + 6 special tokens = 4,102 tokens.
Model: 6 layers, 8 heads, 256 hidden dim, ~6M parameters.
Fine-tuning tasks: haplogroup classification (26 classes), pathogenic variant prediction (binary), heteroplasmy regression.

## Tech stack

uv, PyTorch, HuggingFace Transformers, PEFT/LoRA, MLflow, DVC, pytest, ruff, BioPython, Gradio
