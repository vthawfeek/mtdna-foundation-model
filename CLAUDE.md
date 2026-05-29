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
- Day 4: COMPLETE (commit 78b869a) — preprocessing pipeline + EDA notebook
- Day 5: COMPLETE (commit 41ec618) — variant datasets (gnomAD, ClinVar, PhyloTree)
- Day 6: COMPLETE (commit 33169c2) — PyTorch Dataset classes (MtDNADataset, VariantDataset)
- Day 7: COMPLETE (commit d43cb47) — CI hardening (ruff format gate, lint+test jobs), Week 1 exit criteria verified
- Day 8: COMPLETE — model architecture (MtDNAConfig, CircularPE, MtDNAEmbeddings, transformer, MtDNAModel, MtDNAForMaskedModeling)
- Day 9: COMPLETE — masking collator (BERT 80/10/10 + D-loop blacklist), combined MLM+het loss, 20 new tests
- Day 10: COMPLETE — MtDNATrainer (cosine LR, gradient accum, MLflow, checkpoint rotation, Phase 2 encoder loading), Phase 1+2 configs, 21 new tests
- Day 11: COMPLETE — test suite expansion (76% → 80% coverage, 193 tests), gradient checkpointing added to model
- Day 12: COMPLETE — test suite completion (80% → 97% coverage, 236 tests), full coverage of download clients, variant_downloader, CLI scripts, and model internals
- Day 13: COMPLETE (commit 9544d1f) — pre-training analysis notebook (training curves, attention heatmaps, zero-shot k-NN 9.5% vs 4% random)
- Day 14: COMPLETE (commit 5733c78) — Phase 1 checkpoint verified, zero-shot k-NN confirmed (16% vs 10% random), Phase 2 config + trainer wired, Phase 2 launched on human HmtDB sequences (het_weight=0.3)
- Day 15: COMPLETE — genome embedding API (MtDNAEmbedder: embed_genome, embed_variant, embed_dataset, from_pretrained), zero-shot k-NN 50% vs 12.5% random
- Day 16: COMPLETE — haplogroup classification (MtDNAForHaplogroupClassification, LoRA r=8, HaplogroupWindowDataset, finetune CLI, 264 tests, 88% coverage)
- Day 17: COMPLETE (commit 1c9254d) — pathogenic variant prediction (MtDNAForVariantPathogenicity, variant-token hidden state, LoRA r=4, pos_weight=2.5, PathogenicityVariantDataset, 274 tests)
- Day 18: COMPLETE (commit 8b4dee0) — heteroplasmy regression (MtDNAForHeteroplasmyRegression, Huber loss, 5-fold CV, HeteroplasmyRegressionDataset, 294 tests)
- Day 19: COMPLETE (commit be76f1e) — evaluation framework (haplogroup_eval, variant_eval, viz, mtdna-evaluate CLI, notebook 03, 327 tests)
- Day 20: COMPLETE (commit 8d1abb3) — ancient DNA demonstration (Neanderthal + Denisovan zero-shot, plot_umap_with_ancient_dna, embed_genome length normalization, 346 tests)
- Day 21: COMPLETE (commit e372b1c) — HuggingFace Hub (base model + 2 LoRA adapters pushed, model card, push_to_hub.py)
- Day 22: COMPLETE (commit c25a77f) — Gradio demo (app.py three-tab Spaces demo: haplogroup/pathogenicity/embedding, app_reference.npz, live at vthawfeek/mtdna-fm-demo)
- Day 23: COMPLETE (commit b8665ba) — documentation (5 docs: data pipeline, tokenization, architecture, pre-training, fine-tuning/evaluation)
- Day 24: COMPLETE (commit 6dc0f3f) — DVC pipeline (9 stages: download→preprocess→vocabulary→pretrain→finetune→evaluate, dvc metrics show wired)
- Day 25: COMPLETE (commit 10ecc7b) — showcase notebook (04_showcase.ipynb: t-SNE haplogroup clustering, confusion matrix, pathogenicity ROC AUROC=0.877, ancient DNA UMAP, gene-type recovery without labels, 346 tests)
- Day 26-28: blog posts, community engagement, release

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
