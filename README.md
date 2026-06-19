# mtDNA Foundation Model

[![CI](https://github.com/vthawfeek/mtdna-foundation-model/actions/workflows/ci.yml/badge.svg)](https://github.com/vthawfeek/mtdna-foundation-model/actions/workflows/ci.yml)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-mtdna--fm-yellow)](https://huggingface.co/vthawfeek/mtdna-foundation-model)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

The first dedicated foundation model for mitochondrial DNA. Pre-trained on 152k+ complete mitochondrial genomes (~117k cross-species vertebrate + 34,975 human) with two architectural novelties: circular positional encoding (because mtDNA is circular, not linear) and a heteroplasmy projection channel that encodes per-position variant allele fractions alongside k-mer token IDs.

## Quick Start

```bash
pip install mtdna-fm
```

```python
from mtdna_fm.inference.api import MtDNAEmbedder

embedder = MtDNAEmbedder.from_pretrained("vthawfeek/mtdna-foundation-model")
embedding = embedder.embed_genome(sequence)   # shape: (256,)
```

## Results

| Task | Metric | Random | k-mer PCA + LR | mtDNA-FM (zero-shot) | mtDNA-FM (fine-tuned) |
|------|--------|--------|----------------|---------------------|-----------------------|
| Haplogroup classification | Accuracy | 3.85% | ~65% (26-class) | **37.9%¹ (26-class)** | 1.83%² (26-class) |
| Pathogenic variant prediction | AUROC | 0.50 | ~0.72 | 0.777 (95% CI 0.731–0.821)³ | not evaluated |

¹ Zero-shot 5-NN (cosine) on Phase 2 embeddings, full 26-class haplogroup evaluation (13,884 NCBI-labeled sequences; 3.85% random baseline; **9.8× lift**; 95% CI 34.4–41.2%). Per-class results in `reports/zeroshot_haplogroup_knn.json`.
² LoRA r=8, 1,267 training sequences, 2 epochs on CPU, 26-class evaluation (3.85% random baseline). Partial class collapse (3/26 classes active). Fine-tuning did not converge — CPU compute constraint. Zero-shot k-NN (37.9%) is the more reliable signal of what the pre-training learned. See `reports/eval_summary.json`.
³ Zero-shot 5-fold stratified k-NN (k=5, cosine): 118 ClinVar pathogenic + 419 gnomAD AF≥1% benign mitochondrial SNPs. No pathogenicity labels used during pre-training. Per-type: missense 0.727 (n=56), tRNA 0.718 (n=44). Script: `scripts/zeroshot_patho_eval.py`. Supervised LoRA fine-tuning on real data is future work.

## Architecture

mtDNA is circular: position 16,569 is genomically adjacent to position 1. Standard sinusoidal positional encoding treats these as maximally distant. mtDNA-FM uses a circular sinusoidal encoding:

```
PE[pos, 2i]   = sin(2pi * pos / 16569 * 1/10000^(2i/d))
PE[pos, 2i+1] = cos(2pi * pos / 16569 * 1/10000^(2i/d))
```

This is a fixed, non-learnable buffer. The circular topology is a biological fact, not a hyperparameter.

Model size: 6 layers, 8 heads, 256 hidden dimensions, ~6M parameters. Tokenization: 6-mer overlapping sliding window over the 16,569 bp genome, vocabulary of 4,102 tokens (4,096 6-mers + 6 special tokens).

## Documentation

- [Data Pipeline](docs/01_data_pipeline.md)
- [Tokenization](docs/02_tokenization.md)
- [Architecture](docs/03_architecture.md)
- [Pre-training](docs/04_pretraining.md)
- [Fine-tuning and Evaluation](docs/05_finetuning_and_evaluation.md)

## Reproducibility

```bash
git clone https://github.com/vthawfeek/mtdna-foundation-model
cd mtdna-foundation-model
uv sync
dvc repro
dvc metrics show
```

## Development

```bash
uv sync --extra dev
uv run pytest tests/
uv run ruff check mtdna_fm/
```

## License

Apache 2.0. See [LICENSE](LICENSE).
