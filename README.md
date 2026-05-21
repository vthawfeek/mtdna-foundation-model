# mtDNA Foundation Model

[![CI](https://github.com/YOUR_USERNAME/mtdna-foundation-model/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/mtdna-foundation-model/actions/workflows/ci.yml)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-mtdna--fm-yellow)](https://huggingface.co/YOUR_USERNAME/mtdna-foundation-model)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

The first dedicated foundation model for mitochondrial DNA. Pre-trained on 77k+ complete mitochondrial genomes with two architectural novelties: circular positional encoding (because mtDNA is circular, not linear) and a heteroplasmy projection channel that encodes per-position variant allele fractions alongside k-mer token IDs.

## Quick Start

```bash
pip install mtdna-fm
```

```python
from mtdna_fm.inference.api import MtDNAEmbedder

embedder = MtDNAEmbedder.from_pretrained("YOUR_USERNAME/mtdna-foundation-model")
embedding = embedder.embed_genome(sequence)   # shape: (256,)
```

## Results

| Task | Metric | Majority class | k-mer PCA + LR | mtDNA-FM |
|------|--------|---------------|----------------|---------|
| Haplogroup classification | Accuracy | 15% | ~65% | >95% |
| Pathogenic variant prediction | AUROC | 0.50 | ~0.72 | >0.85 |
| Heteroplasmy estimation | Spearman | 0.00 | ~0.12 | >0.30 |

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
git clone https://github.com/YOUR_USERNAME/mtdna-foundation-model
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
