# Day 15: Genome Embedding API

## What was built

- `mtdna_fm/inference/api.py`: `MtDNAEmbedder` — the stable public interface for embedding mtDNA sequences with the pre-trained model
- `tests/test_inference.py`: 18 tests covering `embed_genome`, `embed_variant`, `embed_dataset`, `from_pretrained`, and k-inference from vocabulary size

## What was learned

- **CLS-mean pooling strategy**: The model produces one 256-dim vector per 512-token window. Mean-pooling the CLS (position-0) token across all overlapping windows gives a full-genome embedding that weights every genomic region equally. This is analogous to how sentence-transformers handle long documents.
- **`from_pretrained` extraction pattern**: The checkpoint saved by `MtDNATrainer` is `MtDNAForMaskedModeling` (pretraining wrapper). The embedder loads the full pretraining model and extracts `.mtdna` (the inner `MtDNAModel`), discarding the prediction heads. This is the standard HuggingFace pattern (`BertModel` extracted from `BertForMaskedLM`).
- **Token-level embedding for variants**: Pathogenicity is local. `embed_variant` returns the hidden state at the token containing the variant position rather than the CLS state, capturing the specific functional context (tRNA fold, codon identity, regulatory motif) that determines pathogenicity.
- **k inferred from vocabulary size**: `k = log4(vocab_size - 6)` derives the k-mer size without requiring it to be stored separately. This keeps the API clean — `from_pretrained` needs only the checkpoint directory.

## Key decisions

- **`from_pretrained` loads `MtDNAForMaskedModeling` then extracts `.mtdna`**: Alternative was to strip the `mtdna.` weight prefix when loading into `MtDNAModel`. The extraction approach is cleaner — no manual state_dict manipulation, and it validates that the checkpoint was actually saved in the expected format.
- **`embed_dataset` delegates to `embed_genome`**: The method loops over sequences and calls `embed_genome` rather than doing any batching across sequences. Full-genome embedding already processes multiple windows per sequence; adding cross-sequence batching would complicate the code for marginal throughput gain on CPU.
- **Circular window wrapping in `embed_genome`**: Windows wrap at the token stream boundary (`% n_tokens`), matching the circular topology that was used during pre-training. This ensures the 16568/0 junction is covered by windows, not silently skipped.
- **Phase 2 checkpoint status**: Phase 2 training was launched on Day 14 but did not produce a saved checkpoint (`models/phase2_v1/` is empty). The zero-shot k-NN check below uses Phase 1 embeddings, which already exceed the ≥40% target.

## Verification

```bash
# Lint
uv run ruff check mtdna_fm/ tests/
# All checks passed!

# Test suite
uv run pytest tests/ -m "not slow and not integration" -q
# 254 passed, 2 warnings in 6.64s
```

Zero-shot k-NN check (Phase 1 embeddings, 64 sequences, 8 major haplogroups):
```
Random baseline:              12.5% (1/8 classes)
Zero-shot 3-NN (4-fold CV):   50.0% ± 6.2%
Folds: [56.2%, 43.8%, 43.8%, 56.2%]
```

The Phase 1 model achieves 50% zero-shot haplogroup accuracy, already exceeding the ≥40% target set for Phase 2. This confirms the pre-trained representations encode phylogenetic structure from sequence alone.

MtDNAEmbedder loads and runs correctly:
```python
from mtdna_fm.inference.api import MtDNAEmbedder
embedder = MtDNAEmbedder.from_pretrained("models/phase1_v1")
vec = embedder.embed_genome(sequence)   # shape: (256,)
```

## Next up

Day 16: Fine-tuning Task 1 — haplogroup classification with LoRA (`r=8`), `MtDNAForHaplogroupClassification`, `configs/finetuning_haplogroup.yaml`.
