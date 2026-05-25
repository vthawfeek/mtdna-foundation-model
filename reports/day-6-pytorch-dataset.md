# Day 6: PyTorch Dataset Classes

## What was built

- `mtdna_fm/data/dataset.py` — `MtDNADataset(Dataset)`: windowed Dataset for pre-training. Pre-tokenises each genome once (stride=1, circular=True → genome_length tokens), then builds a flat index of (seq_idx, window_start) pairs. Windows wrap circularly so the 16568/0 boundary is always covered. Absolute position IDs preserved across windows.
- `mtdna_fm/data/variant_dataset.py` — `VariantDataset(Dataset)`: SNP pathogenicity Dataset. Applies each alt allele to the rCRS reference, tokenises the mutated sequence, and returns a 512-token window centred on the variant position. Indels filtered out automatically.
- `tests/test_data.py` — Added `TestMtDNADataset` (7 tests) and `TestVariantDataset` (5 tests) to the existing test file. Tests cover the four plan-specified functions plus output shapes, label propagation, `from_dataframe`, and SNP-application correctness.

## What was learned

- **Pre-tokenise vs on-the-fly**: All 47k sequences fit in memory (~780 MB as raw strings); pre-tokenising each genome once avoids repeating the k-mer sliding window per epoch. The tokenised arrays add ~2 GB but avoid re-computing 16569 k-mers per genome per batch.
- **Circular windowing with absolute position IDs**: Windows use `(window_start + i) % genome_length` for token index lookup. This naturally produces the junction window (tokens wrap from 16568 back to 0) without any special-case code. Position IDs stay as the original genomic coordinate, not the window-relative index — this is what the circular positional encoding expects.
- **Window count**: `range(0, genome_length, stride)` with genome_length=16569, stride=256 produces 65 windows per genome (≈3.1M training examples from 47k sequences). The plan's "about 63" is the non-circular floor formula; the circular dataset adds two junction-crossing windows.
- **VCF position convention**: gnomAD/ClinVar use 1-based positions. `VariantDataset` converts to 0-based (`pos_0 = int(row["pos"]) - 1`) before indexing into the reference string. Forgetting this off-by-one would silently corrupt every variant.
- **Variant token offset**: The variant-position hidden state (not CLS) is the right classifier input for pathogenicity — it encodes the local k-mer context around the mutation. The `variant_offset` field lets the model head index directly into the right token position.

## Key decisions

- **Circular windows rather than non-circular + one explicit junction window**: cleaner, no special cases, every start position in `range(0, genome_length, stride)` is treated identically. The circularity emerges from modular arithmetic.
- **Filter indels in `__init__` not `__getitem__`**: filtering once at construction time avoids a per-sample branch and keeps `__len__` accurate.
- **`from_dataframe` classmethod**: mirrors scFM's pattern; lets training code go directly from a parquet file to a Dataset without intermediate boilerplate.
- **`het_level_vectors` default to None (not zeros)**: zeros are filled in by `tokenize_sequence` when not provided. Defaulting to None avoids allocating 47k × 16569 float arrays for the cross-species Phase 1 dataset which has no heteroplasmy data.

## Verification

```
$ uv run ruff check mtdna_fm/ tests/
# No output (0 errors)

$ uv run pytest tests/ -m "not slow and not integration" -v
============================= 101 passed in 10.97s ==============================

Key passing tests (Day 6):
  TestMtDNADataset::test_dataset_length          PASSED
  TestMtDNADataset::test_all_positions_covered   PASSED
  TestMtDNADataset::test_circular_junction_window PASSED
  TestMtDNADataset::test_het_values_range        PASSED
  TestVariantDataset::test_snp_applied_to_reference PASSED
```

## Next up

Day 7: GitHub CI workflow (lint + test jobs) and Blog Post 1 — "Why Mitochondrial DNA Needs Its Own Foundation Model".
