# Day 4: Preprocessing Pipeline and EDA Notebook

## What was built

- `mtdna_fm/data/preprocessor.py` — five standalone functions covering the full clean-to-split pipeline:
  - `clean_sequence`: uppercase, replace non-ACGTN with N, strip trailing circular junction duplicate
  - `normalize_length`: pad short sequences at the D-loop start (position 576), trim long ones
  - `stratified_split`: 80/10/10 train/val/test split by haplogroup using `StratifiedShuffleSplit`
  - `build_record_dataframe`: parse FASTA + optional metadata parquet into the canonical schema
  - `preprocess_sequences`: orchestrates clean + normalize, adds `length_raw` and `qc_pass` columns
  - `save_splits`: write `train.parquet`, `val.parquet`, `test.parquet`
- `mtdna_fm/scripts/preprocess.py` — full `mtdna-preprocess` CLI wrapping the above functions
- `tests/test_data.py` — 27 new unit tests across six test classes for all preprocessor functions (67 total, all passing)
- `notebooks/01_data_exploration.ipynb` — EDA notebook with five analyses: haplogroup distribution, per-position Shannon entropy, N-content, length histogram, geographic distribution
- `configs/data.yaml` — data preprocessing parameters tracked by DVC
- `dvc.yaml` — added `preprocess` stage with `deps`, `outs`, and `params`

## What was learned

- **Junction duplicate detection**: HmtDB and some NCBI records append the first 200 bp to the end of circular genome sequences so analysis tools don't miss the junction region. Stripping this is a preprocessing step that is easy to miss and silently corrupts length normalization if left in.
- **Padding position matters**: inserting N padding at the D-loop start (position 576) rather than appending to the 3' end preserves canonical coordinate offsets for all 37 mitochondrial genes. Any downstream tool that maps a variant by position (e.g., `chrM:3243`) still gets the right bases.
- **Rare class handling in stratified splits**: `StratifiedShuffleSplit` raises `ValueError` if a class has fewer than 2 samples. The fix is to merge rare classes into a `_rare` bucket before splitting, then restore the original labels after assignment. This is a common pitfall in genomics datasets where some haplogroups have only one representative.
- **Why function-per-step over a pipeline class**: each function is independently importable, testable, and composable. A future pipeline that skips normalization (e.g., variable-length inference) can reuse `clean_sequence` and `stratified_split` without touching the rest.
- **D-loop entropy contrast**: even with synthetic data the Shannon entropy analysis shows that the D-loop boundary is the sharpest transition in sequence variability across the entire 16,569-bp genome. This is the empirical justification for why circular positional encoding matters — the model must learn that positions 16,024 and 576 are adjacent, not 15,448 positions apart.

## Key decisions

- **Separate functions, not a class**: the plan specifies this explicitly. Three similar preprocessing steps are better encoded as three functions you can unit-test in isolation than as methods on a class that can only be tested end-to-end.
- **QC flagging, not filtering**: sequences that exceed the N threshold are marked `qc_pass=False` but kept in the output parquet. Whether to exclude them is a training-time decision (e.g., mask heavily-N windows during MLM rather than dropping whole genomes), not a preprocessing decision.
- **Padding at position 576**: not position 0 or the 3' end. The D-loop starts at 576; it is the most variable region and most tolerant of inserted N content. Padding at the 3' end would displace the cytochrome b gene (positions 14747–15887) and invalidate position-based variant lookups.
- **`_rare` class merging**: preserves stratification proportions for large haplogroups without crashing on singleton haplogroups. The alternative (dropping rare-class sequences) would quietly remove up to 3–4% of the corpus.

## Verification

```
$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run pytest tests/ -m "not slow and not integration" -q
67 passed in 1.92s
```

Specific function checks:
```python
from mtdna_fm.data.preprocessor import (
    clean_sequence, normalize_length, stratified_split,
    RCRS_LENGTH, DLOOP_PAD_POSITION, JUNCTION_DUPLICATE_CHECK_BASES
)

# Clean
s = clean_sequence("acgtrywskm")
assert s == "ACGTNNNNNN"  # IUPAC codes → N

# Junction duplicate strip
prefix = "A" * 200
assert len(clean_sequence(prefix + "G" * 300 + prefix)) == 500

# Normalize
assert len(normalize_length("A" * 100)) == 16569

# Stratified split fractions
import pandas as pd, numpy as np
df = pd.DataFrame({"accession": [f"s{i}" for i in range(1000)],
                   "haplogroup": ["H"]*500 + ["L"]*500})
result = stratified_split(df)
counts = result["split"].value_counts(normalize=True)
assert abs(counts["train"] - 0.8) < 0.05
assert abs(counts["val"]   - 0.1) < 0.05
assert abs(counts["test"]  - 0.1) < 0.05
```

## Next up

Day 5: variant datasets — download gnomAD chrM, ClinVar pathogenic mtDNA, and PhyloTree Build 17; build `variant_processor.py` producing three clean parquet files for the fine-tuning tasks.
