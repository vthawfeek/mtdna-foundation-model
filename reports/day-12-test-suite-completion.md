# Day 12: Test Suite Completion (80% → 97% Coverage)

## What was built

- `tests/test_data.py` — 9 new tests in `TestHmtdbClientInternals` covering `_download_file`, `_download_fasta_from_hmtdb` (sha256 match/mismatch), `_download_metadata_from_hmtdb`, `_ncbi_fallback`, `_validate_fasta`, and `extract_zip_fasta`
- `tests/test_data.py` — 7 new tests in `TestNcbiClientInternals` covering `_rate_delay` (with/without API key), `_configure_api_key`, `_esearch`, `_efetch_batch`, zero-results error path, force-delete path, and the batch fetch loop
- `tests/test_data.py` — 9 new tests in `TestVariantDownloader` covering idempotency for gnomAD/ClinVar/PhyloTree, `_extract_chrom_from_gz`, `_stream_download`, and all three download functions with mocked network calls
- `tests/test_scripts.py` — `TestPreprocessCLI` (4 tests): no-source error, missing FASTA errors, successful pipeline with mocked preprocessor
- `tests/test_scripts.py` — `TestTrainCLI` (2 tests): missing config error, mocked trainer invocation
- `tests/test_scripts.py` — `TestDownloadScriptInternals` (5 tests): direct calls to `_run_hmtdb`, `_run_ncbi_refseq`, `_run_gnomad`, `_run_clinvar`, `_run_phylotree`
- `tests/test_model.py` — `TestGradientCheckpointing` (3 tests): enable flag, forward pass with checkpointing active, non-encoder module ignored
- `tests/test_model.py` — `TestTransformerValidation` (1 test): `ValueError` for `hidden_size % num_heads != 0`
- `tests/test_model.py` — `TestLearnablePositionalEncoding` (2 tests): fallback PE type check, forward pass shape

## What was learned

- **Mock patch paths matter**: When a function does a local `from X import Y`, the patch must target the original module (`X.Y`), not the caller's namespace. Patching `mtdna_fm.data.hmtdb_client.download_ncbi_mtdna` would miss `_ncbi_fallback` which imports it inside the function body; the correct target is `mtdna_fm.data.ncbi_client.download_ncbi_mtdna`.
- **`subprocess` imported inside a function needs `patch("subprocess.run")`** (the global module), not `patch("mymodule.subprocess.run")`, since `subprocess` is not a module-level attribute when imported inside a conditional branch.
- **pytest.raises as context manager composes cleanly with `patch`** inside a single `with (patch(...), pytest.raises(...)):` block, which satisfies ruff's `SIM117` rule about nested `with` statements.
- **Gradient checkpointing is gated on `self.training`**: the `torch.utils.checkpoint.checkpoint` path in `MtDNAEncoder.forward` only runs when both `gradient_checkpointing=True` AND the model is in training mode. Tests that only do `gradient_checkpointing_enable()` without calling `model.train()` would miss that branch.
- **Learnable PE uses absolute `nn.Embedding(max_seq_len)`** — not `genome_length` — so test position_ids must be within `[0, max_seq_len)`, not the full genome coordinate space.

## Key decisions

- **Targeted the lowest-coverage modules first**: `variant_downloader.py` (0%), `scripts/preprocess.py` (0%), `scripts/train.py` (0%), then `hmtdb_client.py` (49%), `ncbi_client.py` (59%). This produced the maximum coverage gain with the fewest tests.
- **Mocked all network calls**: every test that touches `requests.get`, `Entrez.esearch`, or `Entrez.efetch` patches the network layer. Tests run in <10 seconds with no internet dependency.
- **Used real zip/gz files** for `extract_zip_fasta` and `_extract_chrom_from_gz` tests — creating actual in-memory archives rather than mocking zipfile or gzip. This tests real parsing logic, not just API signatures.

## Verification

```
uv run ruff check mtdna_fm/ tests/
# All checks passed!

uv run pytest tests/ -m "not slow and not integration" --cov=mtdna_fm
# 236 passed, 2 warnings in 8.01s

Coverage summary (before → after):
  variant_downloader.py    0% → 100%
  scripts/download.py     64% → 100%
  scripts/train.py         0% → 100%
  scripts/preprocess.py    0% →  89%
  hmtdb_client.py         49% →  99%
  ncbi_client.py          59% →  98%
  model/model.py          98% → 100%
  model/embeddings.py     98% → 100%
  model/transformer.py    98% → 100%
  TOTAL                   80% →  97%
```

## Next up

Day 13: Training analysis notebook — MLM loss and LR curves from MLflow, attention weight heatmaps, and zero-shot k-NN haplogroup classification using Phase 1 CLS embeddings.
