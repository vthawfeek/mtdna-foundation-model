# Day 3: Data download clients

## What was built

- `mtdna_fm/data/hmtdb_client.py` — Idempotent HmtDB download: checks for existing outputs before attempting any network call, streams FASTA and metadata CSV, verifies SHA256 before writing, falls back automatically to NCBI Entrez if HmtDB is unreachable. Also exports `extract_zip_fasta` for distributions that ship a zip archive.
- `mtdna_fm/data/ncbi_client.py` — Resumable NCBI Entrez client: uses `esearch` + `efetch` with `usehistory=True` (WebEnv) so the server caches the result set. Writes a `.progress.json` file tracking which batches are done; a second run reads it and skips completed batches. Reads `NCBI_API_KEY` from environment for the 10 req/s tier.
- `mtdna_fm/scripts/download.py` — Full Typer CLI replacing the Day 1 stub. Accepts `--source` (hmtdb | ncbi-refseq | gnomad | clinvar | phylotree), `--output`, and `--force`. Dispatches to the appropriate client. gnomad/clinvar/phylotree stubs ready for Day 5.
- `dvc.yaml` — DVC pipeline with `download_hmtdb` and `download_ncbi` stages, both with `persist: true` so `dvc gc` does not delete the downloaded data.
- `tests/test_data.py` — 17 unit tests covering idempotency, progress file round-trip, fallback trigger, SHA256 mismatch, directory creation, and CLI routing. All tests mock network calls and run offline.

## What was learned

- **Idempotency as a first-class property.** A download script that always fetches is a footgun in a bioinformatics pipeline: re-running the pipeline or resuming from a checkpoint re-downloads 200 MB of FASTA. Checking for existing outputs at the top of every download function costs one `Path.exists()` call and saves hours.
- **WebEnv batching eliminates repeated searches.** Without `usehistory=True`, each `efetch` call for a batch would re-run the search against NCBI's index. With WebEnv, the search result set is cached server-side and the batch calls just page through it. This is the right pattern whenever you are fetching >500 records from Entrez.
- **Progress files make long downloads resumable.** A 30k-record NCBI fetch with batch_size=500 is 60 HTTP requests and takes 20 minutes on the free tier. If the process is interrupted at batch 45, the progress file allows resuming from batch 46 rather than starting over.
- **`persist: true` in DVC prevents data loss on `dvc gc`.** DVC's garbage collection removes outputs that are not referenced by the current pipeline state. For raw data downloads, which are expensive to regenerate and not tracked by content hash, `persist: true` keeps them safe regardless of pipeline state.
- **Mocking at the module boundary.** For download tests to run offline and fast, the mock target must be the function's name in the module that *uses* it, not where it is defined. `patch("mtdna_fm.data.hmtdb_client._download_fasta_from_hmtdb")` patches it at the point of call, not at the definition — this is why the fallback test works correctly.

## Key decisions

- **NCBI fallback in `hmtdb_client`, not in `download.py`:** The client owns its fallback logic rather than pushing it up to the CLI layer. This keeps the CLI thin and makes the fallback testable in isolation.
- **Progress file keyed by batch index, not by retstart:** Using `str(batch_idx)` as the key keeps the progress file human-readable and independent of batch_size changes. If batch_size is changed between runs, the progress file is ignored (new batches have no entry) and the download restarts — safer than partial consistency.
- **`_efetch_batch` is a standalone function, not a method:** Testable in isolation without needing a full client object. The same pattern appears in `ncbi_client`'s `_esearch` and `_save_progress` — all the stateless operations are module-level functions that take and return plain values.
- **Day 5 stubs in download.py exit with code 1:** gnomad, clinvar, and phylotree sources raise `typer.Exit(code=1)` with a clear message rather than silently succeeding. This makes it obvious if a pipeline accidentally references an unimplemented source.

## Verification

```
$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run pytest tests/ -m "not slow and not integration" -v
============================= test session starts ==============================
collected 41 items

tests/test_data.py::TestHmtdbClient::test_skips_download_when_outputs_exist PASSED
tests/test_data.py::TestHmtdbClient::test_force_triggers_redownload PASSED
tests/test_data.py::TestHmtdbClient::test_sha256_mismatch_raises PASSED
tests/test_data.py::TestHmtdbClient::test_output_dir_created_if_missing PASSED
tests/test_data.py::TestHmtdbClient::test_fallback_called_on_network_error PASSED
tests/test_data.py::TestNcbiClient::test_progress_file_initialises_empty PASSED
tests/test_data.py::TestNcbiClient::test_progress_file_save_load_roundtrip PASSED
tests/test_data.py::TestNcbiClient::test_progress_complete_all_done PASSED
tests/test_data.py::TestNcbiClient::test_progress_incomplete_missing_batch PASSED
tests/test_data.py::TestNcbiClient::test_skips_all_batches_when_complete PASSED
tests/test_data.py::TestNcbiClient::test_output_dir_created PASSED
tests/test_data.py::TestNcbiClient::test_count_fasta_records PASSED
tests/test_data.py::TestDownloadCLI::test_invalid_source_exits_with_error PASSED
tests/test_data.py::TestDownloadCLI::test_unimplemented_source_exits_with_error PASSED
tests/test_data.py::TestDownloadCLI::test_valid_source_hmtdb_calls_download PASSED
tests/test_data.py::TestDownloadCLI::test_valid_source_ncbi_calls_download PASSED
tests/test_data.py::TestDownloadCLI::test_force_flag_passed_through PASSED
[...24 tokenizer tests, all PASSED...]

41 passed in 2.85s
```

## Next up

Day 4: preprocessing pipeline — sequence cleaning, length normalization to 16,569 bp, stratified 80/10/10 split by haplogroup, and the EDA notebook documenting D-loop entropy and haplogroup distribution.
