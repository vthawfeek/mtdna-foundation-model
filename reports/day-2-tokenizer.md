# Day 2: Tokenizer

## What was built

- `mtdna_fm/tokenizer/vocabulary.py` — `KmerVocabulary` class: builds all 4,096 possible 6-mers from the ACGT alphabet plus 6 special tokens (PAD, CLS, MASK, UNK, SEP, HET) at indices 0–5, for a total vocabulary of 4,102. Sorted lexicographically for determinism. Supports `save_pretrained` / `from_pretrained` following HuggingFace conventions.
- `mtdna_fm/tokenizer/tokenize.py` — `tokenize_sequence` function: sliding k-mer window over DNA, returning `input_ids`, `attention_mask`, `position_ids`, and `het_values`. When `circular=True`, k-mers wrap at the sequence boundary using modular indexing so every genomic position produces exactly one token per stride.
- `mtdna_fm/tokenizer/__init__.py` — public exports: `KmerVocabulary`, `tokenize_sequence`, and all six special-token ID constants.
- `tests/test_tokenizer.py` — 24 tests across `TestKmerVocabulary` (9 tests) and `TestTokenizeSequence` (15 tests): vocabulary size, special-token IDs, encode/decode roundtrip, save/load, circular junction coverage, stride variations, truncation, N-character handling, case normalization, heteroplasmy values.

## What was learned

- **K-mer vs BPE tokenization.** BPE builds a vocabulary from corpus statistics; the vocabulary depends on the training data and is never the same twice. A k-mer vocabulary is completely determined by alphabet size and k: for ACGT with k=6, it is always exactly 4,096 tokens. This determinism means the tokenizer is reproducible on any machine without fitting it first.
- **Circular boundary coverage.** For a standard linear sliding window, the k-mers starting within the last k-1 positions of the sequence would fall off the end. For a circular genome, those positions need to be covered too. Using `seq[(p + j) % L]` for each k-mer base produces wrap-around k-mers without any preprocessing trick and keeps position IDs clean (0 to L-1 in order).
- **Why position_ids matter early.** The model's circular positional encoding indexes into a pre-computed buffer using absolute genomic coordinates. If the tokenizer assigned window-relative positions (0, 1, 2, ...), the circular PE buffer would be indexed incorrectly. By assigning position IDs equal to the k-mer's start position in the genome, the tokenizer and model share a common coordinate system.
- **Heteroplasmy as a continuous channel.** Representing het levels as a float per token (0.0–1.0) rather than discretizing them avoids a design decision that might throw away information. The model can learn whatever threshold matters from the data.

## Key decisions

- **Special tokens at indices 0–5, k-mers from index 6 onward:** This reserves a fixed, stable range for special tokens regardless of k. A `[PAD]` ID of 0 is conventional in PyTorch (where embedding layers typically zero-pad at index 0).
- **Modular indexing for circular wrap rather than string prepending:** Either approach produces the same k-mers, but modular indexing assigns position IDs naturally (p is always the correct genomic start position) and avoids building a modified string. It also makes the code's intent clearer.
- **`max_seq_len` truncates without padding:** Padding is the job of the DataCollator, not the tokenizer. Returning variable-length outputs from `tokenize_sequence` keeps the function stateless and composable.
- **`scope="module"` on vocab fixtures:** Building the k=6 vocabulary takes about 50 ms (itertools.product over 4^6 combinations). With module scope, it is built once and reused across all 24 tests, keeping the suite under 0.2 s.

## Verification

```
$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run pytest tests/ -m "not slow and not integration" -v
============================= test session starts ==============================
collected 24 items

tests/test_tokenizer.py::TestKmerVocabulary::test_vocabulary_size PASSED
tests/test_tokenizer.py::TestKmerVocabulary::test_vocabulary_size_k3 PASSED
tests/test_tokenizer.py::TestKmerVocabulary::test_special_token_ids PASSED
tests/test_tokenizer.py::TestKmerVocabulary::test_kmer_ids_start_at_six PASSED
tests/test_tokenizer.py::TestKmerVocabulary::test_encode_decode_roundtrip PASSED
tests/test_tokenizer.py::TestKmerVocabulary::test_unknown_kmer_maps_to_unk PASSED
tests/test_tokenizer.py::TestKmerVocabulary::test_vocabulary_is_deterministic PASSED
tests/test_tokenizer.py::TestKmerVocabulary::test_vocabulary_save_load PASSED
tests/test_tokenizer.py::TestKmerVocabulary::test_contains PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_circular_junction_covered PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_linear_token_count PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_circular_stride2 PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_output_keys PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_attention_mask_all_ones PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_attention_mask_length_matches_input_ids PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_position_ids_range PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_position_ids_stride1_circular PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_max_seq_len_truncation PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_n_maps_to_unk PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_case_insensitive PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_het_values_default_zeros PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_het_values_provided PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_het_values_length_matches PASSED
tests/test_tokenizer.py::TestTokenizeSequence::test_all_ids_valid PASSED

============================== 24 passed in 0.13s ==============================

$ python -c "
from mtdna_fm.tokenizer import KmerVocabulary, tokenize_sequence
v = KmerVocabulary.build(k=6)
print(f'Vocabulary size: {len(v)}')
tokens = tokenize_sequence('ACGT' * 4142, vocabulary=v, circular=True)
print(f'16568 bp circular -> {len(tokens[\"input_ids\"])} tokens (== 16568)')
print(f'Roundtrip ATGCAT: {v.decode(v.encode(\"ATGCAT\"))}')
"
Vocabulary size: 4102
16568 bp circular -> 16568 tokens (== 16568)
Roundtrip ATGCAT: ATGCAT
```

## Next up

Day 3: idempotent download clients for HmtDB and NCBI Entrez — BioPython-based FASTA parsing, SHA256 verification, batch-resumable Entrez fetching.
