# Day 9: Masking and Loss

## What was built

- **`mtdna_fm/training/masking.py`** — `MtDNAMaskingCollator`: batches tokenised window dicts and applies BERT 80/10/10 masked k-mer modelling. Collation-time masking (not pre-computed) so each epoch sees different masks over the same windows. Blacklists genomic positions 303-315 (the homopolymeric C-tract in the D-loop) from ever being selected as masking targets. Outputs `kmer_labels` (original IDs at masked positions, -100 elsewhere) and `het_labels` (het values at masked positions, -1 elsewhere).
- **`mtdna_fm/training/losses.py`** — `mtdna_mlm_loss()`: combined masked k-mer CE + heteroplasmy MSE loss. `mlm_weight=1.0` always; `het_weight=0.0` for Phase 1 (no het data), `het_weight=0.3` for Phase 2. CE uses `ignore_index=-100`; MSE uses `het_labels != -1` guard.
- **`mtdna_fm/tokenizer/vocabulary.py`** — added convenience property accessors (`pad_token_id`, `cls_token_id`, `mask_token_id`, `unk_token_id`, `sep_token_id`, `het_token_id`, `n_special`) to `KmerVocabulary` so downstream code can use `vocab.mask_token_id` instead of module-level constants.
- **`tests/test_training.py`** — 20 tests across two classes:
  - `TestMtDNAMaskingCollator` (11 tests): output keys/shapes, ~15% masking rate, blacklist enforcement, CLS never masked, 80/10/10 split verified statistically, het_labels alignment with kmer_labels, fallback for missing het_values
  - `TestMtDNAMLMLoss` (9 tests): scalar return, sparse vs dense CE comparison, het_weight=0 isolation, combined > MLM-only, mlm_weight scaling, all-invalid het guard, gradient flow, non-negativity, no NaN

## What was learned

- **BERT 80/10/10 masking prevents shortcut learning**: if all masked positions were replaced with `[MASK]`, the model could learn "whenever I see `[MASK]` I should predict something; for all other positions I can ignore." The 10% random replacement forces the model to represent *all* positions correctly, and the 10% unchanged forces the representation to be useful at inference where no `[MASK]` tokens exist.
- **Collation-time masking is equivalent to 1/mask_prob × data augmentation for free**: pre-computing masks would mean each window sees the same 15% of tokens masked every epoch. By sampling masks at collation time, epoch 1 masks different positions than epoch 2 on the same windows — about 6.67 independent views of each window across epochs at 15% mask rate.
- **Biological blacklisting is a form of domain-specific regularisation**: the C-tract (positions 303-315) is dominated by sequencing noise — homopolymer runs are systematically miscalled by short-read sequencers. Without the blacklist, the model would allocate representation capacity to learning sequencer artefacts rather than biological signal. The blacklist is cheap (a frozenset lookup per token) and eliminates a source of spurious gradients.
- **PyTorch `cross_entropy` with all `ignore_index=-100` returns `nan`, not 0.0**: this is correct behaviour (there are no valid terms to sum), but a test that asserted `loss == 0.0` would fail. The right invariant to test is that a sparse label tensor produces a different loss than a dense one, not the specific NaN/zero edge case.
- **`het_labels` sentinel as -1 (not -100) avoids confusion**: using `-100` for both kmer_labels and het_labels would be confusing since het_labels are floats and -100 could in principle be a valid target if het were unnormalised. Using `-1.0` as the sentinel is unambiguous given het values are bounded [0, 1].

## Key decisions

- **Blacklist as a `frozenset`**: O(1) membership test per token per sample, and the set is constructed once at collator init, not per-call. With a 30-token window at 15% masking, the blacklist is consulted for every eligible position on every forward pass — constant-time lookup matters.
- **Masking applied in collator, not dataset**: if masking were in `MtDNADataset.__getitem__`, the mask would be fixed for each window at dataset load time. Putting it in the collator means `DataLoader` workers apply a fresh random mask to each window in each batch, effectively making each epoch a new augmentation. This follows the scFM design.
- **`het_weight=0.0` default**: the combined loss function is designed for both phases, but Phase 1 (cross-species corpus) has no heteroplasmy data. Setting `het_weight=0.0` skips the MSE term entirely — no computation, no gradient. This avoids needing two separate loss functions.
- **Vocabulary convenience properties over module constants**: callers would need to import `MASK_TOKEN_ID` from the vocabulary module directly if the class had no accessors. Adding properties (`vocab.mask_token_id`) follows the encapsulation pattern used by HuggingFace tokenizers and means the collator only needs to import the class, not the module-level constants.

## Verification

```
$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run pytest tests/ -m "not slow and not integration" -v --tb=short 2>&1 | tail -15
tests/test_training.py::TestMtDNAMaskingCollator::test_output_keys PASSED
tests/test_training.py::TestMtDNAMaskingCollator::test_output_shapes PASSED
tests/test_training.py::TestMtDNAMaskingCollator::test_masking_rate_approximately_15_percent PASSED
tests/test_training.py::TestMtDNAMaskingCollator::test_blacklisted_positions_never_masked PASSED
tests/test_training.py::TestMtDNAMaskingCollator::test_cls_never_masked PASSED
tests/test_training.py::TestMtDNAMaskingCollator::test_80_10_10_split PASSED
tests/test_training.py::TestMtDNAMaskingCollator::test_kmer_labels_only_at_masked_positions PASSED
tests/test_training.py::TestMtDNAMaskingCollator::test_het_labels_at_masked_positions PASSED
tests/test_training.py::TestMtDNAMaskingCollator::test_het_values_range PASSED
tests/test_training.py::TestMtDNAMaskingCollator::test_no_het_values_in_batch PASSED
tests/test_training.py::TestMtDNAMaskingCollator::test_default_blacklist_constant PASSED
tests/test_training.py::TestMtDNAMLMLoss::test_returns_scalar_tensor PASSED
tests/test_training.py::TestMtDNAMLMLoss::test_loss_only_at_masked_positions PASSED
tests/test_training.py::TestMtDNAMLMLoss::test_het_weight_zero_ignores_het PASSED
tests/test_training.py::TestMtDNAMLMLoss::test_combined_loss_larger_than_mlm_alone PASSED
tests/test_training.py::TestMtDNAMLMLoss::test_mlm_weight_scales_loss PASSED
tests/test_training.py::TestMtDNAMLMLoss::test_het_loss_only_at_valid_labels PASSED
tests/test_training.py::TestMtDNAMLMLoss::test_gradient_flows_through_loss PASSED
tests/test_training.py::TestMtDNAMLMLoss::test_loss_is_non_negative PASSED
tests/test_training.py::TestMtDNAMLMLoss::test_no_nan_loss PASSED

153 passed, 1 warning in 1.45s
```

## Next up

Day 10: Pre-training launch — `trainer.py`, `pretraining_phase1.yaml`, and the first `mtdna-train` run that takes MLM loss from 8.3 (random baseline) toward convergence.
