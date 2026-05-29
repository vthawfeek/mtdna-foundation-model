# Day 17: Pathogenic Variant Prediction

## What was built

- `mtdna_fm/model/model.py` — added `MtDNAForVariantPathogenicity` and `VariantPathogenicityOutput`; binary classifier that reads the hidden state at the variant-position token (not CLS)
- `configs/finetuning_pathogenicity.yaml` — LoRA r=4, weight_decay=0.1, pos_weight=2.5 for the 1:2.5 ClinVar/gnomAD class imbalance
- `mtdna_fm/scripts/finetune.py` — added `PathogenicityVariantDataset` (512-token window centered on variant, synthetic fallback for tests) and `finetune_pathogenicity` training loop with sklearn AUROC evaluation
- `tests/test_model.py` — 10 new tests in `TestMtDNAForVariantPathogenicity` covering forward shape, probs range [0,1], BCE loss, gradient flow, pos_weight buffer, clamped variant_token_idx, freeze/unfreeze, loss-decreases-over-steps, LoRA compatibility

## What was learned

- **Local vs global representations**: haplogroup classification benefits from whole-genome pooling (CLS mean) because the haplogroup is determined by the cumulative pattern of variant sites across the genome. Pathogenicity is different — it is about whether *this specific change at this specific position* disrupts a protein, tRNA stem, or rRNA structural element. Reading the hidden state at the variant-position token rather than CLS directly encodes this locality assumption into the architecture.
- **Class imbalance and `pos_weight`**: ClinVar has roughly 2,000 pathogenic mtDNA variants. gnomAD common (AF > 0.01) provides around 5,000 negatives. `pos_weight=2.5` in `BCEWithLogitsLoss` is equivalent to upweighting the positive class by 2.5× — it raises the penalty for missing a true pathogenic variant relative to the penalty for a false alarm. This is appropriate for a screening task where the cost of a false negative (missing a real disease variant) is higher than a false positive.
- **LoRA rank selection**: r=4 for pathogenicity vs r=8 for haplogroup reflects dataset size. The pathogenicity dataset is ~7k variants. With r=4, the LoRA adapter has 4× fewer parameters to fit, which reduces the risk of the adapter memorising the small training set. Heavier `weight_decay=0.1` (vs 0.01 for haplogroup) further regularises.
- **Synthetic fallback in datasets**: `PathogenicityVariantDataset` generates 64 random synthetic variants when the parquet is missing. This keeps the test suite runnable in CI without real data while the actual data pipeline is being built separately.
- **`pos_weight` as a registered buffer**: Storing `pos_weight` via `register_buffer()` instead of `nn.Parameter` means it moves to the correct device with `.to(device)` but is excluded from the optimiser's parameter group. It also gets saved and restored with `save_pretrained`.

## Key decisions

- **Variant-position hidden state, not CLS**: pathogenicity is a local property — what the substitution does to a codon or tRNA arm — so the representation of the surrounding k-mer context at the mutation site is the most informative signal. CLS aggregates the whole window and dilutes the local signal.
- **Window centered on variant (512 tokens)**: centering on the variant gives equal left/right context (≈256 bp each side). For most mtDNA variants, this covers the entire gene they fall in (the longest protein-coding gene, ND5, is about 1,812 bp = 1,806 tokens; pathogenic variants in it are covered by roughly 3 non-overlapping windows). A 512-token window captures the immediate functional context for the majority of variants.
- **LoRA r=4 with weight_decay=0.1**: small dataset (7k variants) → small rank to prevent adapter overfitting; heavier L2 to compensate for the fact that even r=4 adds 4 × hidden_size parameters per wrapped layer.
- **Synthetic dataset fallback**: avoids a hard dependency on the variant parquet being present to run the test suite. The synthetic data shares the same tokenisation path as real data, so it tests the dataset code without testing biological plausibility.

## Verification

```
$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run pytest tests/ -m "not slow and not integration" -x -q
274 passed, 2 warnings in 4.57s
```

10 new tests added (264 → 274). All green. New test class `TestMtDNAForVariantPathogenicity` covers:
- Forward shapes: logits/probs are (batch,)
- Probs bounded in [0, 1]
- BCE loss is a scalar, no NaN
- No labels → no loss
- Gradients reach classifier.weight
- pos_weight is a registered buffer at the correct value
- Out-of-bounds variant_token_idx is clamped without raising
- Freeze/unfreeze encoder works
- Loss decreases over 5 gradient steps
- PEFT LoRA wraps the model correctly

## Next up

Day 18: Heteroplasmy level regression (`MtDNAForHeteroplasmyRegression`, Huber loss, sigmoid head, 5-fold cross-validation on gnomAD variants with ≥50 heteroplasmic carriers).
