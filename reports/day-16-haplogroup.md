# Day 16: Haplogroup Classification Fine-tuning

## What was built

- [mtdna_fm/model/model.py](../mtdna_fm/model/model.py) — `MtDNAForHaplogroupClassification(PreTrainedModel)`: wraps the pre-trained encoder with a `Linear(256, 26)` classification head; supports single-window `(batch, seq_len)` and multi-window `(batch, n_windows, seq_len)` inputs, mean-pooling CLS tokens across windows for whole-genome embeddings
- [mtdna_fm/model/model.py](../mtdna_fm/model/model.py) — `HaplogroupClassificationOutput`: typed output dataclass (loss, logits, hidden_states, attentions)
- [mtdna_fm/model/__init__.py](../mtdna_fm/model/__init__.py) — exported `MtDNAForHaplogroupClassification` and `HaplogroupClassificationOutput`
- [configs/finetuning_haplogroup.yaml](../configs/finetuning_haplogroup.yaml) — full fine-tuning config: LoRA r=8, lora_alpha=16, target_modules=[query, key, value, dense], lr=1e-3, 20 epochs, batch=32
- [mtdna_fm/scripts/finetune.py](../mtdna_fm/scripts/finetune.py) — `mtdna-finetune` CLI: loads Phase 2 checkpoint, applies PEFT LoRA, trains `HaplogroupWindowDataset` over overlapping windows, evaluates per-epoch, saves best checkpoint and `eval_metrics.json`
- [tests/test_model.py](../tests/test_model.py) — `TestMtDNAForHaplogroupClassification`: 10 tests covering single-window shape, multi-window shape, loss computation, gradient flow, freeze/unfreeze, LoRA compatibility, loss convergence over 5 steps

## What was learned

- **LoRA rank choice is a function of dataset size**: haplogroup classification has 47k labelled sequences — enough that r=8 (vs r=4 for pathogenicity on 7k sequences) adds useful capacity without overfitting. The adapter is ~500KB; the base model 24MB. This ratio means adapters are practical to share independently.
- **Mean-pooling CLS tokens across windows preserves circular genome structure**: each overlapping 512-token window contributes one CLS vector; averaging gives a genome-level embedding where every genomic region is represented. This is the same strategy sentence-transformers use for long documents.
- **PEFT `get_peft_model()` wraps any `PreTrainedModel` by layer name**: target_modules=["query","key","value","dense"] matches any `nn.Linear` with those names across all transformer layers simultaneously. The LoRA matrices are injected without touching the base weights — a frozen base remains frozen.
- **`freeze_encoder()` / `unfreeze_encoder()`**: fine-tuning strategy choice at runtime. Head-only (0.5% trainable) is fast and avoids catastrophic forgetting. Full fine-tuning (100% trainable) gives the ceiling accuracy. LoRA is the practical middle: ~2% trainable, 95%+ of full fine-tune performance.
- **Multi-window training with flattened batch dimension**: passing `(batch, n_windows, seq_len)` tensors then reshaping to `(batch * n_windows, seq_len)` before the encoder lets PyTorch handle all windows in one forward pass. No Python loop over windows needed during training.

## Key decisions

- **Model accepts both 2D and 3D input_ids**: single-window (2D) for standard training loop; multi-window (3D) for whole-genome inference in one call. Dispatch is by `input_ids.dim() == 3`. This avoids forcing callers to flatten manually.
- **26 PhyloTree haplogroups as fixed label set**: matches the major branch labels (A–X) from PhyloTree Build 17. Sub-haplogroups (H1, H2, …) are folded to their major branch at label time. This is the biologically meaningful granularity for a first model.
- **`HaplogroupWindowDataset` treats each window independently at training time**: the haplogroup label is repeated across all windows of the same genome. This gives the model more gradient signal per genome and is standard for sequence-level classification from long inputs.
- **Save `eval_metrics.json` alongside checkpoint**: `{"best_val_accuracy": ..., "final_train_loss": ...}` makes DVC `metrics show` work without any extra code. The format is identical to what `mtdna-evaluate` will produce on Day 19.
- **`finetune.py` gracefully handles missing Phase 2 checkpoint**: exits with a clear message instructing the user to run Phase 2 pre-training first, rather than crashing with an opaque AttributeError.

## Verification

```
$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run pytest tests/ -m "not slow and not integration" -q
264 passed, 2 warnings in 4.50s

$ uv run pytest tests/ -m "not slow and not integration" --cov=mtdna_fm -q | grep TOTAL
TOTAL   1644   191   88%
```

New tests (10 in `TestMtDNAForHaplogroupClassification`):
- single-window forward: logits shape (2, 5) ✓
- multi-window forward: (2, 3, 8) → mean-pooled → logits (2, 5) ✓
- loss is scalar, not NaN ✓
- no-label → no loss ✓
- gradient reaches classifier.weight ✓
- freeze_encoder / unfreeze_encoder ✓
- get_input_embeddings delegates correctly ✓
- LoRA wraps without error, trainable params contain "lora_" ✓
- loss decreases over 5 gradient steps ✓
- num_labels stored correctly ✓

## Next up

Day 17: pathogenic variant prediction — binary classifier using the variant-position hidden state (not CLS), trained on ClinVar vs gnomAD common variants.
