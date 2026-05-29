# Day 21: HuggingFace Hub

## What was built

- `mtdna_fm/scripts/push_to_hub.py` — rewrote to push the actual trained adapters (not fresh random ones), patch `base_model_name_or_path` from local path to Hub model ID, write proper READMEs for each adapter repo
- `models/phase1_v1/README.md` — updated model card: Phase 2 completion noted, performance table updated with actual results (50% zero-shot vs 12.5% random, >90% fine-tuned haplogroup), ancient DNA zero-shot results, limitations section
- `vthawfeek/mtdna-foundation-model` (HuggingFace Hub) — base model pushed: config.json, model.safetensors (44.6MB / 6.9M params), tokenizer, vocab, model card
- `vthawfeek/mtdna-fm-haplogroup` (HuggingFace Hub) — LoRA adapter r=8, 400KB, haplogroup classification (26 classes), proper README
- `vthawfeek/mtdna-fm-pathogenicity` (HuggingFace Hub) — LoRA adapter r=4, 203KB, variant pathogenicity prediction, proper README

## What was learned

- **LoRA as a deployment artefact:** The adapter for haplogroup classification is 400KB. The base model is 44.6MB. You can update the classification head without redistributing the base model — each downstream task adds <1% to the storage cost of the base.
- **Patching adapter configs for the Hub:** PEFT saves `base_model_name_or_path` as a local filesystem path. Before pushing to Hub, this must be patched to the Hub model ID so users can load the adapter without having the local filesystem layout. The push script does this in a temp directory.
- **Custom architectures and AutoConfig:** HuggingFace's `AutoConfig.from_pretrained` refuses to load unknown `model_type` values — expected for a custom architecture not registered with Transformers. The workaround is to load via the concrete class (`MtDNAForMaskedModeling.from_pretrained`) directly, which works because the class knows its own config schema.
- **Model card as documentation:** The model card is the first thing users see on the Hub. Treating it as a primary deliverable (architecture novelties with equations, honest performance table with baselines, limitations section) is what distinguishes a research-grade release from a weight dump.

## Key decisions

- **Push phase1_v1 as the base model:** Phase 2 trained on human HmtDB but the checkpoint directory was empty of model files (only adapter files from finetuning were saved there). Phase 1 provides the full base encoder with all necessary files.
- **Real adapters, not fresh ones:** The original push_to_hub.py created fresh LoRA adapters with random weights and pushed those. Changed to upload the actual trained adapters from `finetune_haplogroup_v1` and `finetune_pathogenicity_v1` — the versions that were actually trained, not placeholder weights.
- **Separate repos for adapters:** Following the PEFT convention of separate Hub repos per adapter makes it easy for users to load any combination of base + adapter without downloading everything. The adapter `base_model_name_or_path` points back to the base model Hub ID.
- **AutoConfig warning is expected:** The `model_type: mtdna_fm` is custom and not registered with Transformers. This is fine — users who install the `mtdna-fm` package get the correct class. The warning in the verification step is documented in the model card as a known limitation.

## Verification

```
uv run python mtdna_fm/scripts/push_to_hub.py

Authenticated as: vthawfeek

=== Pushing base model to vthawfeek/mtdna-foundation-model ===
  Repository vthawfeek/mtdna-foundation-model ready
  ✓ config.json
  ✓ model.safetensors      (44.6 MB)
  ✓ tokenizer_config.json
  ✓ vocab.json
  ✓ README.md

=== Pushing adapter to vthawfeek/mtdna-fm-haplogroup ===
  ✓ README.md
  ✓ adapter_model.safetensors  (400 KB)
  ✓ adapter_config.json

=== Pushing adapter to vthawfeek/mtdna-fm-pathogenicity ===
  ✓ README.md
  ✓ adapter_model.safetensors  (203 KB)
  ✓ adapter_config.json

=== Verifying Hub load for vthawfeek/mtdna-foundation-model ===
  AutoConfig failed (expected — custom model_type not registered with Transformers)
  MtDNAForMaskedModeling: 6,907,393 parameters
  ✓ Hub load verified

uv run ruff check mtdna_fm/ tests/   → All checks passed!
uv run pytest tests/ -m "not slow and not integration" -q  → 346 passed in 40.35s
```

## Next up

Day 22: Build a Gradio demo on HuggingFace Spaces with three tabs — haplogroup classification, variant pathogenicity, and genome embedding visualisation.
