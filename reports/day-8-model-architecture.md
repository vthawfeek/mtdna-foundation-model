# Day 8: Model Architecture

## What was built

- [configs/model_small.yaml](../configs/model_small.yaml) — canonical small model config: 6 layers, 8 heads, 256 hidden, ~5.8M parameters
- [mtdna_fm/model/config.py](../mtdna_fm/model/config.py) — `MtDNAConfig(PretrainedConfig)` with two novel fields: `genome_length` (16569) and `use_circular_encoding`
- [mtdna_fm/model/embeddings.py](../mtdna_fm/model/embeddings.py) — `MtDNACircularPositionalEncoding` (fixed buffer) + `MtDNAEmbeddings` (k-mer + circular PE + het projection)
- [mtdna_fm/model/transformer.py](../mtdna_fm/model/transformer.py) — pre-LN bidirectional transformer ported from scFM with mtDNA config types (`MtDNAAttention`, `MtDNAFFN`, `MtDNALayer`, `MtDNAEncoder`)
- [mtdna_fm/model/model.py](../mtdna_fm/model/model.py) — `MtDNAModel(PreTrainedModel)` base encoder + `MtDNAForMaskedModeling` with k-mer prediction head and heteroplasmy regression head
- [mtdna_fm/model/__init__.py](../mtdna_fm/model/__init__.py) — clean public API for the model package
- [tests/conftest.py](../tests/conftest.py) — added `tiny_config` (genome_length=100, hidden=16, 2 layers) and `tiny_vocabulary` fixtures
- [tests/test_model.py](../tests/test_model.py) — 38 new tests across 5 test classes covering config, circular PE, embeddings, base model, and masked modeling

## What was learned

- **Circular PE as a fixed buffer, not a parameter.** The circular topology of mtDNA is not a statistical property of the training data — it is a biological fact. Registering the PE as a `register_buffer` means no gradient will ever update it. Standard BERT's sinusoidal PE treats positions 0 and 16568 as maximally different; the circular formula makes their encodings mathematically identical (sin(0) = sin(2π)).

- **Two-class model separation (MtDNAModel vs MtDNAForMaskedModeling).** Following the BertModel / BertForMaskedLM pattern means the prediction heads used only during pre-training are discarded at fine-tuning time. The base `MtDNAModel` is what gets pushed to HuggingFace Hub and used downstream — no pre-training cruft leaks into fine-tuning.

- **Het projection as a continuous channel, not discretized bins.** The scFM analogue (`value_projection`) inspired mapping the continuous heteroplasmy float directly into embedding space via a Linear + LayerNorm. Discretizing into, say, 10 bins would lose resolution at the boundaries and introduce an arbitrary design choice. The continuous projection is simpler and more expressive.

- **Named linear layers for PEFT compatibility.** Naming the attention projections `query`, `key`, `value`, `dense` is not cosmetic — PEFT's `get_peft_model()` targets modules by name when applying LoRA. Renaming them to `q_proj` or similar would require updating the LoRA config on every fine-tuning task. The test `test_peft_lora_compatibility` verifies this works end-to-end.

- **Pre-LN transformer is more stable than post-LN.** The original BERT applies LayerNorm after the residual addition. Pre-norm applies it before. Modern pre-training literature (GPT-2, LLaMA, Geneformer) universally uses pre-norm because it avoids gradient vanishing in deep networks, which matters especially when training on a laptop with gradient checkpointing.

## Key decisions

- **Fixed circular PE buffer (not learnable):** The circular topology is a biological fact, not a learned pattern. Making it non-learnable means it cannot be corrupted during fine-tuning and is interpretable analytically.

- **`MtDNAForMaskedModeling` het weight defaults to 0.0:** Phase 1 pre-training is cross-species with no heteroplasmy data. The combined loss `mlm_weight * MLM + het_weight * MSE` correctly drops the het term when `het_weight=0`, making the same model class handle both training phases.

- **3-layer MLP for kmer_prediction_head:** `hidden → hidden (GELU) → LayerNorm → vocab_size`. The intermediate GELU layer gives the head capacity to transform contextual embeddings without the final vocab projection becoming a bottleneck for a 4,102-class output. Matches scFM's gene prediction head architecture.

- **`MtDNAModel` parameter count: 5,790,720.** At `hidden=256, 6 layers, 8 heads, intermediate=1024`, this is ~5.8M parameters — trainable on a CPU in 8-12 hours at 50k steps, and fast enough to iterate architecturally without a GPU.

## Verification

```
$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run ruff format --check mtdna_fm/ tests/
26 files already formatted

$ uv run pytest tests/ -m "not slow and not integration" -v
======================== 133 passed, 1 warning in 1.29s ========================

$ python -c "
from mtdna_fm.model.model import MtDNAModel
from mtdna_fm.model.config import MtDNAConfig
import yaml
with open('configs/model_small.yaml') as f:
    cfg = MtDNAConfig(**yaml.safe_load(f))
model = MtDNAModel(cfg)
total = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total:,}')
"
Total parameters: 5,790,720
```

Tests added today:
- `TestMtDNAConfig` (5 tests) — JSON roundtrip, novel field values, special token IDs, save/load
- `TestCircularPE` (5 tests) — output shape, batch shape, non-learnable buffer, circular boundary behaviour, buffer size
- `TestMtDNAEmbeddings` (4 tests) — forward shape, het_values optional (None == zeros), no-het-projection config, dtype
- `TestMtDNAModel` (8 tests) — forward shapes, pooler is CLS token, attention/hidden_states optional outputs, save/load, gradient flow, attention mask effect, CPU-only
- `TestMtDNAForMaskedModeling` (8 tests) — loss is scalar, loss with het, no-labels = no-loss, logits shape, het_preds range, gradient flow, loss decreases over 5 steps, PEFT/LoRA compatibility

## Next up

Day 9: Masking collator (`MtDNAMaskingCollator` with D-loop blacklist at positions 303-315) and combined MLM + heteroplasmy loss function.
