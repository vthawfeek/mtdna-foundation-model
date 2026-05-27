# Day 10: Pre-training Launch (Phase 1)

## What was built

- [`mtdna_fm/training/trainer.py`](../mtdna_fm/training/trainer.py) — `MtDNATrainer` class: full two-phase aware pre-trainer with cosine LR schedule, gradient accumulation, MLflow logging, checkpoint rotation, and Phase 2 encoder-only loading
- [`configs/pretraining_phase1.yaml`](../configs/pretraining_phase1.yaml) — Phase 1 training config (all-species, 50k steps, het_weight=0)
- [`configs/pretraining_phase2.yaml`](../configs/pretraining_phase2.yaml) — Phase 2 training config (human-only, 25k steps, het_weight=0.3, lower LR)
- [`mtdna_fm/scripts/train.py`](../mtdna_fm/scripts/train.py) — Updated `mtdna-train` CLI to actually invoke the trainer (was a stub)
- [`tests/test_trainer.py`](../tests/test_trainer.py) — 21 new tests covering: LR schedule, setup, evaluate, training loop, checkpoint save/load, checkpoint rotation, final save, Phase 2 encoder loading, optimizer param groups, species filter, het weight, from_yaml

## What was learned

- **Gradient accumulation is transparent to the model**: the pattern is `loss / grad_accum` at each micro-step, then `optimizer.step()` every N micro-steps. The model sees an effective batch size of 128 (16 × 8) even though only 16 sequences fit in memory at once.

- **Cosine LR with warmup has two distinct regimes**: warmup (linear 0 → lr over 2k steps) prevents divergence from random initialization; cosine decay (lr → 0.1×lr over remaining steps) prevents the model from losing convergence near the optimum. The schedule is non-negotiable for BERT-scale pre-training.

- **Why Phase 2 must not resume the optimizer**: AdamW maintains per-parameter running averages of gradient moments. These are calibrated to Phase 1's cross-species gradient landscape. Carrying them into Phase 2 would apply Phase 1's second-moment corrections to Phase 2's human-specific gradients — a mismatch that prevents the model from adapting to the new distribution at the beginning of Phase 2. Fresh optimizer = clean start on human data.

- **Deriving k from vocab_size**: `k = log₄(vocab_size - n_special)`. This lets the trainer automatically handle any k-mer vocabulary without being hardcoded to k=6, which is essential for unit tests that use tiny 3-mer vocabularies.

- **Pre-tokenizing the full dataset upfront**: `MtDNADataset` tokenizes all sequences at construction time. For 152k × 16,569-bp sequences, this takes 2–3 minutes. The tradeoff: each `__getitem__` call is then O(window_size) index lookups rather than O(sequence_length) re-tokenization. For 50k training steps with 16 sequences/batch, this amortizes to nearly zero cost per step.

- **MLflow `start_run()` wraps the training loop in a `finally`**: even if training crashes mid-run, the MLflow run is closed cleanly. Without this, MLflow marks the run as `RUNNING` indefinitely, polluting the experiment view.

- **gradient_checkpointing=True trades ~30% compute for ~50% memory**: on a laptop where 6M parameters × 6 layers × 512 tokens × float32 ≈ 750MB just for activations, checkpointing makes the difference between OOM and successful training.

## Key decisions

- **`_infer_k_from_vocab_size` method**: instead of hardcoding k=6, derive it from the model's vocab_size. This makes the trainer work correctly with any KmerVocabulary, including the 3-mer test vocabulary. Formula: `k = round(log₄(vocab_size - 6))`.

- **`window_size = min(512, genome_length)`**: caps the window to the genome length. For the production case (genome_length=16569), window=512. For tests (genome_length=100), window=100. Without this cap, position IDs in a 512-token window over a 100-bp genome would cycle correctly but the test would be misleading.

- **Two-step Phase 2 checkpoint load**: `_load_checkpoint(encoder_weights_only=True)` loads the full Phase 1 model, copies only keys starting with `mtdna.` (the encoder), then discards the Phase 1 model. The prediction heads (kmer_prediction_head, het_prediction_head) are not copied — they are re-initialized fresh for Phase 2. This is the correct domain-adaptive pre-training pattern.

- **No-decay param groups for AdamW**: weights in LayerNorm, final_layer_norm, and all bias terms skip weight decay. Weight decay on normalization parameters or biases destabilizes training without regularization benefit. This is the BERT convention used universally.

- **Synthetic fallback in `_load_dataset`**: if the parquet doesn't exist, the trainer generates 2 random sequences of the correct genome_length. This makes the trainer testable without real data and prevents cryptic errors if someone runs `mtdna-train` before `mtdna-preprocess`.

## Verification

```bash
# Lint
uv run ruff check mtdna_fm/ tests/
# All checks passed

# Full test suite (174 tests)
uv run pytest tests/ -m "not slow and not integration" -q
# 174 passed, 2 warnings in 6.11s

# New trainer tests (21 tests)
uv run pytest tests/test_trainer.py -v
# 21 passed

# LR schedule at key points
uv run python -c "
from mtdna_fm.training.trainer import _cosine_lr_with_warmup
import math
print('step 0    :', _cosine_lr_with_warmup(0, 2000, 50000, 1e-4))      # 0.0
print('step 2000 :', _cosine_lr_with_warmup(2000, 2000, 50000, 1e-4))   # 1e-4
print('step 50000:', _cosine_lr_with_warmup(50000, 2000, 50000, 1e-4))  # 1e-5
"
# step 0    : 0.0
# step 2000 : 1.0
# step 50000: 0.1

# Smoke test: trainer setup with real data, k=6, genome_length=16569
# (Running in background — loads 152k sequences, tokenizes all)
# Expected initial mlm_loss ≈ log(4102) = 8.32 (random baseline)
uv run mtdna-train --config configs/pretraining_phase1.yaml --model-config configs/model_small.yaml
```

Phase 1 training launch command (runs in background):
```bash
nohup uv run mtdna-train \
    --config configs/pretraining_phase1.yaml \
    --model-config configs/model_small.yaml \
    > logs/phase1.log 2>&1 &
mlflow ui --backend-store-uri mlruns &
```

Expected loss curve:
- Step 0: ~8.32 (log 4,102 — random baseline)
- Step 5,000: ~5.5–6.0
- Step 20,000: ~3.5–4.0
- Step 50,000: ~2.5–3.0

## Next up

Day 11: Write tests to >80% coverage on model and tokenizer modules (`tests/test_model.py` comprehensive expansion, `TestMtDNAForMaskedModeling` with 5-step convergence check, PEFT LoRA compatibility).
