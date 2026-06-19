# Pre-training

This document covers the two-phase curriculum rationale, expected MLM loss dynamics, how to run pre-training on a laptop, and how to monitor with MLflow.

---

## Two-Phase Curriculum

### Why two phases?

Phase 1 trains on 117,615 cross-species vertebrate mtDNA sequences. Phase 2 loads those weights and continues on 34,975 human HmtDB sequences with the heteroplasmy loss enabled.

The rationale is domain-adaptive pre-training: start with broad diversity (many species, many evolutionary distances), then specialize on the target domain (human variation, haplogroup structure, heteroplasmy).

Phase 1 teaches the model:
- Conserved structural features that appear across vertebrate mtDNA (protein-coding gene positions, rRNA structures, tRNA anticodons)
- The low-level statistical regularities of the 6-mer vocabulary
- The basic geometry of the circular genome

Without Phase 1, the model would learn human-specific variation first and might overfit to the haplogroup structure of the HmtDB cohort before learning transferable molecular features.

Phase 2 teaches the model:
- Human-specific sequence variation and haplogroup signatures
- Heteroplasmy as a meaningful signal (via the `het_weight` parameter)
- The D-loop hypervariable regions that distinguish haplogroups at fine resolution

### Phase 2 loads encoder weights only

When Phase 2 starts, only the encoder weights are loaded from Phase 1. The optimizer state (Adam moment estimates, learning rate schedule, step count) is discarded. This is intentional:

- Phase 2 uses a lower learning rate (3e-5 vs 1e-4) because the model is already well-initialized.
- The optimizer state from Phase 1 would encode the gradient history from cross-species training, which is the wrong prior for human-specific fine-tuning.
- A fresh warmup period (500 steps) prevents the model from immediately jumping to a high learning rate and disturbing the Phase 1 representations.

In `MtDNATrainer._load_checkpoint`, passing `load_optimizer=False` implements this behavior.

---

## Running Pre-training

### Phase 1

```bash
uv run mtdna-train \
  --config configs/pretraining_phase1.yaml \
  --model-config configs/model_small.yaml
```

Key parameters in `configs/pretraining_phase1.yaml`:

```yaml
batch_size: 16
gradient_accumulation_steps: 8     # effective batch = 128
learning_rate: 1.0e-4
warmup_steps: 2000
max_steps: 50000
fp16: false                         # CPU-safe; set true for CUDA
gradient_checkpointing: true
mask_prob: 0.15
mlm_weight: 1.0
het_weight: 0.0                     # no het data in cross-species corpus
output_dir: "models/phase1_v1"
mlflow_experiment: "mtdna_fm_pretraining_phase1"
```

### Phase 2

```bash
uv run mtdna-train \
  --config configs/pretraining_phase2.yaml \
  --model-config configs/model_small.yaml
```

Key parameters in `configs/pretraining_phase2.yaml`:

```yaml
resume_from: "models/phase1_v1"     # encoder weights only
data:
  species_filter: "homo_sapiens"
learning_rate: 3.0e-5
max_steps: 25000
warmup_steps: 500
het_weight: 0.3
output_dir: "models/phase2_v1"
```

---

## Gradient Accumulation

With `batch_size=16` and `gradient_accumulation_steps=8`, the effective batch size is 128. This simulates training with 128 sequences per update without requiring 128 sequences in GPU/CPU memory simultaneously.

Each actual step:
1. Load 16 sequences (1 batch)
2. Forward pass, compute loss
3. `loss / accumulation_steps` — scale gradient by 1/8
4. `.backward()` — accumulate gradients
5. Repeat 7 more times
6. `optimizer.step()` — one weight update with the accumulated gradient

**Why does this matter?** Adam's variance estimate (the `v` term in `m / (sqrt(v) + ε)`) is more accurate with larger effective batch sizes. With batch=16, the gradient estimate is noisy and the loss curve is jagged. With effective batch=128, convergence is smoother and the final loss is ~0.2 lower.

---

## Expected MLM Loss Dynamics

The random baseline (before any training) is approximately:

```
loss ≈ log(vocab_size) = log(4102) ≈ 8.32
```

This is the cross-entropy when the model predicts a uniform distribution over all 4,102 tokens. As training progresses, the model learns to exploit structural regularities in the sequence.

| Step | Expected loss | What the model is learning |
|---|---|---|
| 0 | ~8.3 | Random baseline |
| 1k | ~7.0-7.5 | Base composition biases (GC content) |
| 5k | ~5.5-6.0 | Common k-mer patterns, basic co-occurrences |
| 10k | ~4.5-5.0 | Local sequence structure (codons, tRNA stems) |
| 25k | ~3.5-4.0 | Domain-specific features, protein-coding patterns |
| 50k | ~2.5-3.0 | Convergence, diminishing returns |

Phase 2 starts from ~2.8 (Phase 1 endpoint) and warms up slightly before decreasing further to ~2.3-2.5 with `het_weight=0.3` contributing additional loss signal.

These numbers are approximate — actual values depend on whether you're training on CPU or GPU, the specific data split, and how many sequences are in your corpus.

---

## Masking Strategy

The `MtDNAMaskingCollator` applies BERT-style masking:

- 15% of k-mer tokens are selected for masking
- Of those, 80% are replaced with `[MASK]` token
- Of those, 10% are replaced with a random k-mer token
- Of those, 10% are left unchanged

The 80/10/10 split prevents the model from learning that `[MASK]` always means "predict something here" — the unchanged 10% forces it to build robust contextual representations even when the input token is real.

### D-loop blacklist

Positions 303-315 (the homopolymeric C-tract in the D-loop) are blacklisted from masking. This region is sequencing noise: it consists of a run of consecutive C bases that is almost impossible to sequence accurately, and the base calls here reflect PCR and sequencing artifacts rather than biological variation. Teaching the model to predict this region would introduce noise into the learned representations.

---

## Gradient Checkpointing

With `gradient_checkpointing: true`, intermediate activations are discarded during the forward pass and recomputed during the backward pass. This approximately halves memory usage at the cost of ~30% slower training.

For a 6M-parameter model on CPU, memory is rarely the bottleneck. However, gradient checkpointing is enabled by default because:

1. It makes the configuration reusable on machines with limited RAM
2. The 30% compute cost is acceptable for a 50k-step training run
3. It becomes important if you scale to deeper models (12+ layers)

To disable: set `gradient_checkpointing: false` in the config YAML.

---

## Monitoring with MLflow

```bash
# Start the MLflow UI (runs at http://localhost:5000 by default)
mlflow ui --backend-store-uri mlruns &

# Or point to a remote tracking server
export MLFLOW_TRACKING_URI=http://your-server:5000
```

The trainer logs to MLflow every `log_steps` (default: 100):
- `train/mlm_loss`: MLM cross-entropy on masked positions
- `train/het_loss`: Heteroplasmy MSE (Phase 2 only, when `het_weight > 0`)
- `train/total_loss`: Weighted sum of the above
- `train/learning_rate`: Current LR after warmup/decay
- `train/grad_norm`: Gradient norm (watch for spikes > 5.0)

Every `eval_steps` (default: 2500), the trainer evaluates on the validation set and logs:
- `eval/mlm_loss`: Validation MLM loss
- `eval/perplexity`: `exp(mlm_loss)` — a more interpretable metric

Checkpoints are saved every `save_steps` (default: 5000). Only the last `keep_last_n_checkpoints` (default: 3) are retained to save disk space.

### What to watch for

**Loss not decreasing after 2k steps:** Learning rate is too high or data loading is broken (all-zero inputs, for example). Check `train/grad_norm` — if it is consistently > 10, reduce `learning_rate` by 3×.

**Loss decreasing then jumping:** A bad batch (e.g., a sequence with high N-content that produces many `[UNK]` tokens) caused an unstable gradient. `max_grad_norm: 1.0` clips this but cannot eliminate all instability. This resolves on its own.

**Validation loss diverging from training loss after step 10k:** Mild overfitting. The cross-species corpus is large enough that this rarely happens in Phase 1. In Phase 2, if the HmtDB corpus is small, increase `warmup_steps` or add dropout.

---

## Timing Estimates

On a modern CPU (e.g., Intel Core i7, 16 GB RAM), with `batch_size=16`, `gradient_accumulation=8`, `gradient_checkpointing=True`:

| Phase | Steps | Estimated time |
|---|---|---|
| Phase 1 | 50,000 | 10-14 hours |
| Phase 2 | 25,000 | 5-7 hours |

On a GPU (e.g., NVIDIA RTX 3080):

| Phase | Steps | Estimated time |
|---|---|---|
| Phase 1 | 50,000 | 1.5-2 hours |
| Phase 2 | 25,000 | 45-60 minutes |

For GPU training, set `fp16: true` in the config. This halves memory and typically speeds up training by 1.5-2×.

If 50k steps is too slow on CPU, reducing to 25k steps gives a usable model. The loss curve flattens noticeably after 25k steps; the marginal improvement from 25k to 50k is smaller than from 10k to 25k.
