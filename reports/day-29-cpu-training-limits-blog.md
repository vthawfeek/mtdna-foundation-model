# Training a Foundation Model on a Laptop: What CPU Limits Actually Look Like

Most deep learning blog posts assume a GPU. This one doesn't. I trained the mtDNA foundation model fine-tuning runs on a laptop CPU, and I want to be specific about what that means: wall-clock times, memory usage, the deadlock I hit, and what I'd do differently.

---

## The numbers

Training dataset: 82,355 windows (1,267 genomes after sliding-window tokenization with window_size=512, stride=256).

Model: 6-layer BERT encoder, 256 hidden dimensions, ~6M total parameters. LoRA r=8 fine-tuning: 98,304 trainable parameters.

| Quantity | Value |
|----------|-------|
| Batches per epoch (batch_size=128) | 644 |
| Forward+backward time per batch (CPU) | ~30 seconds |
| Time per epoch | ~6.5 hours |
| Two-epoch run | ~20 hours |
| Peak RAM (training process) | ~4.4 GB |

After 2 epochs: training loss = 3.266. Random baseline = ln(26) = 3.258. The loss moved by 0.008.

LoRA fine-tuning typically converges in 10–50 epochs. We ran 2. The math here is straightforward: the model needed ~100–200 hours of CPU training to converge. I ran it for 20.

---

## The DataLoader deadlock

The most expensive mistake was adding `num_workers=4` to the PyTorch DataLoader.

The intent was to parallelise data loading — while the model is computing the forward pass on batch N, workers prepare batch N+1. This is a standard optimization that works well on GPU training where the GPU is the bottleneck.

On CPU-only training, it doesn't help for two reasons: (1) data loading is not the bottleneck (transformer arithmetic is), and (2) on Linux, PyTorch's DataLoader uses `fork()` to spawn worker processes by default.

The problem with `fork()` and PyTorch: the forked child processes inherit a full copy of the parent's memory, including any thread mutexes held by PyTorch's internal thread pool. Those threads don't exist in the child process, so the mutexes are inherited in a locked state. The workers deadlock trying to acquire them. The main process blocks indefinitely waiting for a batch that never arrives.

**What this looked like in practice:**

The training log showed normal startup:
```
INFO Device: cpu
INFO Building dataset from data/processed/train_haplogroup.parquet
INFO Training windows: 82355
INFO Class weights computed: min=0.975 max=1.680
```

Then silence. For 13.5 hours.

Process status: 5 Python processes (main + 4 workers), all in sleeping state at 0% CPU. Total RAM: ~20 GB (3.9 GB per worker process holding a full dataset+model copy, plus 4.4 GB for the main process).

The workers were alive — they hadn't crashed. They were sleeping, waiting for a lock they could never acquire.

**Fix:** `num_workers=0`. One line change. Data loading runs in the main process thread, no forking, no deadlock.

```python
# Before
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)

# After
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
```

**Why this doesn't hurt performance:** The data loading step (fetching a pre-tokenized window from the in-memory list, assembling a batch tensor) takes approximately 0.2 seconds per batch. The forward+backward pass takes ~30 seconds. Overlapping data loading with compute would save ~0.7% of wall clock time in exchange for risking a fork deadlock. Not worth it.

If you're using `num_workers > 0` on Linux with PyTorch, use `multiprocessing_context='spawn'` to avoid the fork deadlock. The `spawn` context starts fresh Python processes instead of forking, which avoids inheriting locked state.

---

## Memory usage breakdown

With `num_workers=0`:

| Component | RAM |
|-----------|-----|
| Model weights (float32, ~6M params) | ~24 MB |
| LoRA adapter parameters | <1 MB |
| Optimizer state (Adam: 2 copies of gradients) | ~200 MB |
| Training dataset in memory (82,355 windows) | ~800 MB |
| Activations (forward pass, batch_size=128) | ~1.5 GB |
| PyTorch internal buffers + Python overhead | ~2 GB |
| **Total** | **~4.5 GB** |

With `num_workers=4` (the deadlocked version), each worker spawned a full copy: 4 × 3.9 GB + 4.4 GB = 20 GB for a process that was doing nothing.

---

## What GPU changes

Same 6-layer, 256-dim model fine-tuned on the same 82,355 windows:

| Scenario | Time per epoch | 50-epoch run |
|----------|---------------|-------------|
| CPU (laptop) | ~6.5 hours | ~325 hours |
| V100 (estimated) | ~3 minutes | ~2.5 hours |
| A100 (estimated) | ~1 minute | ~50 minutes |

The 50-epoch run that the LoRA adaptation needs to converge: feasible in a lunch break on GPU, 2+ weeks of continuous CPU time.

The pre-trained base model is 6M parameters — small by foundation model standards — but the bottleneck is pure matrix multiplication throughput. GPUs win by 100–200× on this workload.

---

## What you can do on CPU

If you're stuck on CPU (laptop project, no cloud budget), here's what actually helps:

**Pre-tokenize to disk.** Every training restart rebuilds the 82,355-window dataset from scratch: read the parquet, tokenize each genome, slide the windows. This takes ~57 seconds. Cache the windows to a `.pt` file and load directly on restart. Not a huge saving per run, but matters when you're debugging and restarting frequently.

**Gradient checkpointing.** PyTorch's `torch.utils.checkpoint` recomputes activations during the backward pass instead of storing them. Halves peak activation memory at the cost of one extra forward pass (~33% slower backward, but the memory saving lets you run a larger batch size, which often offsets the time).

**Gradient accumulation.** Running batch_size=128 on CPU requires storing the full batch in memory. If you're memory-constrained, use batch_size=32 with gradient_accumulation_steps=4 — identical gradient update, quarter of the batch memory.

**Profile first.** I measured: data loading = 0.2s/batch, forward pass = 18s/batch, backward pass = 12s/batch. The data loading is 0.6% of wall clock. Don't spend time optimising the fast part.

**Accept the constraint.** The biggest productivity gain was acknowledging that the fine-tuning would not converge at the CPU budget available and choosing the zero-shot evaluation as the honest benchmark instead.

---

## The takeaway

CPU-only training is not a viable path to fine-tuning convergence for a transformer with 6M parameters on an 82k-sample dataset. The wall-clock math doesn't work.

That said, it was completely viable for:
- Pre-training on masked language modelling (losses converge much faster)
- Prototyping the architecture and training loop
- Validating that the data pipeline, tokenizer, and model interfaces all work
- Running zero-shot evaluation (no gradient computation needed)

The zero-shot k-NN result (~50% accuracy with no fine-tuning labels) is real. It doesn't require gradient convergence — just a forward pass over the test set. That's feasible on CPU and gives an honest signal about what the pre-training learned.

The fine-tuned accuracy (1.83% after 2 epochs, below random) is also real. It's a statement about what's achievable at this compute budget, not about the model's potential.
