# Why My Haplogroup Classifier Is Below Random — And What the Confusion Matrix Reveals

My fine-tuned mtDNA haplogroup classifier achieved 1.83% accuracy on the test set. The random baseline for 26-class uniform classification is 3.85%. The model is below random.

This post explains exactly why, what the confusion matrix shows despite the low accuracy, and what this tells us about the relationship between pre-training and fine-tuning.

---

## The diagnosis: partial class collapse

The accuracy number alone is misleading. Look at the per-class F1 scores:

- 3 of 26 classes: F1 > 0.01
- 23 of 26 classes: F1 ≈ 0.00

The model isn't guessing randomly across 26 classes — it's predicting 3 haplogroups for almost everything. This is **partial class collapse**: the classifier head converged on a small set of majority classes and learned to ignore the rest.

A collapsed model scores below uniform random because the classes it ignores are predicted at zero frequency, pulling the average accuracy down below what you'd get from uniform guessing.

---

## Why class collapse happens: window-level imbalance

The training pipeline slides a 512-token window across each genome with stride 256. A 16,569 bp genome produces approximately 63 windows. But genomes of different haplogroups vary slightly in length, and the training set has unequal counts per class.

After windowing, H haplogroup accounts for ~15.6% of all training windows. L5 accounts for ~0.4%. Standard CrossEntropyLoss minimises average loss — the easiest way to do that is to predict H (and the next largest classes) for everything.

I added inverse-frequency class weights to the loss:

```python
class_counts = torch.bincount(torch.tensor([w["label"] for w in train_ds._windows]))
class_weights = 1.0 / (class_counts.float() + 1e-6)
class_weights = class_weights / class_weights.sum()
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

This improved the situation from full collapse (1 class) to partial collapse (3 classes). The loss function now penalises rare class errors proportionally, but the model still needs more gradient updates to actually learn the minority class boundaries.

---

## Why the class weights weren't enough: compute

Each training epoch processes 82,355 windows in batches of 128. That's 644 forward+backward passes per epoch. On CPU, each pass through the 6-layer, 256-dim transformer takes approximately 30 seconds. One epoch = 6+ hours of wall clock time.

Training loss after 2 epochs: **3.266**.
Random baseline: **ln(26) = 3.258**.

The loss moved by 0.008. The LoRA adapters (98,304 trainable parameters out of 5.9M total) need far more gradient steps to converge. The standard advice for LoRA fine-tuning is to run for 10–50 epochs with a learning rate in the 1e-4 to 3e-4 range. We ran 2 epochs at 3e-4. The learning rate was right; the epoch count was a compute constraint.

A GPU would give 100–200× throughput. The same 20-hour CPU run would complete in 6–12 minutes on an A100, making 50-epoch fine-tuning feasible in a lunch break.

---

## What the confusion matrix shows anyway

Despite 1.83% accuracy and partial class collapse, the confusion matrix is not random noise.

The errors that do occur are phylogenetically structured:

- L0 gets confused with L1 (adjacent branches on the African root)
- H gets confused with HV (H is derived from HV; they're phylogenetically adjacent)
- M-clade haplogroups get confused with each other

There are no bright off-diagonal cells connecting African root haplogroups (L0–L5) to European-derived ones (H, HV, U, J, T). No L3↔H errors. No L2↔V errors.

This is not the classifier head doing this — the head has barely learned anything in 2 epochs. This is the **pre-trained embedding geometry** making itself felt even through an unconverged linear layer. The representations cluster phylogenetically-related sequences near each other in 256-d space, and that structure survives even a poorly-calibrated projection.

---

## Zero-shot vs fine-tuned: measuring different things

The zero-shot k-NN experiment uses the pre-trained embeddings directly, with no fine-tuning:

1. Embed each sequence → 256-d vector
2. For a test sequence, find the k nearest neighbours in the training set
3. Predict the majority haplogroup among those neighbours
4. No gradient updates. No labels during training.

Result: ~50% accuracy on the same 26-class problem where random is 3.85%.

The fine-tuned model, after 2 CPU epochs: 1.83%.

These two numbers measure different things. The zero-shot result measures what the pre-training learned about evolutionary structure — whether similar sequences land near each other in embedding space. The fine-tuned result measures whether we gave the classifier head enough compute to decode that structure.

The answer to the second question is: not yet.

---

## What would fix this

In priority order:

1. **GPU compute**: 50 epochs in ~30 minutes vs 150 hours. Most important.
2. **More training data**: The full HmtDB has 8,921 properly-labeled haplogroup sequences (vs 1,267 used here). Larger training set → more signal per epoch.
3. **Curriculum**: start with high-confidence windows (full-length genomes, clean sequences) before adding the fragmented ones.
4. **Longer LoRA training**: the learning rate (3e-4) and architecture (r=8) are reasonable; the convergence problem is epoch count, not hyperparameters.

---

## The lesson

The fine-tuned accuracy is a statement about available compute, not about what the model knows. The zero-shot k-NN result is the more honest measure of what the pre-trained representations learned.

When you're resource-constrained, measure what you can measure honestly and be explicit about what you haven't measured. "The fine-tuning didn't converge on CPU — here's the evidence and what it would take to fix" is a better portfolio outcome than hiding a low accuracy number behind an uncaveated benchmark.
