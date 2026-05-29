(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[10] = {
    topic: "Pre-training Launch (Phase 1)",
    commit: "f674e57",
    status: "complete",
    built: [
      "mtdna_fm/training/trainer.py — MtDNATrainer: cosine LR, gradient accum, MLflow, checkpoint rotation",
      "configs/pretraining_phase1.yaml — all-species, 50k steps, het_weight=0",
      "configs/pretraining_phase2.yaml — human-only, 25k steps, het_weight=0.3, lower LR",
      "mtdna_fm/scripts/train.py — actual trainer invocation (was a stub)",
      "tests/test_trainer.py — 21 new tests (174 total)"
    ],
    learned: [
      "Gradient accumulation: loss / grad_accum each micro-step, optimizer.step() every N steps",
      "Cosine LR warmup: linear 0→lr over 2k steps, then cosine lr→0.1×lr over remaining 50k",
      "Phase 2 must not resume optimizer: Phase 1 moment estimates corrupt Phase 2 gradient landscape",
      "_infer_k_from_vocab_size: k = log4(vocab_size - 6) — works for any KmerVocabulary",
      "gradient_checkpointing=True trades ~30% compute for ~50% memory — enables laptop training"
    ],
    decisions: [
      "_infer_k_from_vocab_size instead of hardcoded k=6 — handles test 3-mer vocabulary",
      "window_size = min(512, genome_length) — caps at genome for tiny test configs",
      "Two-step Phase 2 load: copy only 'mtdna.' keys (encoder), discard prediction heads",
      "No-decay param groups: LayerNorm + biases skip weight decay (BERT convention)"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="140" viewBox="0 0 120 140">
          <line x1="15" y1="100" x2="105" y2="100" stroke="#e2e8f0" stroke-width="1"/>
          <line x1="15" y1="20" x2="15" y2="100" stroke="#e2e8f0" stroke-width="1"/>
          <text x="8" y="24" font-size="5.5" fill="#94a3b8">lr</text>
          <text x="55" y="110" text-anchor="middle" font-size="5.5" fill="#94a3b8">steps (50k)</text>
          <polyline points="15,100 25,35" stroke="#3b82f6" stroke-width="2" fill="none"/>
          <path d="M 25 35 Q 55 34 70 60 Q 85 78 105 90" stroke="#3b82f6" stroke-width="2" fill="none"/>
          <text x="20" y="30" font-size="5.5" fill="#3b82f6">warmup</text>
          <text x="60" y="50" font-size="5.5" fill="#3b82f6">cosine decay</text>
          <line x1="25" y1="20" x2="25" y2="100" stroke="#94a3b8" stroke-width="0.8" stroke-dasharray="2 2"/>
          <text x="25" y="115" text-anchor="middle" font-size="5" fill="#94a3b8">2k</text>
          <text x="60" y="128" text-anchor="middle" font-size="6" fill="#374151">micro-steps → optimizer</text>
          <text x="60" y="138" text-anchor="middle" font-size="6" fill="#64748b">accum=4 → 4× batch</text>
        </svg>
        <div class="eli5-caption">Learning rate warms up then slowly decays over 50k steps</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>We built the <strong>trainer</strong> — the loop that actually teaches the model. It reads windows of DNA, masks them, runs the model, measures the loss (wrongness), and then nudges all 5.8M parameters slightly in the direction of "less wrong." This is called <em>gradient descent</em>.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>The <strong>learning rate</strong> controls how big each nudge is. Too big and the model oscillates; too small and it barely learns. We use a "warmup + cosine decay" schedule: start slow (2,000 steps to ramp up), then gradually decrease over the remaining 50,000 steps, ending at 10% of the peak rate.</div>
        <div class="eli5-step"><span class="eli5-num">3</span><strong>Gradient accumulation</strong> is a memory trick. Our GPU/CPU can only hold 16 windows at once (a "micro-batch"). We run 4 micro-batches and add up their gradients before updating the model — effectively getting the same result as a batch of 64 but using only a batch-of-16's worth of memory.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>Every 1,000 steps, we save a <strong>checkpoint</strong> (a snapshot of all parameters). We keep the 3 most recent ones. If training crashes, we resume from the last checkpoint — no need to start from scratch after 20 hours.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Cosine learning rate schedule</div>
      <div class="math-eq">lr(t) =
  t/T_warm × lr_max                         if t < T_warm  (linear warmup)
  lr_min + ½(lr_max − lr_min)(1 + cos(π×(t−T_warm)/(T_total−T_warm)))
                                             otherwise

lr_max = 1×10⁻⁴,  lr_min = 1×10⁻⁵ (10% of max)
T_warm = 2,000,   T_total = 50,000</div>
      <div class="math-example"><strong>Example at t=25,000 (midpoint):</strong><br>
cos(π × (25000−2000)/(50000−2000)) = cos(π × 23000/48000) = cos(0.477π) ≈ 0.0<br>
lr(25000) = 1×10⁻⁵ + ½×(9×10⁻⁵)×(1+0) = 1×10⁻⁵ + 4.5×10⁻⁵ = 5.5×10⁻⁵</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Gradient accumulation</div>
      <div class="math-eq">effective_batch = micro_batch × n_accumulation_steps
                     = 16 × 4 = 64

Each micro-step: loss = loss / n_accum  (scale to prevent sum bias)
                 loss.backward()         (accumulate grad in .grad)
Every n_accum steps: optimizer.step(); optimizer.zero_grad()</div>
    </div>`
  };
})();
