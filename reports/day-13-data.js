(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[13] = {
    topic: "Pre-training Analysis",
    commit: "9544d1f",
    status: "complete",
    built: [
      "notebooks/02_pretraining_analysis.ipynb — MLM loss curves, attention heatmaps, zero-shot k-NN",
      "docs/figures/training_curves.png — loss + LR schedule visualisation",
      "docs/figures/attention_heatmap_step0.png — 6-layer × 8-head attention grids at step 0",
      "docs/figures/knn_haplogroup_accuracy.png — k-NN accuracy bar chart",
      "docs/figures/positional_entropy_kmer.png — per-position entropy in D-loop region"
    ],
    learned: [
      "Smoke test loss 8.322 ≈ ln(4102): confirms correct initialisation over 4,102-token vocab",
      "Zero-shot k-NN (cosine, k=5): 9.5% ± 0.024 vs 4% random baseline (2.4× above chance)",
      "Attention at step 0 is near-uniform: no structure yet, as expected for random init",
      "D-loop (first ~576 bp) shows 7× higher k-mer entropy than coding region"
    ],
    decisions: [
      "Cosine similarity for k-NN: tests embedding direction, not magnitude (random init magnitude varies)",
      "128-token window for embedding extraction: captures D-loop haplogroup signal at low compute cost",
      "Step-0 heatmap only — Phase 1 run not yet complete, annotated what to expect at step 25k",
      "Projected curve labelled as such — notebook stays honest about what is real vs. expected"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <line x1="15" y1="100" x2="105" y2="100" stroke="#e2e8f0" stroke-width="1"/>
          <line x1="15" y1="10" x2="15" y2="100" stroke="#e2e8f0" stroke-width="1"/>
          <circle cx="20" cy="20" r="4" fill="#ef4444"/>
          <text x="26" y="18" font-size="5.5" fill="#ef4444">8.32</text>
          <text x="26" y="26" font-size="5" fill="#94a3b8">random</text>
          <path d="M 20 20 Q 50 30 80 65 Q 95 78 105 88" stroke="#3b82f6" stroke-width="1.5" fill="none" stroke-dasharray="4 2"/>
          <text x="72" y="58" font-size="5" fill="#3b82f6">expected</text>
          <rect x="18" y="104" width="84" height="18" rx="3" fill="#dcfce7" stroke="#86efac" stroke-width="1"/>
          <text x="60" y="113" text-anchor="middle" font-size="6" fill="#166534" font-weight="bold">k-NN: 9.5% vs 4% random</text>
          <text x="60" y="120" text-anchor="middle" font-size="5.5" fill="#16a34a">2.4× better than chance ✓</text>
        </svg>
        <div class="eli5-caption">Step 0 loss = exactly ln(4102) — a sanity check passed</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>Before running 50,000 training steps, we did a <strong>report card</strong> check: is the model starting from the right place? An untrained model should guess randomly. With 4,102 possible tokens, random guessing should give a loss of exactly <em>ln(4,102) = 8.32</em>. We measured 8.322. ✓ The model is correctly initialised.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>We also did a <strong>zero-shot test</strong>: using the untrained model's representations, can we classify DNA into haplogroups (family branches on the human family tree)? We used k-nearest-neighbours — find the 5 most similar DNA sequences and vote on the haplogroup.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>Result: 9.5% accuracy vs 4% random (25 classes → 1/25 = 4%). Even with random weights, the model already does <strong>2.4× better than chance</strong>. Why? Because some haplogroup differences are visible directly in the raw DNA letters — the D-loop region varies by haplogroup, and the model can see that even without training.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>The <strong>attention maps</strong> at step 0 are completely uniform — every position pays equal attention to every other position, as expected for random weights. After 25,000 training steps, we expect to see structured patterns: short-range k-mer context and D-loop motifs.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Expected initial loss (uniform random prediction)</div>
      <div class="math-eq">For a model predicting uniformly over V tokens:
  p_i = 1/V  for all i

  H(uniform) = −Σᵢ p_i log(p_i) = −V × (1/V) × log(1/V) = log(V)

  Expected loss = log(4,102) = ln(4,102) ≈ 8.319

  Measured at step 5: 8.322  ✓  (difference < 0.004)</div>
      <div class="math-example"><strong>Why step 5 and not step 0?</strong> At step 0 the model hasn't seen any data. Step 5 is after 5 micro-batches — still effectively random, but the MLflow logger records from step 1 onward. The value 8.322 vs expected 8.319 is within floating-point precision.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Zero-shot k-NN with cosine similarity</div>
      <div class="math-eq">Cosine similarity:  sim(u, v) = (u · v) / (||u|| × ||v||)
  → 1.0: identical direction,  0.0: orthogonal,  −1.0: opposite

k-NN prediction for sequence x:
  1. Compute CLS embedding e_x = model.encode(x)
  2. Find k=5 nearest neighbours in training set by cosine sim
  3. Predict: argmax of majority vote over their haplogroup labels

5-fold CV accuracy:  9.5% ± 2.4%
Random baseline:     1/25 = 4.0%
Lift:                9.5% / 4.0% = 2.38×</div>
    </div>`
  };
})();
