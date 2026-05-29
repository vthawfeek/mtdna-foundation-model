(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[9] = {
    topic: "Masking and Loss",
    commit: "f9979a5",
    status: "complete",
    built: [
      "mtdna_fm/training/masking.py — MtDNAMaskingCollator: BERT 80/10/10 + D-loop blacklist pos 303-315",
      "mtdna_fm/training/losses.py — mtdna_mlm_loss(): combined MLM CE + het MSE with weight flags",
      "KmerVocabulary — convenience property accessors (mask_token_id, het_token_id, etc.)",
      "tests/test_training.py — 20 tests: 11 masking + 9 loss (153 total)"
    ],
    learned: [
      "BERT 80/10/10: 10% random + 10% unchanged forces correct representation of all positions",
      "Collation-time masking: ~6.7× augmentation over pre-computed masks at 15% mask rate",
      "D-loop blacklist is domain-specific regularisation: C-tract (303-315) is sequencing noise",
      "PyTorch cross_entropy with all -100 labels returns nan — test the contrast, not the edge case",
      "het_labels sentinel -1 (not -100) — unambiguous for float targets bounded in [0, 1]"
    ],
    decisions: [
      "Blacklist as frozenset: O(1) membership, constructed once at collator init",
      "Masking in collator not dataset: DataLoader workers apply fresh masks per batch per epoch",
      "het_weight=0.0 default: Phase 1 has no het data — single loss function for both phases",
      "Vocabulary convenience properties: vocab.mask_token_id over module-level constants"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <text x="8" y="28" font-family="monospace" font-size="8" fill="#22c55e" font-weight="bold">A</text>
          <rect x="22" y="16" width="14" height="16" rx="2" fill="#dbeafe" stroke="#93c5fd" stroke-width="1.5"/>
          <text x="29" y="27" text-anchor="middle" font-size="7" fill="#1e40af" font-weight="bold">?</text>
          <text x="42" y="28" font-family="monospace" font-size="8" fill="#22c55e" font-weight="bold">C</text>
          <text x="56" y="28" font-family="monospace" font-size="8" fill="#3b82f6" font-weight="bold">G</text>
          <rect x="70" y="16" width="14" height="16" rx="2" fill="#dbeafe" stroke="#93c5fd" stroke-width="1.5"/>
          <text x="77" y="27" text-anchor="middle" font-size="7" fill="#1e40af" font-weight="bold">?</text>
          <text x="90" y="28" font-family="monospace" font-size="8" fill="#ef4444" font-weight="bold">T</text>
          <text x="104" y="28" font-family="monospace" font-size="8" fill="#22c55e" font-weight="bold">A</text>
          <text x="60" y="46" text-anchor="middle" font-size="6.5" fill="#64748b">guess the hidden letters!</text>
          <circle cx="60" cy="88" r="28" fill="none" stroke="#e2e8f0" stroke-width="1"/>
          <path d="M 60 60 L 60 88 A 28 28 0 0 1 37.7 74.6 Z" fill="#3b82f6" opacity="0.8"/>
          <path d="M 60 88 A 28 28 0 0 1 37.7 74.6 L 60 88 Z" fill="#f59e0b" opacity="0.8"/>
          <path d="M 60 60 A 28 28 0 0 1 60 88 Z" fill="#22c55e" opacity="0.8"/>
          <text x="60" y="75" text-anchor="middle" font-size="6" fill="#fff" font-weight="bold">80%</text>
          <text x="43" y="84" text-anchor="middle" font-size="5.5" fill="#fff" font-weight="bold">10%</text>
          <text x="72" y="76" text-anchor="middle" font-size="5.5" fill="#fff" font-weight="bold">10%</text>
          <text x="60" y="122" text-anchor="middle" font-size="6" fill="#64748b">[MASK] / random / keep</text>
        </svg>
        <div class="eli5-caption">Cover 15% of tokens; model guesses what was there</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>We train the model by playing a <strong>guessing game</strong>: we take a DNA window, hide 15% of the tokens (replace them with a [MASK] symbol), and ask the model to guess what was hidden. The model learns by getting corrected when it's wrong.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>But we don't hide all 15% the same way. For each selected token: <strong>80%</strong> of the time → replace with [MASK]; <strong>10%</strong> → swap with a random token; <strong>10%</strong> → leave it unchanged. Why the last two? If everything hidden became [MASK], the model could cheat by just saying "whenever I see [MASK], output whatever is common here." Random swaps and unchanged tokens force the model to understand every position.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>Positions 303–315 are <strong>blacklisted</strong> — never masked. These form the "C-tract" in the D-loop, where sequencing machines notoriously make mistakes. Training on garbage teaches garbage — so we skip these positions entirely.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>The <strong>loss</strong> is how wrong the model is on average. We measure two things: how wrong it was about the masked DNA letters (cross-entropy), and optionally how wrong it was about heteroplasmy levels (mean squared error). In Phase 1 we only measure the DNA guessing; in Phase 2 we add the heteroplasmy component.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">BERT 80/10/10 masking probabilities</div>
      <div class="math-eq">For each token at position p, draw u ~ Uniform(0, 1):
  p_mask   = 0.15  (probability of being selected at all)

If selected:
  if   u < 0.80 → replace with [MASK]
  elif u < 0.90 → replace with random token from vocab
  else          → keep original (unchanged)

Expected augmentation: 1 / p_mask = 1/0.15 ≈ 6.7 independent
views of each window across training epochs.</div>
      <div class="math-example"><strong>Why 80/10/10?</strong> At inference, there are no [MASK] tokens. If 100% of masked positions used [MASK] during training, the model might not learn to represent unmasked positions well. The 10% unchanged forces the model to always produce a useful representation, masked or not.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Combined MLM + heteroplasmy loss</div>
      <div class="math-eq">L = w_mlm × L_MLM  +  w_het × L_het

L_MLM = CrossEntropy(logits_i, label_i)  for all masked positions i
      = − (1/|M|) Σᵢ∈M  log( softmax(logits_i)[label_i] )

L_het = MSE(het_pred_j, het_true_j)  for positions j where het_label ≠ −1
      = (1/|H|) Σⱼ∈H  (het_pred_j − het_true_j)²

Phase 1: w_mlm=1.0, w_het=0.0  →  L = L_MLM only
Phase 2: w_mlm=1.0, w_het=0.3  →  L = L_MLM + 0.3 × L_het</div>
      <div class="math-example"><strong>Example:</strong> 12 positions masked. Correct token at position 5 has logit score 2.1 out of [2.1, 1.3, 0.8, …]. softmax(2.1) ≈ 0.42. CE contribution: −log(0.42) ≈ 0.87 nats. Average over 12 positions = L_MLM.</div>
    </div>`
  };
})();
