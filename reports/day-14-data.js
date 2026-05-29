(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[14] = {
    topic: "Phase 2 Launch",
    commit: "5733c78",
    status: "complete",
    built: [
      "models/phase1_v1/ — Phase 1 checkpoint: config.json, model.safetensors, vocab.json (5.8M params)",
      "configs/pretraining_phase2.yaml — resume_from: models/phase1_v1, het_weight=0.3, lr=3e-5, 25k steps",
      "tokenizer_config.json — HF-compatible tokenizer metadata added to checkpoint directory",
      "Phase 2 training launched on human HmtDB sequences with encoder_weights_only=True"
    ],
    learned: [
      "Selective weight transfer: only 'mtdna.*' encoder keys loaded; prediction heads freshly reinitialised",
      "Fresh optimizer state mandatory: Phase 1 Adam moments calibrated to LR=1e-4 would corrupt Phase 2 (LR=3e-5)",
      "Zero-shot k-NN on Phase 1 checkpoint: 16.0% ± 4.9% vs 10% random (10-class synthetic task)",
      "Species filtering post-load: all 34,974 sequences loaded, then filtered to homo_sapiens"
    ],
    decisions: [
      "Save as MtDNAForMaskedModeling, load selectively: avoids needing a separate encoder-only format",
      "tokenizer_config.json added manually: KmerVocabulary.save_pretrained() doesn't write it automatically",
      "Synthetic fallback for Phase 2 smoke test: real parquet filtered to homo_sapiens not yet available",
      "Phase 2: 25k steps (vs 50k Phase 1) — starting from checkpoint, needs less time to converge"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="140" viewBox="0 0 120 140">
          <rect x="10" y="8" width="100" height="30" rx="5" fill="#dcfce7" stroke="#86efac" stroke-width="1.5"/>
          <text x="60" y="22" text-anchor="middle" font-size="7.5" fill="#166534" font-weight="bold">Phase 1 checkpoint</text>
          <text x="60" y="32" text-anchor="middle" font-size="6" fill="#16a34a">all species, 50k steps</text>
          <line x1="60" y1="38" x2="60" y2="55" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr3)"/>
          <defs><marker id="arr3" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
          <text x="75" y="49" font-size="6" fill="#6b21a8">encoder only ↓</text>
          <rect x="10" y="56" width="100" height="30" rx="5" fill="#dbeafe" stroke="#93c5fd" stroke-width="1.5"/>
          <text x="60" y="70" text-anchor="middle" font-size="7.5" fill="#1e40af" font-weight="bold">Phase 2 training</text>
          <text x="60" y="80" text-anchor="middle" font-size="6" fill="#3b82f6">human-only, 25k steps + het</text>
          <rect x="15" y="98" width="90" height="20" rx="3" fill="#fffbeb" stroke="#fcd34d" stroke-width="1"/>
          <text x="60" y="108" text-anchor="middle" font-size="6" fill="#92400e" font-weight="bold">Phase 1 k-NN: 16% vs 10% ✓</text>
          <text x="60" y="116" text-anchor="middle" font-size="5.5" fill="#b45309">checkpoint verified</text>
          <text x="60" y="132" text-anchor="middle" font-size="6" fill="#64748b">LR: 1×10⁻⁴ → 3×10⁻⁵</text>
        </svg>
        <div class="eli5-caption">Phase 2 loads the Phase 1 brain and specialises it</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>Phase 1 taught the model about DNA from <em>all vertebrates</em> — fish, mammals, birds. It learned the grammar of mitochondrial DNA in general. Phase 2 is <strong>specialist school</strong>: we now train exclusively on human sequences, and we add heteroplasmy supervision (teaching it about mixed DNA copies).</div>
        <div class="eli5-step"><span class="eli5-num">2</span>We don't start from scratch — we <strong>transfer the encoder</strong> from Phase 1. Think of it as a doctor who studied general medicine for 4 years (Phase 1) and now does a 2-year specialisation in cardiology (Phase 2). The general knowledge transfers; only the specialisation is new.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>We <strong>reset the optimizer</strong> completely. The Adam optimizer stores a "memory" of past gradients. Phase 1's memory was calibrated to a learning rate of 0.0001. Phase 2 uses 0.00003 (3×). Using the old memory would cause the wrong-sized updates — like trying to use cruise-control settings from a highway for city driving.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>Before starting Phase 2, we did a <strong>sanity check</strong> on the Phase 1 checkpoint: the model correctly achieves 16% k-NN accuracy on a synthetic 10-class task vs 10% random. The checkpoint is non-trivial and worth building on.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Selective weight transfer (encoder only)</div>
      <div class="math-eq">Phase 1 checkpoint keys:
  mtdna.embeddings.*        ← copy to Phase 2 ✓
  mtdna.encoder.layer.*     ← copy to Phase 2 ✓
  kmer_prediction_head.*    ← discard ✗ (reinitialise)
  het_prediction_head.*     ← discard ✗ (reinitialise)

Code: loaded = {k: v for k, v in ckpt.items() if k.startswith("mtdna.")}
      model.load_state_dict(loaded, strict=False)</div>
      <div class="math-example"><strong>Why discard the heads?</strong> The MLM prediction head learned to map hidden states → 4,102-token logits for Phase 1's objective. Phase 2 has a different task balance (+ het regression). Reinitialising lets the heads adapt to the Phase 2 objective without Phase 1 bias.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Phase 2 combined loss</div>
      <div class="math-eq">L = 1.0 × L_MLM  +  0.3 × L_het

L_MLM = CrossEntropy(masked token predictions)   [same as Phase 1]
L_het = MSE(het_pred, het_true)  for positions with het labels ≠ −1
      = (1/N) Σ (het̂ᵢ − hetᵢ)²

w_het = 0.3: heteroplasmy term contributes ~23% of total loss</div>
    </div>`
  };
})();
