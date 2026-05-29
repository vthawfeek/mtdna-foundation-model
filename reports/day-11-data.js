(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[11] = {
    topic: "Test Suite Expansion (76% → 80%)",
    commit: "26a48d1",
    status: "complete",
    built: [
      "tests/test_tokenizer.py — unk/sep/het property tests added to TestKmerVocabulary",
      "tests/test_model.py — get/set_input_embeddings, forward_without_attention_mask",
      "tests/test_trainer.py — 6 new: invalid vocab, checkpointing, parquet loading, species filter",
      "tests/test_scripts.py — new file: evaluate and finetune CLI stubs coverage",
      "mtdna_fm/model/model.py — supports_gradient_checkpointing + _set_gradient_checkpointing",
      "mtdna_fm/model/transformer.py — gradient_checkpointing flag + checkpoint branching in encoder"
    ],
    learned: [
      "Coverage arithmetic: model/training core 95-100%; overall 76% dragged by data/CLI scripts",
      "Gradient checkpointing requires opt-in to both flag AND checkpoint wiring in encoder",
      "CLI stubs are worth testing: pins the exit-code contract before implementation exists",
      "Parquet branch needs real data: minimal DataFrame in tmp_path tests actual read logic"
    ],
    decisions: [
      "Implement real gradient checkpointing (not mock) — improves production training path too",
      "New tests/test_scripts.py rather than mixing into test_data.py (CliRunner is distinct)",
      "Species filter test uses real parquet written to tmp_path — tests actual filter logic"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <line x1="10" y1="90" x2="110" y2="90" stroke="#374151" stroke-width="3"/>
          <line x1="10" y1="90" x2="35" y2="50" stroke="#374151" stroke-width="2"/>
          <line x1="110" y1="90" x2="85" y2="50" stroke="#374151" stroke-width="2"/>
          <line x1="35" y1="50" x2="85" y2="50" stroke="#374151" stroke-width="2"/>
          <circle cx="35" cy="50" r="5" fill="#22c55e" stroke="#166534" stroke-width="1.2"/>
          <circle cx="60" cy="50" r="5" fill="#22c55e" stroke="#166534" stroke-width="1.2"/>
          <circle cx="85" cy="50" r="5" fill="#22c55e" stroke="#166534" stroke-width="1.2"/>
          <circle cx="22" cy="70" r="5" fill="#22c55e" stroke="#166534" stroke-width="1.2"/>
          <circle cx="98" cy="70" r="5" fill="#22c55e" stroke="#166534" stroke-width="1.2"/>
          <circle cx="60" cy="90" r="5" fill="#22c55e" stroke="#166534" stroke-width="1.2"/>
          <text x="60" y="108" text-anchor="middle" font-size="6.5" fill="#374151">193 tests ✓</text>
          <text x="60" y="120" text-anchor="middle" font-size="6" fill="#64748b">coverage 76% → 80%</text>
        </svg>
        <div class="eli5-caption">Tests are like bolts — every critical joint checked</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>We expanded the test suite from 153 to 193 tests. Tests are mini-experiments: each one calls a specific piece of code with known inputs and checks the output matches expectations. If someone later changes the code and breaks something, the test immediately fails.</div>
        <div class="eli5-step"><span class="eli5-num">2</span><strong>Coverage</strong> went from 76% to 80%. This means 80% of code lines were executed by at least one test. The remaining 20% are mostly in download and CLI scripts that need network calls — we'll cover those in Day 12.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>We also added real <strong>gradient checkpointing</strong> to the model. Normally, the model saves every intermediate calculation during the forward pass (to reuse during backpropagation). Gradient checkpointing throws them away and recomputes them on demand — trading ~30% more compute for ~50% less memory. Essential for training on a laptop.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>New tests covered things like: what happens if you call the CLI with wrong arguments? What if you pass a CSV instead of a parquet? These <em>boundary tests</em> check that the code fails gracefully rather than silently producing wrong output.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Gradient checkpointing memory trade-off</div>
      <div class="math-eq">Standard backprop: store all n-layer activations during forward
  memory = O(n × batch × seq_len × d_model)

Gradient checkpointing: recompute activations in backward
  memory ≈ O(√n × batch × seq_len × d_model)  (checkpoint every √n layers)
  compute ≈ 1.3× (30% overhead to recompute)</div>
      <div class="math-example"><strong>For our model (n=6 layers, d=256, seq=512):</strong><br>
Activation size per layer ≈ batch × 512 × 256 × 4 bytes ≈ 0.5 MB (batch=1).<br>
Without checkpointing: 6 × 0.5 MB = 3 MB. With: √6 ≈ 2.5 layers cached ≈ 1.25 MB. Meaningful on a laptop with limited VRAM.</div>
    </div>`
  };
})();
