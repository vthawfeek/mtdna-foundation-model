(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[8] = {
    topic: "Model Architecture",
    commit: "618ebd1",
    status: "complete",
    built: [
      "configs/model_small.yaml — 6 layers, 8 heads, 256 hidden, ~5.8M parameters",
      "mtdna_fm/model/config.py — MtDNAConfig(PretrainedConfig) with genome_length + use_circular_encoding",
      "mtdna_fm/model/embeddings.py — MtDNACircularPositionalEncoding (fixed buffer) + MtDNAEmbeddings",
      "mtdna_fm/model/transformer.py — pre-LN bidirectional transformer ported from scFM",
      "mtdna_fm/model/model.py — MtDNAModel base encoder + MtDNAForMaskedModeling",
      "tests/test_model.py — 38 new tests across 5 classes (133 total)"
    ],
    learned: [
      "Circular PE as fixed buffer: mtDNA topology is biological fact, not a learned pattern",
      "BertModel / BertForMaskedLM separation: prediction heads discarded at fine-tuning time",
      "Het projection as continuous Linear+LayerNorm: avoids arbitrary discretization thresholds",
      "Named linear layers (query, key, value, dense) required for PEFT LoRA target matching",
      "Pre-LN transformer more stable than post-LN — avoids gradient vanishing in 6-layer net"
    ],
    decisions: [
      "Fixed circular PE buffer (non-learnable) — biological constraint, not a parameter",
      "het_weight defaults 0.0 — same model class handles Phase 1 (no het) and Phase 2",
      "3-layer MLP kmer_prediction_head: intermediate GELU avoids bottleneck for 4,102-class output",
      "5,790,720 parameters — trainable on CPU in 8-12 hours at 50k steps"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="140" viewBox="0 0 120 140">
          <circle cx="60" cy="28" r="18" fill="none" stroke="#3b82f6" stroke-width="1.5"/>
          <circle cx="60" cy="10" r="3.5" fill="#3b82f6"/><text x="60" y="11.5" text-anchor="middle" font-size="4" fill="#fff" font-weight="bold">0</text>
          <circle cx="78" cy="28" r="3.5" fill="#3b82f6"/><text x="78" y="29.5" text-anchor="middle" font-size="4" fill="#fff" font-weight="bold">5k</text>
          <circle cx="60" cy="46" r="3.5" fill="#3b82f6"/><text x="60" y="47.5" text-anchor="middle" font-size="4" fill="#fff" font-weight="bold">11k</text>
          <circle cx="42" cy="28" r="3.5" fill="#3b82f6"/><text x="42" y="29.5" text-anchor="middle" font-size="4" fill="#fff" font-weight="bold">16k</text>
          <text x="60" y="30" text-anchor="middle" font-size="5.5" fill="#1e40af" font-weight="bold">circular</text>
          <text x="60" y="38" text-anchor="middle" font-size="5" fill="#64748b">PE</text>
          <rect x="20" y="58" width="80" height="10" rx="2" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
          <rect x="20" y="71" width="80" height="10" rx="2" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
          <rect x="20" y="84" width="80" height="10" rx="2" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
          <text x="60" y="66.5" text-anchor="middle" font-size="6" fill="#1e40af">Layer 1: 8 attention heads</text>
          <text x="60" y="79.5" text-anchor="middle" font-size="6" fill="#1e40af">Layer 2: 8 attention heads</text>
          <text x="60" y="92.5" text-anchor="middle" font-size="6" fill="#1e40af">… 6 layers total</text>
          <text x="60" y="110" text-anchor="middle" font-size="6.5" fill="#374151">6 × 8 heads × 256 dim</text>
          <text x="60" y="122" text-anchor="middle" font-size="6.5" fill="#94a3b8">= ~5.8M parameters</text>
        </svg>
        <div class="eli5-caption">6 stacked layers, each with 8 "students" reading DNA</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>The AI brain is a <strong>transformer</strong> — imagine 6 classrooms stacked on top of each other. In each classroom, 8 students (<em>attention heads</em>) each read the DNA sequence and discuss: "which other positions matter most for understanding position X right now?"</div>
        <div class="eli5-step"><span class="eli5-num">2</span>Before reading, every position gets a <strong>position tag</strong> (positional encoding) so students know where they are. Standard AI uses straight-line positions (0, 1, 2, …). Ours uses a <em>circle</em> — positions 0 and 16,568 get nearly identical tags because DNA is circular and they're adjacent.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>Each position also has an optional <strong>heteroplasmy channel</strong> — a single number [0.0–1.0] saying "in this person's cells, what fraction of DNA copies have this variant?" This gets projected into the same 256-dimensional space as the letter embedding.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>The full model has <strong>5,790,720 numbers</strong> (parameters) that are adjusted during training. That's about the same as a large image-recognition model from 2015 — modern but compact enough to train on a laptop CPU.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Circular positional encoding</div>
      <div class="math-eq">Standard BERT:   PE(p, 2i)   = sin(p / 10000^(2i/d))
                 PE(p, 2i+1) = cos(p / 10000^(2i/d))
→ PE(0) ≠ PE(16568)  (endpoints maximally different)

Circular (ours): PE(p, 2i)   = sin(2π × p / L)   where L = 16,569
                 PE(p, 2i+1) = cos(2π × p / L)
→ PE(0) = PE(16568+1): sin(0) = sin(2π) = 0  ✓ circular</div>
      <div class="math-example"><strong>Key insight:</strong> position 16,568 and position 0 differ by one letter physically. The standard formula assigns them values sin(0)=0 vs sin(16568/10000)≈sin(1.66)≈1.0 — maximally different. The circular formula assigns sin(0)=0 and sin(2π×16568/16569)≈sin(2π−ε)≈−ε ≈ 0 — correctly similar.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Self-attention (one head)</div>
      <div class="math-eq">Q = X W_Q,  K = X W_K,  V = X W_V     (X: sequence embeddings)

Attention(Q,K,V) = softmax( Q Kᵀ / √d_k ) V

d_k = 256 / 8 heads = 32   →   √d_k ≈ 5.66

The √d_k scaling prevents the dot products from getting too large
before softmax (which would push gradients toward zero).</div>
      <div class="math-example"><strong>Example (tiny):</strong> Q·Kᵀ score for two positions = [4.0, 1.0, 0.5]. Divide by √32 ≈ 5.66 → [0.71, 0.18, 0.09]. Softmax → [0.50, 0.37, 0.13]. Position 0 attends 50% to itself, 37% to position 1, 13% to position 2.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Parameter count</div>
      <div class="math-eq">Embeddings:  vocab_size × d_model = 4,102 × 256 ≈ 1.05M
Per layer:   4 × d² (attention Q,K,V,O) + 2 × d × d_ff (FFN)
           = 4 × 256² + 2 × 256 × 1,024
           = 262,144 + 524,288 = 786,432
6 layers:    6 × 786,432 ≈ 4.72M
Total ≈ 5.79M  ✓</div>
    </div>`
  };
})();
