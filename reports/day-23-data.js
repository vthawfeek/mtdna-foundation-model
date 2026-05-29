(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[23] = {
    topic: "Documentation",
    commit: "",
    status: "complete",
    built: [
      "docs/01_data_pipeline.md — datasets (HmtDB, NCBI, gnomAD, ClinVar, PhyloTree, ancient DNA), preprocessing steps with rationale, full DVC pipeline YAML",
      "docs/02_tokenization.md — 6-mer vocabulary construction, special token table, circular windowing, heteroplasmy channel, vocabulary statistics",
      "docs/03_architecture.md — circular PE derivation from first principles, het projection design, parameter count breakdown (~6M), comparison vs DNABERT2/HyenaDNA",
      "docs/04_pretraining.md — two-phase curriculum rationale, gradient accumulation, expected MLM loss curve, masking strategy, MLflow monitoring, CPU/GPU timing",
      "docs/05_finetuning_and_evaluation.md — three tasks with LoRA rationale, baseline tables, ancient DNA zero-shot, known limitations"
    ],
    learned: [
      "Documentation written from first principles produces content that reveals the 'why', not just the 'what' — the circular PE derivation shows exactly where standard sinusoidal PE fails before introducing the fix",
      "The comparison table against DNABERT2 and HyenaDNA clarifies the value proposition: missing circular topology and heteroplasmy support cannot be patched by model scale",
      "Explicit LoRA rank rationale (r=8 for 47k examples, r=4 for 7k) makes the relationship between dataset size, task complexity, and adapter capacity concrete",
      "A known limitations section is as important as results — documenting population bias, missing indel support, and regression confidence limits sets appropriate expectations"
    ],
    decisions: [
      "One topic per doc, no overlap: prevents the maintenance problem of updating the same fact in two places and makes each doc independently useful",
      "Derivations from first principles rather than final formulas: more useful to a reader implementing a similar model than showing only the result",
      "Concrete baseline numbers in comparison tables: majority class + k-mer frequency + logistic regression + fine-tuned model, so the value added by the foundation model is visible",
      "Limitations section not minimized: population bias, indels not supported, and known R² floor stated directly"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <!-- Book / documentation icon -->
          <rect x="20" y="15" width="70" height="90" rx="4" ry="4" fill="#4a90d9" />
          <rect x="25" y="20" width="60" height="80" rx="2" ry="2" fill="#fff" />
          <!-- Lines of text -->
          <rect x="32" y="30" width="46" height="4" rx="2" fill="#4a90d9" />
          <rect x="32" y="40" width="40" height="3" rx="1" fill="#aac4e8" />
          <rect x="32" y="48" width="44" height="3" rx="1" fill="#aac4e8" />
          <rect x="32" y="56" width="36" height="3" rx="1" fill="#aac4e8" />
          <rect x="32" y="68" width="46" height="4" rx="2" fill="#4a90d9" />
          <rect x="32" y="78" width="42" height="3" rx="1" fill="#aac4e8" />
          <rect x="32" y="86" width="38" height="3" rx="1" fill="#aac4e8" />
          <!-- Spine -->
          <rect x="20" y="15" width="8" height="90" rx="2" ry="2" fill="#2c6fad" />
        </svg>
        <div class="eli5-caption">Five docs, one topic each</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>Imagine you built a complicated LEGO set but lost the instructions. The model files exist, but without documentation, the next person (or future you) has to reverse-engineer every decision. That's what Day 23 fixes.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>Five documents, each covering exactly one thing: where the data comes from, how sequences get chopped into tokens, how the model architecture works, how the model is trained, and how it gets tested on real tasks.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>The most important doc explains the circular positional encoding — why the standard approach breaks for circular DNA, and what the mathematical fix looks like step by step. This is the kind of explanation you can't get from just reading the code.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>Good documentation also admits what the model can't do: it works better on European populations, it can't handle insertions or deletions, and its regression results are noisy. Honest limitations are as valuable as impressive results.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Circular Positional Encoding</div>
      <div class="math-eq">Standard sinusoidal PE:
  PE[pos, 2i]   = sin(pos / 10000^(2i/d))
  PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

Circular PE (mtDNA-FM):
  angle[pos] = 2 * pi * pos / genome_length
  PE[pos, 2i]   = sin(angle[pos] / 10000^(2i/d))
  PE[pos, 2i+1] = cos(angle[pos] / 10000^(2i/d))</div>
      <div class="math-example"><strong>Why it matters:</strong> With standard PE, positions 0 and 16568 have maximally different encodings (far apart). With circular PE, angle(0) = 0 and angle(16569) = 2*pi, so sin(2*pi) = sin(0) = 0 and cos(2*pi) = cos(0) = 1. The junction is seamless.</div>
    </div>`
  };
})();
