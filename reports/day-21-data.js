(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[21] = {
    topic: "HuggingFace Hub",
    commit: "e372b1c",
    status: "complete",
    built: [
      "mtdna_fm/scripts/push_to_hub.py — rewrote to push actual trained adapters, patch base_model_name_or_path, write proper adapter READMEs",
      "models/phase1_v1/README.md — updated model card with Phase 2 completion, actual performance numbers, ancient DNA results, limitations",
      "vthawfeek/mtdna-foundation-model (Hub) — base model: config, 44.6MB weights, tokenizer, model card",
      "vthawfeek/mtdna-fm-haplogroup (Hub) — LoRA r=8 adapter (400KB) for 26-class haplogroup classification",
      "vthawfeek/mtdna-fm-pathogenicity (Hub) — LoRA r=4 adapter (203KB) for binary variant pathogenicity"
    ],
    learned: [
      "LoRA as deployment artefact: haplogroup adapter is 400KB vs 44.6MB base — tasks share the base, each adapter is <1% additional storage",
      "Patching adapter configs for Hub: PEFT saves base_model_name_or_path as local path; must patch to Hub model ID before uploading",
      "Custom architectures and AutoConfig: AutoConfig refuses unknown model_type values — expected for custom architectures, load via concrete class instead",
      "Model card as primary deliverable: equations, honest baselines, limitations section distinguishes a research release from a weight dump"
    ],
    decisions: [
      "Push phase1_v1 as base model: phase2_v1 directory had no model files (only adapter outputs from finetuning were saved there)",
      "Real adapters not fresh ones: original script created random-weight adapters; changed to upload actual trained adapters from finetune_haplogroup_v1 and finetune_pathogenicity_v1",
      "Separate Hub repos per adapter: following PEFT convention, each adapter in its own repo pointing back to base model Hub ID",
      "AutoConfig warning expected and documented: custom model_type not registered with Transformers; users load via mtdna-fm package concrete class"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <!-- Cloud/Hub icon -->
          <rect x="10" y="60" width="100" height="50" rx="8" fill="#ff9d00" opacity="0.15"/>
          <text x="60" y="90" text-anchor="middle" font-size="11" fill="#ff9d00" font-weight="bold">🤗 Hub</text>
          <!-- Base model box -->
          <rect x="30" y="10" width="60" height="28" rx="4" fill="#4a90d9" opacity="0.8"/>
          <text x="60" y="28" text-anchor="middle" font-size="9" fill="white">Base Model</text>
          <text x="60" y="20" text-anchor="middle" font-size="8" fill="white">44.6 MB</text>
          <!-- Adapter boxes -->
          <rect x="5" y="48" width="50" height="18" rx="3" fill="#5cb85c" opacity="0.8"/>
          <text x="30" y="60" text-anchor="middle" font-size="8" fill="white">Haplo 400KB</text>
          <rect x="65" y="48" width="50" height="18" rx="3" fill="#d9534f" opacity="0.8"/>
          <text x="90" y="60" text-anchor="middle" font-size="8" fill="white">Path 203KB</text>
          <!-- Arrows down to cloud -->
          <line x1="60" y1="38" x2="60" y2="60" stroke="#666" stroke-width="1" marker-end="url(#arr)"/>
          <line x1="30" y1="66" x2="45" y2="75" stroke="#666" stroke-width="1"/>
          <line x1="90" y1="66" x2="75" y2="75" stroke="#666" stroke-width="1"/>
          <defs>
            <marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#666"/>
            </marker>
          </defs>
        </svg>
        <div class="eli5-caption">Base + 2 adapters on the Hub</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>Think of HuggingFace Hub like GitHub but for AI models. Instead of code files, you upload weights files — the numbers that make the model work. Anyone can then download and use your model with one line of Python.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>The base model (44.6 MB) is the general-purpose mtDNA encoder trained on 30,000+ genomes. It's the shared foundation that every task builds on top of.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>LoRA adapters are tiny add-ons (400 KB and 203 KB) that specialise the base model for a specific task without changing its weights. Like a lens filter — the camera stays the same, the filter changes what it sees.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>The model card is the documentation page shown on the Hub. It includes the equations for circular positional encoding, a benchmark table with honest baselines, known limitations (European population bias), and a 3-line usage example.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">LoRA: Low-Rank Adaptation</div>
      <div class="math-eq">W' = W + (B × A)  where  A ∈ R^(r×d_in),  B ∈ R^(d_out×r),  r &lt;&lt; d

Trainable params = r × (d_in + d_out)  vs  d_in × d_out  for full fine-tuning</div>
      <div class="math-example"><strong>Example (haplogroup head, r=8):</strong>
One attention layer: d_in=256, d_out=256
Full fine-tuning: 256×256 = 65,536 params
LoRA r=8: 8×256 + 8×256 = 4,096 params (6.3% of full)
Entire haplogroup adapter: ~500K params vs ~6.9M base = 7.2%</div>
    </div>`
  };
})();
