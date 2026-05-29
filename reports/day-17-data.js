(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[17] = {
    topic: "Pathogenicity",
    commit: "1c9254d",
    status: "complete",
    built: [
      "mtdna_fm/model/model.py — MtDNAForVariantPathogenicity + VariantPathogenicityOutput: binary classifier reading variant-position token hidden state",
      "configs/finetuning_pathogenicity.yaml — LoRA r=4, weight_decay=0.1, pos_weight=2.5 for ClinVar/gnomAD class imbalance",
      "mtdna_fm/scripts/finetune.py — PathogenicityVariantDataset (centered 512-token window, synthetic fallback) + finetune_pathogenicity training loop with AUROC evaluation",
      "tests/test_model.py — 10 new tests in TestMtDNAForVariantPathogenicity (264 → 274 total)"
    ],
    learned: [
      "Local vs global representations: pathogenicity depends on one position's effect on a codon or tRNA stem, so reading the variant-token hidden state is more informative than aggregating the whole window via CLS",
      "pos_weight in BCEWithLogitsLoss: equivalent to upweighting positives by 2.5× — appropriate when missing a true pathogenic variant (false negative) costs more than a false alarm",
      "LoRA rank selection by dataset size: r=4 for ~7k variants vs r=8 for 47k genomes; smaller rank reduces adapter overfitting on small datasets",
      "Synthetic fallback in datasets: generating random variants when parquet is absent keeps CI runnable without real data, while testing the same tokenisation path",
      "register_buffer for pos_weight: moves with .to(device), excluded from optimiser, saved/restored by save_pretrained — the right pattern for non-learnable scalar hyperparameters that need device awareness"
    ],
    decisions: [
      "Variant-position hidden state not CLS: pathogenicity is local — the variant's effect on a codon or RNA stem — so the contextual representation at the mutation site is more informative than a whole-window aggregate",
      "512-token window centered on variant: equal left/right context (~256 bp each side) covers the functional element most mtDNA variants fall in without padding overhead",
      "LoRA r=4 + weight_decay=0.1: small dataset (7k) needs smaller rank and heavier L2 to avoid memorising training examples",
      "Synthetic dataset fallback: avoids hard dependency on variant parquet in test suite while testing the same tokenisation code path as real data"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <!-- DNA strand with a marked mutation site -->
          <rect x="10" y="55" width="100" height="8" rx="4" fill="#93c5fd"/>
          <!-- bases along strand -->
          <circle cx="25" cy="59" r="5" fill="#3b82f6"/>
          <circle cx="45" cy="59" r="5" fill="#3b82f6"/>
          <circle cx="65" cy="59" r="5" fill="#ef4444" stroke="#b91c1c" stroke-width="2"/>
          <circle cx="85" cy="59" r="5" fill="#3b82f6"/>
          <circle cx="105" cy="59" r="5" fill="#3b82f6"/>
          <!-- red arrow pointing at the mutant base -->
          <line x1="65" y1="35" x2="65" y2="50" stroke="#ef4444" stroke-width="2" marker-end="url(#arr)"/>
          <defs>
            <marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#ef4444"/>
            </marker>
          </defs>
          <text x="65" y="28" text-anchor="middle" font-size="9" fill="#ef4444">variant</text>
          <!-- classifier box -->
          <rect x="35" y="78" width="50" height="20" rx="4" fill="#fef3c7" stroke="#f59e0b"/>
          <text x="60" y="92" text-anchor="middle" font-size="8" fill="#92400e">pathogenic?</text>
          <!-- output labels -->
          <text x="28" y="115" text-anchor="middle" font-size="8" fill="#16a34a">benign</text>
          <text x="92" y="115" text-anchor="middle" font-size="8" fill="#ef4444">pathogenic</text>
          <line x1="52" y1="98" x2="38" y2="110" stroke="#6b7280" stroke-width="1"/>
          <line x1="68" y1="98" x2="82" y2="110" stroke="#6b7280" stroke-width="1"/>
        </svg>
        <div class="eli5-caption">One mutation, one verdict</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>A mutation is a single letter change at one position in the 16,569-letter mitochondrial genome. Some mutations cause disease (pathogenic); most don't (benign).</div>
        <div class="eli5-step"><span class="eli5-num">2</span>The model reads a 512-letter window centered on the mutation — like zooming in on the page of a book where the typo appears, rather than reading the whole book.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>Instead of summarising the whole window into one number, it reads the model's internal description specifically at the mutated position — the local context matters most.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>A classifier then outputs one probability: how likely is this specific change, in this specific context, to break something important? Above 0.5 → flag as potentially pathogenic.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Binary Cross-Entropy with Positive Weighting</div>
      <div class="math-eq">
        L = -1/N * sum_i [ pos_weight * y_i * log(sigma(z_i)) + (1 - y_i) * log(1 - sigma(z_i)) ]

        where:
          z_i   = raw logit from classifier (variant-token hidden state)
          y_i   = 1 (pathogenic) or 0 (benign)
          sigma = sigmoid function: sigma(z) = 1 / (1 + exp(-z))
          pos_weight = 2.5  (upweights the pathogenic class)
      </div>
      <div class="math-example"><strong>Example:</strong> suppose the model outputs logit z = 1.5 for a ClinVar variant (y=1).
sigma(1.5) = 0.818.
Without pos_weight: loss contribution = -log(0.818) = 0.20.
With pos_weight=2.5:  loss contribution = -2.5 * log(0.818) = 0.50.
The pathogenic class now counts 2.5× more toward the gradient — correcting for the fact there are 2.5× more benign variants in training.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Variant-Token Hidden State Extraction</div>
      <div class="math-eq">
        h_variant = H[b, clamp(variant_token_idx[b], 0, seq_len-1), :]

        where H is last_hidden_state of shape (batch, seq_len, hidden_size)
        and variant_token_idx[b] is the index into the window of the token
        that contains the mutated position for sample b.

        logit = Linear(hidden_size, 1)(dropout(h_variant))  shape: (batch,)
      </div>
      <div class="math-example"><strong>Example:</strong> window starts at genome position 3000, variant is at position 3256.
variant_token_idx = 3256 - 3000 = 256 (the 256th token in the window).
h_variant = H[b, 256, :]  — a vector of length 256 (hidden_size).
This vector encodes the k-mer context at the mutation site, not a summary of the whole window.</div>
    </div>`
  };
})();
