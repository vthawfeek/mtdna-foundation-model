(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[2] = {
    topic: "Tokenizer",
    commit: "3e1f9d9",
    status: "complete",
    built: [
      "mtdna_fm/tokenizer/vocabulary.py — KmerVocabulary: 4,096 6-mers + 6 special tokens = 4,102",
      "mtdna_fm/tokenizer/tokenize.py — tokenize_sequence() with circular wrap via modular indexing",
      "mtdna_fm/tokenizer/__init__.py — public exports: KmerVocabulary, tokenize_sequence, token IDs",
      "tests/test_tokenizer.py — 24 tests across vocabulary and tokenization"
    ],
    learned: [
      "K-mer vocabulary is deterministic (4^k), unlike BPE which depends on corpus statistics",
      "Circular boundary coverage: (p + j) % L produces junction k-mers without string manipulation",
      "position_ids must be absolute genomic coordinates so circular PE buffer indexes correctly",
      "Heteroplasmy as continuous float [0.0, 1.0] avoids arbitrary discretization thresholds"
    ],
    decisions: [
      "Special tokens at indices 0–5 (PAD=0 is PyTorch embedding convention)",
      "Modular indexing over string prepending: cleaner position IDs, identical k-mers",
      "max_seq_len truncates without padding — padding is the DataCollator's responsibility",
      "scope='module' on vocab fixtures: 50ms build amortized across all 24 tests"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <text x="10" y="20" font-size="9" fill="#374151" font-family="monospace" font-weight="bold">ACGTAC</text>
          <line x1="60" y1="24" x2="60" y2="38" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>
          <defs><marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
          <rect x="20" y="42" width="80" height="22" rx="4" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"/>
          <text x="60" y="57" text-anchor="middle" font-size="8" fill="#1e40af">dictionary lookup</text>
          <line x1="60" y1="64" x2="60" y2="78" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>
          <text x="10" y="92" font-size="9" fill="#374151" font-family="monospace" font-weight="bold">[42, 7, 315, ...]</text>
          <text x="60" y="118" text-anchor="middle" font-size="7" fill="#64748b">letters → numbers</text>
          <rect x="5" y="95" width="110" height="18" rx="3" fill="#f0fdf4" stroke="#86efac" stroke-width="1"/>
        </svg>
        <div class="eli5-caption">Every 6-letter DNA word gets its own number</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>The AI can't read letters — it only understands numbers. So we made a <strong>dictionary</strong> that gives every possible 6-letter DNA word its own unique ID number. (<em>Why 6 letters? It's a sweet spot — short enough to be fast, long enough to capture real biology.</em>)</div>
        <div class="eli5-step"><span class="eli5-num">2</span>How many 6-letter words can you make from A, C, G, T? Each position has 4 choices, and there are 6 positions: <strong>4 × 4 × 4 × 4 × 4 × 4 = 4,096 words</strong>. Plus 6 special tokens ([PAD], [CLS], [MASK], etc.) = <strong>4,102 total</strong>.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>To tokenize a sequence, we slide a window of 6 letters along it, one step at a time. Position 0 gives us letters 0–5, position 1 gives letters 1–6, etc. We look up each 6-letter chunk in the dictionary and write down its number.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>DNA is <strong>circular</strong> — the last letter connects back to the first. So at position 16,567 we look up letters 16,567, 16,568, 0, 1, 2, 3 (wrapping around). We do this with modular arithmetic: position <em>p+j</em> becomes <em>(p+j) % 16,569</em>.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">K-mer vocabulary size</div>
      <div class="math-eq">|V| = 4^k + n_special
     = 4^6 + 6
     = 4,096 + 6
     = 4,102</div>
      <div class="math-example"><strong>Why 4^k?</strong> Each of the k positions can independently be A, C, G, or T — 4 choices. By the multiplication principle: 4 × 4 × … (k times) = 4^k.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Circular tokenization with modular indexing</div>
      <div class="math-eq">token_id[p] = vocab[ seq[(p+0)%L], seq[(p+1)%L], …, seq[(p+k-1)%L] ]

where L = 16,569 (genome length), k = 6</div>
      <div class="math-example"><strong>Example (L=5, k=3, seq="ATCGA"):</strong><br>
p=3 → chars at (3%5, 4%5, 5%5) = seq[3], seq[4], seq[0] = G, A, A = "GAA"<br>
p=4 → chars at (4%5, 5%5, 6%5) = seq[4], seq[0], seq[1] = A, A, T = "AAT"<br>
The circular wrap means position 4 seamlessly joins back to the start.</div>
    </div>`
  };
})();
