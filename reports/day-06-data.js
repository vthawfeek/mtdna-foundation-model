(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[6] = {
    topic: "PyTorch Dataset Classes",
    commit: "33169c2",
    status: "complete",
    built: [
      "mtdna_fm/data/dataset.py — MtDNADataset: windowed circular Dataset (stride=256, ~65 windows/genome)",
      "mtdna_fm/data/variant_dataset.py — VariantDataset: 512-token SNP pathogenicity windows",
      "tests/test_data.py — TestMtDNADataset (7) + TestVariantDataset (5), 101 total"
    ],
    learned: [
      "Pre-tokenize once: 47k genomes fit in memory; avoids re-running k-mer window per epoch",
      "Circular windowing: (window_start + i) % genome_length produces junction windows naturally",
      "Window count: range(0, 16569, 256) gives 65 windows/genome ≈ 3.1M training examples",
      "gnomAD/ClinVar positions are 1-based — convert to 0-based before string indexing",
      "variant_offset field lets model head index the variant-position token directly"
    ],
    decisions: [
      "Circular windows (not non-circular + junction hack): no special cases, uniform treatment",
      "Filter indels in __init__ not __getitem__: accurate __len__ and no per-sample branch",
      "from_dataframe classmethod: mirrors scFM pattern for direct parquet-to-Dataset path",
      "het_level_vectors default to None (not zeros): avoids 47k × 16,569 float array allocation"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <circle cx="60" cy="46" r="30" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3 2"/>
          <text x="60" y="50" text-anchor="middle" font-size="7" fill="#64748b">genome</text>
          <text x="60" y="60" text-anchor="middle" font-size="6.5" fill="#94a3b8">16,569 bp</text>
          <path d="M 35 26 A 30 30 0 0 1 85 26" fill="none" stroke="#3b82f6" stroke-width="3" stroke-linecap="round"/>
          <text x="60" y="20" text-anchor="middle" font-size="6.5" fill="#1e40af" font-weight="bold">window 1</text>
          <path d="M 42 74 A 30 30 0 0 0 88 62" fill="none" stroke="#16a34a" stroke-width="3" stroke-linecap="round"/>
          <text x="85" y="82" text-anchor="middle" font-size="6.5" fill="#166534" font-weight="bold">window 2</text>
          <text x="60" y="110" text-anchor="middle" font-size="7" fill="#374151">65 windows / genome</text>
          <text x="60" y="122" text-anchor="middle" font-size="6.5" fill="#64748b">stride = 256 bp</text>
        </svg>
        <div class="eli5-caption">Each genome is cut into 65 overlapping windows</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>The AI can't process the whole 16,569-letter genome at once — it's too long. Instead, we cut it into <strong>windows of 512 letters</strong>, like looking at a vinyl record through a magnifying glass that shows 512 grooves at a time.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>The magnifying glass moves <strong>256 letters at a time</strong> (the stride). So window 1 shows letters 0–511, window 2 shows letters 256–767, etc. The 50% overlap means each letter appears in two windows — giving the model two chances to learn from it.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>Since DNA is circular, the last window wraps around: it shows letters 16,313–16,568 and then loops back to letters 0–256. We handle this with modular indexing — no special cases needed.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>47,000 genomes × 65 windows = <strong>~3.1 million training examples</strong>. Each window is pre-tokenized once (turned into a list of numbers) and stored in RAM — much faster than re-tokenizing every epoch.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Window count per genome</div>
      <div class="math-eq">n_windows = ceil(L / stride)
            = ceil(16,569 / 256)
            = ceil(64.72)
            = 65

Window start positions: s_i = i × stride  for i = 0, 1, …, 64
Window content: seq[(s_i + j) % L]  for j = 0, 1, …, window_size−1</div>
      <div class="math-example"><strong>Last window (i=64):</strong> s_64 = 64 × 256 = 16,384. Covers positions 16,384 … 16,568 (185 bp) then wraps to 0 … 326 (327 bp). Total = 512 bp. The circular modulus makes the junction seamless.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Total training examples</div>
      <div class="math-eq">total = n_sequences × n_windows_per_genome
      ≈ 47,000 × 65
      ≈ 3,055,000 examples</div>
      <div class="math-example"><strong>Variant windows</strong> are separate: each variant gets a 512-token window centred on its position. The <code>variant_offset</code> field records which token index holds the variant so the fine-tuning head knows where to look.</div>
    </div>`
  };
})();
