(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[5] = {
    topic: "Variant Datasets",
    commit: "41ec618",
    status: "complete",
    built: [
      "mtdna_fm/data/variant_processor.py — 6 functions for gnomAD/ClinVar/PhyloTree parsing",
      "mtdna_fm/data/variant_downloader.py — 3 idempotent download functions",
      "mtdna_fm/scripts/download.py — replaced stubs with real gnomad/clinvar/phylotree dispatch",
      "tests/test_data.py — 22 new tests across 4 classes (89 total)"
    ],
    learned: [
      "gnomAD chrM INFO fields differ from autosomal: mean_hl and n_hom_var instead of standard names",
      "ClinVar CLNSIG is multi-valued with | and , delimiters — regex split on both required",
      "AF ≥ 0.01 as benign proxy is standard but introduces survivorship bias at rare pathogenic variants",
      "PhyloTree mutation strings: insertions, deletions, back-mutations all mixed in one column",
      "Tabix for gnomAD (5 GB), Python gzip filter for ClinVar (small) — tool matches file size"
    ],
    decisions: [
      "Standalone functions not a class — parse_clinvar importable without the gnomAD parser",
      "Idempotency on output parquets (not input VCFs) — re-parsing unchanged VCFs is wasteful",
      "BENIGN_AF_THRESHOLD = 0.01 as named constant — visible in API surface, overridable",
      "het_level_vector column deferred to Dataset class — populated via join on position"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <rect x="8" y="8" width="50" height="30" rx="4" fill="#fee2e2" stroke="#fca5a5" stroke-width="1.5"/>
          <text x="33" y="22" text-anchor="middle" font-size="7" fill="#991b1b" font-weight="bold">DANGEROUS</text>
          <text x="33" y="32" text-anchor="middle" font-size="6.5" fill="#b91c1c">ClinVar path.</text>
          <rect x="62" y="8" width="52" height="30" rx="4" fill="#dcfce7" stroke="#86efac" stroke-width="1.5"/>
          <text x="88" y="22" text-anchor="middle" font-size="7" fill="#166534" font-weight="bold">HARMLESS</text>
          <text x="88" y="32" text-anchor="middle" font-size="6.5" fill="#16a34a">gnomAD AF≥1%</text>
          <rect x="30" y="50" width="60" height="28" rx="4" fill="#fef3c7" stroke="#fcd34d" stroke-width="1.5"/>
          <text x="60" y="63" text-anchor="middle" font-size="7.5" fill="#92400e" font-weight="bold">❓ UNKNOWN</text>
          <text x="60" y="73" text-anchor="middle" font-size="6.5" fill="#b45309">rare, no label</text>
          <text x="60" y="100" text-anchor="middle" font-size="6.5" fill="#64748b">label = 1 / 0 / NaN</text>
          <line x1="33" y1="38" x2="50" y2="50" stroke="#94a3b8" stroke-width="1.2"/>
          <line x1="88" y1="38" x2="70" y2="50" stroke="#94a3b8" stroke-width="1.2"/>
        </svg>
        <div class="eli5-caption">Each DNA variant is labelled: dangerous, harmless, or unknown</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>A <strong>variant</strong> is a single-letter typo in DNA — position 1,234 might be 'A' in most people but 'G' in others. Some typos cause diseases; most don't. We gathered lists of known typos from three sources: <em>gnomAD</em> (population), <em>ClinVar</em> (disease), <em>PhyloTree</em> (ancestry tree).</div>
        <div class="eli5-step"><span class="eli5-num">2</span>We <strong>label</strong> each variant: if ClinVar says "Pathogenic" or "Likely pathogenic" → label 1 (dangerous). If gnomAD shows it in ≥1% of healthy people → label 0 (harmless). Everything else gets label NaN (we don't know yet).</div>
        <div class="eli5-step"><span class="eli5-num">3</span>The 1% threshold (AF ≥ 0.01) for "harmless" is a rule of thumb: if a DNA change is common in healthy people, it's probably not destroying anything vital. It's not perfect — rare diseases exist — but it's the standard in genetics.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>ClinVar can label the same variant multiple ways at once (e.g., "Pathogenic|Uncertain_significance"). We split on both <code>|</code> and <code>,</code> separators and take the most severe label.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Pathogenicity labelling rule</div>
      <div class="math-eq">label(v) =
  1   if CLNSIG(v) ∩ {"Pathogenic","Likely_pathogenic"} ≠ ∅
  0   else if AF(v) ≥ 0.01
  NaN otherwise

CLNSIG parsing:  clnsig_values = re.split(r"[|,]", raw_clnsig)</div>
      <div class="math-example"><strong>Example:</strong> variant at pos 3,243 has CLNSIG="Pathogenic|Uncertain_significance" and AF=0.0001. Split gives ["Pathogenic","Uncertain_significance"]. "Pathogenic" is in the set → label=1.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Allele frequency (AF) threshold</div>
      <div class="math-eq">AF = (count of alt alleles in population) / (total alleles)
   = n_alt / (2 × n_samples)   [diploid autosomal; for mtDNA: n_het / n_samples]

Benign proxy: AF ≥ 0.01  →  variant appears in ≥1% of healthy individuals</div>
      <div class="math-example"><strong>Example:</strong> 500 people sequenced, 30 carry the variant → AF = 30/500 = 0.06 ≥ 0.01 → label=0 (benign).</div>
    </div>`
  };
})();
