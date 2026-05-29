(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[4] = {
    topic: "Preprocessing Pipeline + EDA Notebook",
    commit: "78b869a",
    status: "complete",
    built: [
      "mtdna_fm/data/preprocessor.py — 5 standalone functions: clean, normalize, split, build_df, preprocess",
      "mtdna_fm/scripts/preprocess.py — full mtdna-preprocess CLI",
      "tests/test_data.py — 27 new tests, 67 total",
      "notebooks/01_data_exploration.ipynb — haplogroup distribution, D-loop entropy, N-content, geo distribution",
      "configs/data.yaml + dvc.yaml preprocess stage"
    ],
    learned: [
      "Junction duplicate detection: some databases append first 200 bp — must strip before normalization",
      "Padding position matters: D-loop start (576) is most tolerant, preserves canonical gene coordinates",
      "Rare haplogroup handling in StratifiedShuffleSplit: merge &lt;2 samples into _rare bucket",
      "Function-per-step over pipeline class: independently testable and composable",
      "D-loop entropy is 7× higher than coding regions — empirical justification for circular PE"
    ],
    decisions: [
      "Separate functions not a class — three similar steps better than premature abstraction",
      "QC flagging (qc_pass=False) not filtering — downstream training decides what to exclude",
      "Padding at position 576 (D-loop start), not pos 0 or 3'-end",
      "_rare class merging preserves stratification without dropping up to 4% of corpus"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <circle cx="60" cy="36" r="22" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4 2"/>
          <text x="60" y="30" text-anchor="middle" font-size="7" fill="#64748b">MESSY</text>
          <text x="60" y="40" text-anchor="middle" font-size="6" fill="#64748b">variable len</text>
          <line x1="60" y1="58" x2="60" y2="72" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr2)"/>
          <defs><marker id="arr2" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
          <text x="60" y="70" text-anchor="middle" font-size="7" fill="#374151">clean + pad/trim</text>
          <circle cx="60" cy="98" r="22" fill="none" stroke="#16a34a" stroke-width="2"/>
          <text x="60" y="93" text-anchor="middle" font-size="7" fill="#166534" font-weight="bold">CLEAN</text>
          <text x="60" y="103" text-anchor="middle" font-size="6" fill="#16a34a">16,569 bp</text>
          <path d="M 50 81 A 22 22 0 0 1 70 81" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linecap="round"/>
          <text x="60" y="79" text-anchor="middle" font-size="5.5" fill="#d97706">D-loop</text>
        </svg>
        <div class="eli5-caption">Messy sequences cleaned and padded to exactly 16,569 bp</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>Raw DNA sequences are messy: some are too long (databases append a duplicate fragment at the end), some have ambiguous letters like 'R' or 'Y'. We <strong>clean</strong> them: uppercase everything, replace ambiguous letters with 'N', cut off the duplicate tail.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>All sequences must be <strong>exactly 16,569 letters long</strong> (the true length of the human mitochondrial genome). If a sequence is shorter, we pad it with Ns; if longer, we trim it. We pad at position 576 — the start of the D-loop, the most variable region — to disturb the important coding genes as little as possible.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>We then <strong>split the data</strong> into three piles: 80% for training, 10% for validation (checking during training), 10% for testing (final score). We do this <em>stratified by haplogroup</em> — meaning each of the 26 haplogroup categories keeps its proportions in all three piles. No category gets accidentally left out of the test set.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>Rare haplogroups (fewer than 2 samples) get merged into a <strong>"_rare" bucket</strong> before splitting — otherwise we'd have only 1 sample in a class and couldn't put any in both train and test.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Length normalization</div>
      <div class="math-eq">target_L = 16,569

if len(seq) > target_L:
    seq = seq[:target_L]           # trim from the 3' end

if len(seq) < target_L:
    pad_len = target_L - len(seq)
    seq = seq[:576] + "N"*pad_len + seq[576:]   # insert at D-loop</div>
      <div class="math-example"><strong>Example:</strong> seq has 16,400 bp → pad_len = 169 → insert 169 × 'N' at position 576. The 16,569 − 576 = 15,993 coding-region bases are untouched.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Stratified split</div>
      <div class="math-eq">For each class c with n_c samples:
  train_c = floor(0.80 × n_c)
  val_c   = floor(0.10 × n_c)
  test_c  = n_c − train_c − val_c   (remainder)

Classes with n_c < 2 → merged into "_rare" before split.</div>
      <div class="math-example"><strong>Example:</strong> haplogroup H has 8,000 samples → 6,400 train / 800 val / 800 test. Haplogroup Z has 1 sample → absorbed into _rare, which gets the same 80/10/10 treatment.</div>
    </div>`
  };
})();
