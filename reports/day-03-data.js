(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[3] = {
    topic: "Data Download Clients",
    commit: "dabd8c5",
    status: "complete",
    built: [
      "mtdna_fm/data/hmtdb_client.py — idempotent HmtDB download with SHA256 + NCBI fallback",
      "mtdna_fm/data/ncbi_client.py — resumable Entrez client with WebEnv batching + progress.json",
      "mtdna_fm/scripts/download.py — full Typer CLI replacing Day 1 stub",
      "dvc.yaml — download_hmtdb and download_ncbi stages with persist:true",
      "tests/test_data.py — 17 offline unit tests (all network calls mocked)"
    ],
    learned: [
      "Idempotency as first-class property: check outputs exist before any network call",
      "WebEnv batching: server-side result caching eliminates repeated search overhead",
      "Progress files make long NCBI downloads resumable at batch granularity",
      "persist:true in DVC prevents raw downloads from dvc gc deletion",
      "Mock target must be the function in the module that *uses* it, not where defined"
    ],
    decisions: [
      "NCBI fallback lives in hmtdb_client, not download.py — client owns its fallback logic",
      "Progress file keyed by batch index (not retstart) — human-readable, stable on batch_size changes",
      "_efetch_batch as standalone function: testable without a full client object",
      "Day 5 stubs exit code=1 with clear message rather than silently succeeding"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <rect x="10" y="10" width="46" height="34" rx="5" fill="#dbeafe" stroke="#93c5fd" stroke-width="1.5"/>
          <text x="33" y="25" text-anchor="middle" font-size="7.5" fill="#1e40af" font-weight="bold">HmtDB</text>
          <text x="33" y="37" text-anchor="middle" font-size="6.5" fill="#3b82f6">library</text>
          <rect x="64" y="10" width="46" height="34" rx="5" fill="#f3e8ff" stroke="#d8b4fe" stroke-width="1.5"/>
          <text x="87" y="25" text-anchor="middle" font-size="7.5" fill="#6b21a8" font-weight="bold">NCBI</text>
          <text x="87" y="37" text-anchor="middle" font-size="6.5" fill="#9333ea">backup</text>
          <line x1="33" y1="44" x2="50" y2="64" stroke="#94a3b8" stroke-width="1.2"/>
          <line x1="87" y1="44" x2="70" y2="64" stroke="#94a3b8" stroke-width="1.2"/>
          <rect x="35" y="64" width="50" height="26" rx="4" fill="#f0fdf4" stroke="#86efac" stroke-width="1.5"/>
          <text x="60" y="76" text-anchor="middle" font-size="7" fill="#166534" font-weight="bold">robot</text>
          <text x="60" y="86" text-anchor="middle" font-size="6.5" fill="#16a34a">downloads</text>
          <line x1="60" y1="90" x2="60" y2="104" stroke="#94a3b8" stroke-width="1.2"/>
          <rect x="30" y="104" width="60" height="20" rx="4" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.5"/>
          <text x="60" y="117" text-anchor="middle" font-size="7" fill="#92400e" font-weight="bold">data/ folder</text>
        </svg>
        <div class="eli5-caption">Robots fetch DNA from two libraries; save locally</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>We need tens of thousands of real DNA sequences from the internet. We built two <strong>download robots</strong>: one for HmtDB (the main mitochondrial DNA library) and one for NCBI (a backup library run by the US government).</div>
        <div class="eli5-step"><span class="eli5-num">2</span>The robots are <strong>idempotent</strong> — a fancy word for "don't repeat yourself." Before downloading anything, they check if the file already exists and if its fingerprint (SHA256 hash) matches. If yes, they skip it. Run them 10 times, you still only download once.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>NCBI delivers sequences in <strong>batches</strong> of 500. The robot saves a progress file after every batch. If the internet cuts out halfway through, next time it restarts from the last completed batch — not from zero.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>All tests run <strong>offline</strong> — every internet call is replaced with a fake in tests. This means CI doesn't need a real internet connection, and tests run in seconds rather than hours.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">SHA-256 idempotency check</div>
      <div class="math-eq">valid = sha256(file_bytes) == expected_hash

sha256 maps any input → a 256-bit (64 hex char) digest.
Probability of collision: 1/2^256 ≈ 10^-77 (effectively zero).</div>
      <div class="math-example"><strong>Example:</strong> expected = "a3f2…bc91". Compute sha256 of downloaded file. If they match, file is intact. If not, re-download.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Resumable batch index</div>
      <div class="math-eq">batch_index = floor(sequences_downloaded / batch_size)
retstart   = batch_index × batch_size

To resume: read progress.json → get last batch_index
→ skip to retstart = batch_index × batch_size</div>
      <div class="math-example"><strong>Example:</strong> batch_size=500, downloaded 3,000 → batch_index=6 → resume at retstart=3,000. No sequences are fetched twice.</div>
    </div>`
  };
})();
