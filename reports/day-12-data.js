(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[12] = {
    topic: "Test Suite Completion (80% → 97%)",
    commit: "6c7c57a",
    status: "complete",
    built: [
      "TestHmtdbClientInternals (9 tests): _download_file, SHA256 match/mismatch, _validate_fasta, zip extraction",
      "TestNcbiClientInternals (7 tests): rate delay, API key config, _esearch, _efetch_batch, force-delete",
      "TestVariantDownloader (9 tests): idempotency for all 3 sources, _extract_chrom_from_gz, _stream_download",
      "TestPreprocessCLI (4) + TestTrainCLI (2) + TestDownloadScriptInternals (5)",
      "TestGradientCheckpointing (3) + TestTransformerValidation (1) + TestLearnablePositionalEncoding (2)"
    ],
    learned: [
      "Mock patch paths: patch target must be the original module, not the caller's imported namespace",
      "subprocess imported inside a function: patch('subprocess.run') (global module, not attribute)",
      "pytest.raises + patch compose cleanly in a single with block (satisfies ruff SIM117)",
      "Gradient checkpointing gated on self.training — model.train() needed to hit that branch",
      "Learnable PE uses nn.Embedding(max_seq_len): position_ids must be in [0, max_seq_len)"
    ],
    decisions: [
      "Targeted lowest-coverage modules first: variant_downloader 0%→100%, train.py 0%→100%",
      "Mocked all network calls — suite runs offline in under 10 seconds",
      "Real zip/gz archives (not mocked) for archive tests: tests actual parsing, not API signatures"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <line x1="20" y1="10" x2="20" y2="95" stroke="#e2e8f0" stroke-width="1"/>
          <line x1="20" y1="95" x2="105" y2="95" stroke="#e2e8f0" stroke-width="1"/>
          <rect x="28" y="37" width="26" height="58" rx="3" fill="#fbbf24" opacity="0.8"/>
          <text x="41" y="33" text-anchor="middle" font-size="7" fill="#92400e" font-weight="bold">80%</text>
          <text x="41" y="108" text-anchor="middle" font-size="6" fill="#64748b">before</text>
          <rect x="66" y="15" width="26" height="80" rx="3" fill="#22c55e" opacity="0.8"/>
          <text x="79" y="11" text-anchor="middle" font-size="7" fill="#166534" font-weight="bold">97%</text>
          <text x="79" y="108" text-anchor="middle" font-size="6" fill="#64748b">after</text>
          <text x="60" y="120" text-anchor="middle" font-size="6.5" fill="#374151">236 tests passing</text>
        </svg>
        <div class="eli5-caption">Coverage jumped from 80% to 97% — nearly everything tested</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>We systematically found the modules with lowest coverage and wrote tests for them until we reached 97% overall. This meant writing tests for the download clients, variant downloader, CLI scripts, and model internals — all the pieces that were previously untested.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>Testing download code is tricky because it makes <strong>internet requests</strong>. We used "mocking" — replacing the real internet call with a fake that returns pre-written responses instantly. Tests run offline in under 10 seconds.</div>
        <div class="eli5-step"><span class="eli5-num">3</span><strong>Mock patching</strong> has a subtle rule: you must intercept the function <em>where it's used</em>, not where it's defined. If module B imports <code>requests.get</code> from module A, you patch <code>B.requests.get</code> — not <code>A.requests.get</code>. Getting this wrong is a very common testing mistake.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>For tests that parse ZIP/gzip archives, we used <strong>real archives</strong> written to a temporary folder — not mocks. This is important: mocking the archive parser itself would only test that our mock returns the right thing, not that our actual parsing code works.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Mock patch target rule</div>
      <div class="math-eq">Given: module_b.py contains
  from requests import get   (or: import requests; requests.get)
  def fetch(): return get(url)

Correct patch target:
  patch("module_b.get")             # where it is USED
  # NOT patch("requests.get")       # where it is DEFINED

Python's mock.patch replaces the name in the given namespace.
If module_b already imported 'get' into its own namespace,
patching 'requests.get' has no effect on 'module_b.get'.</div>
      <div class="math-example"><strong>subprocess special case:</strong> if <code>subprocess.run</code> is imported inside a function (not at module top), the correct target is always <code>patch("subprocess.run")</code> — because no local alias was created.</div>
    </div>`
  };
})();
