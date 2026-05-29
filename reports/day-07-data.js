(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[7] = {
    topic: "CI Hardening + Week 1 Exit Criteria",
    commit: "d43cb47",
    status: "complete",
    built: [
      ".github/workflows/ci.yml — two-job CI: lint (ruff check + format) + test (pytest)",
      "README.md — CI and HuggingFace badges with correct GitHub username",
      "ruff format applied across 9 files (whitespace only, no semantic changes)",
      "Week 1 exit criteria verified: 101 tests, 16,569 bp sequences, vocab=4102"
    ],
    learned: [
      "ruff format --check is separate CI gate from ruff check — style vs semantic quality",
      "CI is a Week 1 task, not afterthought: regression detection is automatic from here",
      "Week 1 verified: all 101 tests pass, 152k train sequences exactly 16,569 bp"
    ],
    decisions: [
      "Two-job CI: lint and test separate so format failure doesn't hide test failure",
      "ruff format run before hardening the gate — avoids immediate CI failure on prior commits",
      "Coverage computed in CI but not a gate — premature before model architecture tests"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <rect x="15" y="8" width="90" height="28" rx="5" fill="#1a1a2e" stroke="#3b82f6" stroke-width="1.5"/>
          <text x="60" y="22" text-anchor="middle" font-size="7.5" fill="#93c5fd" font-weight="bold">git push</text>
          <text x="60" y="32" text-anchor="middle" font-size="6.5" fill="#64748b">triggers CI</text>
          <line x1="60" y1="36" x2="40" y2="52" stroke="#94a3b8" stroke-width="1.2"/>
          <line x1="60" y1="36" x2="80" y2="52" stroke="#94a3b8" stroke-width="1.2"/>
          <rect x="10" y="52" width="45" height="24" rx="4" fill="#fef3c7" stroke="#fcd34d" stroke-width="1.5"/>
          <text x="32" y="63" text-anchor="middle" font-size="6.5" fill="#92400e" font-weight="bold">ruff lint</text>
          <text x="32" y="72" text-anchor="middle" font-size="6" fill="#b45309">style check</text>
          <rect x="65" y="52" width="45" height="24" rx="4" fill="#f3e8ff" stroke="#d8b4fe" stroke-width="1.5"/>
          <text x="87" y="63" text-anchor="middle" font-size="6.5" fill="#6b21a8" font-weight="bold">pytest</text>
          <text x="87" y="72" text-anchor="middle" font-size="6" fill="#9333ea">101 tests</text>
          <text x="32" y="94" text-anchor="middle" font-size="12" fill="#22c55e">✓</text>
          <text x="87" y="94" text-anchor="middle" font-size="12" fill="#22c55e">✓</text>
          <text x="60" y="118" text-anchor="middle" font-size="7" fill="#166534" font-weight="bold">merge allowed</text>
        </svg>
        <div class="eli5-caption">Every push automatically runs two checks</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>We set up a <strong>robot teacher</strong> that automatically grades our work every time we save code to GitHub. This is called CI — Continuous Integration. It takes about 30 seconds and tells us if we broke anything.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>The robot runs <strong>two separate checks</strong>: a style checker (ruff) that makes sure the code is formatted consistently, and a test runner (pytest) that runs all 101 tests. They run in parallel and fail independently — a style failure won't hide a broken test.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>At the end of Week 1, we verified the <strong>exit criteria</strong>: all 101 tests pass, all 152,000 training sequences are exactly 16,569 bp, and the vocabulary has exactly 4,102 tokens.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>Coverage (what % of code lines are tested) was 76% at this point — computed but not enforced yet. We'll add a coverage gate later once the model code is fully tested.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Test coverage formula</div>
      <div class="math-eq">coverage = (lines_executed_during_tests / total_lines) × 100%

At Day 7:  coverage = 76%
→ means 24% of code lines were not reached by any test</div>
      <div class="math-example"><strong>Interpretation:</strong> If total_lines = 1,200 and coverage = 76%, then 912 lines were executed. The 288 uncovered lines are mostly in download/variant scripts that need network mocking.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Week 1 exit criteria (verified)</div>
      <div class="math-eq">✓ 101 tests pass  (pytest exit code 0)
✓ vocab_size = 4,102 = 4^6 + 6
✓ all sequences: len(seq) == 16,569
✓ n_train = 152,358  (80% of 190,448 total sequences)</div>
      <div class="math-example">These are <strong>invariants</strong> — facts that must remain true throughout the project. The test suite enforces them automatically on every commit from here on.</div>
    </div>`
  };
})();
