(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[1] = {
    topic: "Project Scaffold",
    commit: "12d614c",
    status: "complete",
    built: [
      "pyproject.toml — full project config, 5 CLI entry points, ruff &amp; pytest configured",
      "uv.lock — deterministic dependency lockfile for reproducible installs",
      "mtdna_fm/ package with 8 submodules (model, tokenizer, data, training, etc.)",
      "CLI stubs: download, preprocess, train, finetune, evaluate (Typer)",
      "tests/conftest.py — synthetic_sequence and synthetic_sequence_16569 fixtures",
      ".github/workflows/ci.yml — lint + test jobs on push/PR",
      ".gitignore excludes data/, models/, mlruns/ (DVC-tracked)"
    ],
    learned: [
      "Packaging first: CLI entry points in pyproject.toml make interface stable across installs",
      "uv lockfiles pin every transitive dependency — reproducible environments across machines",
      "Typer generates --help from type annotations automatically",
      "Test markers (slow, integration) defined at project level decouple CI from test files"
    ],
    decisions: [
      "hatchling over setuptools: faster, no setup.py, everything in pyproject.toml",
      "dev extras not requirements-dev.txt: single 'uv sync --extra dev' command",
      "5 separate CLI entry points (not one with subcommands) for clean DVC stage mapping",
      ".gitignore excludes data/ and models/ from day one — DVC-tracked, not git"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="120" viewBox="0 0 120 120">
          <rect x="10" y="10" width="100" height="100" rx="8" fill="#f0fdf4" stroke="#86efac" stroke-width="2"/>
          <rect x="22" y="28" width="36" height="26" rx="4" fill="#dbeafe" stroke="#93c5fd" stroke-width="1.5"/>
          <rect x="66" y="28" width="36" height="26" rx="4" fill="#fef3c7" stroke="#fcd34d" stroke-width="1.5"/>
          <rect x="22" y="66" width="36" height="26" rx="4" fill="#f3e8ff" stroke="#d8b4fe" stroke-width="1.5"/>
          <rect x="66" y="66" width="36" height="26" rx="4" fill="#ffe4e6" stroke="#fca5a5" stroke-width="1.5"/>
          <text x="40" y="45" text-anchor="middle" font-size="8" fill="#1e40af" font-weight="bold">model/</text>
          <text x="84" y="45" text-anchor="middle" font-size="8" fill="#92400e" font-weight="bold">data/</text>
          <text x="40" y="83" text-anchor="middle" font-size="8" fill="#6b21a8" font-weight="bold">tests/</text>
          <text x="84" y="83" text-anchor="middle" font-size="8" fill="#9f1239" font-weight="bold">CLI</text>
          <text x="60" y="16" text-anchor="middle" font-size="7" fill="#64748b">📦 project layout</text>
        </svg>
        <div class="eli5-caption">The project's "toy box" — every tool has its drawer</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>Before writing any AI code, we set up the <strong>workshop</strong>. Imagine you're going to build a LEGO model — you first need to sort all the bricks into labelled boxes so you can find them later.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>We created <strong>four main drawers</strong>: one for the AI brain (<code>model/</code>), one for all the DNA data (<code>data/</code>), one for checks (<code>tests/</code>), and one for command-line buttons (<code>scripts/</code>).</div>
        <div class="eli5-step"><span class="eli5-num">3</span>We also wrote a <strong>shopping list</strong> (<code>pyproject.toml</code>) that tells the computer exactly which tools to download — so anyone else can set up an identical workshop with one command.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>Finally, we made five <strong>doorbell buttons</strong> (CLI entry points): <em>download</em>, <em>preprocess</em>, <em>train</em>, <em>finetune</em>, <em>evaluate</em>. Press a button → the right code runs. They're empty stubs for now, but the wiring exists.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">No new math today — process &amp; tooling</div>
      <div class="math-eq">project_root/
├── pyproject.toml     ← single source of truth for deps + CLI names
├── mtdna_fm/          ← importable Python package
│   ├── model/
│   ├── data/
│   └── training/
└── tests/             ← pytest discovers test_*.py here</div>
      <div class="math-example"><strong>Key idea:</strong> separating code into modules means each piece can be tested independently. The CLI entry points defined in <code>[project.scripts]</code> map a name like <code>mtdna-train</code> directly to <code>mtdna_fm.scripts.train:main</code>.</div>
    </div>`
  };
})();
