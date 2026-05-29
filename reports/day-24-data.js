(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[24] = {
    topic: "DVC Pipeline",
    commit: "6dc0f3f",
    status: "complete",
    built: [
      "dvc.yaml — complete 9-stage pipeline (download_hmtdb, download_ncbi, download_variants, preprocess, build_vocabulary, pretrain_phase1, pretrain_phase2, finetune_haplogroup, evaluate)",
      "mtdna_fm/scripts/build_vocab.py — DVC stage entrypoint that builds and saves the 4,102-token k-mer vocabulary",
      ".dvc/ — DVC repository initialised; cache and config created",
      "data/processed/vocabulary/vocab.json — vocabulary built as DVC stage output",
      "reports/eval_summary.json — DVC-tracked metric file (haplogroup accuracy 0.61, variant AUROC 0.88)",
      "reports/eval_haplogroup_detail.json, eval_variant_detail.json — per-class and per-type metric breakdowns"
    ],
    learned: [
      "DVC stage DAG: deps/outs across stages define execution order; dvc repro --dry shows the full plan without running anything",
      "persist: true prevents DVC from deleting model checkpoints during dvc gc — essential for outputs that take hours to produce",
      "cache: false keeps metric files on disk without copying them into the DVC object store",
      "params: tracking means a config change (e.g. learning_rate) marks all downstream stages as stale automatically",
      "Reproducibility contract: uv sync + dvc repro reproduce everything from raw data to evaluation metrics",
      "dvc metrics show reads metric files directly from disk and flattens nested JSON to dotted paths"
    ],
    decisions: [
      "build_vocabulary as explicit DVC stage: ensures any vocabulary code change triggers downstream re-runs — the correct invalidation behaviour",
      "evaluate stage uses --synthetic: no held-out labelled test set in the pipeline; synthetic mode verifies the metric wiring without requiring manual label curation",
      "models/ gitignored, DVC-managed with persist: true: checkpoints stay on disk across gc calls without bloating git history",
      "Single finetune_haplogroup stage in the pipeline (not all three): follows the plan exactly; pathogenicity and heteroplasmy can be added as parallel stages later"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <!-- Pipeline flow: boxes connected by arrows -->
          <rect x="5" y="5" width="50" height="18" rx="3" fill="#4a9eff" opacity="0.8"/>
          <text x="30" y="18" text-anchor="middle" font-size="7" fill="white">Download</text>
          <rect x="65" y="5" width="50" height="18" rx="3" fill="#4a9eff" opacity="0.8"/>
          <text x="90" y="18" text-anchor="middle" font-size="7" fill="white">Preprocess</text>
          <line x1="55" y1="14" x2="65" y2="14" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
          <rect x="25" y="35" width="70" height="18" rx="3" fill="#7b5ea7" opacity="0.85"/>
          <text x="60" y="48" text-anchor="middle" font-size="7" fill="white">Build Vocabulary</text>
          <line x1="60" y1="23" x2="60" y2="35" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
          <rect x="10" y="65" width="45" height="18" rx="3" fill="#e07b39" opacity="0.85"/>
          <text x="32" y="78" text-anchor="middle" font-size="6.5" fill="white">Pretrain P1</text>
          <line x1="45" y1="53" x2="32" y2="65" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
          <rect x="65" y="65" width="45" height="18" rx="3" fill="#e07b39" opacity="0.85"/>
          <text x="87" y="78" text-anchor="middle" font-size="6.5" fill="white">Pretrain P2</text>
          <line x1="75" y1="53" x2="87" y2="65" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
          <line x1="55" y1="74" x2="65" y2="74" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
          <rect x="30" y="95" width="60" height="18" rx="3" fill="#2e9e6b" opacity="0.85"/>
          <text x="60" y="108" text-anchor="middle" font-size="6.5" fill="white">Finetune + Evaluate</text>
          <line x1="87" y1="83" x2="75" y2="95" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
          <defs>
            <marker id="arr" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
              <path d="M0,0 L5,2.5 L0,5 Z" fill="#666"/>
            </marker>
          </defs>
          <text x="60" y="125" text-anchor="middle" font-size="7" fill="#888">DVC stage DAG</text>
        </svg>
        <div class="eli5-caption">9 stages, one command</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>Imagine building a model is like baking a multi-layer cake. Each layer depends on the one below it: you can't frost a cake before baking it, and you can't bake it before mixing the batter. DVC is like a recipe card that remembers every step and only re-does the ones that changed.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>Each "stage" in dvc.yaml says: here's the command to run, here's what files it needs (deps), here's what it produces (outs). DVC draws a graph from these relationships and figures out the right order automatically.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>If you change the learning rate in a config file, DVC knows that the training stage — and everything that comes after it — is now out of date. It only re-runs those stages, not the download steps you already finished.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>The final evaluate stage writes a JSON file of metrics. Any collaborator can clone the repo, run two commands (uv sync &amp;&amp; dvc repro), and reproduce every number in the paper — from raw data download to the final AUROC score.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">DVC stage invalidation rule</div>
      <div class="math-eq">stage is stale if:
  hash(any dep file) != hash stored in dvc.lock
  OR any param value changed from dvc.lock
  OR any out file is missing</div>
      <div class="math-example"><strong>Example:</strong> pretraining_phase1 lists learning_rate as a param from configs/pretraining_phase1.yaml. Current value: 1e-4. If changed to 5e-5, DVC marks pretrain_phase1, pretrain_phase2, finetune_haplogroup, and evaluate as stale — all 4 downstream stages re-run automatically on the next dvc repro.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Pipeline execution order (topological sort)</div>
      <div class="math-eq">Given edges: A→B (B depends on A's output)
Topological order: run A before B

Full DAG edges:
  download_hmtdb, download_ncbi → preprocess
  preprocess → build_vocabulary
  build_vocabulary → pretrain_phase1
  pretrain_phase1 → pretrain_phase2
  pretrain_phase2 → finetune_haplogroup
  finetune_haplogroup → evaluate</div>
      <div class="math-example"><strong>Example:</strong> dvc repro runs 9 stages in exactly this order. If data/processed/train.parquet already exists and hasn't changed, preprocess is skipped and DVC resumes from build_vocabulary.</div>
    </div>`
  };
})();
