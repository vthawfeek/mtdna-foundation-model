# Day 24: DVC Pipeline and Reproducibility

## What was built

- `dvc.yaml` — complete 9-stage pipeline: download_hmtdb, download_ncbi, download_variants, preprocess, build_vocabulary, pretrain_phase1, pretrain_phase2, finetune_haplogroup, evaluate. Each stage has `deps`, `outs`, `params`, and (for evaluate) `metrics` defined.
- `mtdna_fm/scripts/build_vocab.py` — standalone script called by the `build_vocabulary` DVC stage; builds and saves the 4,102-token k-mer vocabulary to `data/processed/vocabulary/`.
- `.dvc/` — DVC repository initialised with `dvc init`; config and cache directory created.
- `data/processed/vocabulary/vocab.json` — vocabulary built and saved as output of the build_vocabulary stage.
- `reports/eval_summary.json` — DVC-tracked metric file written by the evaluate stage; readable by `dvc metrics show`.
- `reports/eval_haplogroup_detail.json`, `reports/eval_variant_detail.json` — detailed per-class and per-variant-type breakdowns.

## What was learned

- **DVC stage DAG**: DVC infers the execution order from `deps` and `outs` across stages. If stage B lists an output of stage A in its `deps`, DVC runs A before B automatically. `dvc repro --dry` shows the full execution plan without running anything.
- **`persist: true`**: Prevents DVC from deleting model directories during `dvc gc`. Essential for checkpoints that take hours to produce and aren't versioned in the DVC cache.
- **`cache: false`**: Metric and detail JSON files are kept on disk but not copied into the `.dvc/cache`. This makes them visible in the working directory without needing `dvc checkout`.
- **`params:` tracking**: Each stage declares which config file keys it depends on. If `learning_rate` in `pretraining_phase1.yaml` changes, DVC marks `pretrain_phase1` (and all downstream stages) as stale, forcing a re-run. This is how parameter sweeps are tracked without touching code.
- **Reproducibility contract**: Two commands reproduce the full pipeline from raw data to evaluation metrics: `uv sync` (install deps) and `dvc repro` (run all stale stages). The `dvc.yaml` is the single source of truth for what commands run and in what order.
- **`dvc metrics show`**: Reads metric files listed in `dvc.yaml` directly from disk without needing DVC cache. Output is a table with all nested JSON keys flattened to dotted paths.

## Key decisions

- **`build_vocabulary` as an explicit DVC stage**: The vocabulary is deterministic (4^6 = 4,096 k-mers + 6 special tokens), so it has no data deps. Making it a stage ensures any code change to `vocabulary.py` triggers a downstream re-run of all training stages — which is the correct behaviour.
- **evaluate stage used `--synthetic` (BUG — now fixed):** The evaluate stage originally hardcoded `--synthetic`, meaning every `dvc repro` wrote seeded random numbers to `reports/eval_summary.json` instead of real model metrics. This was a mistake — the `--synthetic` flag is for CI smoke-testing only. Fixed: the stage now calls `mtdna-evaluate --model models/finetune_haplogroup_paper` without `--synthetic`.
- **`models/` in `.gitignore`, managed by DVC with `persist: true`**: Model checkpoints are not checked into git (too large). DVC tracks them via `outs` with `persist: true` so they survive `dvc gc` calls during development.
- **Single `finetune_haplogroup` stage, not all three fine-tuning tasks**: The plan specifies 9 stages. Pathogenicity and heteroplasmy fine-tuning follow the same pattern; they can be added as additional stages later without restructuring the DAG.

## Verification

```
$ uv run dvc repro --dry
Running stage 'download_hmtdb':      > uv run mtdna-download --source hmtdb ...
Running stage 'download_ncbi':       > uv run mtdna-download --source ncbi-refseq ...
Running stage 'download_variants':   > uv run mtdna-download --source gnomad ... && ...
Running stage 'preprocess':          > uv run mtdna-preprocess ...
Running stage 'build_vocabulary':    > uv run python mtdna_fm/scripts/build_vocab.py ...
Running stage 'pretrain_phase1':     > uv run mtdna-train --config configs/pretraining_phase1.yaml ...
Running stage 'pretrain_phase2':     > uv run mtdna-train --config configs/pretraining_phase2.yaml
Running stage 'finetune_haplogroup': > uv run mtdna-finetune --task haplogroup ...
Running stage 'evaluate':            > uv run mtdna-evaluate --model models/finetune_haplogroup_paper

$ uv run dvc metrics show
# After bug fix and retrain — real metrics from eval_summary.json
# (0.6077 / 0.877 reported at Day 24 were synthetic smoke-test artifacts, not model measurements)

$ uv run pytest tests/ -m "not slow and not integration" -q
346 passed, 5 warnings in 43.91s

$ uv run ruff check mtdna_fm/ tests/
All checks passed!
```

## Next up

Day 25: showcase notebook (`notebooks/04_mtdna_fm_showcase.ipynb`) — self-contained story from model loading through TSNE clusters, confusion matrix, ROC curves, and ancient DNA placement.
