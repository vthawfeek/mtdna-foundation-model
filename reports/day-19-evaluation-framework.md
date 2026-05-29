# Day 19: Evaluation Framework

## What was built
- `mtdna_fm/evaluation/haplogroup_eval.py`: accuracy, macro-F1, per-haplogroup precision/recall/F1, and confusion matrix; pure NumPy, no sklearn dependency
- `mtdna_fm/evaluation/variant_eval.py`: AUROC, AUPRC, per-variant-type breakdown (missense/tRNA/rRNA/D-loop/other) using rCRS gene coordinate ranges; hand-coded trapezoid-rule AUC
- `mtdna_fm/evaluation/viz.py`: UMAP embedding plot with phylogenetic colour scheme, ROC curve, confusion matrix heatmap, per-layer attention weight heatmap
- `mtdna_fm/evaluation/__init__.py`: clean public surface for all four evaluation utilities
- `mtdna_fm/scripts/evaluate.py`: `mtdna-evaluate` CLI — loads model, runs all evaluations, writes `reports/eval_summary.json` + `eval_haplogroup_detail.json` + `eval_variant_detail.json` + ROC figure; `--synthetic` flag for smoke-testing without a real checkpoint
- `notebooks/build_notebook_03.py` + `notebooks/03_finetuning_results.ipynb`: haplogroup confusion matrix, ROC curve, UMAP of genome embeddings, attention heatmap, per-variant-type breakdown
- `tests/test_evaluation.py`: 33 new tests covering haplogroup metrics, variant metrics, variant-type classification, and all viz functions
- Updated `tests/test_scripts.py`: replaced stale "not yet implemented" CLI tests with four real evaluate-CLI tests

## What was learned
- NumPy 2.x removed `np.trapz` — must use `np.trapezoid`; this is a common silent breakage when upgrading environments
- AUROC can be computed without sklearn by building the ROC curve from scratch and applying the trapezoidal rule; useful when the dependency graph must stay minimal
- AUPRC requires careful handling of the boundary: at recall=0 there is no well-defined precision, so anchoring at precision=1 at threshold=max avoids underestimating the curve
- Per-variant-type breakdowns give more interpretable diagnostic signal than a single scalar — tRNA variants and D-loop variants have very different pathogenicity signal profiles compared to missense variants
- The UMAP figure is the single most important diagnostic: phylogenetic topology emerging from sequence alone (no labels) validates that pre-training learned evolutionary structure, not just statistical regularities
- Writing evaluation code without sklearn removes the temptation to call `roc_auc_score` as a black box — understanding the curve construction is important for debugging edge cases (all-one-class, ties in scores)

## Key decisions
- No sklearn dependency in evaluation: evaluation code must be auditable and dependency-minimal for a foundation model that others may use; hand-coded metrics also make it easy to inspect what exactly is being computed
- `--synthetic` flag on the CLI: allows smoke-testing the full evaluate pipeline in CI without a trained checkpoint; the synthetic data mimics realistic score distributions so the figures are visually meaningful
- Variant-type classification by coordinate range (not annotation columns): avoids needing a VEP annotation at evaluation time; the rCRS gene coordinate ranges are stable and well-documented
- Downsample ROC/PR curve arrays to ≤200 points before JSON storage: full curves at 10,000-threshold resolution would make `eval_summary.json` unreadably large and slow to load in the notebook

## Verification

```
$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run pytest tests/ -m "not slow and not integration" -q
327 passed, 2 warnings in 46.84s

$ uv run mtdna-evaluate --model /tmp/fake --synthetic --output-dir /tmp/eval_test_day19
[evaluate] Running synthetic smoke-test evaluation …
[evaluate] Wrote /tmp/eval_test_day19/eval_summary.json

── Evaluation Summary ─────────────────────────
  Haplogroup accuracy : 0.6077
  Haplogroup macro-F1 : 0.6025
  Variant AUROC       : 0.8770
  Variant AUPRC       : 0.8628
───────────────────────────────────────────────

[evaluate] Saved ROC curve → /tmp/eval_test_day19/eval_roc_curve.png
[evaluate] Done.
```

## Next up
Day 20: Ancient DNA demonstration — embed Neanderthal (NC_011137.1) and Denisovan (FR695060.1) mtDNA zero-shot and place them on the human phylogeny UMAP.
