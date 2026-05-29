# Day 23: Documentation

## What was built

- `docs/01_data_pipeline.md` — complete coverage of HmtDB, NCBI, gnomAD, ClinVar, PhyloTree, and ancient DNA datasets; preprocessing steps with rationale; full DVC pipeline YAML
- `docs/02_tokenization.md` — 6-mer vocabulary construction, special token table, circular windowing mechanic, heteroplasmy channel design, vocabulary statistics
- `docs/03_architecture.md` — circular PE derivation from first principles with the standard sinusoidal baseline and why it fails, het projection design, parameter count breakdown (~6M trainable), comparison table against DNABERT2 and HyenaDNA on mtDNA-relevant axes
- `docs/04_pretraining.md` — two-phase curriculum rationale, gradient accumulation math, expected MLM loss curve (step 0 to 50k), masking strategy and D-loop blacklist, monitoring with MLflow, CPU/GPU timing estimates
- `docs/05_finetuning_and_evaluation.md` — three downstream tasks with LoRA configuration choices and rationale, baseline tables for haplogroup and pathogenicity tasks, ancient DNA zero-shot setup, known limitations (population bias, indels not supported, haplogroup resolution)

## What was learned

- Documentation written from first principles produces different content than documentation written as a post-hoc summary. Walking through the circular PE derivation from the standard sinusoidal formula to the circular version reveals exactly what the formula is doing and why — this is not obvious from reading the code.
- The comparison table against DNABERT2 and HyenaDNA clarifies the model's value proposition: the features that matter for mtDNA (circular topology, heteroplasmy) are absent from both, and domain-specific pre-training compensates for smaller model size.
- Writing LoRA configuration rationale explicitly (`r=8` for haplogroup with 47k examples, `r=4` for pathogenicity with 7k examples) makes the relationship between dataset size, task complexity, and LoRA rank concrete.
- The known limitations section is as important as the results section. Documenting population bias, lack of indel support, and regression confidence limits sets appropriate expectations and shows the project is honest about what the model doesn't do.

## Key decisions

- **Each doc covers exactly one topic**: No overlap between the five docs. Data pipeline does not explain tokenization; tokenization does not explain the model. This makes each doc independently useful and avoids the maintenance problem of updating the same fact in two places.
- **Derivations written from first principles**: The circular PE section walks through why standard sinusoidal PE fails before introducing the fix. This is more useful to a reader implementing a similar model than showing only the final formula.
- **Baseline comparisons use concrete numbers**: The haplogroup and pathogenicity tables compare majority class, k-mer frequency, logistic regression, and the fine-tuned model. These numbers are meaningful; a table with only the fine-tuned model result tells you nothing about whether the model is actually adding value.
- **Limitations section is not minimized**: Population bias, missing indel support, and known R² floor for heteroplasmy regression are stated directly. A model card or documentation that omits these would be misleading.

## Verification

```bash
$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run pytest tests/ -m "not slow and not integration" -q
346 passed, 5 warnings in 65.96s

$ ls docs/
01_data_pipeline.md  02_tokenization.md  03_architecture.md
04_pretraining.md    05_finetuning_and_evaluation.md  figures/
```

All five documentation files created. No Python changes were made, so linting and tests pass unchanged from Day 22.

## Next up

Day 24: Complete `dvc.yaml` with all pipeline stages (download through evaluate), each with deps/outs/params/metrics, so `dvc repro` reproduces the full experiment from raw data to evaluation metrics.
