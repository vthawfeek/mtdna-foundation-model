# Day 18: Heteroplasmy Regression

## What was built
- `mtdna_fm/model/model.py`: `MtDNAForHeteroplasmyRegression` — regression model with `Linear(256,64)->GELU->Linear(64,1)->Sigmoid` head on variant-position token hidden state, Huber loss (delta=0.1)
- `mtdna_fm/model/model.py`: `HeteroplasmyRegressionOutput` dataclass
- `mtdna_fm/model/__init__.py`: exported `MtDNAForHeteroplasmyRegression` and `HeteroplasmyRegressionOutput`
- `mtdna_fm/scripts/finetune.py`: `HeteroplasmyRegressionDataset` — 512-token windowed dataset with mean het_level as float target; synthetic fallback when parquet absent
- `mtdna_fm/scripts/finetune.py`: `finetune_heteroplasmy()` — 5-fold cross-validation loop reporting R-squared and Spearman rho per fold; saves final model on all data
- `mtdna_fm/scripts/finetune.py`: wired `heteroplasmy` task in CLI dispatch (was `typer.Exit(code=1)` placeholder)
- `configs/finetuning_heteroplasmy.yaml`: training config (LoRA r=4, n_folds=5, huber_delta=0.1, batch=16, 15 epochs)
- `tests/test_model.py`: 12 tests for `MtDNAForHeteroplasmyRegression` (shape, range, loss, gradients, freeze/unfreeze, LoRA, architecture)
- `tests/test_scripts.py`: 8 tests for `HeteroplasmyRegressionDataset` (synthetic fallback, parquet loading, window size, label range) and CLI dispatch

## What was learned
- **Huber vs MSE**: Huber loss (delta=0.1) down-weights the squared penalty for large residuals, making it robust to outlier heteroplasmy estimates from gnomAD (where "mean het level" across >=50 carriers is still a noisy statistic). MSE would let a handful of badly-estimated carriers dominate the gradient.
- **5-fold CV vs held-out test set**: With ~1,000 data points, a fixed 80/20 split wastes 20% of an already small dataset. 5-fold CV uses all data for both training and evaluation; the variance across folds tells you how stable the result is, which matters more at this scale than a slightly cleaner evaluation procedure.
- **Spearman vs R-squared for regression evaluation**: R-squared measures absolute scale accuracy (sensitive to systematic bias); Spearman rho measures ranking accuracy (whether variants with higher true het levels also get higher predictions). For biological applications, the ranking matters more — a model that correctly ranks variants by constraint is useful even if its absolute predictions are off.
- **Variant-token vs CLS hidden state**: Heteroplasmy level is a local property of the variant's nucleotide context (codon position, tRNA loop structure, rRNA accessibility), not a global genome property. Using the variant-token hidden state is the correct inductive bias here, same as for pathogenicity.
- **LoRA r=4 for small datasets**: ~1,000 training examples warrant heavier regularisation than haplogroup fine-tuning (~10,000 windows). r=4 (vs r=8 for haplogroup) limits the effective rank of the weight updates, preventing overfitting on the training folds.

## Key decisions
- **Huber delta=0.1**: gnomAD het level estimates are bounded [0,1] and residuals rarely exceed 0.5; delta=0.1 keeps the transition from squared to linear loss at a biologically meaningful scale.
- **Sigmoid output head**: bounds predictions to [0,1] to match the physical range of heteroplasmy, avoiding need for clipping at inference time.
- **5-fold CV (not train/val split)**: dataset is too small (~1,000 points) to afford a held-out split; cross-validation maximises both training signal and evaluation reliability.
- **HeteroplasmyRegressionDataset synthetic fallback**: matches the PathogenicityVariantDataset pattern — enables full test coverage without shipping real gnomAD data in the repo.
- **Spearman > 0.30 as "real signal" threshold**: a modest bar, but meaningful for a regression on noisy gnomAD estimates from a model not trained on heteroplasmy data.

## Verification
```
$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run pytest tests/ -m "not slow and not integration" -q
294 passed, 2 warnings in 6.84s
```

274 → 294 tests (+20 new tests across test_model.py and test_scripts.py).

## Next up
Day 19: evaluation framework — haplogroup accuracy/F1, pathogenicity AUROC/AUPRC, UMAP visualisation, `mtdna-evaluate` CLI.
