# Day 1: Project Scaffold

## What was built

- `pyproject.toml` — full project config: hatchling build backend, all runtime and dev dependencies pinned, 5 CLI entry points, ruff and pytest configured
- `uv.lock` — deterministic dependency lockfile (reproducible installs across machines)
- `mtdna_fm/` — Python package with 8 submodules: `model`, `tokenizer`, `data`, `training`, `inference`, `evaluation`, `scripts`
- `mtdna_fm/scripts/download.py` — Typer CLI stub for dataset downloads (`mtdna-download`)
- `mtdna_fm/scripts/preprocess.py` — Typer CLI stub for preprocessing (`mtdna-preprocess`)
- `mtdna_fm/scripts/train.py` — Typer CLI stub for pre-training (`mtdna-train`)
- `mtdna_fm/scripts/finetune.py` — Typer CLI stub for fine-tuning (`mtdna-finetune`)
- `mtdna_fm/scripts/evaluate.py` — Typer CLI stub for evaluation (`mtdna-evaluate`)
- `tests/conftest.py` — `synthetic_sequence` (100 bp) and `synthetic_sequence_16569` (16,569 bp) fixtures
- `.github/workflows/ci.yml` — GitHub Actions: lint (ruff check + format) and test (pytest, skipping slow/integration) jobs on push and PR
- `.gitignore` — excludes data/, models/, mlruns/, .venv/ from git (these are tracked by DVC or too large)
- `README.md` — skeleton with architecture description, expected results table, and quick-start

## What was learned

- **Packaging first:** defining CLI entry points in `pyproject.toml` means every script is callable as `mtdna-train` rather than `python -m mtdna_fm.scripts.train`. This makes the interface stable regardless of where the code is installed.
- **uv lockfiles:** unlike `requirements.txt`, the lockfile pins every transitive dependency, not just direct ones. This means a fresh `uv sync` on any machine gets exactly the same environment, which is what makes DVC pipeline reproducibility reliable.
- **Typer over argparse:** Typer generates `--help` text automatically from type annotations and docstrings. The stubs are functional from day one even though the logic is not yet implemented.
- **Test markers:** defining `slow` and `integration` markers in `pyproject.toml` from the start means CI can skip expensive tests without modifying the test files themselves.

## Key decisions

- **hatchling over setuptools:** hatchling is faster and does not require a `setup.py`. It reads everything from `pyproject.toml`, keeping configuration in one place.
- **dev extras rather than a separate requirements-dev.txt:** `uv sync --extra dev` installs everything in one command. No separate file to keep in sync.
- **5 dedicated CLI entry points:** instead of one `mtdna` command with subcommands, each pipeline stage gets its own top-level command. This mirrors the scFM pattern and makes DVC stage configuration simpler (each stage just calls one command).
- **.gitignore excludes data and models:** these directories will be tracked by DVC, not git. Keeping them out of git from day one avoids the common mistake of accidentally committing a large dataset.

## Verification

```
$ uv run pytest tests/ -v
0 tests collected — no errors

$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run ruff format --check mtdna_fm/ tests/
15 files already formatted

$ uv run python -c "import mtdna_fm; print(mtdna_fm.__version__)"
0.1.0
```

Git commit: `12d614c feat: project scaffold for mtDNA foundation model`

## Next up

Day 2: Build `KmerVocabulary` and `tokenize_sequence` — the reusable tokenizer that works for any short circular genome, with `save_pretrained`/`from_pretrained` following HuggingFace conventions and a full `tests/test_tokenizer.py` suite.
