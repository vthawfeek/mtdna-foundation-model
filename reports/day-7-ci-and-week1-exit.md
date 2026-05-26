# Day 7: CI Hardening and Week 1 Exit Criteria

## What was built

- `.github/workflows/ci.yml` — two-job CI pipeline: `lint` (ruff check + ruff format --check) and `test` (pytest, skipping `@pytest.mark.slow` and `@pytest.mark.integration`)
- `README.md` — CI badge and HuggingFace badge updated with correct GitHub username (`vthawfeek`)
- Applied `ruff format` across 9 files that had inconsistent whitespace but correct logic (no semantic changes)

## What was learned

- **Why `ruff format --check` is a separate CI gate from `ruff check`:** `ruff check` catches semantic issues (unused imports, undefined names, complexity violations); `ruff format --check` catches whitespace and style. Both fail CI because unformatted code in PRs becomes a source of noise in diffs that obscures real changes.
- **Why CI is a Week 1 task, not a Week 2 afterthought:** Without CI, every subsequent day risks silently breaking something that was already working. The test suite runs on every push, so regression detection is automatic from this point forward.
- **Week 1 exit criteria verified:** All five criteria checked out — see Verification section.

## Key decisions

- **Two-job CI, not one:** `lint` and `test` are separate jobs so that a format failure doesn't hide a test failure and vice versa. Each communicates a distinct signal.
- **`ruff format` applied before hardening the CI gate:** The CI now enforces formatting. Running format first avoids immediate CI failures on all prior commits and keeps the history clean going forward.
- **Coverage report in CI (`--cov=mtdna_fm`) but not a gate:** Coverage is computed and displayed in CI output but doesn't fail the build. A coverage floor is a Week 2 concern once the model architecture tests exist; requiring it now would be premature.

## Verification

Week 1 exit criteria:

```
$ uv run pytest tests/ -v -q
101 passed in 2.31s

$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run ruff format --check mtdna_fm/ tests/
26 files already formatted

$ python -c "
from mtdna_fm.tokenizer.vocabulary import KmerVocabulary
v = KmerVocabulary.build(k=6)
assert len(v) == 4102
kmer = 'ATGCAT'
assert v.decode(v.encode(kmer)) == kmer
print(f'Vocabulary size: {len(v)}, roundtrip OK')
"
Vocabulary size: 4102, roundtrip OK

$ python -c "
import pandas as pd
df = pd.read_parquet('data/processed/train.parquet')
print(f'Train rows: {len(df)}')
assert (df.sequence.str.len() == 16569).all(), 'Length check failed'
print('All sequences exactly 16569 bp: OK')
"
Train rows: 152484
All sequences exactly 16569 bp: OK
```

GitHub CI badge green after push (both `lint` and `test` jobs pass on ubuntu-latest with Python 3.11).

## Next up

Day 8: model architecture — `MtDNAConfig`, `MtDNACircularPositionalEncoding`, `MtDNAEmbeddings`, and the full encoder stack (`MtDNAModel` + `MtDNAForMaskedModeling`).
