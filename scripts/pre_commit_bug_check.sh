#!/bin/bash
# Critical bug scanner — runs before every commit.
# Blocks commits that would introduce synthetic metrics into the production pipeline
# or that contain stale evaluation data.
#
# Install:
#   cp scripts/pre_commit_bug_check.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit

set -euo pipefail

FAIL=0
WARN=0

echo "── mtDNA-FM pre-commit bug check ──────────────────────────────"

# Check 1: --synthetic not in dvc.yaml pipeline commands
if grep -q "\-\-synthetic" dvc.yaml 2>/dev/null; then
    echo "BLOCKED [1]: dvc.yaml contains --synthetic in a pipeline stage."
    echo "  Remove it — synthetic flags belong in pytest only, not in DVC."
    FAIL=1
else
    echo "PASS    [1]: dvc.yaml has no --synthetic flag"
fi

# Check 2: seeded RNG (seed=0) not in production code outside tests/
if grep -rn "default_rng(0)\|default_rng(seed=0)\|manual_seed(0)" mtdna_fm/ 2>/dev/null | grep -v "^Binary"; then
    echo "BLOCKED [2]: seeded RNG with seed=0 found in mtdna_fm/ (production code)."
    echo "  Move to tests/conftest.py. Production code must not generate fake data."
    FAIL=1
else
    echo "PASS    [2]: no fixed-seed RNG in production code"
fi

# Check 3: eval_summary.json must not have source: synthetic
if [ -f "reports/eval_summary.json" ]; then
    SOURCE=$(python3 -c "import json; d=json.load(open('reports/eval_summary.json')); print(d.get('source','missing'))" 2>/dev/null || echo "unreadable")
    if [ "$SOURCE" = "synthetic" ]; then
        echo "BLOCKED [3]: reports/eval_summary.json has source: synthetic."
        echo "  Run real evaluation: uv run mtdna-evaluate --model models/finetune_haplogroup_paper --output-dir reports"
        FAIL=1
    elif [ "$SOURCE" = "missing" ] || [ "$SOURCE" = "unreadable" ]; then
        echo "WARN    [3]: reports/eval_summary.json missing source field (run evaluation to add it)"
        WARN=1
    else
        echo "PASS    [3]: eval_summary.json source=${SOURCE}"
    fi
else
    echo "SKIP    [3]: no eval_summary.json yet"
fi

# Check 4: class collapse warning in eval results
if [ -f "reports/eval_summary.json" ]; then
    COLLAPSE=$(python3 -c "
import json, sys
d = json.load(open('reports/eval_summary.json'))
per_class = d.get('haplogroup', {}).get('per_class', {})
if per_class:
    nonzero = sum(1 for v in per_class.values() if isinstance(v, dict) and v.get('f1', 0) > 0.01)
    total = len(per_class)
    if nonzero < total // 2:
        print(f'collapse:{nonzero}/{total}')
    else:
        print(f'ok:{nonzero}/{total}')
else:
    print('skip')
" 2>/dev/null || echo "skip")
    if [[ "$COLLAPSE" == collapse* ]]; then
        echo "WARN    [4]: class collapse detected (${COLLAPSE#collapse:} classes with F1>0.01)"
        echo "  Consider adding inverse-frequency class weights to CrossEntropyLoss before reporting metrics."
        WARN=1
    elif [[ "$COLLAPSE" == ok* ]]; then
        echo "PASS    [4]: class balance ok (${COLLAPSE#ok:} classes active)"
    else
        echo "SKIP    [4]: no per-class data to check"
    fi
fi

# Check 5: no 0.877 or 60.8 in published content (these were the known synthetic artifacts)
SUSPECTS=$(grep -rn "0\.877\|60\.8%" reports/ README.md models/*/README.md 2>/dev/null | grep -v "synthetic\|correction\|was fake\|artifact" || true)
if [ -n "$SUSPECTS" ]; then
    echo "BLOCKED [5]: known synthetic metric values (0.877 or 60.8%) found in published content:"
    echo "$SUSPECTS" | head -5
    FAIL=1
else
    echo "PASS    [5]: no known synthetic metric values in published content"
fi

echo "───────────────────────────────────────────────────────────────"

if [ $FAIL -ne 0 ]; then
    echo "COMMIT BLOCKED: fix the issues above before committing."
    exit 1
elif [ $WARN -ne 0 ]; then
    echo "WARNINGS present — commit allowed but review the warnings above."
    exit 0
else
    echo "All checks passed. Commit allowed."
    exit 0
fi
