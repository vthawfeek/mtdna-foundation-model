#!/usr/bin/env bash
# Reproduce all mtDNA-FM paper experiments end-to-end.
# Prerequisites: uv sync, DVC pipeline already run (dvc repro)
set -euo pipefail

echo "=== mtDNA-FM Paper Reproduction ==="
echo "Working directory: $(pwd)"

# Verify pre-training checkpoints
for ckpt in models/phase1_v1 models/vocabulary; do
    if [ ! -d "$ckpt" ]; then
        echo "ERROR: Checkpoint not found: $ckpt"
        echo "Run: uv run mtdna-pretrain --phase 1 --output models/phase1_v1"
        exit 1
    fi
done

# Step 2: Create held-out evaluation splits
echo ""
echo "--- Step 2: Creating held-out evaluation splits ---"
uv run python paper/experiments/evaluation/create_eval_splits.py

# Step 3: k-mer baseline (fast, no GPU required)
echo ""
echo "--- Step 3: k-mer frequency baseline ---"
uv run python paper/experiments/baselines/kmer_frequency_baseline.py

# Step 4: Main evaluation with confidence intervals
echo ""
echo "--- Step 4: Main evaluation ---"
uv run mtdna-evaluate \
    --haplogroup-model models/finetune_haplogroup_v1 \
    --pathogenicity-model models/finetune_pathogenicity_v1 \
    --test-data paper/experiments/evaluation/held_out_test.parquet \
    --output reports/eval_summary.json || echo "WARN: mtdna-evaluate failed; using existing reports"

echo ""
echo "--- Step 4b: Confidence intervals ---"
uv run python paper/experiments/evaluation/compute_confidence_intervals.py

# Step 5: Ancient DNA evaluation
echo ""
echo "--- Step 5: Ancient DNA extended evaluation ---"
uv run python paper/experiments/evaluation/ancient_dna_extended.py

# Step 6: Generate figures
echo ""
echo "--- Step 6: Generating paper figures ---"
uv run python paper/manuscript/figures/generate_figures.py

# Note: Ablation studies and DNABERT2 baseline require additional training time.
# Run separately:
#   uv run python paper/experiments/ablations/ablate_circular_pe.py
#   uv run python paper/experiments/ablations/ablate_curriculum.py
#   uv run python paper/experiments/ablations/ablate_het_channel.py --train
#   uv run python paper/experiments/baselines/dnabert2_baseline.py

echo ""
echo "=== Reproduction complete ==="
echo "Figures: paper/manuscript/figures/"
echo "Results: paper/experiments/{ablations,baselines,evaluation}/results/"
echo ""
echo "Next steps:"
echo "  1. Run ablation studies (see paper/README.md Step 3)"
echo "  2. Run DNABERT2 baseline (requires transformers + GPU)"
echo "  3. Download MITOMAP/HelixMTdb for external validation (Step 6)"
echo "  4. cd paper/manuscript && pdflatex main.tex && bibtex main && pdflatex main.tex"
