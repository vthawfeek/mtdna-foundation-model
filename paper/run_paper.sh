#!/usr/bin/env bash
# =============================================================================
# mtDNA-FM paper — master run script
# Acts as a PhD student: fixes data issues, trains models, evaluates, generates figures.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."  # always run from project root

LOG_DIR="paper/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/run.log"; }
die() { log "ERROR: $*"; exit 1; }

log "=== mtDNA-FM paper pipeline starting ==="
log "Working directory: $(pwd)"

# Check prerequisites
[ -d models/phase1_v1 ] || die "models/phase1_v1 not found. Pre-training must be complete."
[ -f data/processed/train.parquet ] || die "data/processed/train.parquet not found."
[ -f data/processed/test.parquet ]  || die "data/processed/test.parquet not found."

# =============================================================================
# STEP 1: Fix haplogroup label mapping (critical bug fix)
# Only 247 of 152k sequences had exact major-haplogroup labels.
# This adds a proper major_haplogroup column to all parquets.
# =============================================================================
log "STEP 1: Fixing haplogroup label mapping..."
uv run python paper/experiments/evaluation/fix_haplogroup_labels.py \
    2>&1 | tee "$LOG_DIR/step1_fix_labels.log"
log "STEP 1 complete."

# =============================================================================
# STEP 2: Prepare variant data for pathogenicity fine-tuning
# Downloads gnomAD/ClinVar if available; falls back to synthetic variants.
# =============================================================================
log "STEP 2: Preparing variant data..."
uv run python paper/experiments/evaluation/prepare_variant_data.py \
    2>&1 | tee "$LOG_DIR/step2_variant_data.log"
log "STEP 2 complete."

# =============================================================================
# STEP 3: Re-fine-tune haplogroup model with proper labels
# ~30 min on CPU (20 epochs × 1,263 sequences)
# =============================================================================
log "STEP 3: Fine-tuning haplogroup model (proper labels)..."
uv run mtdna-finetune \
    --task haplogroup \
    --config configs/finetuning_haplogroup_paper.yaml \
    2>&1 | tee "$LOG_DIR/step3_finetune_haplogroup.log"
log "STEP 3 complete."

# =============================================================================
# STEP 4: Re-fine-tune pathogenicity model with real data
# ~30 min on CPU (20 epochs)
# =============================================================================
log "STEP 4: Fine-tuning pathogenicity model..."
uv run mtdna-finetune \
    --task pathogenicity \
    --config configs/finetuning_pathogenicity_paper.yaml \
    2>&1 | tee "$LOG_DIR/step4_finetune_pathogenicity.log"
log "STEP 4 complete."

# =============================================================================
# STEP 5: Fine-tune heteroplasmy regression model
# ~15 min on CPU (5-fold CV, 15 epochs each)
# =============================================================================
log "STEP 5: Fine-tuning heteroplasmy regression model..."
uv run mtdna-finetune \
    --task heteroplasmy \
    --config configs/finetuning_heteroplasmy_paper.yaml \
    2>&1 | tee "$LOG_DIR/step5_finetune_heteroplasmy.log"
log "STEP 5 complete."

# =============================================================================
# STEP 6: Generate real model predictions
# Runs PEFT models on test data; saves per-sample parquets for CI computation.
# =============================================================================
log "STEP 6: Generating real model predictions..."
uv run python paper/experiments/evaluation/generate_predictions.py \
    2>&1 | tee "$LOG_DIR/step6_predictions.log"
log "STEP 6 complete."

# =============================================================================
# STEP 7: Create stratified haplogroup eval splits
# =============================================================================
log "STEP 7: Creating evaluation splits..."
uv run python paper/experiments/evaluation/create_eval_splits.py \
    2>&1 | tee "$LOG_DIR/step7_eval_splits.log"
log "STEP 7 complete."

# =============================================================================
# STEP 8: Bootstrap confidence intervals
# =============================================================================
log "STEP 8: Computing confidence intervals..."
uv run python paper/experiments/evaluation/compute_confidence_intervals.py \
    2>&1 | tee "$LOG_DIR/step8_ci.log"
log "STEP 8 complete."

# =============================================================================
# STEP 9: k-mer frequency baseline
# ~5 min on CPU
# =============================================================================
log "STEP 9: k-mer frequency baseline..."
uv run python paper/experiments/baselines/kmer_frequency_baseline.py \
    2>&1 | tee "$LOG_DIR/step9_kmer_baseline.log"
log "STEP 9 complete."

# =============================================================================
# STEP 10: Ancient DNA extended evaluation
# Fetches sequences from NCBI (needs internet)
# =============================================================================
log "STEP 10: Ancient DNA evaluation..."
uv run python paper/experiments/evaluation/ancient_dna_extended.py \
    2>&1 | tee "$LOG_DIR/step10_ancient_dna.log"
log "STEP 10 complete."

# =============================================================================
# STEP 11: Generate paper figures
# =============================================================================
log "STEP 11: Generating paper figures..."
uv run python paper/manuscript/figures/generate_figures.py \
    2>&1 | tee "$LOG_DIR/step11_figures.log"
log "STEP 11 complete."

# =============================================================================
# STEP 12: Auto-fill paper numbers
# =============================================================================
log "STEP 12: Filling paper placeholders..."
uv run python paper/fill_paper_numbers.py \
    2>&1 | tee "$LOG_DIR/step12_fill_numbers.log"
log "STEP 12 complete."

# =============================================================================
# PHASE 2 TRAINING (background — ~4h CPU)
# Start after Tier 1 is done. Results go into a v2 paper update.
# =============================================================================
log ""
log "=== Starting Phase 2 training in background (4-6h) ==="
log "Config: configs/pretraining_phase2.yaml"
log "Logs: $LOG_DIR/phase2_training.log"
log "When complete, re-run Steps 3-12 with base_model: models/phase2_v1"
log ""
nohup uv run mtdna-train \
    --config configs/pretraining_phase2.yaml \
    > "$LOG_DIR/phase2_training.log" 2>&1 &
PHASE2_PID=$!
log "Phase 2 training PID: $PHASE2_PID"
echo $PHASE2_PID > "$LOG_DIR/phase2.pid"

# =============================================================================
# Summary
# =============================================================================
log ""
log "=== Tier 1 pipeline complete ==="
log ""
log "Results:"
log "  reports/real_eval_summary.json     — real model metrics"
log "  reports/eval_haplogroup_predictions.parquet"
log "  reports/eval_variant_predictions.parquet"
log "  paper/experiments/evaluation/confidence_intervals.json"
log "  paper/experiments/baselines/results/"
log "  paper/manuscript/figures/fig{1..6}.{pdf,png}"
log "  paper/manuscript/main_filled.tex   — paper with numbers filled in"
log "  paper/numbers.json                 — all extracted numbers"
log ""
log "Next steps:"
log "  1. Review paper/manuscript/main_filled.tex"
log "  2. Manually fill remaining \\todo{} entries"
log "  3. cd paper/manuscript && pdflatex main_filled.tex && bibtex main_filled && pdflatex main_filled.tex"
log "  4. When Phase 2 finishes (check: tail -f $LOG_DIR/phase2_training.log):"
log "     bash paper/run_paper.sh  (re-runs with phase2 model)"
log ""
log "For overnight ablations, run separately:"
log "  nohup uv run python paper/experiments/ablations/ablate_circular_pe.py &"
log "  nohup uv run python paper/experiments/ablations/ablate_curriculum.py &"
