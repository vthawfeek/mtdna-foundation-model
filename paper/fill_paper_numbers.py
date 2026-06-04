"""
Auto-fill paper LaTeX \todo{X} placeholders with real results.

Reads from:
  - reports/eval_summary.json            (primary: real model evaluation, source="real")
  - reports/eval_haplogroup_detail.json  (per-class metrics, confusion matrix)
  - reports/eval_variant_detail.json     (ROC/PR data — absent until variant data prepared)
  - paper/experiments/evaluation/confidence_intervals.json  (bootstrap CIs)
  - paper/experiments/baselines/results/kmer_frequency_baseline.json
  - paper/experiments/baselines/results/dnabert2_baseline.json
  - paper/experiments/evaluation/ancient_dna_results.json

Replaces \todo{X} tokens in paper/manuscript/main.tex and writes:
  - paper/manuscript/main_filled.tex   (filled version for review)
  - paper/numbers.json                 (all extracted numbers for record)

Usage:
    uv run python paper/fill_paper_numbers.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MANUSCRIPT = Path("paper/manuscript/main.tex")
OUTPUT = Path("paper/manuscript/main_filled.tex")
NUMBERS_OUT = Path("paper/numbers.json")


# ---------------------------------------------------------------------------
# Load all result files
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def collect_numbers() -> dict:
    """Extract all paper-relevant numbers from result files."""
    nums = {}

    # Real eval summary — always load from reports/eval_summary.json (source: "real")
    real = load_json("reports/eval_summary.json")
    hap_detail = load_json("reports/eval_haplogroup_detail.json")
    var_detail = load_json("reports/eval_variant_detail.json")
    ci = load_json("paper/experiments/evaluation/confidence_intervals.json")
    kmer = load_json("paper/experiments/baselines/results/kmer_frequency_baseline.json")
    dna2 = load_json("paper/experiments/baselines/results/dnabert2_baseline.json")
    ancient = load_json("paper/experiments/evaluation/ancient_dna_results.json")
    ablation_pe = load_json("paper/experiments/ablations/results/circular_pe_ablation.json")
    ablation_cur = load_json("paper/experiments/ablations/results/curriculum_ablation.json")

    # Haplogroup accuracy (from real eval_summary.json, source="real")
    hap_summary = real.get("haplogroup", {})
    if hap_summary.get("accuracy") is not None:
        nums["haplogroup_accuracy"] = f"{hap_summary['accuracy']:.1%}"
    elif hap_detail.get("accuracy") is not None:
        nums["haplogroup_accuracy"] = f"{hap_detail['accuracy']:.1%}"

    if hap_detail.get("macro_f1") is not None:
        nums["haplogroup_macro_f1"] = f"{hap_detail['macro_f1']:.3f}"

    # Pathogenicity — no real evaluation data available; AUROC is not reported
    var_auroc = real.get("variant_pathogenicity", {}).get("auroc")
    if var_auroc is not None:
        nums["pathogenicity_auroc"] = f"{var_auroc:.3f}"
    else:
        nums["pathogenicity_auroc"] = "N/A (no labeled evaluation data)"

    if var_detail.get("auprc"):
        nums["pathogenicity_aupr"] = f"{var_detail['auprc']:.3f}"

    if var_detail.get("n_positive"):
        nums["pathogenicity_n_pos"] = str(var_detail["n_positive"])
    if var_detail.get("n_negative"):
        nums["pathogenicity_n_neg"] = str(var_detail["n_negative"])

    # Per-type AUROC
    per_type = var_detail.get("per_type", {})
    for vtype in ["tRNA", "missense", "rRNA", "d_loop"]:
        vdata = per_type.get(vtype, {})
        if vdata.get("auroc"):
            nums[f"auroc_{vtype}"] = f"{vdata['auroc']:.3f}"

    # CI
    path_ci = ci.get("pathogenicity", {})
    if path_ci.get("auroc", {}).get("ci_95"):
        ci_data = path_ci["auroc"]["ci_95"]
        nums["pathogenicity_auroc_ci"] = f"{ci_data['lower']:.3f}--{ci_data['upper']:.3f}"

    hap_ci = ci.get("haplogroup_accuracy", {})
    if hap_ci.get("bootstrap_ci"):
        bci = hap_ci["bootstrap_ci"]
        nums["haplogroup_accuracy_ci"] = f"{bci['lower']:.1%}--{bci['upper']:.1%}"

    # k-mer baselines
    k6_hap = kmer.get("k6_haplogroup", {})
    if k6_hap.get("status") == "completed":
        v = k6_hap["lr_5fold"]
        nums["kmer_haplogroup_acc"] = f"{v['mean']:.1%} ± {v['std']:.1%}"

    k6_path = kmer.get("k6_pathogenicity", {})
    if k6_path.get("status") == "completed":
        v = k6_path["auroc_5fold"]
        nums["kmer_pathogenicity_auroc"] = f"{v['mean']:.3f} ± {v['std']:.3f}"

    # DNABERT2 baselines
    d2_hap = dna2.get("haplogroup", {})
    if d2_hap.get("knn_5fold"):
        v = d2_hap["knn_5fold"]
        nums["dna2_haplogroup_acc"] = f"{v['mean']:.1%} ± {v['std']:.1%}"

    d2_path = dna2.get("pathogenicity", {}).get("position_token", {})
    if d2_path.get("auroc_5fold"):
        v = d2_path["auroc_5fold"]
        nums["dna2_pathogenicity_auroc"] = f"{v['mean']:.3f} ± {v['std']:.3f}"

    # Ancient DNA
    if ancient.get("modern_baseline_l2"):
        nums["modern_baseline_l2"] = f"{ancient['modern_baseline_l2']:.4f}"
    concordance = ancient.get("concordance_by_type", {})
    for atype in ["neanderthal", "denisovan", "early_modern"]:
        c = concordance.get(atype, {})
        if c.get("fraction") is not None:
            nums[f"ancient_concordance_{atype}"] = f"{c['fraction']:.0%}"
        if c.get("n"):
            nums[f"ancient_n_{atype}"] = str(c["n"])

    per_sample = ancient.get("per_sample", [])
    neanderthals = [s for s in per_sample if s.get("type") == "neanderthal"]
    if neanderthals:
        ratios = [s["ratio_vs_modern_baseline"] for s in neanderthals]
        nums["neanderthal_dist_ratio"] = f"{sum(ratios)/len(ratios):.2f}"

    # Ablation PE
    circ = ablation_pe.get("circular", {})
    if circ.get("zero_shot_knn"):
        v = circ["zero_shot_knn"]
        nums["ablation_circular_pe_acc"] = f"{v['mean']:.1%} ± {v['std']:.1%}"

    sinus = ablation_pe.get("sinusoidal", {})
    if sinus.get("zero_shot_knn"):
        v = sinus["zero_shot_knn"]
        nums["ablation_sinusoidal_pe_acc"] = f"{v['mean']:.1%} ± {v['std']:.1%}"

    # Curriculum ablation
    two_phase = ablation_cur.get("two_phase", {})
    if two_phase.get("zero_shot_knn"):
        v = two_phase["zero_shot_knn"]
        nums["ablation_two_phase_acc"] = f"{v['mean']:.1%} ± {v['std']:.1%}"

    single_phase = ablation_cur.get("single_phase", {})
    if single_phase.get("zero_shot_knn"):
        v = single_phase["zero_shot_knn"]
        nums["ablation_single_phase_acc"] = f"{v['mean']:.1%} ± {v['std']:.1%}"

    return nums


# ---------------------------------------------------------------------------
# Fill LaTeX placeholders
# ---------------------------------------------------------------------------

# Map todo labels found in main.tex to number keys
# Format: \todo{label} → nums[key]
TODO_MAP = {
    # Abstract / results numbers
    "X\\%": "haplogroup_accuracy",
    "X": "haplogroup_accuracy",
    "pathogenicity_auroc": "pathogenicity_auroc",  # will be N/A until real data prepared
    "50\\%": "ablation_circular_pe_acc",  # zero-shot claim — real
    # CI
    "X--Y": "pathogenicity_auroc_ci",
}


def fill_latex(tex: str, nums: dict) -> str:
    """Replace \todo{X} patterns with real numbers where mappings exist."""
    def replace_todo(m):
        content = m.group(1)
        # Try exact match first
        if content in TODO_MAP and TODO_MAP[content] in nums:
            return nums[TODO_MAP[content]]
        # Return as-is (still todo) but flag it
        return m.group(0)

    filled = re.sub(r"\\todo\{([^}]+)\}", replace_todo, tex)
    return filled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not MANUSCRIPT.exists():
        logger.error(f"Manuscript not found: {MANUSCRIPT}")
        return

    nums = collect_numbers()
    logger.info(f"Collected {len(nums)} numbers from result files")
    for k, v in sorted(nums.items()):
        logger.info(f"  {k}: {v}")

    # Save numbers for record
    NUMBERS_OUT.write_text(json.dumps(nums, indent=2))
    logger.info(f"Numbers saved → {NUMBERS_OUT}")

    # Fill LaTeX
    tex = MANUSCRIPT.read_text()
    filled = fill_latex(tex, nums)
    OUTPUT.write_text(filled)
    logger.info(f"Filled manuscript → {OUTPUT}")

    # Count remaining todos
    remaining = re.findall(r"\\todo\{[^}]+\}", filled)
    logger.info(f"{len(remaining)} \\todo{{}} placeholders still need manual filling")
    if remaining:
        print("\n=== Remaining \\todo{} placeholders ===")
        for t in remaining[:20]:
            print(f"  {t}")
        if len(remaining) > 20:
            print(f"  ... and {len(remaining) - 20} more")

    print(f"\n=== Extracted Numbers ===")
    for k, v in sorted(nums.items()):
        print(f"  {k:<40} {v}")
    print(f"\nFilled manuscript: {OUTPUT}")
    print("Open in LaTeX and manually fill remaining \\todo{{}} entries.")


if __name__ == "__main__":
    main()
