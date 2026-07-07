"""
Region-only baseline for the pathogenicity task.

Reproduces the manuscript claim (Section 4.2, "The overall metric is confounded by
gene region"): a classifier that uses ONLY the five-way gene-region label of each
variant --- with no sequence embedding whatsoever --- matches the mtDNA-FM model's
ranking performance. This shows the reported AUROC (0.777) is largely attributable
to the differing gene-region composition of the pathogenic vs. benign sets, not to
variant-level representation.

The per-region pathogenic/benign counts are taken directly from the model's own
report (reports/zeroshot_pathogenicity_knn.json). A region-only classifier cannot
distinguish variants within a region, so reconstructing per-variant (region, label)
pairs from these counts is exact for this baseline.

Run: uv run python paper/experiments/evaluation/region_baseline_pathogenicity.py

Verified result (2026-07-07):
  IN-SAMPLE region-only : AUROC=0.7788  AUPRC=0.4776
  5-FOLD-CV region-only : AUROC=0.7541  AUPRC=0.4628
  MODEL (from report)   : AUROC=0.7772  AUPRC=0.4398   random AUPRC=0.2197
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports" / "zeroshot_pathogenicity_knn.json"


def main() -> None:
    d = json.loads(REPORT.read_text())
    pt = d["per_type"]
    regions = {r: (pt[r]["n_pos"], pt[r]["n_neg"]) for r in
               ["d_loop", "rRNA", "missense", "tRNA", "other"]}

    # Reconstruct per-variant (region, label). Exact for a region-only classifier.
    x_region: list[str] = []
    y: list[int] = []
    for r, (n_pos, n_neg) in regions.items():
        x_region += [r] * (n_pos + n_neg)
        y += [1] * n_pos + [0] * n_neg
    x_region = np.array(x_region)
    y = np.array(y)

    prior = {r: n_pos / (n_pos + n_neg) for r, (n_pos, n_neg) in regions.items()}

    # In-sample region prior (ceiling on what region alone can explain)
    s_in = np.array([prior[r] for r in x_region])
    print("regions (n_pos, n_neg):", regions)
    print(f"n_pos={int(y.sum())} n_neg={int((1 - y).sum())} total={len(y)}")
    print("P(pathogenic | region):",
          {r: round(v, 3) for r, v in sorted(prior.items(), key=lambda kv: kv[1])})
    print(f"IN-SAMPLE region-only : AUROC={roc_auc_score(y, s_in):.4f}  "
          f"AUPRC={average_precision_score(y, s_in):.4f}")

    # 5-fold stratified CV region prior (matches the model's exact protocol)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, te in skf.split(x_region, y):
        fold_prior = {
            r: (y[tr][x_region[tr] == r].mean()
                if (x_region[tr] == r).sum() > 0 else y[tr].mean())
            for r in regions
        }
        for i in te:
            oof[i] = fold_prior[x_region[i]]
    print(f"5-FOLD-CV region-only : AUROC={roc_auc_score(y, oof):.4f}  "
          f"AUPRC={average_precision_score(y, oof):.4f}")
    print(f"MODEL (from report)   : AUROC={d['auroc']:.4f}  AUPRC={d['auprc']:.4f}  "
          f"random AUPRC={d['random_auprc']:.4f}")


if __name__ == "__main__":
    main()
