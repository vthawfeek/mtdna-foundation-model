"""
Replace RESULT_* placeholders in main.tex and populate supplementary per-class table
in supplementary.tex from the zero-shot evaluation JSON.

Usage (run after run_zeroshot_haplogroup.py completes):
    uv run python paper/update_paper_results.py

Reads:  reports/zeroshot_haplogroup_knn.json
Writes: paper/manuscript/main.tex  (in-place, replaces RESULT_* placeholders)
        paper/manuscript/supplementary.tex  (in-place, populates S3.1 table)
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT      = Path(__file__).parent.parent
JSON_PATH = ROOT / "reports" / "zeroshot_haplogroup_knn.json"
MAIN_TEX  = ROOT / "paper" / "manuscript" / "main.tex"
SUPP_TEX  = ROOT / "paper" / "manuscript" / "supplementary.tex"

if not JSON_PATH.exists():
    raise FileNotFoundError(
        f"Run paper/run_zeroshot_haplogroup.py first. Missing:\n  {JSON_PATH}"
    )

with open(JSON_PATH) as f:
    data = json.load(f)

acc_pct  = f"{data['accuracy'] * 100:.1f}"
ci_lo    = f"{data['accuracy_ci_95_lo'] * 100:.1f}"
ci_hi    = f"{data['accuracy_ci_95_hi'] * 100:.1f}"
lift     = f"{data['lift_over_random']:.1f}"
macro_f1 = f"{data['macro_f1']:.4f}"
n_test   = data["n_test_queries"]
n_train  = data["n_train_library"]
knn_k    = data.get("knn_k", 5)


# ── 1. Update main.tex ────────────────────────────────────────────────────────

with open(MAIN_TEX, encoding="utf-8") as f:
    tex = f.read()

substitutions = [
    (r"RESULT\_ACC\%",          f"{acc_pct}\\%"),
    (r"\textbf{RESULT\_ACC}",   f"\\textbf{{{acc_pct}}}"),
    (r"RESULT\_ACC",            acc_pct),
    (r"RESULT\_CI\_LO",         ci_lo),
    (r"RESULT\_CI\_HI",         ci_hi),
    (r"RESULT\_LIFT",           lift),
    (r"RESULT\_F1",             macro_f1),
]

replaced = 0
for old, new in substitutions:
    count = tex.count(old)
    if count:
        tex = tex.replace(old, new)
        print(f"  [{count}x] {old!r}  ->  {new!r}")
        replaced += count

if replaced == 0:
    print("INFO: No RESULT_* placeholders in main.tex (already substituted?)")
else:
    with open(MAIN_TEX, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"\nUpdated main.tex:")
    print(f"  accuracy = {acc_pct}%  (95% CI {ci_lo}–{ci_hi}%)")
    print(f"  lift     = {lift}×  over 3.85% random baseline")
    print(f"  macro-F1 = {macro_f1}")


# ── 2. Populate supplementary Table S3.1 ─────────────────────────────────────

per_class = sorted(data["per_class"], key=lambda r: r["label"])

rows = []
for pc in per_class:
    label = pc["label"]
    prec  = f"{pc['precision']:.3f}"
    rec   = f"{pc['recall']:.3f}"
    f1    = f"{pc['f1']:.3f}"
    sup   = str(pc["support"])
    rows.append(f"{label} & {prec} & {rec} & {f1} & {sup} \\\\")

table_body = "\n".join(rows)

new_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Per-class zero-shot {knn_k}-NN haplogroup metrics ({n_test} test queries, {n_train} train
library sequences, cosine distance). No haplogroup labels used in pre-training.}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Haplogroup}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1}} & \\textbf{{N test}} \\\\
\\midrule
{table_body}
\\midrule
\\textbf{{Macro avg}} & \\textbf{{{macro_f1}}} & & & {n_test} \\\\
\\bottomrule
\\end{{tabular}}
\\label{{tab:per_class_zeroshot}}
\\end{{table}}"""

with open(SUPP_TEX, encoding="utf-8") as f:
    supp = f.read()

placeholder = (
    "\\begin{center}\n"
    "\\textit{[Results pending; table will be populated from "
    "\\texttt{reports/zeroshot\\_haplogroup\\_knn.json}]}\n"
    "\\end{center}"
)

if placeholder in supp:
    supp = supp.replace(placeholder, new_table)
    with open(SUPP_TEX, "w", encoding="utf-8") as f:
        f.write(supp)
    print(f"\nPopulated supplementary Table S3.1 ({len(per_class)} haplogroup rows)")
else:
    print("\nINFO: supplementary.tex placeholder not found (already populated or edited manually)")

print("\nAll done.")
