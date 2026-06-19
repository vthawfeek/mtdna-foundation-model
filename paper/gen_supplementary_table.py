"""Generate LaTeX rows for S3 per-class haplogroup table from eval_haplogroup_detail.json."""
import json
from pathlib import Path

data = json.loads((Path("reports/eval_haplogroup_detail.json")).read_text())

print("% Auto-generated from eval_haplogroup_detail.json")
print("% Paste into supplementary.tex S3 table body\n")

totals = {"precision": 0, "recall": 0, "f1": 0, "n": 0}
for cls in data["per_class"]:
    lbl = cls["label"]
    p = cls["precision"]
    r = cls["recall"]
    f = cls["f1"]
    n = cls["support"]
    print(f"{lbl} & {p:.3f} & {r:.3f} & {f:.3f} & {n:,} \\\\")
    totals["precision"] += p
    totals["recall"] += r
    totals["f1"] += f
    totals["n"] += n

nclass = len(data["per_class"])
print(r"\midrule")
print(f"\\textbf{{Macro avg}} & {totals['precision']/nclass:.3f} & "
      f"{totals['recall']/nclass:.3f} & {totals['f1']/nclass:.3f} & "
      f"{totals['n']:,} \\\\")
