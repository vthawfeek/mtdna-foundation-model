import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

with open(ROOT / "reports" / "zeroshot_haplogroup_knn.json") as f:
    mtdna = json.load(f)
with open(ROOT / "reports" / "dnabert2_haplogroup_knn.json") as f:
    db2 = json.load(f)

print("=== Observation 4 (C, F, E) spot-check ===")
for hg in ["C", "F", "E"]:
    m = next(x for x in mtdna["per_class"] if x["label"] == hg)
    d = next(x for x in db2["per_class"] if x["label"] == hg)
    chk = "PASS" if m["f1"] > d["f1"] else "FAIL"
    print(f"  [{chk}] {hg}: mtDNA-FM F1={m['f1']:.4f}  DNABERT-2 F1={d['f1']:.4f}")

print()
print("=== European HV average F1 ===")
hv_hgs = ["H","HV","J","K","T","U","V","W","X"]
mtdna_hv = [next(x for x in mtdna["per_class"] if x["label"]==hg)["f1"] for hg in hv_hgs]
db2_hv   = [next(x for x in db2["per_class"] if x["label"]==hg)["f1"] for hg in hv_hgs]
print(f"  mtDNA-FM avg={sum(mtdna_hv)/9:.4f}  range=[{min(mtdna_hv):.4f},{max(mtdna_hv):.4f}]")
print(f"  DNABERT-2 avg={sum(db2_hv)/9:.4f}  range=[{min(db2_hv):.4f},{max(db2_hv):.4f}]")

print()
print("=== African L average F1 ===")
l_hgs = ["L0","L1","L2","L3","L4","L5"]
mtdna_l = [next(x for x in mtdna["per_class"] if x["label"]==hg)["f1"] for hg in l_hgs]
db2_l   = [next(x for x in db2["per_class"] if x["label"]==hg)["f1"] for hg in l_hgs]
print(f"  mtDNA-FM avg={sum(mtdna_l)/6:.4f}  range=[{min(mtdna_l):.4f},{max(mtdna_l):.4f}]")
print(f"  DNABERT-2 avg={sum(db2_l)/6:.4f}  range=[{min(db2_l):.4f},{max(db2_l):.4f}]")
print(f"  Ratio mtDNA-FM:  {sum(mtdna_l)/6 / (sum(mtdna_hv)/9):.2f}x")
print(f"  Ratio DNABERT-2: {sum(db2_l)/6 / (sum(db2_hv)/9):.2f}x")

print()
print("=== DNABERT-2 H and J (n_correct) ===")
for hg in ["H","J"]:
    d = next(x for x in db2["per_class"] if x["label"]==hg)
    n_correct = round(d["recall"] * d["support"])
    print(f"  {hg}: recall={d['recall']:.4f}  support={d['support']}  correct={n_correct}")

print()
print("=== mtDNA-FM H and J (n_correct) ===")
for hg in ["H","J"]:
    m = next(x for x in mtdna["per_class"] if x["label"]==hg)
    n_correct = round(m["recall"] * m["support"])
    print(f"  {hg}: recall={m['recall']:.4f}  support={m['support']}  correct={n_correct}")

print()
print("=== Overall gap ===")
print(f"  DNABERT-2 accuracy: {db2['accuracy']*100:.1f}%")
print(f"  mtDNA-FM accuracy:  {mtdna['accuracy']*100:.1f}%")
print(f"  Gap: {(db2['accuracy']-mtdna['accuracy'])*100:.1f}pp")
