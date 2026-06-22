import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

with open(ROOT / "reports" / "zeroshot_haplogroup_knn.json") as f:
    data = json.load(f)

# Supplementary Table S3.1 values from the LaTeX
table = {
    "A": (0.218, 0.475, 0.299, 40),
    "B": (0.667, 0.800, 0.727, 40),
    "C": (0.667, 0.600, 0.632, 40),
    "D": (0.068, 0.100, 0.081, 40),
    "E": (0.273, 0.750, 0.400, 4),
    "F": (0.774, 0.667, 0.716, 36),
    "G": (0.098, 0.400, 0.157, 10),
    "H": (0.061, 0.050, 0.055, 40),
    "HV": (0.036, 0.040, 0.038, 25),
    "I": (0.762, 0.800, 0.780, 40),
    "J": (0.185, 0.125, 0.149, 40),
    "K": (0.182, 0.150, 0.164, 40),
    "L0": (0.857, 0.750, 0.800, 40),
    "L1": (0.727, 0.571, 0.640, 28),
    "L2": (0.861, 0.775, 0.816, 40),
    "L3": (0.485, 0.400, 0.438, 40),
    "L4": (0.474, 0.818, 0.600, 11),
    "L5": (1.000, 0.750, 0.857, 4),
    "M": (0.412, 0.175, 0.246, 40),
    "N": (0.133, 0.095, 0.111, 21),
    "R": (0.857, 0.316, 0.462, 19),
    "T": (0.133, 0.050, 0.073, 40),
    "U": (0.389, 0.175, 0.241, 40),
    "V": (0.056, 0.091, 0.069, 11),
    "W": (0.000, 0.000, 0.000, 16),
    "X": (0.071, 0.083, 0.077, 12),
}

print("=== Table S3.1 cross-check (supplementary vs JSON) ===")
all_pass = True
for hg, (t_prec, t_rec, t_f1, t_n) in table.items():
    row = next((x for x in data["per_class"] if x["label"] == hg), None)
    if row is None:
        print(f"  [FAIL] {hg}: not found in JSON")
        all_pass = False
        continue
    j_prec = round(row["precision"], 3)
    j_rec  = round(row["recall"], 3)
    j_f1   = round(row["f1"], 3)
    j_n    = row["support"]
    ok = (j_prec == t_prec and j_rec == t_rec and j_f1 == t_f1 and j_n == t_n)
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
        print(f"  [{status}] {hg}: table=({t_prec},{t_rec},{t_f1},{t_n})  json=({j_prec},{j_rec},{j_f1},{j_n})")
    else:
        print(f"  [{status}] {hg}")

print()
if all_pass:
    print("ALL 26 haplogroup rows match JSON ✓")
else:
    print("MISMATCHES FOUND — see above")

# Also check macro-F1
j_macro = round(data["macro_f1"], 4)
t_macro = 0.3703
print(f"\nMacro-F1: table={t_macro}  json={j_macro}  {'PASS' if j_macro == t_macro else 'FAIL'}")
