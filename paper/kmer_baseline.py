"""
Supervised 6-mer frequency logistic regression baseline for 26-class haplogroup
classification. Uses the IDENTICAL data loading, split, and balanced sampling
as run_zeroshot_haplogroup.py (same random_state=42, same 1509/757 sets).

Run: uv run python paper/kmer_baseline.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Import the exact same data-prep functions from the zero-shot script
from paper.run_zeroshot_haplogroup import (
    load_data, prepare_splits, sample_balanced,
    TRAIN_PER_CLASS, TEST_PER_CLASS, RANDOM_STATE,
)

REPORT_OUT = ROOT / "reports" / "kmer_baseline_haplogroup.json"

K = 6
ALPHABET = "ACGT"
ALL_KMERS = [a+b+c+d+e+f
             for a in ALPHABET for b in ALPHABET for c in ALPHABET
             for d in ALPHABET for e in ALPHABET for f in ALPHABET]
KMER_INDEX = {km: i for i, km in enumerate(ALL_KMERS)}


def kmer_vector(seq: str) -> np.ndarray:
    import re
    seq = re.sub(r"[^ACGT]", "A", seq.upper())
    vec = np.zeros(len(KMER_INDEX), dtype=np.float32)
    for i in range(len(seq) - K + 1):
        km = seq[i:i+K]
        if km in KMER_INDEX:
            vec[KMER_INDEX[km]] += 1
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    print("Loading data (same as zero-shot eval)...")
    df = load_data()

    print("Splitting (random_state=42, same as zero-shot eval)...")
    train_df, test_df = prepare_splits(df)
    classes = sorted(set(train_df["major_hg"]) & set(test_df["major_hg"]))
    n_classes = len(classes)

    train_lib = sample_balanced(train_df, TRAIN_PER_CLASS, RANDOM_STATE)
    test_q    = sample_balanced(test_df,  TEST_PER_CLASS,  RANDOM_STATE + 1)
    print(f"  Train: {len(train_lib)}  Test: {len(test_q)}  Classes: {n_classes}")

    print("Extracting 6-mer frequency vectors...")
    X_train = np.vstack([kmer_vector(s) for s in train_lib["sequence"].tolist()])
    y_train = train_lib["major_hg"].tolist()
    X_test  = np.vstack([kmer_vector(s) for s in test_q["sequence"].tolist()])
    y_test  = test_q["major_hg"].tolist()

    print("Training logistic regression (max_iter=2000)...")
    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)

    y_pred   = clf.predict(X_test)
    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    random_baseline = 1.0 / n_classes

    result = {
        "model": f"6-mer frequency logistic regression (supervised, C=1.0)",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "n_train": len(train_lib),
        "n_test": len(test_q),
        "n_classes": n_classes,
        "random_baseline": round(random_baseline, 4),
        "kmer_k": K,
    }

    print()
    print("=" * 60)
    print("  SUPERVISED 6-MER BASELINE")
    print("=" * 60)
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    print(f"  Random:    {random_baseline*100:.2f}%")
    print(f"  Lift:      {acc/random_baseline:.1f}x")
    print("=" * 60)

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(REPORT_OUT), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {REPORT_OUT}")


if __name__ == "__main__":
    main()
