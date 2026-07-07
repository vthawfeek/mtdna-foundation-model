"""
Evaluation-leakage / overlap quantification for the zero-shot haplogroup result.

The manuscript's open caveat is that evaluation sequences may have been seen during
Phase-1 MLM pre-training, so 37.9% "may partly reflect sequence-level memorisation".
The full 117,615-sequence Phase-1 cross-species corpus is not stored locally and the
documented Entrez query no longer resolves, so the exact Phase-1-intersect-eval fraction
cannot be reconstructed. We therefore measure the decisive, fully-local quantity: how
much the k-NN result is inflated by near-duplicate sequences shared between the reference
LIBRARY and the TEST queries. This is the mechanism by which memorised / duplicated
sequences make the zero-shot number trivial, and it is independent of Phase-1.

Reproduces the exact published 1,509-library / 757-test split (random_state 42/43) from
data/hmtdb_labeled/, then, for every test sequence, finds its nearest library sequence by
genome-wide Hamming distance and reports:
  - library-vs-test accession overlap (sanity: should be 0)
  - distribution of nearest-library sequence identity
  - fraction of test with an exact / near-duplicate library twin
  - a raw-sequence 1-NN haplogroup baseline (no model at all)

Run: uv run python paper/experiments/evaluation/eval_overlap_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))

import run_zeroshot_haplogroup as ez  # noqa: E402

L_GENOME = 16569


def encode(seqs: list[str]) -> np.ndarray:
    """ASCII-byte encode length-normalised sequences to a (n, L) uint8 array."""
    arr = np.zeros((len(seqs), L_GENOME), dtype=np.uint8)
    for r, s in enumerate(seqs):
        b = s[:L_GENOME].ljust(L_GENOME, "N").encode("ascii")
        arr[r] = np.frombuffer(b, dtype=np.uint8)
    return arr


def main() -> None:
    # 1. Reproduce the exact published split (same code path as the paper's result)
    df = ez.load_data()
    train_df, test_df = ez.prepare_splits(df)
    lib = ez.sample_balanced(train_df, ez.TRAIN_PER_CLASS, ez.RANDOM_STATE)
    test = ez.sample_balanced(test_df, ez.TEST_PER_CLASS, ez.RANDOM_STATE + 1)
    print(f"\nReproduced split: library={len(lib)}  test={len(test)}")

    lib_acc = set(lib["accession"])
    test_acc = set(test["accession"])
    print(f"library-test accession overlap: {len(lib_acc & test_acc)} "
          f"(expected 0 for a disjoint split)")

    lib_arr = encode(lib["sequence"].tolist())
    test_arr = encode(test["sequence"].tolist())
    lib_hg = lib["major_hg"].to_numpy()
    test_hg = test["major_hg"].to_numpy()

    # 2. For each test sequence: nearest library sequence by Hamming distance,
    #    computed over positions where BOTH sequences are non-N (A/C/G/T).
    N_BYTE = ord("N")
    lib_isN = lib_arr == N_BYTE
    nn_dist = np.zeros(len(test), dtype=np.int32)
    nn_ident = np.zeros(len(test), dtype=np.float64)
    nn_hg_match = np.zeros(len(test), dtype=bool)
    for i in range(len(test)):
        tv = test_arr[i]
        t_isN = tv == N_BYTE
        comparable = ~(lib_isN | t_isN[None, :])          # (n_lib, L) both non-N
        mismatch = (lib_arr != tv[None, :]) & comparable   # differing, both non-N
        d = mismatch.sum(axis=1)
        c = comparable.sum(axis=1).clip(min=1)
        ident = 1.0 - d / c
        j = int(np.argmax(ident))                          # nearest by identity
        nn_dist[i] = int(d[j])
        nn_ident[i] = float(ident[j])
        nn_hg_match[i] = bool(lib_hg[j] == test_hg[i])

    pct = nn_ident * 100
    print("\n=== Test-vs-library nearest-neighbour SEQUENCE identity ===")
    print(f"  median identity : {np.median(pct):.3f}%")
    print(f"  mean   identity : {pct.mean():.3f}%")
    print(f"  min    identity : {pct.min():.3f}%")
    print(f"  exact duplicates in library (100% ident, 0 SNP): "
          f"{int((nn_dist == 0).sum())}/{len(test)} ({(nn_dist==0).mean()*100:.1f}%)")
    for thr, lbl in [(0.99999, ">=99.999%"), (0.9997, ">=99.97% (<=~5 SNP)"),
                     (0.999, ">=99.9% (<=~16 SNP)"), (0.995, ">=99.5%"),
                     (0.99, ">=99%")]:
        frac = (nn_ident >= thr).mean() * 100
        print(f"  test seqs with a library twin {lbl:22s}: {frac:5.1f}%")

    print("\n=== Raw-sequence 1-NN haplogroup baseline (NO model) ===")
    print(f"  nearest library sequence shares the test haplogroup: "
          f"{nn_hg_match.mean()*100:.1f}% of test queries")
    print("  (this is a model-free lower bound on how trivially the k-NN task is")
    print("   solved by nearest-sequence lookup; compare to mtDNA-FM's 37.9%)")

    # 3. Exact-duplicate sequences anywhere in the combined eval set
    all_seqs = lib["sequence"].tolist() + test["sequence"].tolist()
    uniq = len(set(all_seqs))
    print(f"\n=== Exact sequence duplication in the eval set ({len(all_seqs)} seqs) ===")
    print(f"  distinct sequences: {uniq}  |  exact-duplicate rate: "
          f"{(1 - uniq/len(all_seqs))*100:.1f}%")


if __name__ == "__main__":
    main()
