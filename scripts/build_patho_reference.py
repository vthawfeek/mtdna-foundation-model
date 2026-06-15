"""Build app_patho_reference.npz for the Gradio pathogenicity tab.

Combines:
  - reports/zeroshot_patho_embeddings.npz  (X, y, positions)
  - rCRS sequence from data/raw/ncbi/vertebrate_mtdna.fasta  (NC_012920.1)

Output: app_patho_reference.npz in the repo root.
"""

from pathlib import Path

import numpy as np
from Bio import SeqIO

REPO = Path(__file__).parent.parent
EMBEDDINGS = REPO / "reports" / "zeroshot_patho_embeddings.npz"
FASTA = REPO / "data" / "raw" / "ncbi" / "vertebrate_mtdna.fasta"
OUTPUT = REPO / "app_patho_reference.npz"


def load_rcrs(fasta_path: Path) -> str:
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        if "NC_012920" in rec.id:
            return str(rec.seq).upper()
    raise FileNotFoundError(f"NC_012920.1 not found in {fasta_path}")


def main() -> None:
    data = np.load(EMBEDDINGS)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)
    positions = data["positions"].astype(np.int32)
    print(f"Embeddings: X={X.shape}, pathogenic={y.sum()}, benign={(y == 0).sum()}")

    print("Loading rCRS...")
    rcrs = load_rcrs(FASTA)
    assert len(rcrs) == 16569, f"Expected 16569 bp, got {len(rcrs)}"
    print(f"rCRS: {len(rcrs)} bp")

    np.savez(
        OUTPUT,
        X=X,
        y=y,
        positions=positions,
        rcrs=np.frombuffer(rcrs.encode("ascii"), dtype=np.uint8),
    )
    kb = OUTPUT.stat().st_size // 1024
    print(f"Saved {OUTPUT} ({kb} KB)")


if __name__ == "__main__":
    main()
