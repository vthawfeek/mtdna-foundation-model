"""Check overlap between evaluation set (NCBI haplogroup-labeled) and Phase 1 pre-training FASTA."""
from pathlib import Path
import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).parent.parent
meta_path = ROOT / "data" / "hmtdb_labeled" / "metadata.parquet"
fasta_path = ROOT / "data" / "raw" / "ncbi" / "vertebrate_mtdna.fasta"

print("Loading evaluation metadata ...")
meta = pd.read_parquet(meta_path)
eval_accessions = set(meta["accession"].str.split(".").str[0])
print(f"  Evaluation sequences: {len(eval_accessions)}")

print(f"Phase 1 FASTA: {fasta_path}")
if fasta_path.exists():
    pretrain_accessions = set()
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        acc = rec.id.split(".")[0]
        pretrain_accessions.add(acc)
    print(f"  Phase 1 sequences: {len(pretrain_accessions)}")

    overlap = eval_accessions & pretrain_accessions
    pct = 100 * len(overlap) / len(eval_accessions)
    print(f"\n=== OVERLAP RESULT ===")
    print(f"  Overlapping accessions: {len(overlap)}")
    print(f"  Overlap rate:           {pct:.1f}% of evaluation set")
    if pct > 50:
        print("  WARNING: >50% overlap")
    elif pct > 10:
        print("  CAUTION: >10% overlap — note in Limitations with exact count")
    else:
        print("  LOW overlap (<10%) — minimal memorization risk")
    print(f"\n  Note: any leakage inflates mtDNA-FM score only (DNABERT-2 used nuclear genomes).")
    print(f"  So the true gap is >= 28.4pp regardless of leakage direction.")
else:
    print("  FASTA not found — searching ...")
    for p in ROOT.rglob("vertebrate_mtdna.fasta"):
        print(f"  Found: {p} ({p.stat().st_size / 1e6:.0f} MB)")
    # Check if data/raw/ncbi exists at all
    ncbi_dir = ROOT / "data" / "raw" / "ncbi"
    if ncbi_dir.exists():
        for f in ncbi_dir.iterdir():
            print(f"  ncbi dir contains: {f.name} ({f.stat().st_size / 1e6:.0f} MB)")
    else:
        print(f"  data/raw/ncbi/ does not exist. Phase 1 FASTA not available locally.")
        print("  CONCLUSION: Cannot compute overlap from local data.")
        print("  The manuscript Limitations section already acknowledges this as unquantified.")
        print("  Add note: 'We estimate overlap is substantial (NCBI eval corpus subset of Phase 1'")
        print("  'pre-training NCBI query), making any inflation conservative relative to DNABERT-2.'")
