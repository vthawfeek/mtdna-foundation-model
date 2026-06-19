"""
Download NCBI human mtDNA sequences that have haplogroup annotations in their title.

These are the most useful records for zero-shot k-NN evaluation because they
come with haplogroup labels that can be extracted directly from the description.

Query: human[Organism] AND mitochondrion[Filter] AND complete genome[Title] AND haplogroup[Title]
Expected count: ~15,000 records

Usage:
    uv run python paper/download_labeled_mtdna.py

Output:
    data/hmtdb_labeled/sequences.fasta
    data/hmtdb_labeled/metadata.parquet   (accession, description, haplogroup)
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUT_DIR = ROOT / "data" / "hmtdb_labeled"
FASTA_OUT = OUT_DIR / "sequences.fasta"
META_OUT  = OUT_DIR / "metadata.parquet"

# Query targeting only records with haplogroup annotations
QUERY = (
    "human[Organism] AND mitochondrion[Filter] "
    "AND complete genome[Title] AND haplogroup[Title]"
)
EMAIL = "vthawfeek@gmail.com"

# Patterns to parse haplogroup from NCBI description
_HG_PATTERNS = [
    re.compile(r'\bhaplogroup\s+([A-Z][A-Za-z0-9]+)', re.IGNORECASE),
    re.compile(r'\bmt-haplogroup\s+([A-Z][A-Za-z0-9]+)', re.IGNORECASE),
    re.compile(r'\bmitotype\s+([A-Z][A-Za-z0-9]+)', re.IGNORECASE),
]


def parse_haplogroup(description: str) -> str | None:
    """Extract haplogroup label from an NCBI FASTA description line."""
    for pat in _HG_PATTERNS:
        m = pat.search(description)
        if m:
            return m.group(1)
    return None


def download_labeled(max_records: int = 20_000) -> tuple[Path, Path]:
    import pandas as pd
    from Bio import Entrez, SeqIO
    from tqdm import tqdm

    Entrez.email = EMAIL
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if FASTA_OUT.exists() and META_OUT.exists():
        log.info("Already downloaded. Loading metadata …")
        meta = pd.read_parquet(META_OUT)
        log.info("  %d records in metadata.", len(meta))
        return FASTA_OUT, META_OUT

    # 1. Search
    log.info("Searching NCBI: %s", QUERY)
    handle = Entrez.esearch(db="nucleotide", term=QUERY, retmax=max_records, usehistory="y")
    record = Entrez.read(handle)
    handle.close()
    count    = int(record["Count"])
    web_env  = record["WebEnv"]
    query_key = record["QueryKey"]
    ids      = record["IdList"]
    log.info("  Found %d records (fetching up to %d).", count, max_records)

    # 2. Fetch in batches of 500
    batch_size = 500
    records_out = []
    rows = []

    with open(FASTA_OUT, "w") as fout:
        for start in tqdm(range(0, min(count, max_records), batch_size), desc="NCBI fetch"):
            fetch_handle = Entrez.efetch(
                db="nucleotide",
                rettype="fasta",
                retmode="text",
                retstart=start,
                retmax=batch_size,
                webenv=web_env,
                query_key=query_key,
            )
            fasta_text = fetch_handle.read()
            fetch_handle.close()
            fout.write(fasta_text)

            # Parse descriptions in this batch
            for line in fasta_text.splitlines():
                if line.startswith(">"):
                    parts = line[1:].split(None, 1)
                    acc  = parts[0] if parts else ""
                    desc = parts[1] if len(parts) > 1 else ""
                    hg   = parse_haplogroup(desc)
                    rows.append({"accession": acc, "description": desc, "haplogroup": hg})

    log.info("Wrote %d description rows.", len(rows))

    # 3. Save metadata
    meta_df = pd.DataFrame(rows)
    labeled = meta_df[meta_df["haplogroup"].notna()]
    log.info("  Records with parseable haplogroup: %d / %d (%.1f%%)",
             len(labeled), len(meta_df), 100 * len(labeled) / max(len(meta_df), 1))
    meta_df.to_parquet(META_OUT, index=False)

    # Haplogroup distribution
    import collections
    dist = collections.Counter(labeled["haplogroup"].tolist())
    log.info("  Top haplogroups: %s", sorted(dist.items(), key=lambda x: -x[1])[:15])

    return FASTA_OUT, META_OUT


if __name__ == "__main__":
    fasta_path, meta_path = download_labeled()
    log.info("Done. FASTA: %s   Meta: %s", fasta_path, meta_path)
