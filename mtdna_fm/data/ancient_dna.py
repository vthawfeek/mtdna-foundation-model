"""
Ancient DNA download utilities for mtDNA-FM.

Downloads specific NCBI accessions for ancient hominid mtDNA sequences:
  - NC_011137.1  Neanderthal (Vindija Cave, Croatia)
  - FR695060.1   Denisovan  (Altai Cave, Russia)

These sequences were never in the training data and are used for
zero-shot demonstration: embedding them with the pre-trained model
and placing them on the modern human UMAP to verify that evolutionary
relationships emerge from sequence alone.

WHY these two accessions:
  Both are the best-coverage ancient hominid mtDNA sequences available
  on NCBI, assembled from high-quality ancient DNA libraries. Scientific
  literature places them clearly outside modern human haplogroup diversity
  but within the broader Homo clade — exactly the kind of verifiable
  out-of-distribution test that makes a compelling zero-shot demonstration.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)

Entrez.email = os.getenv("NCBI_EMAIL", "thawfeek.varusai@biorelate.com")

# Canonical ancient hominid mtDNA accessions
ANCIENT_ACCESSIONS: dict[str, dict[str, str]] = {
    "neanderthal": {
        "accession": "NC_011137.1",
        "label": "Neanderthal",
        "site": "Vindija Cave, Croatia",
    },
    "denisovan": {
        "accession": "FR695060.1",
        "label": "Denisovan",
        "site": "Altai Cave, Russia",
    },
}


def download_ancient_accession(
    accession: str,
    output_dir: str | Path = "data/raw/ancient",
    db: str = "nucleotide",
    force: bool = False,
) -> Path:
    """
    Download a single NCBI accession as a FASTA file. Idempotent.

    Parameters
    ----------
    accession:
        NCBI accession string (e.g. "NC_011137.1").
    output_dir:
        Directory to write the FASTA file.
    db:
        Entrez database (default: nucleotide).
    force:
        Re-download even if the file already exists.

    Returns
    -------
    Path to the written FASTA file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = output_dir / f"{accession}.fasta"

    if fasta_path.exists() and not force:
        logger.info("Already downloaded: %s", fasta_path)
        return fasta_path

    api_key = os.getenv("NCBI_API_KEY", "")
    if api_key:
        Entrez.api_key = api_key

    logger.info("Fetching %s from NCBI...", accession)
    handle = Entrez.efetch(db=db, id=accession, rettype="fasta", retmode="text")
    data = handle.read()
    handle.close()

    fasta_path.write_text(data if isinstance(data, str) else data.decode())
    logger.info("Saved %s to %s", accession, fasta_path)

    # NCBI polite rate limit
    time.sleep(0.34 if not api_key else 0.11)

    return fasta_path


def download_all_ancient(
    output_dir: str | Path = "data/raw/ancient",
    force: bool = False,
) -> dict[str, Path]:
    """
    Download Neanderthal and Denisovan mtDNA. Idempotent.

    Returns
    -------
    Dict mapping name ("neanderthal", "denisovan") to FASTA path.
    """
    output_dir = Path(output_dir)
    paths: dict[str, Path] = {}
    for name, meta in ANCIENT_ACCESSIONS.items():
        paths[name] = download_ancient_accession(
            accession=meta["accession"],
            output_dir=output_dir,
            force=force,
        )
    return paths


def load_ancient_sequence(fasta_path: str | Path) -> SeqRecord:
    """
    Load the first (and typically only) record from a FASTA file.

    Parameters
    ----------
    fasta_path:
        Path to a single-sequence FASTA file.

    Returns
    -------
    Bio.SeqRecord.SeqRecord with the sequence.
    """
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}")
    return records[0]


def prepare_ancient_sequences(
    output_dir: str | Path = "data/raw/ancient",
    force: bool = False,
) -> dict[str, str]:
    """
    Download (if needed) and return Neanderthal + Denisovan sequences as strings.

    Returns
    -------
    Dict mapping label to uppercase DNA sequence string, e.g.
    {"Neanderthal": "ATGC...", "Denisovan": "ATGC..."}
    """
    paths = download_all_ancient(output_dir=output_dir, force=force)
    result: dict[str, str] = {}
    for name, path in paths.items():
        record = load_ancient_sequence(path)
        seq_str = str(record.seq).upper().replace("N", "N")
        label = ANCIENT_ACCESSIONS[name]["label"]
        result[label] = seq_str
        logger.info(
            "Loaded %s: %d bp (accession %s, %s)",
            label,
            len(seq_str),
            ANCIENT_ACCESSIONS[name]["accession"],
            ANCIENT_ACCESSIONS[name]["site"],
        )
    return result
