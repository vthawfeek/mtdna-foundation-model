"""
Idempotent NCBI Entrez download client for mitochondrial DNA sequences.

Uses esearch + efetch with usehistory=True (WebEnv) so the server holds
the result set and we batch-fetch in chunks of 500 without repeating the
search. The NCBI_API_KEY environment variable increases the rate limit from
3 to 10 requests/second.

Idempotency: a JSON progress file tracks which batches are complete. A
second run reads the file, skips finished batches, and resumes from where
it stopped.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from Bio import Entrez, SeqIO
from tqdm import tqdm

logger = logging.getLogger(__name__)

# NCBI guidelines: set a contact email so they can reach you if you abuse the API
Entrez.email = os.getenv("NCBI_EMAIL", "thawfeek.varusai@biorelate.com")

# Vertebrate complete mtDNA genomes: the cross-species pre-training corpus
VERTEBRATE_QUERY = (
    "vertebrata[Organism] AND complete genome[Title] AND mitochondrion[Filter]"
)
# Human mtDNA: matches HmtDB coverage, used as fallback
HUMAN_QUERY = "human[Organism] AND mitochondrion[Filter] AND complete genome[Title]"

DEFAULT_BATCH_SIZE = 500
# NCBI polite rate: 3/s without key, 10/s with key
_BASE_DELAY = 0.34  # seconds between requests without API key


def _rate_delay() -> float:
    api_key = os.getenv("NCBI_API_KEY", "")
    return 0.11 if api_key else _BASE_DELAY


def _configure_api_key() -> None:
    api_key = os.getenv("NCBI_API_KEY", "")
    if api_key:
        Entrez.api_key = api_key
        logger.debug("NCBI_API_KEY found — using 10 req/s rate limit")
    else:
        logger.debug("No NCBI_API_KEY — using 3 req/s rate limit")


def download_ncbi_mtdna(
    query: str = VERTEBRATE_QUERY,
    output_dir: str | Path = "data/raw/ncbi",
    output_filename: str = "vertebrate_mtdna.fasta",
    batch_size: int = DEFAULT_BATCH_SIZE,
    db: str = "nucleotide",
    force: bool = False,
) -> Path:
    """
    Fetch complete mitochondrial genomes from NCBI Entrez. Idempotent.

    Parameters
    ----------
    query:
        Entrez search query string.
    output_dir:
        Directory to write the FASTA file and progress state.
    output_filename:
        Name of the output FASTA file.
    batch_size:
        Records per efetch call. NCBI recommends ≤500.
    db:
        Entrez database name (default: nucleotide).
    force:
        Re-download even if the output file already exists.

    Returns
    -------
    Path to the written FASTA file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = output_dir / output_filename
    progress_path = output_dir / f"{output_filename}.progress.json"

    _configure_api_key()

    # --- Count total hits ---
    total, web_env, query_key = _esearch(query, db)
    if total == 0:
        raise RuntimeError(f"Entrez query returned 0 results: {query!r}")
    logger.info("Entrez query matched %d records", total)

    # --- Load or initialise progress ---
    progress = _load_progress(progress_path)
    if force:
        progress = {}
        if fasta_path.exists():
            fasta_path.unlink()

    if not force and fasta_path.exists() and _progress_complete(progress, total, batch_size):
        logger.info("All batches already downloaded — skipping")
        return fasta_path

    # --- Fetch in batches ---
    n_batches = (total + batch_size - 1) // batch_size
    mode = "ab" if fasta_path.exists() else "wb"

    with open(fasta_path, mode) as out_fasta, tqdm(
        total=total, desc="NCBI fetch", unit="seq"
    ) as bar:
        for batch_idx in range(n_batches):
            batch_key = str(batch_idx)
            if progress.get(batch_key) == "done":
                bar.update(min(batch_size, total - batch_idx * batch_size))
                continue

            retstart = batch_idx * batch_size
            retmax = min(batch_size, total - retstart)

            records = _efetch_batch(
                db=db,
                web_env=web_env,
                query_key=query_key,
                retstart=retstart,
                retmax=retmax,
            )
            out_fasta.write(records.encode() if isinstance(records, str) else records)

            progress[batch_key] = "done"
            _save_progress(progress_path, progress)
            bar.update(retmax)
            time.sleep(_rate_delay())

    logger.info("Download complete: %s", fasta_path)
    return fasta_path


def _esearch(query: str, db: str) -> tuple[int, str, str]:
    """Run esearch with usehistory=True; return (count, WebEnv, query_key)."""
    handle = Entrez.esearch(db=db, term=query, usehistory="y")
    record = Entrez.read(handle)
    handle.close()
    return int(record["Count"]), record["WebEnv"], record["QueryKey"]


def _efetch_batch(
    db: str,
    web_env: str,
    query_key: str,
    retstart: int,
    retmax: int,
    rettype: str = "fasta",
    retmode: str = "text",
) -> str:
    handle = Entrez.efetch(
        db=db,
        rettype=rettype,
        retmode=retmode,
        retstart=retstart,
        retmax=retmax,
        webenv=web_env,
        query_key=query_key,
    )
    data = handle.read()
    handle.close()
    return data


def _load_progress(path: Path) -> dict[str, str]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_progress(path: Path, progress: dict[str, str]) -> None:
    with open(path, "w") as f:
        json.dump(progress, f)


def _progress_complete(
    progress: dict[str, str], total: int, batch_size: int
) -> bool:
    n_batches = (total + batch_size - 1) // batch_size
    return all(progress.get(str(i)) == "done" for i in range(n_batches))


def count_fasta_records(fasta_path: Path) -> int:
    """Count sequences in a FASTA file without loading them all into memory."""
    return sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
