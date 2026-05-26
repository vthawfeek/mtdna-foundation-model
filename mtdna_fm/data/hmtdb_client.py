"""
Idempotent HmtDB download client.

Running this module a second time checks what already exists and skips
completed work — a production habit worth baking in from day one.

HmtDB (https://www.hmtdb.uniba.it/) provides bulk FASTA and metadata for
~47k human mitochondrial genomes. If the site is unavailable, the client
falls back to fetching the equivalent dataset from NCBI Entrez.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO
from tqdm import tqdm

logger = logging.getLogger(__name__)

# HmtDB bulk download endpoints (as of 2024)
HMTDB_FASTA_URL = "https://www.hmtdb.uniba.it/hmtdb2/allsequences.fasta"
HMTDB_METADATA_URL = "https://www.hmtdb.uniba.it/hmtdb2/allmetadata.csv"

# NCBI fallback: same dataset via Entrez
NCBI_FALLBACK_QUERY = "human[Organism] AND mitochondrion[Filter] AND complete genome[Title]"

# Expected output filenames
FASTA_FILENAME = "sequences.fasta"
METADATA_FILENAME = "metadata.parquet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    """Stream a file from URL to dest, showing a tqdm progress bar."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))


def download_hmtdb(
    output_dir: str | Path,
    expected_sha256: str | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """
    Download HmtDB bulk FASTA and metadata. Idempotent.

    Parameters
    ----------
    output_dir:
        Directory to write sequences.fasta and metadata.parquet.
    expected_sha256:
        If provided, the SHA256 of the raw FASTA file is verified before writing.
        Pass None to skip verification (default).
    force:
        Re-download even if output files already exist.

    Returns
    -------
    (fasta_path, metadata_path) on success.

    Raises
    ------
    RuntimeError
        If HmtDB is unreachable and the NCBI fallback also fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = output_dir / FASTA_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    if not force and fasta_path.exists() and metadata_path.exists():
        logger.info("HmtDB outputs already exist at %s — skipping download", output_dir)
        return fasta_path, metadata_path

    logger.info("Downloading HmtDB FASTA from %s", HMTDB_FASTA_URL)
    try:
        _download_fasta_from_hmtdb(fasta_path, expected_sha256)
        _download_metadata_from_hmtdb(metadata_path)
    except (requests.RequestException, OSError) as exc:
        logger.warning("HmtDB download failed (%s); falling back to NCBI", exc)
        fasta_path, metadata_path = _ncbi_fallback(output_dir)

    _validate_fasta(fasta_path)
    return fasta_path, metadata_path


def _download_fasta_from_hmtdb(dest: Path, expected_sha256: str | None) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
        tmp_path = Path(tmp.name)

    try:
        _download_file(HMTDB_FASTA_URL, tmp_path, desc="HmtDB FASTA")
        if expected_sha256 is not None:
            actual = _sha256(tmp_path)
            if actual != expected_sha256.lower():
                raise ValueError(f"SHA256 mismatch: expected {expected_sha256}, got {actual}")
        shutil.move(str(tmp_path), str(dest))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _download_metadata_from_hmtdb(dest: Path) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp_path = Path(tmp.name)
    try:
        _download_file(HMTDB_METADATA_URL, tmp_path, desc="HmtDB metadata")
        df = pd.read_csv(tmp_path)
        df.to_parquet(dest, index=False)
        logger.info("Metadata: %d rows written to %s", len(df), dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _ncbi_fallback(output_dir: Path) -> tuple[Path, Path]:
    """Pull the human mtDNA dataset from NCBI when HmtDB is unreachable."""
    from mtdna_fm.data.ncbi_client import download_ncbi_mtdna

    logger.info("NCBI fallback: query=%r", NCBI_FALLBACK_QUERY)
    fasta_path = download_ncbi_mtdna(
        query=NCBI_FALLBACK_QUERY,
        output_dir=output_dir,
        output_filename=FASTA_FILENAME,
    )
    # Build a minimal metadata parquet from the FASTA headers
    records = list(SeqIO.parse(fasta_path, "fasta"))
    df = pd.DataFrame(
        [
            {"accession": rec.id, "description": rec.description, "length": len(rec.seq)}
            for rec in records
        ]
    )
    metadata_path = output_dir / METADATA_FILENAME
    df.to_parquet(metadata_path, index=False)
    logger.info("NCBI fallback: wrote %d records to %s", len(df), output_dir)
    return fasta_path, metadata_path


def _validate_fasta(path: Path) -> None:
    """Quick sanity check: count sequences and log the result."""
    n = sum(1 for _ in SeqIO.parse(path, "fasta"))
    if n == 0:
        raise ValueError(f"FASTA at {path} contains 0 records — download may have failed")
    logger.info("Validated %s: %d sequences", path.name, n)


def extract_zip_fasta(zip_path: Path, output_dir: Path, expected_sha256: str) -> Path:
    """
    Verify a downloaded zip by SHA256, then extract the FASTA inside.
    Used when HmtDB distributes a zip archive rather than a bare FASTA.
    """
    actual = _sha256(zip_path)
    if actual != expected_sha256.lower():
        raise ValueError(f"SHA256 mismatch: expected {expected_sha256}, got {actual}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        fasta_names = [n for n in zf.namelist() if n.endswith((".fasta", ".fa", ".fas"))]
        if not fasta_names:
            raise ValueError(f"No FASTA file found inside {zip_path}")
        dest = output_dir / FASTA_FILENAME
        with zf.open(fasta_names[0]) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    logger.info("Extracted %s -> %s", fasta_names[0], dest)
    return dest
