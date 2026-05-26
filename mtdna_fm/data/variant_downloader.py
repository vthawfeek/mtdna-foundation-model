"""
Idempotent download clients for gnomAD chrM, ClinVar mtDNA, and PhyloTree Build 17.

Each function checks for existing output files before attempting any network
call — the same pattern used by hmtdb_client and ncbi_client on Days 3–4.

gnomAD chrM
-----------
gnomAD v3.1 stores mitochondrial variants in a separate VCF that must be
extracted with tabix before processing:

    tabix -h gnomad.genomes.v3.1.sites.chrM.vcf.bgz chrM > chrM.vcf

We download the .bgz and .tbi files together, then run tabix to extract
chrM. The extracted plain-text VCF is the input to parse_gnomad_chrm_vcf().

ClinVar
-------
ClinVar ships a GRCh38 VCF that covers all chromosomes. We download the
full VCF (gzipped) and filter for CHROM == "chrM" using Python line-by-line
reading — no tabix dependency required because the file is small enough.

PhyloTree Build 17
------------------
PhyloTree is distributed as a static CSV mapping each variant to its
defining haplogroup. We maintain a canonical URL for the rCRS-annotated
variant table.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants (filenames written to disk)
# ---------------------------------------------------------------------------

GNOMAD_VCF_FILENAME = "gnomad_chrM.vcf"
GNOMAD_BGZ_FILENAME = "gnomad_chrM.vcf.bgz"
GNOMAD_TBI_FILENAME = "gnomad_chrM.vcf.bgz.tbi"

CLINVAR_VCF_FILENAME = "clinvar_chrM.vcf"

PHYLOTREE_CSV_FILENAME = "phylotree_build17.csv"

# ---------------------------------------------------------------------------
# Remote URLs
# ---------------------------------------------------------------------------

GNOMAD_BGZ_URL = (
    "https://gnomad-public-us-east-1.s3.amazonaws.com/release/3.1/vcf/genomes/"
    "gnomad.genomes.v3.1.sites.chrM.vcf.bgz"
)
GNOMAD_TBI_URL = GNOMAD_BGZ_URL + ".tbi"

CLINVAR_VCF_GZ_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"

# Flat CSV derived from PhyloTree Build 17 (rCRS-referenced, SNPs only)
PHYLOTREE_CSV_URL = (
    "https://raw.githubusercontent.com/smartos-dev/phylotree-data/"
    "main/build17/phylotree_build17_variants.csv"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stream_download(url: str, dest: Path, desc: str = "") -> None:
    """Stream a remote file to *dest* with a tqdm progress bar."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with (
        open(dest, "wb") as fh,
        tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
            bar.update(len(chunk))


def _extract_chrom_from_gz(gz_path: Path, dest: Path, chrom: str = "chrM") -> None:
    """Filter a gzipped VCF, writing header + chrM lines to *dest*."""
    chrom_bytes = chrom.encode()
    with gzip.open(gz_path, "rb") as src, open(dest, "wb") as out:
        for line in src:
            if line.startswith(b"#") or line.startswith(chrom_bytes):
                out.write(line)


# ---------------------------------------------------------------------------
# Public download functions
# ---------------------------------------------------------------------------


def download_gnomad_chrm(output_dir: Path, force: bool = False) -> Path:
    """Download gnomAD chrM VCF.

    Downloads the .bgz and .tbi files, then uses tabix (if available) to
    extract a plain-text chrM VCF.  If tabix is not installed, the raw .bgz
    is kept and the caller is responsible for extraction.

    Returns the path to the extracted VCF (or .bgz if tabix is unavailable).
    Idempotent — returns immediately if the target file already exists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vcf_path = output_dir / GNOMAD_VCF_FILENAME
    if vcf_path.exists() and not force:
        logger.info("gnomAD chrM VCF already exists: %s", vcf_path)
        return vcf_path

    bgz_path = output_dir / GNOMAD_BGZ_FILENAME
    tbi_path = output_dir / GNOMAD_TBI_FILENAME

    logger.info("Downloading gnomAD chrM .bgz …")
    _stream_download(GNOMAD_BGZ_URL, bgz_path, desc="gnomAD .bgz")
    logger.info("Downloading gnomAD chrM .tbi …")
    _stream_download(GNOMAD_TBI_URL, tbi_path, desc="gnomAD .tbi")

    tabix = shutil.which("tabix")
    if tabix:
        import subprocess

        logger.info("Extracting chrM with tabix …")
        with open(vcf_path, "w") as out:
            subprocess.run(
                [tabix, "-h", str(bgz_path), "chrM"],
                stdout=out,
                check=True,
            )
        logger.info("gnomAD chrM VCF written to %s", vcf_path)
    else:
        logger.warning(
            "tabix not found — keeping %s. Install tabix and re-run with --force "
            "to extract the plain-text VCF.",
            bgz_path,
        )
        return bgz_path

    return vcf_path


def download_clinvar_chrm(output_dir: Path, force: bool = False) -> Path:
    """Download ClinVar VCF and extract chrM variants.

    Streams the full GRCh38 ClinVar VCF (gzipped), filters for CHROM==chrM,
    and writes a plain-text VCF to *output_dir/clinvar_chrM.vcf*.

    Idempotent — returns immediately if the target file already exists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vcf_path = output_dir / CLINVAR_VCF_FILENAME
    if vcf_path.exists() and not force:
        logger.info("ClinVar chrM VCF already exists: %s", vcf_path)
        return vcf_path

    with tempfile.TemporaryDirectory() as tmp:
        gz_path = Path(tmp) / "clinvar.vcf.gz"
        logger.info("Downloading ClinVar VCF (this may take a few minutes) …")
        _stream_download(CLINVAR_VCF_GZ_URL, gz_path, desc="ClinVar VCF")
        logger.info("Filtering for chrM …")
        _extract_chrom_from_gz(gz_path, vcf_path, chrom="chrM")

    logger.info("ClinVar chrM VCF written to %s (%d bytes)", vcf_path, vcf_path.stat().st_size)
    return vcf_path


def download_phylotree(output_dir: Path, force: bool = False) -> Path:
    """Download PhyloTree Build 17 variant CSV.

    Idempotent — returns immediately if the target file already exists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / PHYLOTREE_CSV_FILENAME
    if csv_path.exists() and not force:
        logger.info("PhyloTree CSV already exists: %s", csv_path)
        return csv_path

    logger.info("Downloading PhyloTree Build 17 CSV …")
    _stream_download(PHYLOTREE_CSV_URL, csv_path, desc="PhyloTree Build 17")
    logger.info("PhyloTree CSV written to %s", csv_path)
    return csv_path
