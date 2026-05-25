"""
mtdna-download CLI — idempotent download of all raw datasets.

Usage examples:
    mtdna-download --source ncbi-refseq --output data/raw/ncbi
    mtdna-download --source hmtdb --output data/raw/hmtdb
    mtdna-download --source ncbi-refseq --force
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

app = typer.Typer(
    help="Download raw datasets (HmtDB, NCBI RefSeq, gnomAD, ClinVar, PhyloTree).",
    no_args_is_help=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_SOURCES = ("hmtdb", "ncbi-refseq", "gnomad", "clinvar", "phylotree")


@app.command()
def main(
    source: str = typer.Option(
        ...,
        "--source",
        "-s",
        help=f"Dataset to download. One of: {', '.join(VALID_SOURCES)}",
    ),
    output: str = typer.Option(
        "data/raw",
        "--output",
        "-o",
        help="Output directory",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-download even if output files already exist",
    ),
) -> None:
    """Download a raw dataset. Idempotent by default."""
    if source not in VALID_SOURCES:
        typer.echo(
            f"[error] Unknown source '{source}'. "
            f"Valid options: {', '.join(VALID_SOURCES)}",
            err=True,
        )
        raise typer.Exit(code=1)

    output_dir = Path(output)

    if source == "hmtdb":
        _run_hmtdb(output_dir, force)
    elif source == "ncbi-refseq":
        _run_ncbi_refseq(output_dir, force)
    elif source == "gnomad":
        typer.echo("[gnomad] Not yet implemented — coming Day 5")
        raise typer.Exit(code=1)
    elif source == "clinvar":
        typer.echo("[clinvar] Not yet implemented — coming Day 5")
        raise typer.Exit(code=1)
    elif source == "phylotree":
        typer.echo("[phylotree] Not yet implemented — coming Day 5")
        raise typer.Exit(code=1)


def _run_hmtdb(output_dir: Path, force: bool) -> None:
    from mtdna_fm.data.hmtdb_client import download_hmtdb

    fasta, metadata = download_hmtdb(output_dir, force=force)
    typer.echo(f"[hmtdb] FASTA    → {fasta}")
    typer.echo(f"[hmtdb] metadata → {metadata}")


def _run_ncbi_refseq(output_dir: Path, force: bool) -> None:
    from mtdna_fm.data.ncbi_client import VERTEBRATE_QUERY, download_ncbi_mtdna

    fasta = download_ncbi_mtdna(
        query=VERTEBRATE_QUERY,
        output_dir=output_dir,
        output_filename="vertebrate_mtdna.fasta",
        force=force,
    )
    typer.echo(f"[ncbi-refseq] FASTA → {fasta}")
