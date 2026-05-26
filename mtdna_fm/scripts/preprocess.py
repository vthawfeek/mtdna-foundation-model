"""CLI entry point: mtdna-preprocess."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Preprocess raw sequence data into clean train/val/test parquet files.")
console = Console()


@app.command()
def main(
    hmtdb: str = typer.Option(None, help="Path to HmtDB raw directory"),
    ncbi: str = typer.Option(None, help="Path to NCBI raw directory"),
    output_dir: str = typer.Option("data/processed", help="Output directory"),
    target_length: int = typer.Option(16569, help="Target sequence length (rCRS is 16569 bp)"),
    min_n_fraction: float = typer.Option(0.1, help="Max N fraction for QC pass"),
) -> None:
    """Preprocess HmtDB and/or NCBI FASTA files into stratified parquet splits."""
    import pandas as pd

    from mtdna_fm.data.preprocessor import (
        build_record_dataframe,
        preprocess_sequences,
        save_splits,
        stratified_split,
    )

    if hmtdb is None and ncbi is None:
        console.print("[red]Error: at least one of --hmtdb or --ncbi must be provided[/red]")
        raise typer.Exit(code=1)

    frames: list[pd.DataFrame] = []

    if hmtdb is not None:
        hmtdb_path = Path(hmtdb)
        fasta = hmtdb_path / "sequences.fasta"
        metadata_parquet = hmtdb_path / "metadata.parquet"
        if not fasta.exists():
            console.print(f"[red]FASTA not found: {fasta}[/red]")
            raise typer.Exit(code=1)
        metadata_df = pd.read_parquet(metadata_parquet) if metadata_parquet.exists() else None
        console.print(f"Loading HmtDB sequences from {fasta} ...")
        df = build_record_dataframe(fasta, metadata_df=metadata_df, default_species="homo_sapiens")
        frames.append(df)
        console.print(f"  {len(df):,} sequences loaded")

    if ncbi is not None:
        ncbi_path = Path(ncbi)
        fasta = ncbi_path / "vertebrate_mtdna.fasta"
        if not fasta.exists():
            console.print(f"[red]FASTA not found: {fasta}[/red]")
            raise typer.Exit(code=1)
        console.print(f"Loading NCBI sequences from {fasta} ...")
        df = build_record_dataframe(fasta, default_species="vertebrate")
        frames.append(df)
        console.print(f"  {len(df):,} sequences loaded")

    combined = pd.concat(frames, ignore_index=True)
    console.print(f"\nTotal: {len(combined):,} sequences")

    console.print("Cleaning and normalizing sequences ...")
    combined = preprocess_sequences(
        combined, target_length=target_length, min_n_fraction=min_n_fraction
    )
    qc_pass = int(combined["qc_pass"].sum())
    qc_fail = len(combined) - qc_pass
    console.print(f"QC pass: {qc_pass:,} / {len(combined):,}")
    if qc_fail:
        console.print(f"  Removing {qc_fail:,} sequences with >{min_n_fraction:.0%} N content")
        combined = combined[combined["qc_pass"]].reset_index(drop=True)

    console.print("Stratifying train/val/test split (80/10/10 by haplogroup) ...")
    combined = stratified_split(combined, label_col="haplogroup")

    split_counts = combined["split"].value_counts()
    for split_name in ("train", "val", "test"):
        console.print(f"  {split_name}: {split_counts.get(split_name, 0):,}")

    console.print(f"\nSaving to {output_dir}/ ...")
    paths = save_splits(combined, output_dir)
    for _split_name, path in paths.items():
        console.print(f"  {path}")

    console.print("\n[green]Preprocessing complete.[/green]")
