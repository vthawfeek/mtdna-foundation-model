import typer

app = typer.Typer(help="Preprocess raw sequence data into clean train/val/test parquet files.")


@app.command()
def main(
    hmtdb: str = typer.Option(None, help="Path to HmtDB raw directory"),
    ncbi: str = typer.Option(None, help="Path to NCBI raw directory"),
    output_dir: str = typer.Option("data/processed", help="Output directory"),
    target_length: int = typer.Option(16569, help="Target sequence length (rCRS is 16569 bp)"),
) -> None:
    typer.echo("[preprocess] not yet implemented")
    raise typer.Exit(code=1)
