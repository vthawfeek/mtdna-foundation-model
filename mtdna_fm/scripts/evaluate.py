import typer

app = typer.Typer(help="Evaluate a trained mtDNA-FM model and write metrics to reports/.")


@app.command()
def main(
    model: str = typer.Option(..., help="Path to model directory"),
    output_dir: str = typer.Option("reports", help="Output directory for metrics"),
) -> None:
    typer.echo("[evaluate] not yet implemented")
    raise typer.Exit(code=1)
