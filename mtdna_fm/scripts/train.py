import typer

app = typer.Typer(help="Pre-train mtDNA-FM (Phase 1 cross-species or Phase 2 human-specific).")


@app.command()
def main(
    config: str = typer.Option(..., help="Path to training config YAML"),
    model_config: str = typer.Option(None, help="Path to model config YAML (required for Phase 1)"),
    device: str = typer.Option("cpu", help="Device: cpu | cuda | mps"),
) -> None:
    typer.echo("[train] not yet implemented")
    raise typer.Exit(code=1)
