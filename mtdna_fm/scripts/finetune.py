import typer

app = typer.Typer(
    help="Fine-tune mtDNA-FM on downstream tasks (haplogroup, pathogenicity, heteroplasmy)."
)


@app.command()
def main(
    task: str = typer.Option(..., help="Task: haplogroup | pathogenicity | heteroplasmy"),
    config: str = typer.Option(..., help="Path to fine-tuning config YAML"),
    model_config: str = typer.Option(None, help="Path to model config YAML"),
) -> None:
    typer.echo("[finetune] not yet implemented")
    raise typer.Exit(code=1)
