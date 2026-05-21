import typer

app = typer.Typer(help="Download raw datasets (HmtDB, NCBI RefSeq, gnomAD, ClinVar, PhyloTree).")


@app.command()
def main(
    source: str = typer.Option(
        ..., help="Dataset source: hmtdb | ncbi-refseq | gnomad | clinvar | phylotree"
    ),
    output: str = typer.Option("data/raw", help="Output directory"),
) -> None:
    typer.echo(f"[download] source={source}  output={output}  (not yet implemented)")
    raise typer.Exit(code=1)
