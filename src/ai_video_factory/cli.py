import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def doctor() -> None:
    """Report readiness without modifying the host."""


@app.command()
def benchmark() -> None:
    """Probe installed tools without downloading models."""


@app.command("test-pipeline")
def test_pipeline() -> None:
    """Run the synthetic local video fixture."""
