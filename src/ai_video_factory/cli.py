import typer

from ai_video_factory.benchmark import BenchmarkReport, run_benchmarks
from ai_video_factory.doctor import collect_doctor_report

app = typer.Typer(no_args_is_help=True)


@app.command()
def doctor() -> None:
    """Report readiness without modifying the host."""
    typer.echo(collect_doctor_report().model_dump_json(indent=2))


@app.command()
def benchmark() -> None:
    """Probe installed tools without downloading models."""
    report = BenchmarkReport(schema_version=1, probes=run_benchmarks())
    typer.echo(report.model_dump_json(indent=2))


@app.command("test-pipeline")
def test_pipeline() -> None:
    """Run the synthetic local video fixture."""
