import json
from pathlib import Path
from typing import Literal

import typer

from ai_video_factory.benchmark import BenchmarkReport, run_benchmarks
from ai_video_factory.doctor import collect_doctor_report
from ai_video_factory.inference_benchmark import run_capability_benchmark
from ai_video_factory.inference_config import load_inference_config
from ai_video_factory.inference_models import InferenceCheck, InferenceResult
from ai_video_factory.inference_service import InferenceService
from ai_video_factory.pipeline import failed_pipeline_result, run_synthetic_pipeline
from ai_video_factory.run_store import RunStore
from ai_video_factory.sanitization import sanitize_diagnostic

app = typer.Typer(no_args_is_help=True)
inference_app = typer.Typer(no_args_is_help=True)
app.add_typer(inference_app, name="inference")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _PROJECT_ROOT / "data"
_MODEL_IDENTIFIER = "avf-qwen36-executor"
_InferenceCommand = Literal[
    "doctor", "estimate", "start", "status", "benchmark", "stop"
]


def build_inference_service() -> InferenceService:
    """Build the configured local inference boundary from repository data."""
    config = load_inference_config(_PROJECT_ROOT / "config" / "inference.toml")
    return InferenceService(config)


def _failed_inference_result(
    command: _InferenceCommand, error: BaseException
) -> InferenceResult:
    return InferenceResult(
        command=command,
        status="fail",
        retryable=False,
        model_identifier=_MODEL_IDENTIFIER,
        checks={"operation": InferenceCheck(status="not_ready", detail=None)},
        metrics={},
        artifacts={},
        error=sanitize_diagnostic(error),
    )


def _run_inference_command(command: _InferenceCommand) -> None:
    try:
        service = build_inference_service()
        if command == "benchmark":
            project_data = _DATA_ROOT / "projects" / "system"
            store = RunStore(
                project_data / "state",
                artifact_root=project_data / "runs",
            )
            result = run_capability_benchmark(service, store, _DATA_ROOT)
        else:
            operation = getattr(service, command)
            result = operation()
    except Exception as error:
        result = _failed_inference_result(command, error)

    typer.echo(result.model_dump_json())
    if result.status != "pass":
        raise typer.Exit(code=2)


@inference_app.command("doctor")
def inference_doctor() -> None:
    """Inspect local LM Studio readiness without changing residency."""
    _run_inference_command("doctor")


@inference_app.command("estimate")
def inference_estimate() -> None:
    """Estimate configured model memory without loading it."""
    _run_inference_command("estimate")


@inference_app.command("start")
def inference_start() -> None:
    """Memory-gate and load only the configured stable identifier."""
    _run_inference_command("start")


@inference_app.command("status")
def inference_status() -> None:
    """Inspect configured model residency without changing it."""
    _run_inference_command("status")


@inference_app.command("benchmark")
def inference_benchmark() -> None:
    """Run the deterministic three-probe local capability benchmark."""
    _run_inference_command("benchmark")


@inference_app.command("stop")
def inference_stop() -> None:
    """Unload only the configured stable identifier."""
    _run_inference_command("stop")


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
def test_pipeline(json_output: bool = typer.Option(False, "--json")) -> None:
    """Run the synthetic local video fixture."""
    try:
        project_root = Path(__file__).resolve().parents[2]
        result = run_synthetic_pipeline(project_root, project_root / "data")
    except Exception as error:
        result = failed_pipeline_result(error)
    if json_output:
        typer.echo(json.dumps(result.to_dict(), sort_keys=True))
    else:
        typer.echo(result.status)
    if result.status != "pass":
        raise typer.Exit(code=2)
