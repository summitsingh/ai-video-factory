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
from ai_video_factory.video_pipeline import VideoJob, run_video_pipeline


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


@app.command("video-pipeline")
def video_pipeline(
    topic: str = typer.Argument(..., help="The trending topic for the video"),
    description: str = typer.Option(None, "--description", "-d", help="Video description"),
    source_url: str = typer.Option(None, "--source", "-s", help="Source URL for the topic"),
    output: str = typer.Option("data/projects/generated", "--output", "-o", help="Output directory"),
    script_file: str = typer.Option(None, "--script-file", help="Pre-made worker script JSON to use instead of generating"),
    duration: int = typer.Option(90, "--duration", help="Target video duration in seconds"),
    assets_dir: str = typer.Option(None, "--assets-dir", help="Per-scene stock asset directories (scene-00/, ...)"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
) -> None:
    """Run a complete video production pipeline for a trending topic.
    
    This pipeline:
    1. Research the trending topic
    2. Generate a sourced script using LM Studio
    3. Create a storyboard/edit document
    4. Render video with Remotion
    5. Run quality checks
    6. Render a draft
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        output_path = Path(output)
        
        # Create a video job
        job = VideoJob(
            topic=topic,
            description=description or topic,
            source_url=source_url or "https://example.com",
            output_path=output_path,
            duration_seconds=duration,
            assets_dir=Path(assets_dir) if assets_dir else None,
        )
        
        # Run the pipeline
        result = run_video_pipeline(
            project_root=project_root,
            data_root=project_root / "data",
            job=job,
            script_path=Path(script_file) if script_file else None,
        )
        
        if json_output:
            typer.echo(json.dumps(result.to_dict(), indent=2))
        else:
            typer.echo(f"Video pipeline status: {result.status}")
            typer.echo(f"Run ID: {result.run_id}")
            typer.echo(f"Artifacts: {list(result.artifacts.keys())}")
            
            if result.metadata.get("selected_topic"):
                typer.echo(f"Selected topic: {result.metadata['selected_topic']}")
        
        if result.status != "pass":
            raise typer.Exit(code=2)
            
    except Exception as error:
        typer.echo(sanitize_diagnostic(error), err=True)
        raise typer.Exit(code=2)


@app.command("daily")
def daily(
    topic: str | None = typer.Option(None, "--topic", "-t", help="Fixed topic to research (skips trend discovery)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what a run would do; no network or rendering"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume completed production stages from a prior run"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
) -> None:
    """Run the daily production job (operator-scheduled, local only).

    Discovers a trending topic, researches it with verified facts, builds a
    15-20 minute documentary candidate, runs final QC on real artifacts, and
    produces an upload-ready package pending explicit human approval. No system
    cron or service is created; schedule this command yourself.
    """
    from ai_video_factory.daily_job import JobConfig, run_daily_job

    try:
        project_root = Path(__file__).resolve().parents[2]
        config = JobConfig(dry_run=dry_run)
        if topic:
            config.topic_override = topic
        result = run_daily_job(project_root, config=config)
    except Exception as error:
        from ai_video_factory.sanitization import sanitize_diagnostic

        typer.echo(sanitize_diagnostic(error), err=True)
        raise typer.Exit(code=2)

    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        typer.echo(f"Run ID: {result.run_id or '(dry-run)'}")
        typer.echo(f"Status: {result.status}")
        typer.echo(f"Topic: {result.topic}")
        if result.gates_passed and result.approval_path:
            typer.echo(f"Approval package: {result.approval_path}")
        elif result.failure_reason:
            typer.echo(f"Failed: {result.failure_reason}")

    if result.status != "completed":
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()