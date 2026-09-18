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
from ai_video_factory.theme import resolve_theme
from ai_video_factory.video_pipeline import VideoJob, run_video_pipeline
from ai_video_factory.youtube import (
    DEFAULT_PRIVACY,
    PRIVACY_CHOICES,
    YouTubeError,
    build_service,
    load_credentials,
    quota_path,
    request_from_run,
    run_auth_flow,
    token_path,
    upload_package,
    set_video_privacy,
    QuotaGuard,
    UploadRequest,
    _default_data_dir,
)


app = typer.Typer(no_args_is_help=True)
inference_app = typer.Typer(no_args_is_help=True)
app.add_typer(inference_app, name="inference")
youtube_app = typer.Typer(no_args_is_help=True)
app.add_typer(youtube_app, name="youtube")
scheduler_app = typer.Typer(no_args_is_help=True)
app.add_typer(scheduler_app, name="scheduler")
review_app = typer.Typer(no_args_is_help=True)
app.add_typer(review_app, name="review")

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
    trend_source: str = typer.Option("all", "--trend-source", help="Trend providers to query: reddit, gnews, or all"),
    theme: str = typer.Option("space", "--theme", help="Theming preset: space (default) or generic"),
    theme_json: str = typer.Option(None, "--theme-json", help="Path to a custom theme JSON file (overrides --theme)"),
    longform: bool = typer.Option(False, "--longform", help="Generate a 20-30 minute documentary script (three-act structure) instead of a 90-second short"),
    duration_minutes: float = typer.Option(25.0, "--duration-minutes", help="Target runtime in minutes for --longform (20-30)"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
    llm_url: str = typer.Option("http://localhost:1234/v1/chat/completions", "--llm-url", help="OpenAI-compatible chat completions URL for script generation (LM Studio default, or llama-server e.g. http://localhost:8080/v1/chat/completions)"),
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

        # Resolve the theming preset (defaults to the space theme, which
        # preserves the original pipeline behavior).
        theme_config = resolve_theme(theme, Path(theme_json) if theme_json else None)
        
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
            theme=theme_config,
            trend_source=trend_source,
            longform=longform,
            target_duration_minutes=duration_minutes,
            llm_url=llm_url,
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




def _youtube_data_dir(data_dir):
    if data_dir:
        return Path(data_dir)
    return _default_data_dir(Path(__file__).resolve().parents[2])


@youtube_app.command("auth")
def youtube_auth(
    client_secrets: str = typer.Option(
        ..., "--client-secrets", help="Path to the OAuth client_secrets.json from Google Cloud Console"
    ),
    data_dir: str = typer.Option(None, "--data-dir", help="YouTube state directory (default: data/youtube)"),
) -> None:
    """Run the one-time installed-app OAuth flow and store the refresh token."""
    try:
        stored = run_auth_flow(Path(client_secrets), _youtube_data_dir(data_dir))
    except YouTubeError as error:
        typer.echo(f"YouTube auth failed: {error}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"Stored YouTube credentials at {stored}")


@youtube_app.command("status")
def youtube_status(
    data_dir: str = typer.Option(None, "--data-dir", help="YouTube state directory (default: data/youtube)"),
    json_output: bool = typer.Option(False, "--json", help="Output status as JSON"),
) -> None:
    """Show YouTube auth state and today's quota usage."""
    directory = _youtube_data_dir(data_dir)
    quota = QuotaGuard(directory)
    try:
        load_credentials(directory)
        authorized = True
        auth_error = None
    except YouTubeError as error:
        authorized = False
        auth_error = str(error)
    report = {
        "authorized": authorized,
        "auth_error": auth_error,
        "token_path": str(token_path(directory)),
        "quota_path": str(quota_path(directory)),
        "quota_budget": quota.daily_budget,
        "quota_used_today": quota.units_used_today(),
        "quota_remaining": quota.units_remaining(),
    }
    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        typer.echo(f"Authorized: {'yes' if authorized else 'no'}")
        if auth_error:
            typer.echo(f"Auth: {auth_error}")
        typer.echo(
            f"Quota today: {report['quota_used_today']}/{report['quota_budget']} "
            f"used ({report['quota_remaining']} remaining)"
        )


def _parse_tags(raw):
    if not raw:
        return []
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def _load_chapters_file(path):
    if not path:
        return None
    chapters_path = Path(path)
    if not chapters_path.is_file():
        raise YouTubeError(f"chapters file not found: {chapters_path}")
    try:
        payload = json.loads(chapters_path.read_text(encoding="utf-8"))
        chapters = payload["chapters"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise YouTubeError(f"chapters file is not a chapters JSON manifest: {chapters_path}") from error
    if not isinstance(chapters, list):
        raise YouTubeError(f"chapters file has no chapter list: {chapters_path}")
    return chapters


@youtube_app.command("upload")
def youtube_upload(
    run_id: str = typer.Option(None, "--run-id", help="Completed pipeline run id to upload"),
    video: str = typer.Option(None, "--video", help="Video file to upload (alternative to --run-id)"),
    title: str = typer.Option(None, "--title", help="Video title (defaults to the run topic)"),
    description: str = typer.Option(None, "--description", help="Video description (defaults to the run description)"),
    tags: str = typer.Option(None, "--tags", help="Comma-separated tags"),
    category_id: str = typer.Option("28", "--category-id", help="YouTube category id (default 28, Science and Technology)"),
    language: str = typer.Option("en", "--language", help="Video language code"),
    privacy: str = typer.Option(DEFAULT_PRIVACY, "--privacy", help="Privacy status: unlisted, private, or public"),
    confirm_public: bool = typer.Option(False, "--confirm-public", help="Required to publish as public"),
    thumbnail: str = typer.Option(None, "--thumbnail", help="Thumbnail PNG/JPEG (alternative to --thumbnail-choice)"),
    thumbnail_choice: int = typer.Option(1, "--thumbnail-choice", help="Which pipeline thumbnail to use (1-3)"),
    captions: str = typer.Option(None, "--captions", help="Captions SRT file (alternative to the run SRT)"),
    chapters_json: str = typer.Option(None, "--chapters-json", help="Chapters JSON manifest (alternative to the run chapters)"),
    data_dir: str = typer.Option(None, "--data-dir", help="YouTube state directory (default: data/youtube)"),
    project_data: str = typer.Option(None, "--project-data", help="Pipeline data root (default: data)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate everything without calling the YouTube API"),
    json_output: bool = typer.Option(False, "--json", help="Output the result as JSON"),
) -> None:
    """Upload a finished video to YouTube (default privacy: unlisted)."""
    try:
        youtube_dir = _youtube_data_dir(data_dir)
        project_root = Path(__file__).resolve().parents[2]
        data_root = Path(project_data) if project_data else project_root / "data"

        if privacy not in PRIVACY_CHOICES:
            raise YouTubeError(f"privacy must be one of {', '.join(PRIVACY_CHOICES)}")

        if run_id is not None:
            request = request_from_run(
                data_root,
                run_id,
                title=title,
                description=description,
                thumbnail_choice=thumbnail_choice,
                privacy_status=privacy,
                confirm_public=confirm_public,
                tags=_parse_tags(tags),
                dry_run=dry_run,
            )
            if thumbnail is not None:
                request.thumbnail_path = Path(thumbnail)
            if captions is not None:
                request.captions_path = Path(captions)
        else:
            if video is None:
                raise YouTubeError("either --run-id or --video is required")
            if title is None:
                raise YouTubeError("either --run-id or --title is required")
            request = UploadRequest(
                video_path=Path(video),
                title=title,
                description=description or "",
                tags=_parse_tags(tags),
                category_id=category_id,
                language=language,
                privacy_status=privacy,
                confirm_public=confirm_public,
                thumbnail_path=Path(thumbnail) if thumbnail else None,
                captions_path=Path(captions) if captions else None,
                chapters=_load_chapters_file(chapters_json),
                dry_run=dry_run,
            )

        def progress(uploaded: int, total: int) -> None:
            if total > 0:
                typer.echo(f"Upload progress: {uploaded * 100 // total}%", err=True)

        quota = QuotaGuard(youtube_dir)
        if dry_run:
            result = upload_package(request, service=None, quota=quota, progress=None)
        else:
            credentials = load_credentials(youtube_dir)
            service = build_service(credentials)
            result = upload_package(request, service=service, quota=quota, progress=progress)

        if json_output:
            typer.echo(json.dumps(result.to_dict(), indent=2))
        elif result.dry_run:
            typer.echo("Dry run passed: video, thumbnail, captions, and chapters are valid.")
            typer.echo(f"Title: {result.title}")
            typer.echo(f"Privacy: {result.privacy_status}")
        else:
            typer.echo(f"Uploaded: {result.url}")
            typer.echo(f"Video id: {result.video_id} (privacy: {result.privacy_status})")
            typer.echo(f"Thumbnail: {'yes' if result.thumbnail_uploaded else 'no'}")
            typer.echo(f"Captions: {'yes' if result.captions_uploaded else 'no'}")
            typer.echo(f"Quota used today: {result.quota_units_used}")
    except YouTubeError as error:
        typer.echo(f"YouTube upload failed: {error}", err=True)
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Scheduler + review queue (automated channel publishing)
# ---------------------------------------------------------------------------

def _scheduler_dirs(data_dir: str | None) -> tuple[Path, Path]:
    youtube_dir = _youtube_data_dir(data_dir)
    project_root = Path(__file__).resolve().parents[2]
    data_root = Path(data_dir).parent if data_dir else project_root / "data"
    return youtube_dir, data_root


def _load_scheduler_configs(channels_config: str | None) -> list:
    from ai_video_factory.scheduler import load_channels_config

    default = Path(__file__).resolve().parents[2] / "config" / "channels.toml"
    return load_channels_config(Path(channels_config) if channels_config else default)


def _research_for_channel(config) -> list[dict]:
    from ai_video_factory.research import research_trending_topics

    result = research_trending_topics(
        max_topics=10, min_engagement=100, trend_source=config.trend_source
    )
    candidates = [
        {"title": t.title, "description": t.description, "url": t.url}
        for t in result.topics
    ]
    if config.research_queries:
        queries = [q.lower() for q in config.research_queries]

        def _score(candidate: dict) -> int:
            text = (candidate["title"] + " " + candidate["description"]).lower()
            return sum(1 for q in queries if q in text)

        candidates.sort(key=_score, reverse=True)
    return candidates


def _pipeline_for_channel(config, data_root: Path):
    def _run(topic_title: str, topic_description: str) -> dict:
        from ai_video_factory.video_pipeline import VideoJob, run_video_pipeline

        job = VideoJob(
            topic=topic_title,
            description=topic_description or topic_title,
            source_url="https://example.com",
            output_path=data_root / "projects" / "generated",
            duration_seconds=int(config.target_duration_minutes * 60),
        )
        result = run_video_pipeline(
            project_root=Path(__file__).resolve().parents[2],
            data_root=data_root,
            job=job,
            theme=resolve_theme(config.theme),
            trend_source=config.trend_source,
            longform=True,
            target_duration_minutes=config.target_duration_minutes,
        )
        return {
            "status": result.status,
            "run_id": result.run_id,
            "artifacts": result.artifacts,
            "error": result.error,
        }

    return _run


def _upload_for_channel(config, youtube_dir: Path, data_root: Path, dry_run: bool):
    def _upload(pipeline_result: dict) -> dict:
        quota = QuotaGuard(youtube_dir)
        request = request_from_run(
            data_root,
            str(pipeline_result["run_id"]),
            privacy_status=config.privacy,
            tags=list(config.tags),
        )
        request.category_id = config.category_id
        request.language = config.language
        if dry_run:
            result = upload_package(request, service=None, quota=quota)
        else:
            credentials = load_credentials(youtube_dir)
            service = build_service(credentials)
            result = upload_package(request, service=service, quota=quota)
        return {"video_id": result.video_id or "", "title": result.title}

    return _upload


@scheduler_app.command("run")
def scheduler_run(
    channels_config: str = typer.Option(None, "--channels-config", help="Path to channels TOML (default: config/channels.toml)"),
    channel: str = typer.Option(None, "--channel", help="Only run this channel slug"),
    force: bool = typer.Option(False, "--force", help="Run even if the channel is not due yet"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Pick topics and report; do not run the pipeline or upload"),
    data_dir: str = typer.Option(None, "--data-dir", help="YouTube state directory (default: data/youtube)"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
) -> None:
    """Run due channels: research, long-form render, unlisted upload, review queue."""
    from ai_video_factory.scheduler import (
        ReviewQueue,
        SchedulerBusy,
        TopicLedger,
        is_channel_due,
        run_channel,
        scheduler_lock,
    )

    try:
        youtube_dir, data_root = _scheduler_dirs(data_dir)
        configs = _load_scheduler_configs(channels_config)
        if channel:
            configs = [c for c in configs if c.slug == channel]
            if not configs:
                raise typer.BadParameter(f"unknown channel slug: {channel}")
        state_dir = youtube_dir / "state"
        due = [c for c in configs if force or is_channel_due(c, state_dir)]
        if dry_run:
            ledger = TopicLedger(youtube_dir / "topics.jsonl")
            plan = []
            for config in due:
                candidate = None
                try:
                    from ai_video_factory.scheduler import pick_topic

                    candidate = pick_topic(config, ledger, _research_for_channel)
                except Exception as error:
                    plan.append({"channel": config.slug, "topic": None, "error": sanitize_diagnostic(error)})
                    continue
                plan.append(
                    {
                        "channel": config.slug,
                        "topic": candidate["title"] if candidate else None,
                        "target_duration_minutes": config.target_duration_minutes,
                    }
                )
            typer.echo(json.dumps(plan, indent=2))
            return

        results = []
        with scheduler_lock(youtube_dir):
            ledger = TopicLedger(youtube_dir / "topics.jsonl")
            review_queue = ReviewQueue(youtube_dir / "review.jsonl")
            for config in due:
                try:
                    result = run_channel(
                        config,
                        data_dir=youtube_dir,
                        ledger=ledger,
                        review_queue=review_queue,
                        research_fn=_research_for_channel,
                        pipeline_fn=_pipeline_for_channel(config, data_root),
                        upload_fn=_upload_for_channel(config, youtube_dir, data_root, dry_run=False),
                    )
                except (SchedulerBusy, Exception) as error:
                    result = {"channel": config.slug, "status": "error", "error": sanitize_diagnostic(error)}
                results.append(result)
        if json_output:
            typer.echo(json.dumps(results, indent=2))
        else:
            for result in results:
                status = result.get("status")
                extra = result.get("video_id") or result.get("reason") or result.get("error") or ""
                typer.echo(f"{result.get('channel')}: {status} {extra}".rstrip())
    except SchedulerBusy:
        typer.echo("Another scheduler run is already in progress; exiting.", err=True)
        raise typer.Exit(code=3)
    except Exception as error:
        typer.echo(sanitize_diagnostic(error), err=True)
        raise typer.Exit(code=2)


@scheduler_app.command("due")
def scheduler_due(
    channels_config: str = typer.Option(None, "--channels-config", help="Path to channels TOML"),
    data_dir: str = typer.Option(None, "--data-dir", help="YouTube state directory (default: data/youtube)"),
) -> None:
    """List channels that are due for their daily run."""
    from ai_video_factory.scheduler import is_channel_due

    youtube_dir, _ = _scheduler_dirs(data_dir)
    state_dir = youtube_dir / "state"
    for config in _load_scheduler_configs(channels_config):
        due = is_channel_due(config, state_dir)
        typer.echo(f"{config.slug}: {'DUE' if due else 'not due'} (scheduled {config.schedule_time} {config.timezone})")


@scheduler_app.command("status")
def scheduler_status(
    channels_config: str = typer.Option(None, "--channels-config", help="Path to channels TOML"),
    data_dir: str = typer.Option(None, "--data-dir", help="YouTube state directory (default: data/youtube)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show per-channel monitoring: last run, due state, topics used, pending reviews."""
    from ai_video_factory.scheduler import channel_status

    youtube_dir, _ = _scheduler_dirs(data_dir)
    rows = channel_status(_load_scheduler_configs(channels_config), youtube_dir)
    if json_output:
        typer.echo(json.dumps(rows, indent=2))
        return
    for row in rows:
        typer.echo(
            f"{row['channel']}: due={row['due']} last={row['last_run_date']} "
            f"status={row['last_status']} topics={row['topics_used']} "
            f"pending_review={row['pending_review']}"
        )


@review_app.command("list")
def review_list(
    status: str = typer.Option(None, "--status", help="Filter by status (default: pending_review)"),
    data_dir: str = typer.Option(None, "--data-dir", help="YouTube state directory (default: data/youtube)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List videos awaiting human review."""
    from ai_video_factory.scheduler import ReviewQueue

    youtube_dir, _ = _scheduler_dirs(data_dir)
    entries = ReviewQueue(youtube_dir / "review.jsonl").list(status=status or "pending_review")
    if json_output:
        typer.echo(json.dumps(entries, indent=2))
        return
    if not entries:
        typer.echo("No videos awaiting review.")
        return
    for entry in entries:
        typer.echo(f"{entry['video_id']} [{entry['status']}] {entry['channel']}: {entry['title']}")
        typer.echo(f"  {entry['url']}")


@review_app.command("approve")
def review_approve(
    video_id: str = typer.Argument(..., help="YouTube video id to approve"),
    publish: bool = typer.Option(False, "--publish", help="Also set the video public on YouTube"),
    data_dir: str = typer.Option(None, "--data-dir", help="YouTube state directory (default: data/youtube)"),
) -> None:
    """Approve a reviewed video. Add --publish to make it public on YouTube."""
    from ai_video_factory.scheduler import ReviewQueue
    from ai_video_factory.youtube import YouTubeError

    try:
        youtube_dir, _ = _scheduler_dirs(data_dir)
        queue = ReviewQueue(youtube_dir / "review.jsonl")
        if publish:
            credentials = load_credentials(youtube_dir)
            service = build_service(credentials)
            quota = QuotaGuard(youtube_dir)
            privacy = set_video_privacy(service, video_id, "public", quota)
            entry = queue.set_status(video_id, "published")
            typer.echo(f"Published {video_id} (privacy now {privacy}).")
        else:
            entry = queue.set_status(video_id, "approved")
            typer.echo(f"Approved {video_id}; still unlisted until published.")
        _ = entry
    except YouTubeError as error:
        typer.echo(f"Review approve failed: {error}", err=True)
        raise typer.Exit(code=2)
    except Exception as error:
        typer.echo(sanitize_diagnostic(error), err=True)
        raise typer.Exit(code=2)


@review_app.command("reject")
def review_reject(
    video_id: str = typer.Argument(..., help="YouTube video id to reject"),
    data_dir: str = typer.Option(None, "--data-dir", help="YouTube state directory (default: data/youtube)"),
) -> None:
    """Reject a reviewed video (keeps it unlisted; record the decision)."""
    from ai_video_factory.scheduler import ReviewQueue, SchedulerError

    try:
        youtube_dir, _ = _scheduler_dirs(data_dir)
        queue = ReviewQueue(youtube_dir / "review.jsonl")
        queue.set_status(video_id, "rejected")
        typer.echo(f"Rejected {video_id}.")
    except SchedulerError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2)



if __name__ == "__main__":
    app()