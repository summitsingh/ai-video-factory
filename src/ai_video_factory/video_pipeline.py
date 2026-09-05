"""Video Production Pipeline for AI Video Factory.

Orchestrates research, script generation, storyboard creation,
video rendering, and quality checks for trending topic videos.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ai_video_factory.edit_schema import EditDocument, EditScene, load_edit, save_edit
from ai_video_factory.lm_studio import LmStudioError
from ai_video_factory.media_probe import MediaProbeError, probe_media
from ai_video_factory.narration import NarrationError, mix_scenes_to_track, synthesize_to_wav
from ai_video_factory.qc import evaluate_qc, write_qc_reports
from ai_video_factory.research import TrendingTopic, research_trending_topics, save_research_result
from ai_video_factory.run_store import RunStore
from ai_video_factory.sanitization import first_diagnostic_line, sanitize_diagnostic
from ai_video_factory.script_generator import ScriptOutput, generate_script


# ========== Types and Models ==========

RenderStage = Callable[[Path, Path], None]
ValidateStage = Callable[[Path, Path], dict[str, str]]


class PipelineCommandError(RuntimeError):
    """Raised when one of the fixed local media commands fails."""


class LocalBrowserError(PipelineCommandError):
    """Raised when no verified local browser can be supplied to Remotion."""


@dataclass(frozen=True)
class PipelineResult:
    schema_version: Literal[1]
    command: Literal["video-pipeline"]
    status: Literal["pass", "fail"]
    run_id: str | None
    resumed: bool
    retryable: bool
    artifacts: dict[str, str]
    error: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VideoJob:
    """A complete video production job."""
    topic: str
    description: str
    source_url: str
    output_path: Path
    title: str | None = None
    description_override: str | None = None
    
    # Output paths
    research_path: Path | None = None
    script_path: Path | None = None
    edit_path: Path | None = None
    master_path: Path | None = None
    qc_report_path: Path | None = None
    
    # Runtime state
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"
    error: str | None = None


# ========== Helper Functions ==========

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pipeline_inputs(
    project_root: Path,
    fixture: Path,
    lockfile: Path,
    tools: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "fixture_sha256": _sha256(fixture),
        "remotion_lockfile_sha256": _sha256(lockfile),
        "provenance": {
            "files": _renderer_file_digests(project_root, fixture, lockfile),
            "tools": dict(tools),
            "command": {
                "render": "npm run render -- --props <fixture> <temporary-output>",
                "browser": {
                    "required": True,
                    "environment": "REMOTION_CHROME_EXECUTABLE",
                    "candidates": [
                        "google-chrome",
                        "google-chrome-stable",
                        "chromium",
                        "chromium-browser",
                    ],
                },
                "audio_filter": "espeak-ng scene narration mixed with adelay/amix, apad to video length (silent anullsrc fallback)",
                "audio_codec": "aac",
                "audio_bitrate": "192k",
                "shortest": True,
            },
        },
    }


def _renderer_file_digests(
    project_root: Path, fixture: Path, lockfile: Path
) -> dict[str, dict[str, Any]]:
    remotion_root = project_root / "remotion"
    required = [
        fixture,
        lockfile,
        remotion_root / "package.json",
        remotion_root / "tsconfig.json",
        remotion_root / "render.ts",
    ]
    source_root = remotion_root / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"renderer source directory is missing: {source_root}")
    files = required + sorted(path for path in source_root.rglob("*") if path.is_file())
    files.append(Path(__file__).resolve())

    records: dict[str, dict[str, Any]] = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"render-determining file is missing: {path}")
        try:
            name = path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            name = "python:ai_video_factory/pipeline.py"
        records[name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    return dict(sorted(records.items()))


def _collect_pipeline_toolchain() -> dict[str, dict[str, Any]]:
    tools: dict[str, dict[str, Any]] = {}
    
    # Find browser
    configured = os.environ.get("REMOTION_CHROME_EXECUTABLE")
    if configured:
        browser = _verified_executable(Path(configured))
        if browser is not None:
            tools["browser"] = _executable_identity(browser, ("--version",))
        else:
            tools["browser"] = {"status": "not_ready", "error": str(configured)}
    else:
        for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            located = shutil.which(candidate)
            if located is not None:
                browser = _verified_executable(Path(located))
                if browser is not None:
                    tools["browser"] = _executable_identity(browser, ("--version",))
                    break
    
    # Find other tools
    for name, executable, version_arguments in (
        ("ffmpeg", "ffmpeg", ("-version",)),
        ("ffprobe", "ffprobe", ("-version",)),
        ("node", "node", ("--version",)),
        ("npm", "npm", ("--version",)),
    ):
        try:
            tools[name] = _executable_identity(
                _find_local_executable(executable), version_arguments
            )
        except (OSError, PipelineCommandError, subprocess.TimeoutExpired) as error:
            tools[name] = {"status": "not_ready", "error": str(error)}
    
    return tools


def _find_local_executable(name: str) -> Path:
    located = shutil.which(name)
    if located is None:
        raise PipelineCommandError(f"{name} executable was not found")
    executable = _verified_executable(Path(located))
    if executable is None:
        raise PipelineCommandError(f"{name} does not resolve to an executable file")
    return executable


def _executable_identity(path: Path, version_arguments: Sequence[str]) -> dict[str, Any]:
    resolved = _verified_executable(path)
    if resolved is None:
        raise PipelineCommandError(f"tool does not resolve to an executable file: {path}")
    try:
        completed = subprocess.run(
            (str(resolved), *version_arguments),
            shell=False,
            timeout=15,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PipelineCommandError(sanitize_diagnostic(
            f"tool identity command could not run: {sanitize_diagnostic(error)}"
        )) from error
    if completed.returncode != 0:
        detail = (
            first_diagnostic_line(completed.stderr)
            or first_diagnostic_line(completed.stdout)
            or f"exit code {completed.returncode}"
        )
        raise PipelineCommandError(
            sanitize_diagnostic(f"tool identity command failed: {detail}")
        )
    version = first_diagnostic_line(completed.stdout) or first_diagnostic_line(completed.stderr)
    if version is None:
        raise PipelineCommandError("tool identity command returned no version")
    return {
        "status": "ready",
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "version": version,
    }


def _verified_executable(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    if resolved.is_file() and os.access(resolved, os.X_OK):
        return resolved
    return None


def _required_tool(tools: Mapping[str, Mapping[str, Any]], name: str) -> Path:
    record = tools.get(name)
    if record is None or record.get("status") != "ready" or not record.get("path"):
        detail = record.get("error") if record is not None else f"{name} identity is missing"
        raise PipelineCommandError(sanitize_diagnostic(detail))
    return Path(str(record["path"]))


def _render_with_remotion(
    project_root: Path,
    fixture: Path,
    output: Path,
    *,
    browser: Path | None = None,
    npm: Path | None = None,
) -> None:
    remotion_directory = project_root / "remotion"
    browser = browser or _find_local_browser()
    _run_command(
        (
            str(npm) if npm is not None else "npm",
            "run",
            "render",
            "--",
            "--props",
            os.path.relpath(fixture, remotion_directory),
            os.path.relpath(output, remotion_directory),
            "--browser-executable",
            str(browser),
        ),
        cwd=remotion_directory,
        name="Remotion render",
        timeout=600,
    )


def _find_local_browser() -> Path:
    configured = os.environ.get("REMOTION_CHROME_EXECUTABLE")
    if configured:
        browser = _verified_executable(Path(configured))
        if browser is not None:
            return browser
        raise LocalBrowserError(
            "REMOTION_CHROME_EXECUTABLE does not name an existing executable browser"
        )

    for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        located = shutil.which(candidate)
        if located is not None:
            browser = _verified_executable(Path(located))
            if browser is not None:
                return browser
    raise LocalBrowserError(
        "no verified local browser executable; install or configure REMOTION_CHROME_EXECUTABLE"
    )


def _mux_narration_audio(
    source: Path,
    narration_track: Path | None,
    destination: Path,
    *,
    ffmpeg: Path | None = None,
) -> None:
    """Mux the video with the TTS narration track (or silence as fallback).

    The narration track is padded with silence so the video length governs
    the output duration via -shortest.
    """
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = str(ffmpeg) if ffmpeg is not None else "ffmpeg"
    if narration_track is not None:
        _run_command(
            (
                ffmpeg_bin, "-y",
                "-i", str(source),
                "-i", str(Path(narration_track).resolve()),
                "-filter_complex", "[1:a]apad[aud]",
                "-map", "0:v", "-map", "[aud]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-ar", "48000", "-ac", "2",
                "-shortest", str(destination),
            ),
            cwd=destination.parent,
            name="FFmpeg narration mux",
        )
    else:
        _mux_silent_audio(source, destination, ffmpeg=ffmpeg)


def _synthesize_narration_track(
    edit: EditDocument, workdir: Path, *, ffmpeg: Path | None = None
) -> Path | None:
    """Synthesize per-scene narration and mix into one padded track.

    Returns None when no scene carries speakable narration (caller falls
    back to a silent track).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    segments: list[tuple[Path, float]] = []
    for i, scene in enumerate(edit.scenes):
        if not scene.narration or not scene.narration.strip():
            continue
        wav = workdir / f"scene-{i}.wav"
        synthesize_to_wav(scene.narration, wav)
        segments.append((wav, scene.from_frame / edit.fps))
    if not segments:
        return None
    return mix_scenes_to_track(
        segments, workdir / "narration.wav",
        ffmpeg=str(ffmpeg) if ffmpeg is not None else "ffmpeg",
    )


def _mux_silent_audio(
    source: Path, destination: Path, *, ffmpeg: Path | None = None
) -> None:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        (
            str(ffmpeg) if ffmpeg is not None else "ffmpeg",
            "-y",
            "-i", str(source),
            "-f", "lavfi",
            "-i", "anullsrc=r=48000:cl=stereo",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(destination),
        ),
        cwd=destination.parent,
        name="FFmpeg audio mux",
    )


def _run_command(argv: Sequence[str], *, cwd: Path, name: str, timeout: int = 180) -> None:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            shell=False,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PipelineCommandError(sanitize_diagnostic(
            f"{name} could not run: {sanitize_diagnostic(error)}"
        )) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise PipelineCommandError(sanitize_diagnostic(f"{name} failed: {detail}"))


def _validate_master(
    master_path: Path,
    fixture: Path,
    *,
    tools: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    media = (
        probe_media(master_path, _media_runner(tools))
        if tools is not None
        else probe_media(master_path)
    )
    report = evaluate_qc(media, load_edit(fixture))
    json_path, markdown_path = write_qc_reports(report, master_path.parent)
    return {
        "status": report.status,
        "report": str(json_path),
        "report_markdown": str(markdown_path),
    }


def _media_runner(
    tools: Mapping[str, Mapping[str, Any]],
) -> Callable[[Sequence[str]], tuple[int, str, str]]:
    def run(argv: Sequence[str]) -> tuple[int, str, str]:
        if not argv:
            raise ValueError("media command cannot be empty")
        executable = _required_tool(tools, argv[0])
        completed = subprocess.run(
            (str(executable), *argv[1:]),
            shell=False,
            timeout=15,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr
    return run


def _status(result: Mapping[str, Any]) -> Literal["pass", "fail"]:
    return result.get("status", "fail")


# ========== Main Pipeline Function ==========

def run_video_pipeline(
    project_root: Path,
    data_root: Path,
    job: VideoJob,
    *,
    render: RenderStage | None = None,
    validate: ValidateStage | None = None,
    script_path: Path | None = None,
) -> PipelineResult:
    """Run the complete video production pipeline.

    Steps:
    1. Research trending topic
    2. Generate script using LM Studio (or load a pre-made worker script)
    3. Create storyboard/edit document
    4. Render video with Remotion
    5. Run quality checks
    6. Report results

    Args:
        project_root: Root directory of the AI Video Factory project
        data_root: Data directory for storing intermediate files
        job: VideoJob configuration with topic and output paths
        render: Optional custom render stage function
        validate: Optional custom validate stage function
        script_path: Optional pre-made script JSON (worker output) used in
            place of LM Studio generation. Still copied to the job script path.
    """
    state_store: RunStore | None = None
    run_id: str | None = None
    artifacts: dict[str, str] = {}
    metadata: dict[str, Any] = {
        "topic": job.topic,
        "created_at": job.created_at,
    }

    try:
        root = Path(project_root).resolve()
        data_root = Path(data_root).resolve()

        # Setup paths
        project_data = data_root / "projects" / "generated"
        artifact_root = project_data / "runs"
        state_store = RunStore(project_data / "state", artifact_root=artifact_root)

        # Step 1: Research trending topic
        metadata["research_status"] = "in_progress"
        job.research_path = project_data / f"{job.topic.replace(' ', '_')}_research.json"
        research_result = research_trending_topics(max_topics=1, min_engagement=100)
        if not research_result.topics:
            raise PipelineCommandError("No trending topics found for research")
        save_research_result(research_result, job.research_path)
        metadata["research_status"] = "complete"
        metadata["selected_topic"] = research_result.topics[0].title
        artifacts["research"] = str(job.research_path)

        # Step 2: Generate script with LM Studio (or load a worker-made script)
        metadata["script_status"] = "in_progress"
        job.script_path = project_data / f"{job.topic.replace(' ', '_')}_script.json"
        trending_topic = research_result.topics[0]
        if script_path is not None:
            worker_data = json.loads(Path(script_path).read_text(encoding="utf-8"))
            script = ScriptOutput(
                title=str(worker_data["title"]),
                narration=str(worker_data["narration"]),
                scenes=list(worker_data["scenes"]),
                sources=list(worker_data.get("sources", [trending_topic.url])),
                captions=list(worker_data.get("captions", [])),
            )
            job.script_path.write_text(
                json.dumps(worker_data, indent=2), encoding="utf-8"
            )
            metadata["script_origin"] = "worker"
        else:
            script = generate_script(
                topic_title=trending_topic.title,
                topic_description=trending_topic.description,
                source_url=trending_topic.url,
                output_path=job.script_path,
                use_local_model=True,
            )
            metadata["script_origin"] = "local_model"
        job.title = script.title
        metadata["script_status"] = "complete"
        artifacts["script"] = str(job.script_path)

        # Step 3: Create storyboard/edit document
        metadata["edit_status"] = "in_progress"
        job.edit_path = project_data / f"{job.topic.replace(' ', '_')}_edit.json"

        # Create edit document from script
        fps = 30
        duration_seconds = 90  # 90 seconds video
        total_frames = duration_seconds * fps

        # Distribute the full composition duration evenly across scenes so no
        # blank tail remains when the script returns fewer/shorter scenes.
        scene_count = max(len(script.scenes), 1)
        base_frames, remainder = divmod(total_frames, scene_count)
        scenes = []
        cursor = 0
        for i, scene_data in enumerate(script.scenes):
            span = base_frames + (1 if i < remainder else 0)
            scene = EditScene(
                id=f"scene-{i}",
                from_frame=cursor,
                duration_frames=span,
                title=scene_data.get("title", f"Scene {i+1}"),
                caption=scene_data.get("caption", ""),
                visual=scene_data.get("visual"),
                narration=scene_data.get("narration"),
            )
            scenes.append(scene)
            cursor += span

        edit_doc = EditDocument(
            schema_version=1,
            width=1280,
            height=720,
            fps=fps,
            duration_frames=total_frames,
            scenes=scenes if scenes else [
                EditScene(
                    id="scene-0",
                    from_frame=0,
                    duration_frames=total_frames,
                    title=job.title or f"Trending: {job.topic}",
                    caption=job.description or job.topic,
                )
            ],
            title=job.title,
            description=job.description,
            sources=script.sources,
            created_at=job.created_at,
        )
        save_edit(edit_doc, job.edit_path)
        metadata["edit_status"] = "complete"
        artifacts["edit"] = str(job.edit_path)

        # Step 4: Render video with Remotion
        metadata["render_status"] = "in_progress"
        tools = _collect_pipeline_toolchain()

        output_dir = Path(job.output_path)
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        if job.master_path is None:
            job.master_path = output_dir / "master.mp4"
        else:
            job.master_path = Path(job.master_path)
            if not job.master_path.is_absolute():
                job.master_path = root / job.master_path
            job.master_path.parent.mkdir(parents=True, exist_ok=True)

        render_run = state_store.start(
            "video-render",
            _pipeline_inputs(
                root,
                job.edit_path,
                root / "remotion" / "package-lock.json",
                tools,
            ),
        )
        run_id = render_run.run_id
        run_directory = artifact_root / render_run.run_id
        temporary_path = run_directory / "render.tmp.mp4"

        if render_run.resumed:
            # Reuse the previously rendered master instead of re-rendering.
            job.master_path = Path(render_run.artifacts["master"])
        else:
            state_store.event(render_run.run_id, "render_started", {"topic": job.topic})
            run_directory.mkdir(parents=True, exist_ok=True)
            run_master = run_directory / "master.mp4"

            if render is None:
                _render_with_remotion(
                    root,
                    job.edit_path,
                    temporary_path,
                    browser=_required_tool(tools, "browser"),
                    npm=_required_tool(tools, "npm"),
                )
                edit_doc_render = load_edit(job.edit_path)
                narration_track = _synthesize_narration_track(
                    edit_doc_render,
                    run_directory / "narration",
                    ffmpeg=_required_tool(tools, "ffmpeg"),
                )
                if narration_track is not None:
                    metadata["narration_status"] = "complete"
                    metadata["narration_segments"] = len([
                        s for s in edit_doc_render.scenes if (s.narration or "").strip()
                    ])
                else:
                    metadata["narration_status"] = "silent_fallback"
                _mux_narration_audio(
                    temporary_path,
                    narration_track,
                    run_master,
                    ffmpeg=_required_tool(tools, "ffmpeg"),
                )
                temporary_path.unlink(missing_ok=True)
            else:
                render(job.edit_path, run_master)

            render_run = state_store.complete(
                render_run.run_id,
                {"master": str(run_master)},
                expected_artifacts={"master": run_master},
            )
            # Publish a stable draft copy outside the run directory.
            shutil.copy2(run_master, job.master_path)

            job.master_path = run_master

        artifacts["master"] = str(job.master_path)
        artifacts["draft"] = str(output_dir / "master.mp4")
        metadata["render_status"] = "complete"

        # Step 5: Run quality checks
        metadata["qc_status"] = "in_progress"
        qc_inputs = {
            "edit_sha256": _sha256(job.edit_path),
            "render_run_id": render_run.run_id,
            "master_sha256": _sha256(job.master_path),
            "provenance": {
                "tools": tools,
                "full_decode": True,
            },
        }
        qc_run = state_store.start("video-qc", qc_inputs)
        run_id = qc_run.run_id

        if qc_run.resumed:
            result = dict(qc_run.artifacts)
            status = _status(result)
            artifacts["qc_report"] = result.get("report", "")
            artifacts["qc_report_markdown"] = result.get("report_markdown", "")
            job.qc_report_path = Path(result["report"]) if result.get("report") else None
        else:
            state_store.event(qc_run.run_id, "qc_started", {"master": str(job.master_path)})
            result = _validate_master(job.master_path, job.edit_path, tools=tools)
            status = _status(result)
            artifacts["qc_report"] = result.get("report", "")
            artifacts["qc_report_markdown"] = result.get("report_markdown", "")
            job.qc_report_path = Path(result.get("report", "")) if result.get("report") else None

            qc_run = state_store.complete(
                qc_run.run_id,
                {"status": status, **result},
                expected_artifacts={
                    key: Path(value)
                    for key, value in result.items()
                    if key != "status"
                },
            )

        metadata["qc_status"] = "complete"
        metadata["qc_results"] = artifacts.get("qc_report", "not_found")

        return PipelineResult(
            schema_version=1,
            command="video-pipeline",
            status=status,
            run_id=render_run.run_id,
            resumed=render_run.resumed and qc_run.resumed,
            retryable=False,
            artifacts=artifacts,
            error=None if status == "pass" else "technical QC failed",
            metadata=metadata,
        )

    except Exception as error:
        metadata["error"] = str(error)
        job.error = sanitize_diagnostic(error)
        if state_store is not None and run_id is not None:
            try:
                state_store.fail(run_id, error)
            except Exception:
                pass
        return PipelineResult(
            schema_version=1,
            command="video-pipeline",
            status="fail",
            run_id=run_id,
            resumed=False,
            retryable=isinstance(error, (PipelineCommandError, LmStudioError, MediaProbeError, NarrationError)),
            artifacts=artifacts,
            error=job.error,
            metadata=metadata,
        )