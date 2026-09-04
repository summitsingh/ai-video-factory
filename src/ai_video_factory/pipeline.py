"""Synthetic local render orchestration exposed through the Hermes CLI."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ai_video_factory.edit_schema import load_edit
from ai_video_factory.media_probe import MediaProbeError, probe_media
from ai_video_factory.qc import evaluate_qc, write_qc_reports
from ai_video_factory.run_store import RunStore


RenderStage = Callable[[Path, Path], None]
ValidateStage = Callable[[Path, Path], dict[str, str]]


class PipelineCommandError(RuntimeError):
    """Raised when one of the fixed local media commands fails."""


class LocalBrowserError(PipelineCommandError):
    """Raised when no verified local browser can be supplied to Remotion."""


@dataclass(frozen=True)
class PipelineResult:
    schema_version: Literal[1]
    command: Literal["test-pipeline"]
    status: Literal["pass", "fail"]
    run_id: str | None
    resumed: bool
    retryable: bool
    artifacts: dict[str, str]
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def failed_pipeline_result(
    error: Exception,
    *,
    run_id: str | None = None,
    resumed: bool = False,
    artifacts: dict[str, str] | None = None,
) -> PipelineResult:
    """Create the stable failure response used by the pipeline and its CLI guard."""
    return PipelineResult(
        schema_version=1,
        command="test-pipeline",
        status="fail",
        run_id=run_id,
        resumed=resumed,
        retryable=_is_retryable(error),
        artifacts=artifacts or {},
        error=str(error),
    )


def run_synthetic_pipeline(
    project_root: Path,
    data_root: Path,
    *,
    render: RenderStage | None = None,
    validate: ValidateStage | None = None,
) -> PipelineResult:
    """Render and technically validate the checked-in synthetic fixture.

    Injected stages are deliberately narrow test seams.  The production path
    always creates a muted Remotion intermediate and muxes a silent AAC track
    before probing and evaluating QC.
    """
    state_store: RunStore | None = None
    render_run = None
    qc_run = None
    artifacts: dict[str, str] = {}

    try:
        root = Path(project_root).resolve()
        fixture = root / "fixtures" / "synthetic-edit.json"
        lockfile = root / "remotion" / "package-lock.json"
        inputs = _pipeline_inputs(fixture, lockfile)
        state_store = RunStore(Path(data_root) / "projects" / "synthetic" / "state")
        load_edit(fixture)
        render_run = state_store.start("synthetic-render", inputs)
        run_directory = Path(data_root) / "projects" / "synthetic" / "runs" / render_run.run_id
        master_path = run_directory / "master.mp4"

        if not render_run.resumed:
            state_store.event(render_run.run_id, "render_started", {"fixture": str(fixture)})
            run_directory.mkdir(parents=True, exist_ok=True)
            if render is None:
                temporary_path = run_directory / "render.tmp.mp4"
                _render_with_remotion(root, fixture, temporary_path)
                _mux_silent_audio(temporary_path, master_path)
                temporary_path.unlink(missing_ok=True)
            else:
                render(fixture, master_path)
            render_run = state_store.complete(
                render_run.run_id,
                {"master": str(master_path)},
            )

        master_path = Path(render_run.artifacts["master"])
        artifacts["master"] = str(master_path)
        qc_inputs = {**inputs, "render_run_id": render_run.run_id, "master_sha256": _sha256(master_path)}
        qc_run = state_store.start("synthetic-qc", qc_inputs)

        if not qc_run.resumed:
            state_store.event(qc_run.run_id, "qc_started", {"master": str(master_path)})
            result = (
                validate(master_path, master_path.parent)
                if validate is not None
                else _validate_master(master_path, fixture)
            )
            status = _status(result)
            artifacts.update({key: str(value) for key, value in result.items() if key != "status"})
            state_store.complete(qc_run.run_id, {"status": status, **artifacts})
            if status == "fail":
                state_store.event(qc_run.run_id, "qc_failed", {"status": status})
        else:
            status = _status(qc_run.artifacts)
            artifacts.update(
                {key: str(value) for key, value in qc_run.artifacts.items() if key != "status"}
            )

        return PipelineResult(
            schema_version=1,
            command="test-pipeline",
            status=status,
            run_id=render_run.run_id,
            resumed=render_run.resumed and qc_run.resumed,
            retryable=False,
            artifacts=artifacts,
            error=None if status == "pass" else "technical QC failed",
        )
    except Exception as error:
        active_run = qc_run or render_run
        if active_run is not None and state_store is not None:
            state_store.event(
                active_run.run_id,
                "stage_failed",
                {"error": str(error), "retryable": _is_retryable(error)},
            )
        return failed_pipeline_result(
            error,
            run_id=render_run.run_id if render_run is not None else None,
            resumed=bool(active_run and active_run.resumed),
            artifacts=artifacts,
        )


def _pipeline_inputs(fixture: Path, lockfile: Path) -> dict[str, Any]:
    return {
        "fixture_sha256": _sha256(fixture),
        "remotion_lockfile_sha256": _sha256(lockfile),
        "command": {
            "render": "npm run render -- --props <fixture> <temporary-output>",
            "browser": {
                "required": True,
                "environment": "REMOTION_CHROME_EXECUTABLE",
                "candidates": ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"],
            },
            "audio_filter": "anullsrc=r=48000:cl=stereo",
            "audio_codec": "aac",
            "audio_bitrate": "192k",
            "shortest": True,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_with_remotion(project_root: Path, fixture: Path, output: Path) -> None:
    remotion_directory = project_root / "remotion"
    browser = _find_local_browser()
    _run_command(
        (
            "npm",
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


def _verified_executable(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    if resolved.is_file() and os.access(resolved, os.X_OK):
        return resolved
    return None


def _mux_silent_audio(source: Path, destination: Path) -> None:
    _run_command(
        (
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(destination),
        ),
        cwd=destination.parent,
        name="FFmpeg audio mux",
    )


def _run_command(argv: Sequence[str], *, cwd: Path, name: str) -> None:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            shell=False,
            timeout=180,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PipelineCommandError(f"{name} could not run: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise PipelineCommandError(f"{name} failed: {detail}")


def _validate_master(master_path: Path, fixture: Path) -> dict[str, str]:
    report = evaluate_qc(probe_media(master_path), load_edit(fixture))
    json_path, markdown_path = write_qc_reports(report, master_path.parent)
    return {
        "status": report.status,
        "report": str(json_path),
        "report_markdown": str(markdown_path),
    }


def _status(result: Mapping[str, Any]) -> Literal["pass", "fail"]:
    status = result.get("status")
    if status not in {"pass", "fail"}:
        raise ValueError("validation result must contain status 'pass' or 'fail'")
    return status


def _is_retryable(error: Exception) -> bool:
    return isinstance(error, (PipelineCommandError, MediaProbeError, OSError, subprocess.TimeoutExpired))
