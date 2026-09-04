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
from ai_video_factory.sanitization import first_diagnostic_line, sanitize_diagnostic


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
        error=sanitize_diagnostic(error),
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
    active_run_id: str | None = None
    artifacts: dict[str, str] = {}

    try:
        root = Path(project_root).resolve()
        fixture = root / "fixtures" / "synthetic-edit.json"
        lockfile = root / "remotion" / "package-lock.json"
        project_data = Path(data_root) / "projects" / "synthetic"
        artifact_root = project_data / "runs"
        state_store = RunStore(project_data / "state", artifact_root=artifact_root)
        load_edit(fixture)
        tools = (
            _collect_pipeline_toolchain()
            if render is None or validate is None
            else {
                "render_stage": {"status": "injected", "version": "test seam"},
                "validate_stage": {"status": "injected", "version": "test seam"},
            }
        )
        render_inputs = _pipeline_inputs(root, fixture, lockfile, tools)
        render_run = state_store.start("synthetic-render", render_inputs)
        run_directory = artifact_root / render_run.run_id
        master_path = run_directory / "master.mp4"

        if not render_run.resumed:
            active_run_id = render_run.run_id
            state_store.event(render_run.run_id, "render_started", {"fixture": str(fixture)})
            run_directory.mkdir(parents=True, exist_ok=True)
            if render is None:
                temporary_path = run_directory / "render.tmp.mp4"
                _render_with_remotion(
                    root,
                    fixture,
                    temporary_path,
                    browser=_required_tool(tools, "browser"),
                    npm=_required_tool(tools, "npm"),
                )
                _mux_silent_audio(
                    temporary_path,
                    master_path,
                    ffmpeg=_required_tool(tools, "ffmpeg"),
                )
                temporary_path.unlink(missing_ok=True)
            else:
                render(fixture, master_path)
            render_run = state_store.complete(
                render_run.run_id,
                {"master": str(master_path)},
                expected_artifacts={"master": master_path},
            )
            active_run_id = None

        master_path = Path(render_run.artifacts["master"])
        artifacts["master"] = str(master_path)
        qc_inputs = {
            "fixture_sha256": _sha256(fixture),
            "render_run_id": render_run.run_id,
            "master_sha256": _sha256(master_path),
            "provenance": {
                "tools": tools,
                "full_decode": True,
                "ffprobe_arguments": "-v error -show_streams -show_format -of json <master>",
                "ffmpeg_arguments": "-v error -i <master> -f null -",
            },
        }
        qc_run = state_store.start("synthetic-qc", qc_inputs)

        if not qc_run.resumed:
            active_run_id = qc_run.run_id
            state_store.event(qc_run.run_id, "qc_started", {"master": str(master_path)})
            result = (
                validate(master_path, master_path.parent)
                if validate is not None
                else _validate_master(master_path, fixture, tools=tools)
            )
            status = _status(result)
            qc_artifacts = {
                key: str(value) for key, value in result.items() if key != "status"
            }
            expected_qc_artifacts = {
                key: Path(value) for key, value in qc_artifacts.items()
            }
            artifacts.update(qc_artifacts)
            qc_run = state_store.complete(
                qc_run.run_id,
                {"status": status, **qc_artifacts},
                expected_artifacts=expected_qc_artifacts,
            )
            active_run_id = None
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
        if active_run_id is not None and state_store is not None:
            try:
                state_store.fail(active_run_id, error)
            except Exception:
                # Preserve the stable public result even if persistence itself failed.
                pass
        return failed_pipeline_result(
            error,
            run_id=render_run.run_id if render_run is not None else None,
            resumed=bool((qc_run or render_run) and (qc_run or render_run).resumed),
            artifacts=artifacts,
        )


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
                "audio_filter": "anullsrc=r=48000:cl=stereo",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_pipeline_toolchain() -> dict[str, dict[str, Any]]:
    tools: dict[str, dict[str, Any]] = {}
    try:
        browser = _find_local_browser()
        tools["browser"] = _executable_identity(browser, ("--version",))
    except (OSError, PipelineCommandError, subprocess.TimeoutExpired) as error:
        tools["browser"] = _unavailable_tool(error)

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
            tools[name] = _unavailable_tool(error)
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


def _unavailable_tool(error: object) -> dict[str, str]:
    return {"status": "not_ready", "error": sanitize_diagnostic(error)}


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


def _mux_silent_audio(
    source: Path, destination: Path, *, ffmpeg: Path | None = None
) -> None:
    _run_command(
        (
            str(ffmpeg) if ffmpeg is not None else "ffmpeg",
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
    status = result.get("status")
    if status not in {"pass", "fail"}:
        raise ValueError("validation result must contain status 'pass' or 'fail'")
    return status


def _is_retryable(error: Exception) -> bool:
    return isinstance(error, (PipelineCommandError, MediaProbeError, OSError, subprocess.TimeoutExpired))
