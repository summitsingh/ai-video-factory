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
    duration_seconds: int = 90
    assets_dir: Path | None = None
    
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


def cache_assets_for_remotion(
    assets_dir: Path,
    remotion_public: Path,
) -> None:
    """Copy scene assets to Remotion's public directory for rendering.
    
    The remotion render spawns a process that copies files from its own
    public directory. We copy all scene media there beforehand so they
    are available during rendering.
    """
    remotion_public.mkdir(parents=True, exist_ok=True)
    for idx, scene_dir in enumerate(sorted(assets_dir.glob("scene-*"))):
        if not scene_dir.is_dir():
            continue
        # Copy clips using a scene-specific name so each scene resolves to its
        # own media file (see attach_scene_assets).
        for clip in sorted(scene_dir.glob("*clip*.mp4")):
            dst = remotion_public / f"scene-{idx:02d}-clip{clip.suffix}"
            if not dst.exists():
                shutil.copy2(clip, dst)
        # Copy images using a scene-specific name. Prefer the first image so it
        # matches what attach_scene_assets stored; copy any extras too.
        for img in sorted(scene_dir.glob("*")):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            dst = remotion_public / f"scene-{idx:02d}-image{img.suffix}"
            if not dst.exists():
                shutil.copy2(img, dst)


def attach_scene_assets(edit: EditDocument, assets_dir: Path | None) -> EditDocument:
    """Attach downloaded stock image/clip paths to scenes by directory index.

    Expects per-scene directories ``scene-00/``, ``scene-01/``, ... beneath
    assets_dir. Prefers one clip then one image per scene; scenes without
    assets keep the title-card look. Paths are stored as filenames that
    will be copied to Remotion's public directory prior to rendering.

    Scenes are matched by their numeric id suffix (``scene-0`` -> ``scene-00/``),
    so intro/outro sequences and any reordering do not shift asset lookup.
    """
    if assets_dir is None:
        return edit
    assets_dir = Path(assets_dir)
    if not assets_dir.is_dir():
        return edit
    scenes: list[EditScene] = []
    for scene in edit.scenes:
        update: dict[str, Any] = {}
        # Only normal content scenes have per-scene asset directories.
        if scene.id.startswith("scene-"):
            try:
                idx = int(scene.id.split("-", 1)[1])
            except ValueError:
                idx = None
            if idx is not None:
                scene_dir = assets_dir / f"scene-{idx:02d}"
                if scene_dir.is_dir():
                    clips = sorted(scene_dir.glob("*clip*.mp4"))
                    images = sorted(
                        [p for p in scene_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                         and p.is_file()]
                    )
                    # Use a scene-specific filename so each scene resolves to
                    # its own media file in Remotion's public/ directory.
                    if clips:
                        update["clip"] = f"scene-{idx:02d}-clip{clips[0].suffix}"
                    if images:
                        update["image"] = f"scene-{idx:02d}-image{images[0].suffix}"
        scenes.append(scene.model_copy(update=update) if update else scene)
    return edit.model_copy(update={"scenes": scenes})


def _chunk_edit(edit: EditDocument, max_frames: int) -> list[EditDocument]:
    """Split an edit into chunk documents of at most max_frames each.

    Scene offsets are re-based to zero within each chunk so chunks render
    independently and concatenate in order.
    """
    chunks: list[EditDocument] = []
    current: list[EditScene] = []
    current_frames = 0
    for scene in edit.scenes:
        if current and current_frames + scene.duration_frames > max_frames:
            chunks.append(_finish_chunk(edit, current))
            current = []
            current_frames = 0
        offset = current_frames
        current.append(scene.model_copy(update={
            "from_frame": offset,
            "id": f"{scene.id}",
        }))
        current_frames += scene.duration_frames
    if current:
        chunks.append(_finish_chunk(edit, current))
    return chunks


def _finish_chunk(edit: EditDocument, scenes: list[EditScene]) -> EditDocument:
    total = sum(s.duration_frames for s in scenes)
    return EditDocument(
        schema_version=1,
        width=edit.width,
        height=edit.height,
        fps=edit.fps,
        duration_frames=total,
        scenes=scenes,
        title=edit.title,
        description=edit.description,
        sources=edit.sources,
        created_at=edit.created_at,
    )


def _concat_chunks(chunks: list[Path], destination: Path, *, ffmpeg: Path | None = None) -> None:
    """Concatenate rendered chunk MP4s.

    Prefers an xfade-based re-encode so transitions between chunks are smooth
    cross-dissolves instead of hard cuts. Falls back to stream-copy concat if
    the xfade path fails (e.g. codec incompatibility), preserving the original
    behavior in that case.
    """
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = str(ffmpeg) if ffmpeg is not None else "ffmpeg"

    # xfade transition duration in seconds (must be < shortest chunk duration).
    xfadeseconds = 0.8
    n = len(chunks)

    try:
        _concat_with_xfade(chunks, destination, ffmpeg_bin, xfadeseconds)
    except PipelineCommandError as error:
        # Fall back to stream-copy concat on any xfade failure.
        filelist = destination.parent / "chunks.txt"
        filelist.write_text(
            "".join(f"file '{Path(c).resolve()}'\n" for c in chunks), encoding="utf-8"
        )
        _run_command(
            (
                ffmpeg_bin,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(filelist),
                "-c",
                "copy",
                str(destination),
            ),
            cwd=destination.parent,
            name="FFmpeg chunk concat (fallback)",
            timeout=600,
        )


def _concat_with_xfade(
    chunks: list[Path], destination: Path, ffmpeg_bin: str, xfadeseconds: float
) -> None:
    """Concatenate chunks using xfade transitions between each pair."""
    n = len(chunks)
    if n == 1:
        # Single chunk: just copy it.
        shutil.copy2(chunks[0], destination)
        return

    # Probe durations to compute correct xfade offsets.
    import subprocess

    probe_bin = "ffprobe" if ffmpeg_bin == "ffmpeg" else ffmpeg_bin
    durations = []
    for chunk in chunks:
        probe = subprocess.run(
            [
                probe_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(chunk),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            durations.append(float(probe.stdout.strip()))
        except (ValueError, AttributeError):
            # Probe failed: fall back to stream-copy concat.
            raise PipelineCommandError("xfade duration probe failed") from None

    # Build the xfade filter chain. Each transition overlaps the tail of one
    # chunk with the head of the next by xfadeseconds seconds.
    inputs = []
    for i, chunk in enumerate(chunks):
        inputs += ["-i", str(chunk)]

    filters = ""
    prev_label = "0:v"  # first input video, referenced without extra brackets
    # acc tracks the accumulated duration of the merged stream so far. The
    # first chunk contributes durations[0]; each transition overlaps by
    # xfadeseconds, so transition i's offset is (acc - xfadeseconds) and the
    # new accumulated duration becomes acc + durations[i] - xfadeseconds.
    acc = durations[0]
    for i in range(1, n):
        offset = acc - xfadeseconds
        filters += (
            f"[{prev_label}][{i}:v]xfade=transition=dissolve:duration={xfadeseconds}"
            f":offset={offset}[v{i}];"
        )
        prev_label = f"v{i}"  # store label without brackets for next wrap
        acc = acc + durations[i] - xfadeseconds

    # Audio xfade: chain acrossfade filters, each taking exactly two inputs.
    filters += "[0:a][1:a]acrossfade=d={xfadeseconds}[a1];".format(xfadeseconds=xfadeseconds)
    for i in range(2, n):
        filters += f"[a{i-1}][{i}:a]acrossfade=d={xfadeseconds}[a{i}];"

    # Map video and audio outputs.
    argv = [
        ffmpeg_bin,
        "-y",
        *inputs,
        "-filter_complex",
        filters,
        "-map",
        f"[{prev_label}]",
        "-map",
        f"[a{n-1}]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        str(destination),
    ]

    _run_command(
        tuple(argv),
        cwd=destination.parent,
        name="FFmpeg chunk concat (xfade)",
        timeout=600,
    )


_CHUNK_MAX_FRAMES = 2700  # ~90s per chunk keeps headless Chrome stable
_RENDER_TIMEOUT_SECONDS = 1200


def _render_chunked(
    project_root: Path,
    edit_path: Path,
    output: Path,
    run_directory: Path,
    *,
    browser: Path,
    npm: Path,
    ffmpeg: Path,
) -> None:
    """Render long compositions chunk by chunk, then concatenate."""
    edit = load_edit(edit_path)
    chunks = _chunk_edit(edit, _CHUNK_MAX_FRAMES)
    if len(chunks) == 1:
        _render_with_remotion(
            project_root, edit_path, output,
            browser=browser, npm=npm, timeout=_RENDER_TIMEOUT_SECONDS,
        )
        return
    chunk_dir = run_directory / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for i, chunk in enumerate(chunks):
        chunk_edit_path = chunk_dir / f"chunk-{i:02d}.json"
        save_edit(chunk, chunk_edit_path)
        chunk_out = chunk_dir / f"chunk-{i:02d}.mp4"
        _render_with_remotion(
            project_root, chunk_edit_path, chunk_out,
            browser=browser, npm=npm, timeout=_RENDER_TIMEOUT_SECONDS,
        )
        rendered.append(chunk_out)
    _concat_chunks(rendered, output, ffmpeg=ffmpeg)


def _render_with_remotion(
    project_root: Path,
    fixture: Path,
    output: Path,
    *,
    browser: Path | None = None,
    npm: Path | None = None,
    timeout: int = 600,
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
        timeout=timeout,
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


def _generate_music_bed(
    duration_seconds: float,
    output: Path,
    *,
    ffmpeg: Path | None = None,
) -> Path:
    """Generate a subtle ambient space music bed procedurally (no downloads).

    Uses layered sine tones with slow amplitude modulation and gentle filtering
    to create a calm, cinematic drone suitable under narration. Output is padded
    to the requested duration so it can be mixed across the full video length.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = str(ffmpeg) if ffmpeg is not None else "ffmpeg"
    # Layered soft tones (A220, E261, A329, C#392) with slow tremolo and a low
    # pass filter for warmth. Volume kept quiet so narration stays dominant.
    _run_command(
        (
            ffmpeg_bin, "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency=220:duration={duration_seconds}",
            "-filter_complex",
            "[0:a]volume=0.15,tremolo=f=0.1:d=0.8[a0];"
            "[0:a]adelay=100|100,volume=0.12,tremolo=f=0.13:d=0.7[a1];"
            "[a0][a1]amix=inputs=2:normalize=1,"
            "lowpass=f=1200,highpass=f=80,"
            f"apad=pad_len={int(duration_seconds * 1000)},apad="
            f"pad_len={int(duration_seconds * 1000)}[bed]",
            "-map", "[bed]",
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            str(output),
        ),
        cwd=output.parent,
        name="FFmpeg music bed generation",
        timeout=300,
    )
    return output


def _mux_narration_audio(
    source: Path,
    narration_track: Path | None,
    destination: Path,
    *,
    ffmpeg: Path | None = None,
    music_track: Path | None = None,
) -> None:
    """Mux the video with narration and optional ducked music bed.

    When a music track is provided it is mixed at low volume and ducked via
    sidechain compression (narration triggers the compression), so the score
    dips automatically whenever someone is speaking. The result is padded to
    match the video length, which governs output duration via -shortest.

    ffmpeg filter semantics: ``sidechaincompress`` takes [signal][sidechain],
    so music is the signal and narration is the sidechain trigger.
    """
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = str(ffmpeg) if ffmpeg is not None else "ffmpeg"

    # Build input list and filter chain based on which tracks are available.
    inputs: list[str] = ["-i", str(source)]  # video (may have audio stream)
    map_args: list[str] = ["-map", "0:v"]
    narr_idx: int | None = None
    music_idx: int | None = None

    if narration_track is not None:
        inputs += ["-i", str(Path(narration_track).resolve())]
        narr_idx = len(inputs) // 2 - 1  # index in combined input list

    if music_track is not None:
        inputs += ["-i", str(Path(music_track).resolve())]
        music_idx = len(inputs) // 2 - 1

    filter_parts: list[str] = []

    if narr_idx is not None and music_idx is not None:
        # Duck the music using narration as the sidechain trigger. When
        # narration is loud, compress (lower) the music; when quiet, music
        # returns to its base volume. Then mix ducked music over narration.
        filter_parts.append(
            f"[{music_idx}:a]volume=0.25[music_base];"
            f"[music_base][{narr_idx}:a]"
            "sidechaincompress=threshold=0.03:ratio=15:attack=20:release=250,"
            "acompressor=threshold=0.02:ratio=4[music_ducked];"
            "[music_ducked]apad=pad_len=60000[music_padded];"
            f"[{narr_idx}:a][music_padded]"
            "amix=inputs=2:normalize=1[mix]"
        )
    elif narr_idx is not None:
        # Narration only (no music): pad to video length.
        filter_parts.append(
            f"[{narr_idx}:a]apad=pad_len=60000,mix"
        )
    elif music_idx is not None:
        # Music only (no narration): lower volume, pad to video length.
        filter_parts.append(
            f"[{music_idx}:a]volume=0.25,apad=pad_len=60000,mix"
        )
    else:
        # Neither track available: fall back to silent audio.
        _mux_silent_audio(source, destination, ffmpeg=ffmpeg)
        return

    filter_chain = ";".join(filter_parts)
    map_args += ["-map", "[mix]"]
    argv = [ffmpeg_bin, "-y", *inputs, "-filter_complex", filter_chain,
            *map_args, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-ar", "48000", "-ac", "2", "-shortest", str(destination)]

    _run_command(
        tuple(argv),
        cwd=destination.parent,
        name="FFmpeg narration + music mux",
    )


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

        # Step 1: Research trending topic (skip if script provided)
        metadata["research_status"] = "in_progress"
        job.research_path = project_data / f"{job.topic.replace(' ', '_')}_research.json"
        if script_path is None:
            research_result = research_trending_topics(max_topics=1, min_engagement=100)
            if not research_result.topics:
                raise PipelineCommandError("No trending topics found for research")
            save_research_result(research_result, job.research_path)
            trending_topic = research_result.topics[0]
            metadata["selected_topic"] = trending_topic.title
            metadata["source_url"] = trending_topic.url
            artifacts["research"] = str(job.research_path)
        else:
            # When script provided, just create minimal research stub
            trending_topic = TrendingTopic(
                title=job.topic,
                description=job.description or job.topic,
                source=job.source_url,
                url=job.source_url or "https://example.com",
                timestamp=datetime.now(),
            )
            job.research_path.write_text(json.dumps({"topic": job.topic, "sources": []}))
            metadata["research_status"] = "skipped_script_provided"
            metadata["selected_topic"] = job.topic

        # Step 2: Generate script with LM Studio (or load a worker-made script)
        metadata["script_status"] = "in_progress"
        job.script_path = project_data / f"{job.topic.replace(' ', '_')}_script.json"
        if script_path is not None:
            worker_data = json.loads(Path(script_path).read_text(encoding="utf-8"))
            script = ScriptOutput(
                title=str(worker_data["title"]),
                narration=str(worker_data["narration_full"]),
                scenes=list(worker_data["scenes"]),
                sources=list(worker_data.get("sources", [trending_topic.url])),
                captions=list(worker_data.get("captions", [])),
            )
            job.script_path.write_text(
                json.dumps(worker_data, indent=2), encoding="utf-8"
            )
            metadata["script_origin"] = "worker"
            artifacts["script"] = str(job.script_path)
        else:
            script = generate_script(
                topic_title=trending_topic.title,
                topic_description=trending_topic.description,
                source_url=trending_topic.url,
                output_path=job.script_path,
                use_local_model=True,
            )
            metadata["script_origin"] = "local_model"
            artifacts["script"] = str(job.script_path)
        job.title = script.title
        metadata["script_status"] = "complete"

        # Step 3: Create storyboard/edit document
        metadata["edit_status"] = "in_progress"
        job.edit_path = project_data / f"{job.topic.replace(' ', '_')}_edit.json"

        # Create edit document from script
        fps = 30
        duration_seconds = job.duration_seconds
        total_frames = duration_seconds * fps

        # Reserve fixed durations for branded intro/outro sequences. These are
        # dedicated scenes so they render once (chunk-safe) and concatenate in
        # order with the normal content.
        INTRO_FRAMES = 150   # 5s branded title card
        OUTRO_FRAMES = 180   # 6s credits sequence

        # Distribute remaining frames across normal scenes so no blank tail
        # remains when the script returns fewer/shorter scenes.
        content_frames = total_frames - INTRO_FRAMES - OUTRO_FRAMES
        scene_count = max(len(script.scenes), 1)
        base_frames, remainder = divmod(content_frames, scene_count)

        scenes: list[EditScene] = []
        cursor = 0

        # Intro sequence first.
        intro_scene = EditScene(
            id="intro",
            from_frame=cursor,
            duration_frames=INTRO_FRAMES,
            title=job.title or f"Trending: {job.topic}",
            caption=job.description or job.topic,
            kind="intro",
        )
        scenes.append(intro_scene)
        cursor += INTRO_FRAMES

        # Normal content scenes.
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

        # Outro sequence last.
        outro_scene = EditScene(
            id="outro",
            from_frame=cursor,
            duration_frames=OUTRO_FRAMES,
            title=job.title or f"Trending: {job.topic}",
            caption="The Eyes That See Everything",
            kind="outro",
        )
        scenes.append(outro_scene)

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
        if job.assets_dir is not None:
            edit_doc = attach_scene_assets(edit_doc, job.assets_dir)
            save_edit(edit_doc, job.edit_path)
            metadata["assets_attached"] = sum(
                1 for s in edit_doc.scenes if s.clip or s.image
            )
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
            
            # Copy assets to Remotion public directory for rendering
            if job.assets_dir is not None:
                remotion_public = root / "remotion" / "public"
                cache_assets_for_remotion(Path(job.assets_dir), remotion_public)

            if render is None:
                _render_chunked(
                    root,
                    job.edit_path,
                    temporary_path,
                    run_directory,
                    browser=_required_tool(tools, "browser"),
                    npm=_required_tool(tools, "npm"),
                    ffmpeg=_required_tool(tools, "ffmpeg"),
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
                # Generate a subtle ambient music bed and duck it under narration.
                duration_seconds = edit_doc_render.duration_frames / edit_doc_render.fps
                music_track = _generate_music_bed(
                    duration_seconds,
                    run_directory / "music.wav",
                    ffmpeg=_required_tool(tools, "ffmpeg"),
                )
                metadata["music_status"] = "complete"
                _mux_narration_audio(
                    temporary_path,
                    narration_track,
                    run_master,
                    ffmpeg=_required_tool(tools, "ffmpeg"),
                    music_track=music_track,
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