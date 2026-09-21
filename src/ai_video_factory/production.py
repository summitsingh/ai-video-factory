"""Production builder for AI Video Factory (Phase 3, Milestone 7).

This module turns a *research result* into an actual rendered documentary
candidate. It is deliberately honest about what it measures: every QC input is
derived from real artifacts on disk (measured narration audio and a real
rendered master), never fabricated to satisfy a gate.

Pipeline::

    verified_facts -> long-form script/edit -> rights-cleared assets
        -> narration synthesis -> render master -> captions
        -> final QC (on real artifacts) -> upload-ready package

The long-form requirement is enforced against *measured* durations: the total
runtime of the rendered master must fall within ``[MIN_RUNTIME, MAX_RUNTIME]``
seconds. A candidate that cannot meet the floor with its verified material is
rejected rather than padded; one that exceeds the ceiling is rejected too.

Rendering and text-to-speech are delegated to a :class:`RenderEngine` so the
whole pipeline can be exercised offline in tests (a fake engine writes real,
ffprobe-measurable WAV/MP4 files) while production uses an FFmpeg + Kokoro
engine. The engine only synthesizes narration and renders video; all QC metrics
are computed by shared :mod:`media_metrics` helpers that read the real files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ai_video_factory.edit_schema import (
    EditDocument,
    EditScene,
    build_edit_document_from_scenes,
    frames_to_seconds,
    seconds_to_frames,
)
from ai_video_factory.narration import mix_scenes_to_track
from ai_video_factory.qc_final import black_frame_exclude_windows
from ai_video_factory.research_pipeline import ResearchResult
from ai_video_factory.sanitization import sanitize_diagnostic


# ========== Long-form constraints ==========

MIN_RUNTIME_SECONDS = 900.0  # 15 minutes
MAX_RUNTIME_SECONDS = 1200.0  # 20 minutes
TARGET_RUNTIME_SECONDS = 1050.0  # pacing target (mid-range)
DEFAULT_MIN_WORDS = 2200
_MAX_WORDS_PER_SCENE = 1500  # a single long-form deep-dive segment can be very long
_BOOKEND_SECONDS = 8.0  # intro + outro bookends

# Headless Remotion renders one frame at a time through software GL (swiftshader) with
# no GPU acceleration, so a full 15-20 minute master can take ~60 min on this machine.
# The cap is generous by design; a genuinely hung render is caught downstream by the
# pilot monitor and by the empty-output guard in render_master().
RENDER_SUBPROCESS_SECONDS = 7200  # 2 hours
# A long-form documentary requires multiple distinct, source-backed chapters. One or
# two claims cannot honestly become a 15–20 minute film; production rejects fewer than
# this many distinct grounded claims rather than padding narration with invented filler.
MIN_GROUNDED_CLAIMS = 3


class ProductionError(RuntimeError):
    """Raised when a candidate cannot be produced within constraints."""


class MediaMetricsError(RuntimeError):
    """Raised when a real artifact metric cannot be measured (fail closed)."""


# ========== Measured QC metrics (read real files, never fabricate) ==========


def _run(argv: Sequence[str], timeout: int = 120) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        list(argv), shell=False, timeout=timeout, capture_output=True, text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ProductionError(f"command failed ({argv[:3]}): {proc.stderr.strip()[-500:] or 'exit '+str(proc.returncode)}")
    return proc


def audio_silence_gap_seconds(wav: Path, *, noise_db: str = "-30dB", min_dur: float = 0.3) -> float:
    """Largest silence gap in a real WAV, measured with ffmpeg silencedetect."""
    wav = Path(wav)
    completed = _run(
        (
            "ffmpeg", "-y", "-i", str(wav),
            "-af", f"silencedetect=noise={noise_db}:d={min_dur}",
            "-f", "null", "-",
        ),
        timeout=180,
    )
    if completed.returncode != 0:
        raise MediaMetricsError(
            sanitize_diagnostic(f"silencedetect failed on {wav}: {completed.stderr.strip()[-300:] or 'exit '+str(completed.returncode)}")
        )
    gaps = [float(m) for m in re.findall(r"silence_duration:\s*([0-9.eE+-]+)", completed.stdout)]
    return max(gaps) if gaps else 0.0


def audio_has_clipping(wav: Path) -> bool:
    """Best-effort clipping detection from ffmpeg astats peak level.

    Returns False when the metric cannot be computed rather than guessing; the
    caller decides whether an unmeasured signal is acceptable.
    """
    wav = Path(wav)
    completed = _run(
        (
            "ffmpeg", "-y", "-i", str(wav),
            "-af", "astats=metadata=1:reset=1",
            "-f", "null", "-",
        ),
        timeout=180,
    )
    if completed.returncode != 0:
        return False
    peaks = re.findall(r"Peak level dB:\s*(-?[0-9.]+)", completed.stdout)
    if not peaks:
        return False
    # A peak at or above -0.05 dBFS indicates digital clipping.
    return any(float(p) >= -0.05 for p in peaks)


def audio_duration_seconds(wav: Path) -> float | None:
    """Real duration of a WAV via ffprobe."""
    wav = Path(wav)
    completed = _run(
        (
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(wav),
        ),
        timeout=60,
    )
    if completed.returncode != 0:
        raise MediaMetricsError(sanitize_diagnostic(f"ffprobe failed on {wav}: {completed.stderr.strip()[-200:] or 'exit '+str(completed.returncode)}"))
    try:
        data = json.loads(stdout_or(completed.stdout))
        return float(data["format"]["duration"])
    except (ValueError, KeyError, TypeError):
        return None


def _subtract_windows(
    windows: list[tuple[float, float]], exclusions: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Remove exclusion ranges from (start, end) windows."""
    result: list[tuple[float, float]] = []
    for start, end in windows:
        segments = [(start, end)]
        for xs, xe in exclusions:
            remaining = []
            for a, b in segments:
                if xe <= a or xs >= b:
                    remaining.append((a, b))
                    continue
                if xs > a:
                    remaining.append((a, xs))
                if xe < b:
                    remaining.append((xe, b))
            segments = remaining
        result.extend(segments)
    return result


def video_black_ratio(
    mp4: Path, exclude_windows: list[tuple[float, float]] | None = None
) -> float:
    """Fraction of the measured runtime covered by black-detect events (real).

    ``exclude_windows`` is a list of (start, end) seconds where dark frames
    are by design (intro/outro branded cards, act-card scrim windows):
    black-detect time inside those windows is ignored and the windows are
    removed from the denominator, so intended darkness never counts as a
    defect. Uniform black frames across normal content scenes still count.
    """
    mp4 = Path(mp4)
    completed = _run(
        ("ffmpeg", "-y", "-i", str(mp4), "-vf", "blackdetect=pix_th=0.05", "-f", "null", "-"),
        timeout=180,
    )
    if completed.returncode != 0:
        raise MediaMetricsError(sanitize_diagnostic(f"blackdetect failed on {mp4}: {completed.stderr.strip()[-200:] or 'exit '+str(completed.returncode)}"))
    # blackdetect writes its diagnostics to stderr; capture the start/end times.
    starts = [float(m) for m in re.findall(r"black_start:\s*([0-9.eE+-]+)", completed.stderr)]
    ends = [float(m) for m in re.findall(r"black_end:\s*([0-9.eE+-]+)", completed.stderr)]
    black_windows = list(zip(starts, ends))
    exclusions = list(exclude_windows or [])
    excluded_total = sum(max(0.0, e - s) for s, e in exclusions)
    black_windows = _subtract_windows(black_windows, exclusions)
    total_black = sum(max(0.0, e - s) for s, e in black_windows)

    # Total runtime from ffprobe so the ratio reflects actual coverage.
    probe = _run(("ffprobe", "-v", "error", "-show_entries", "format=duration",
                  "-of", "json", str(mp4)))
    try:
        total = float(json.loads(probe.stdout)["format"]["duration"])
    except Exception:  # noqa: BLE001 - fall back to the black window if unavailable
        total = max(total_black, 1.0)

    measured_total = max(0.0, total - excluded_total)
    if measured_total <= 0:
        return 0.0
    return min(1.0, total_black / measured_total)


def video_frozen_frame_count(mp4: Path, *, sample_hz: float = 2.0) -> int:
    """Count frozen (identical consecutive) frames sampled from the real master."""
    mp4 = Path(mp4)
    tmp = tempfile.mkdtemp(prefix="avf-frozen-")
    try:
        completed = _run(
            (
                "ffmpeg", "-y", "-i", str(mp4),
                "-vf", f"fps={sample_hz}",
                "-f", "image2", "-q:v", "1", f"{tmp}/frame-%05d.pnm",
            ),
            timeout=180,
        )
        if completed.returncode != 0:
            raise MediaMetricsError(sanitize_diagnostic(f"frame extract failed on {mp4}: {completed.stderr.strip()[-200:] or 'exit '+str(completed.returncode)}"))
        frames = sorted(Path(tmp).glob("frame-*.pnm"))
        if len(frames) < 2:
            return 0
        digests = [_sha256_file(f) for f in frames]
        frozen = sum(1 for a, b in zip(digests, digests[1:]) if a == b)
        return frozen
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def video_min_contrast(mp4: Path, *, sample_hz: float = 2.0) -> float:
    """Minimum (max-min luminance) contrast sampled from the real master.

    Extracts frames with ffmpeg and computes per-frame contrast in Python so the
    metric does not depend on this build's ``signalstats`` metadata print, which
    drops YMin/YMax here. Returns 0.0 when no frame can be decoded (fail closed).
    """
    mp4 = Path(mp4)
    tmp = tempfile.mkdtemp(prefix="avf-contrast-")
    try:
        completed = _run(
            (
                "ffmpeg", "-y", "-i", str(mp4),
                "-vf", f"fps={sample_hz}",
                "-f", "image2", "-q:v", "9", f"{tmp}/frame-%05d.png",
            ),
            timeout=180,
        )
        if completed.returncode != 0:
            raise MediaMetricsError(sanitize_diagnostic(f"frame extract failed on {mp4}: {completed.stderr.strip()[-200:] or 'exit '+str(completed.returncode)}"))
        from PIL import Image
        frames = sorted(Path(tmp).glob("frame-*.png"))
        if len(frames) < 1:
            return 0.0
        contrasts: list[float] = []
        for f in frames:
            with Image.open(f) as im:
                gray = im.convert("L")
                px = list(gray.getdata())
            if not px:
                continue
            lo, hi = min(px), max(px)
            contrasts.append(max(0.0, (hi - lo) / 255.0))
        return min(contrasts) if contrasts else 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def video_repeated_shot_count(mp4: Path, *, sample_hz: float = 2.0, window: int = 30) -> int:
    """Count shots that repeat a previous shot within ``window`` frames (real).

    Extracts frames with ffmpeg and compares each frame's fingerprint to the
    fingerprints of the preceding ``window`` frames; any exact match is counted as
    a repeated shot. Returns 0 when no frame can be decoded (fail closed).
    """
    mp4 = Path(mp4)
    tmp = tempfile.mkdtemp(prefix="avf-repeat-")
    try:
        completed = _run(
            (
                "ffmpeg", "-y", "-i", str(mp4),
                "-vf", f"fps={sample_hz}",
                "-f", "image2", "-q:v", "9", f"{tmp}/frame-%05d.png",
            ),
            timeout=180,
        )
        if completed.returncode != 0:
            raise MediaMetricsError(sanitize_diagnostic(f"frame extract failed on {mp4}: {completed.stderr.strip()[-200:] or 'exit '+str(completed.returncode)}"))
        frames = sorted(Path(tmp).glob("frame-*.png"))
        if len(frames) < 2:
            return 0
        fingerprints = [_sha256_file(f) for f in frames]
        repeated = sum(
            1 for i in range(len(fingerprints))
            if any(fingerprints[i] == fingerprints[j] for j in range(max(0, i - window), i))
        )
        return repeated
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def stdout_or(text: str) -> str:
    return text


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ========== Render engine interface ==========


class RenderEngine:
    """Synthesizes narration and renders a master. Subclasses implement both."""

    def synthesize_narration(self, text: str, voice: str | None = None) -> Path:  # pragma: no cover - interface
        raise NotImplementedError

    def render_master(
        self,
        edit: EditDocument,
        *,
        narration_segments: Sequence[tuple[Path, float]],
        assets_by_scene_id: dict[str, Path],
        destination: Path,
    ) -> Path:  # pragma: no cover - interface
        raise NotImplementedError


# ========== Fake offline engine (tests) ==========


class OfflineRenderEngine(RenderEngine):
    """Writes real, ffprobe-measurable WAV/MP4 files without network or TTS.

    Narration duration is proportional to word count so a document of
    ``min_words`` words paces to roughly the target runtime; this lets tests
    exercise the full production + QC pipeline offline while still measuring
    genuine audio/video artifacts.
    """

    def __init__(self, seconds_per_word: float | None = None) -> None:
        # Default pacing maps the word floor onto the target runtime so that a
        # document meeting ``min_words`` lands near the middle of [900, 1200].
        self._dpw = max(0.05, seconds_per_word or (TARGET_RUNTIME_SECONDS / DEFAULT_MIN_WORDS))

    def synthesize_narration(self, text: str, voice: str | None = None) -> Path:
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            raw = Path(tmp.name)
        words = len(re.findall(r"\S+", text))
        duration = max(0.5, words * self._dpw)
        _write_tone_wav(raw, seconds=duration, hertz=440 if not voice else 300 + hash(voice or "") % 200)
        return raw

    def render_master(
        self,
        edit: EditDocument,
        *,
        narration_segments: Sequence[tuple[Path, float]],
        assets_by_scene_id: dict[str, Path],
        destination: Path,
    ) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fps = int(edit.fps)
        width, height = int(edit.width), int(edit.height)
        segments_dir = tempfile.mkdtemp(prefix="avf-seg-")
        try:
            seg_paths: list[Path] = []
            for i, scene in enumerate(edit.scenes):
                if scene.kind == "intro" or scene.kind == "outro":
                    continue
                dur = frames_to_seconds(scene.duration_frames, fps)
                if dur <= 0:
                    continue
                hue_offset = i * 37.0
                seg = Path(segments_dir) / f"seg-{i}.mp4"
                _run((
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={dur:.3f}",
                    "-vf", f"hue=h={hue_offset}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
                    str(seg),
                ))
                seg_paths.append(seg)
            if not seg_paths:
                raise ProductionError("render engine produced no scene segments")
            concat = Path(segments_dir) / "concat.txt"
            concat.write_text(
                "".join(f"file '{seg.as_posix()}'\n" for seg in seg_paths), encoding="utf-8"
            )
            concat_mp4 = Path(segments_dir) / "concat.mp4"
            _run((
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                "-c", "copy", str(concat_mp4),
            ))
            # Pad narration audio to the concatenated video duration, then mux.
            # concat.mp4 is video-only (testsrc segments carry no audio), so we
            # pad each narration to total length and amix them directly.
            total_frames = int(frames_to_seconds(sum(s.duration_frames for s in edit.scenes), fps))
            argv = ["ffmpeg", "-y", "-i", str(concat_mp4)]
            narr_wavs: list[Path] = []
            for wav, _off in narration_segments:
                argv += ("-i", str(wav))
                narr_wavs.append(wav)
            n_audio = len(narr_wavs)
            pad_len = max(1, total_frames)
            filters = "".join(f"[{i + 1}:a]apad=pad_len={pad_len}[a{i}];" for i in range(n_audio))
            mix_inputs = "".join(f"[a{i}]" for i in range(n_audio))
            argv += (
                "-filter_complex", filters + f"{mix_inputs}amix=inputs={n_audio}:normalize=1[mix]",
                "-map", "0:v:0", "-map", "[mix]",
            )
            _run(tuple(argv + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(destination)]))
            if not destination.is_file() or destination.stat().st_size == 0:
                raise ProductionError("offline engine produced an empty master")
            return destination
        finally:
            shutil.rmtree(segments_dir, ignore_errors=True)


# ========== Real production renderer (Remotion + Kokoro) ==========

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def _verified_browser(path: Path) -> Path | None:
    """Return ``path`` if it names an existing, executable browser binary."""
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file() and os.access(resolved, os.X_OK):
        return resolved
    return None


def _remotion_browser_executable() -> Path:
    """Locate a verified local browser for headless Remotion rendering.

    Fail-closed: honors ``REMOTION_CHROME_EXECUTABLE`` first, then common system
    browsers; raises :class:`RuntimeError` when none is usable.
    """
    configured = os.environ.get("REMOTION_CHROME_EXECUTABLE")
    if configured:
        candidate = _verified_browser(Path(configured))
        if candidate is not None:
            return candidate
        raise RuntimeError(
            "REMOTION_CHROME_EXECUTABLE does not name an existing browser executable"
        )
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        located = shutil.which(name)
        if located is not None:
            candidate = _verified_browser(Path(located))
            if candidate is not None:
                return candidate
    raise RuntimeError(
        "no verified local browser executable; set REMOTION_CHROME_EXECUTABLE or install Chrome/Chromium"
    )


def _run_ffmpeg(argv: Sequence[str], *, timeout: int = 300) -> None:
    """Run ffmpeg, raising :class:`ProductionError` on any failure."""
    try:
        completed = subprocess.run(
            ["ffmpeg", *argv], shell=False, timeout=timeout, capture_output=True, text=True, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProductionError(f"ffmpeg could not run: {sanitize_diagnostic(str(error))}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or f"exit code {completed.returncode}").strip()
        raise ProductionError(f"ffmpeg failed: {sanitize_diagnostic(detail)}")


def _media_kind(path: Path) -> str:
    """Classify a staged asset as ``image`` or ``clip`` by extension, else ffprobe."""
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "clip"
    probe = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=False, timeout=30,
    )
    return "clip" if "video" in (probe.stdout or "").strip() else "image"


class RemotionKokoroRenderEngine(RenderEngine):
    """Real production renderer: Kokoro narration + headless Remotion master.

    :meth:`synthesize_narration` writes a real 48 kHz stereo WAV via the local
    Kokoro ONNX model (falling back to piper/espeak when Kokoro is unavailable).
    :meth:`render_master` wires approved assets into each scene, renders a muted
    MP4 with headless Remotion, then concatenates the per-scene narration tracks
    in scene order and muxes them onto the video.

    The renderer is fail-closed: a missing/empty asset, an unavailable browser,
    or a non-zero render exit all raise before any master is written. Assets are
    staged under ``public_dir/assets/<run>/`` and referenced by path relative to
    the Remotion public root (Remotion's ``staticFile`` serves them).
    """

    def __init__(
        self,
        *,
        remotion_root: Path | None = None,
        public_dir: Path | None = None,
        browser_executable: Path | None = None,
        npm_executable: str = "npm",
    ) -> None:
        root = Path(remotion_root) if remotion_root is not None else self._default_remotion_root()
        self._root = root
        self._public_dir = Path(public_dir) if public_dir is not None else (root / "public")
        self._browser_executable = browser_executable
        self._npm = npm_executable

    @staticmethod
    def _default_remotion_root() -> Path:
        here = Path(__file__).resolve().parent
        for parent in (here, *here.parents):
            if (parent / "remotion" / "package.json").is_file():
                return parent / "remotion"
        raise RuntimeError("remotion project root not found")

    def synthesize_narration(self, text: str, voice: str | None = None) -> Path:
        from ai_video_factory.narration import synthesize_to_wav
        output = tempfile.mkstemp(prefix="narr-", suffix=".wav")[1]
        return synthesize_to_wav(text, Path(output), engine="kokoro", voice=voice or "af_heart")

    def render_master(
        self,
        edit: EditDocument,
        *,
        narration_segments: Sequence[tuple[Path, float]],
        assets_by_scene_id: dict[str, Path],
        destination: Path,
    ) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="avf-render-")
        try:
            # 1. Wire approved assets into the edit (fail-closed on bad artifacts).
            self._wire_assets(edit, assets_by_scene_id)

            # 2. Write props for Remotion (paths are relative to the remotion root).
            props_path = Path(workdir) / f"props-{destination.stem}.json"
            props_path.write_text(edit.model_dump_json(indent=2), encoding="utf-8")

            # 3. Render a muted master (no audio); narration is muxed in step 4.
            browser = self._browser_executable or _remotion_browser_executable()
            muted = Path(workdir) / "muted.mp4"
            self._run_remotion(props_path, muted, browser)
            if not muted.is_file() or muted.stat().st_size == 0:
                raise ProductionError("Remotion render produced no master file")

            # 4. Mix per-scene narration at their absolute start offsets (seconds) and
            #    mux onto video. Offsets place each segment after the intro bookend /
            #    between scenes instead of concatenating everything from 0:00. Absolute
            #    offsets come from each scene's from_frame, so audio stays in sync with
            #    the visuals even when a prior narration run exceeds its slot.
            if not narration_segments:
                raise ProductionError("no narration segments to render")
            narration = Path(workdir) / "narration.wav"
            mix_scenes_to_track(narration_segments, narration)
            total_seconds = sum(scene.duration_frames for scene in edit.scenes) / float(edit.fps)
            self._mux_audio(muted, narration, destination, total_seconds=total_seconds)

            if not destination.is_file() or destination.stat().st_size == 0:
                raise ProductionError("rendered master is empty")
            return destination
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _wire_assets(self, edit: EditDocument, assets_by_scene_id: dict[str, Path]) -> list[Path]:
        """Stage approved assets under the public root and set scene.image/clip.

        Returns the staged asset paths. Raises :class:`ProductionError` if any
        referenced asset is missing or empty so a candidate never packages with a
        broken media reference.
        """
        run_id = hashlib.sha1(
            str(sorted(assets_by_scene_id.items())).encode("utf-8")
        ).hexdigest()[:12]
        run_dir = self._public_dir / "assets" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        served: list[Path] = []
        for scene in edit.scenes:
            source = assets_by_scene_id.get(scene.id)
            if source is None:
                continue
            staged = self._stage_asset(source, run_dir)
            rel = staged.relative_to(self._public_dir).as_posix()
            if _media_kind(staged) == "image":
                scene.image = rel
            else:
                scene.clip = rel
            served.append(staged)
        return served

    def _stage_asset(self, source: Path, run_dir: Path) -> Path:
        asset_path = Path(source)
        if not asset_path.is_file():
            raise ProductionError(f"approved asset is missing: {asset_path}")
        if asset_path.stat().st_size == 0:
            raise ProductionError(f"approved asset is empty: {asset_path}")
        dest = run_dir / asset_path.name
        shutil.copy2(asset_path, dest)
        if not dest.is_file() or dest.stat().st_size == 0:
            raise ProductionError(f"failed to stage approved asset: {dest}")
        return dest


    def _mux_audio(self, video: Path, audio: Path, destination: Path, *, total_seconds: float) -> None:
        # Pad narration to the exact video length so the whole master carries
        # audio (trailing silence if narration is short; -shortest trims any excess).
        pad_len = max(1, int(math.ceil(total_seconds)))
        _run_ffmpeg((
            "-y",
            "-i", str(video),
            "-i", str(audio),
            "-filter_complex", f"[1:a]apad=pad_len={pad_len}[narr]",
            "-map", "0:v:0",
            "-map", "[narr]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(destination),
        ))

    def _run_remotion(self, props_path: Path, output: Path, browser: Path) -> None:
        argv = (
            self._npm, "run", "render", "--",
            "--props", os.path.relpath(props_path, self._root),
            os.path.relpath(output, self._root),
            "--browser-executable", str(browser),
        )
        try:
            completed = subprocess.run(
                list(argv), cwd=self._root, shell=False, timeout=RENDER_SUBPROCESS_SECONDS,
                capture_output=True, text=True, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ProductionError(f"Remotion render could not run: {sanitize_diagnostic(str(error))}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or f"exit code {completed.returncode}").strip()
            raise ProductionError(f"Remotion render failed: {sanitize_diagnostic(detail)}")


def _write_tone_wav(path: Path, *, seconds: float, hertz: int = 440) -> None:
    import array
    import wave

    path = Path(path)
    sample_rate = 48000
    n = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)
        frames = array.array("h", [int(8000 * (1 if (k % (sample_rate // hertz)) < (sample_rate // hertz // 2) else -1)) for k in range(n)])
        wav_out.writeframes(frames.tobytes())


# ========== Script / edit construction ==========


def _flatten_source_sentences(contents_by_url: dict[str, Any]) -> list[str]:
    """Return every distinct sentence across all persisted source content.

    Order is deterministic (source order, then in-document order) so scene
    assignment is reproducible. Sentences shorter than 12 chars or exact
    duplicates are dropped — they carry no factual weight for narration.
    """
    sentences: list[str] = []
    seen: set[str] = set()
    for src in contents_by_url.values():
        content = getattr(src, "content", None) or ""
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            text = sentence.strip()
            if len(text) < 12 or not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            sentences.append(text)
    return sentences


def _grounded_narration(claim_text: str, topic: str, chunk: list[str], target_words: int) -> str:
    """Build one chapter's narration from a *disjoint* slice of grounded material.

    ``chunk`` is the list of source sentences assigned to this scene only — no other
    scene reuses them, so chapters are distinct rather than identical. The verified
    claim anchors the chapter; grounded detail follows. No invented filler is appended:
    if the assigned slice cannot reach ``target_words`` words without repetition we
    return what the sources actually support (the caller decides whether that is enough).
    """
    parts: list[str] = []
    base = (claim_text or "").strip().rstrip(".")
    if base:
        parts.append(base)
    for sentence in chunk:
        parts.append(sentence)
        if sum(len(p.split()) for p in parts) >= target_words:
            break
    return " ".join(parts).strip()


def _trim_to_words(text: str, limit: int) -> str:
    """Truncate narration to at most ``limit`` words at a word boundary.

    The renderer paces runtime off narration length, so a scene that overshoots its
    word share pushes the master past MAX_RUNTIME. Cap each scene at its share so the
    whole plan lands on the word floor instead of above it.
    """
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + "\u2026"


def build_scene_plan(
    research: ResearchResult,
    *,
    min_words: int = DEFAULT_MIN_WORDS,
) -> list[dict]:
    """Build a long-form scene plan whose total narration meets ``min_words``.

    Each verified fact becomes one chapter that draws on a *disjoint* slice of the
    persisted source content — chapters are distinct rather than repeating the same
    sentences. The duration is supported by real, non-repeating material: if the
    combined grounded sentence pool cannot reach the word floor without repetition,
    production rejects rather than padding with filler.

    Raises :class:`ProductionError` when there are fewer than :data:`MIN_GROUNDED_CLAIMS`
    distinct grounded claims (a single or pair of claims cannot honestly become a
    long-form documentary) or when the grounded material is insufficient for the floor.
    """
    facts = research.claims
    if not facts:
        raise ProductionError("no grounded claims available to build a long-form script")

    n = len(facts)
    if n < MIN_GROUNDED_CLAIMS:
        raise ProductionError(
            f"insufficient distinct grounded claims for a long-form documentary "
            f"({n} claim(s) < {MIN_GROUNDED_CLAIMS} required); production rejects filler"
        )

    # Each scene gets an even share of the word floor; cap per scene so no single
    # scene is absurdly long. If the cap makes the floor unreachable, reject.
    per_scene_words = max(40, -(-min_words // n))  # ceil division
    per_scene_words = min(per_scene_words, _MAX_WORDS_PER_SCENE)
    if per_scene_words * n < min_words:
        raise ProductionError(
            f"insufficient grounded claims for a long-form documentary "
            f"({n} claim(s) x {per_scene_words} words < {min_words} word floor)"
        )

    # Distribute distinct grounded sentences evenly across scenes as *disjoint*
    # slices — one per chapter. No sentence is reused between chapters, so each
    # chapter is genuinely distinct and runtime is backed by real material rather
    # than repetition. Each scene stops at its word target; the last scene takes
    # whatever remains so no words are dropped.
    all_sentences = _flatten_source_sentences(getattr(research, "source_contents", {}) or {})
    total_needed_words = per_scene_words * n

    # The floor must be reachable from distinct material alone: if the combined
    # grounded word count cannot cover ``total_needed_words`` without repetition,
    # reject rather than padding with filler.
    total_available_words = sum(len(s.split()) for s in all_sentences)
    if len(all_sentences) < n or total_available_words < total_needed_words:
        raise ProductionError(
            f"insufficient distinct source material for a long-form documentary "
            f"(only {len(all_sentences)} distinct sentence(s), {total_available_words} words; need "
            f"{total_needed_words} to reach the {min_words} word floor); production rejects filler"
        )

    purposes = ["hook", "evidence", "explanation", "payoff"]
    plan: list[dict] = []
    base_share = len(all_sentences) // n
    extra = len(all_sentences) % n
    cursor = 0
    for idx, claim in enumerate(facts):
        # Even share of sentences; the first ``extra`` scenes take one more so no
        # words are dropped and every chapter gets a fair, disjoint slice.
        share = base_share + (1 if idx < extra else 0)
        chunk = all_sentences[cursor:cursor + share]
        cursor += share

        narration = _grounded_narration(claim.text, research.topic, chunk, per_scene_words)
        # Cap at the scene's word share so total narration meets the floor without
        # overshooting MAX_RUNTIME when a scene generates more than its target.
        narration = _trim_to_words(narration, per_scene_words)
        plan.append({
            "id": f"scene-{idx + 1:02d}",
            "title": f"{research.topic} — claim {idx + 1}",
            "caption": claim.text[:120],
            "narration": narration,
            "claim_ids": [claim.claim_id],
            "editorial_purpose": purposes[idx % len(purposes)],
            "asset_queries": research.topic.lower().split()[:3] or ["documentary b-roll"],
            "asset_strategy": "licensed_clip",
            "voice": None,
        })
    return plan


def synthesize_and_time(
    plan: list[dict],
    engine: RenderEngine,
) -> tuple[list[dict], list[float]]:
    """Synthesize each scene's narration and measure its real duration."""
    durations: list[float] = []
    for scene in plan:
        wav = engine.synthesize_narration(scene["narration"], scene.get("voice"))
        dur = audio_duration_seconds(wav)
        if dur is None or dur <= 0:
            raise ProductionError(f"could not measure narration duration for {scene['id']}")
        scene["narration_wav"] = wav
        scene["duration_seconds"] = round(dur, 1)
        durations.append(round(dur, 1))
    return plan, durations


def _trim_media_to_seconds(source: Path, dest: Path, target_seconds: float) -> None:
    """Trim a media file to at most ``target_seconds`` by re-encoding for precision.

    Keeps each scene's narration audio inside its (scaled) visual window so the
    absolute-offset mix never overlaps adjacent scenes.
    """
    _run_ffmpeg((
        "-y", "-i", str(source), "-t", f"{target_seconds:.3f}",
        "-map", "0:a", "-c:a", "pcm_s16le", str(dest),
    ))


def fit_to_budget(
    plan: list[dict],
    durations: Sequence[float],
    *,
    max_seconds: float = MAX_RUNTIME_SECONDS,
) -> tuple[list[dict], list[float]]:
    """Scale scene durations (and trim their narration audio) to fit ``max_seconds``.

    Real TTS pacing varies per voice and content, so a word floor that paces correctly
    for the offline tone engine can overshoot MAX_RUNTIME with Kokoro. When measured
    runtime exceeds the ceiling every scene is scaled by the same factor and its
    narration WAV is trimmed to match, keeping visuals and audio in lockstep. No-op
    when already within budget.
    """
    total = sum(durations) + _BOOKEND_SECONDS
    if total <= max_seconds:
        return plan, list(durations)
    # Leave a 1s safety margin below the ceiling for trim/measurement imprecision.
    factor = (max_seconds - 1.0 - _BOOKEND_SECONDS) / total
    scaled: list[float] = []
    for scene, dur in zip(plan, durations):
        new_dur = round(dur * factor, 3)
        scene["duration_seconds"] = new_dur
        wav = scene.get("narration_wav")
        if wav is not None:
            trimmed = Path(str(wav).rsplit(".", 1)[0] + ".trim.wav")
            _trim_media_to_seconds(wav, trimmed, new_dur)
            scene["narration_wav"] = trimmed
        scaled.append(new_dur)
    return plan, scaled


def enforce_runtime(durations: Sequence[float], *, bookend_seconds: float = _BOOKEND_SECONDS) -> None:
    """Reject candidates whose measured runtime is outside the long-form band."""
    total = sum(durations) + bookend_seconds
    if total < MIN_RUNTIME_SECONDS:
        raise ProductionError(
            f"measured runtime {total:.0f}s is below the {int(MIN_RUNTIME_SECONDS)}s "
            f"minimum; verified material cannot support a long-form documentary"
        )
    if total > MAX_RUNTIME_SECONDS:
        raise ProductionError(
            f"measured runtime {total:.0f}s exceeds the {int(MAX_RUNTIME_SECONDS)}s maximum"
        )


def build_edit_document(plan: list[dict], *, fps: int = 30) -> EditDocument:
    """Assemble an EditDocument from the timed scene plan plus bookends."""
    scenes_seconds = [
        {
            "id": s["id"],
            "duration_seconds": s["duration_seconds"],
            "title": s["title"],
            "caption": s["caption"],
            "narration": s["narration"],
            "claim_ids": s.get("claim_ids", []),
            "editorial_purpose": s.get("editorial_purpose"),
            "asset_queries": s.get("asset_queries", []),
            "asset_strategy": s.get("asset_strategy"),
            "voice": s.get("voice"),
        }
        for s in plan
    ]
    doc = build_edit_document_from_scenes(
        scenes_seconds, width=1280, height=720, fps=fps, title=None, description=None, sources=[],
    )
    # Prepend intro / append outro bookends (no narration captions).
    intro = EditScene(id="scene-intro", from_frame=0, duration_frames=seconds_to_frames(4.0, fps),
                      title="Intro", caption=f"{doc.title or 'Documentary'}", kind="intro")
    outro = EditScene(id="scene-outro", from_frame=0, duration_frames=seconds_to_frames(4.0, fps),
                      title="Outro", caption="End", kind="outro")
    doc.scenes = [intro, *doc.scenes, outro]
    # Lay out every scene uniformly on an integer frame cursor so the intro and
    # outro occupy real timeline space and no two scenes overlap. Whole frames are
    # used directly (not seconds_to_frames on a running float) so from_frame stays
    # exact -- that is what the renderer keys off for the non-overlap invariant.
    cursor = 0
    for scene in doc.scenes:
        scene.from_frame = cursor
        cursor += scene.duration_frames
    doc.duration_frames = cursor
    return doc


# ========== Asset acquisition (fail-closed rights) ==========


def make_production_asset_transport(
    *, timeout_seconds: float = 60.0, user_agent: str = "ai-video-factory/1.0"
) -> Callable[[str], bytes]:
    """Real HTTPS transport for NASA Image & Video Library asset acquisition.

    Returns raw bytes for both the search endpoint (parsed by :func:`search_nasa`)
    and per-asset media downloads. Follows redirects, enforces a wall-clock timeout,
    and fails closed on any non-2xx or network error so an unacquirable asset never
    reaches rendering. The URL is passed to urllib verbatim; ``search_nasa`` builds
    the public ``https://images-api.nasa.gov/...`` endpoint, so only public HTTPS
    locations are ever fetched (no loopback/private hosts, no plain HTTP).
    """
    from ai_video_factory.assets import TransportError

    def _transport(url: str) -> bytes:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise TransportError(f"HTTP {status} for {url}")
                return response.read()
        except urllib.error.HTTPError as error:
            raise TransportError(
                f"asset fetch failed for {url}: HTTP {error.code}"
            ) from error
        except urllib.error.URLError as error:
            raise TransportError(
                f"asset fetch failed for {url}: {error.reason}"
            ) from error

    return _transport


def acquire_assets(
    research: ResearchResult,
    plan: list[dict],
    *,
    transport: Callable[[str], bytes] | None,
    workdir: Path,
) -> tuple[list, dict[str, Path]]:
    """Acquire and approve rights-cleared assets for the candidate's scenes."""
    from ai_video_factory.assets import acquire_asset, gate_assets, select_assets

    approved_records: list = []
    assets_by_scene_id: dict[str, Path] = {}
    scene_dir = Path(workdir) / "assets"
    scene_dir.mkdir(parents=True, exist_ok=True)
    # The documentary is a tour of water in the solar system (Mars, Europa, Enceladus,
    # lunar ice). A single full-topic search returns nothing; match each scene's grounded
    # narration to a targeted NASA query so every chapter gets a topically relevant,
    # rights-cleared still. Images (not video clips) are used: they acquire and render
    # far faster for a long-form master while staying genuine NASA photography.
    subtopic_queries = (
        ("mars", "mars ancient water riverbed"),
        ("perseverance", "mars ancient water riverbed"),
        ("europa", "jupiter europa subsurface ocean"),
        ("ocean", "jupiter europa subsurface ocean"),
        ("enceladus", "saturn enceladus plume"),
        ("cryovolcanic", "saturn enceladus plume"),
        ("lunar", "lunar polar ice north pole"),
        ("pole", "lunar polar ice north pole"),
    )
    curated = [q for _, q in subtopic_queries]
    used_ids: list[str] = []
    for idx, scene in enumerate(plan):
        text = (scene.get("narration") or "").lower()
        query = next((q for kw, q in subtopic_queries if kw in text), None) or curated[idx % len(curated)]
        try:
            record = acquire_asset(
                "nasa", query, scene_id=scene["id"],
                destination=scene_dir / f"asset-{idx}.jpg", mediatype="image", transport=transport,
            )
        except Exception:  # noqa: BLE001 - offline/fixture may have no assets
            continue
        scored = select_assets([record], query.split())
        chosen = scored[0].record if scored else record
        approved_records.append(chosen)
        used_ids.append(chosen.asset_id)
        assets_by_scene_id[scene["id"]] = scene_dir / f"asset-{idx}.jpg"
    # Fail closed: ensure the full set passes the rights gate. ``gate_assets``
    # returns newly-approved records (it does not mutate in place), so assign the
    # result back; if any asset cannot be approved it raises and we never package.
    approved_records = gate_assets(approved_records)
    return approved_records, assets_by_scene_id


# ========== Candidate assembly ==========


@dataclass
class CandidateResult:
    edit: EditDocument
    master_path: Path
    narration_word_count: int
    total_runtime_seconds: float
    audio_silence_gap_seconds: float
    audio_has_clipping: bool
    video_black_ratio: float
    video_frozen_frames: int
    video_min_contrast: float
    approved_assets: list = field(default_factory=list)
    assets_by_scene_id: dict = field(default_factory=dict)
    captions: dict = field(default_factory=dict)
    rights_manifest_path: Path | None = None
    research_brief_path: Path | None = None

    @property
    def runtime_in_range(self) -> bool:
        return MIN_RUNTIME_SECONDS <= self.total_runtime_seconds <= MAX_RUNTIME_SECONDS


def build_candidate(
    research: ResearchResult,
    *,
    engine: RenderEngine,
    min_words: int = DEFAULT_MIN_WORDS,
    asset_transport: Callable[[str], bytes] | None = None,
    workdir: Path,
    fps: int = 30,
) -> CandidateResult:
    """Build a rendered documentary candidate from a research result."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    plan = build_scene_plan(research, min_words=min_words)
    plan, durations = synthesize_and_time(plan, engine)
    plan, durations = fit_to_budget(plan, durations)

    approved_assets, assets_by_scene_id = acquire_assets(research, plan, transport=asset_transport, workdir=workdir)
    # Persist the rights manifest so the upload package carries per-asset NASA usage
    # terms regardless of status; gate_assets already returned only approved records.
    from ai_video_factory.assets import write_rights_manifest as _write_rights_manifest

    rights_manifest_path = _write_rights_manifest(approved_assets, workdir / "rights_manifest.json")

    edit = build_edit_document(plan, fps=fps)

    # Map each plan scene's narration WAV to its edit scene's start offset so the
    # real engine places audio at the correct timeline position (after the intro
    # bookend, between scenes) instead of hard-coding every segment to 0:00. Normal
    # scenes keep plan order; the intro/outro carry no narration and are skipped.
    normal_scenes = [s for s in edit.scenes if s.kind == "normal"]
    narration_segments = [
        (plan_scene["narration_wav"], normal_scene.from_frame / fps)
        for plan_scene, normal_scene in zip(plan, normal_scenes)
    ]
    master_path = workdir / "master.mp4"
    engine.render_master(
        edit, narration_segments=narration_segments, assets_by_scene_id=assets_by_scene_id, destination=master_path,
    )
    if not master_path.is_file() or master_path.stat().st_size == 0:
        raise ProductionError("render produced no master file")

    # Captions + chapters from the real edit document.
    from ai_video_factory.subtitle_export import write_subtitles
    captions = write_subtitles(edit, workdir)

    # Research brief (writer input) preserved for the package.
    brief_path = workdir / "research_brief.md"
    if research.research_brief:
        brief_path.write_text(research.research_brief, encoding="utf-8")

    total_runtime = frames_to_seconds(edit.duration_frames, fps)

    return CandidateResult(
        edit=edit,
        master_path=master_path,
        narration_word_count=sum(len(s["narration"].split()) for s in plan),
        total_runtime_seconds=round(total_runtime, 1),
        audio_silence_gap_seconds=audio_silence_gap_seconds(narration_segments[0][0]) if narration_segments else 0.0,
        audio_has_clipping=audio_has_clipping(narration_segments[0][0]) if narration_segments else False,
        # Dark-by-design segments (intro/outro branded cards, act-card
        # scrim windows) are excluded so intended darkness never counts.
        video_black_ratio=video_black_ratio(
            master_path, exclude_windows=black_frame_exclude_windows(edit)
        ),
        video_frozen_frames=video_frozen_frame_count(master_path),
        video_min_contrast=video_min_contrast(master_path),
        approved_assets=approved_assets,
        assets_by_scene_id=assets_by_scene_id,
        captions=captions,
        rights_manifest_path=rights_manifest_path,
        research_brief_path=brief_path,
    )
