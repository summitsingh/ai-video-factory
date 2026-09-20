"""Background music bed for AI Video Factory (pipeline item #8).

Takes a real royalty-free music track, fits it to the full video duration
(loop or trim), loudness-normalizes the bed, ducks it under the narration
with sidechain compression, and loudness-normalizes the complete final mix
(dual-pass EBU R128 when feasible, single-pass fallback otherwise).

Track sourcing
--------------
The pipeline prefers a real, licensed music track supplied by the operator.
Resolution order in :func:`resolve_music_track`:

1. an explicit ``music_bed_path`` argument, then
2. the ``MUSIC_BED_PATH`` environment variable pointing at an audio file
   (mp3/wav/ogg/m4a/flac).

If no track resolves, the caller falls back to the procedural ambient drone
(``video_pipeline._generate_music_bed``) so the mix never goes out silent;
either way the bed is fitted and normalized here before the mux.

Why there is no Pixabay music fetch
-----------------------------------
The project's ``PIXABAY_API_KEY`` only grants Pixabay's public image/video
endpoints. Tested 2026-09-20:

- ``GET https://pixabay.com/api/audio/?key=<key>&q=ambient`` ->
  ``403 [ERROR 403] Access denied`` (endpoint exists, key not granted)
- ``GET https://pixabay.com/api/music/`` -> 404
- ``GET https://pixabay.com/api/music/search/`` -> 404
- ``GET https://pixabay.com/api/?key=<key>&q=ambient`` -> 200 (images work)

Pixabay's published API docs (https://pixabay.com/api/docs/) cover images
and videos only; music search is a website feature that needs a separate
grant on the Pixabay account. When the user obtains music API access, add a
``fetch_pixabay_music()`` helper here that searches, picks an
instrumental/calm track by tags and duration, and downloads it to
``data/cache/music/`` - the rest of this module (fit, duck, normalize)
already accepts any local file.

Level plan
----------
- Bed pre-normalized to ``DEFAULT_BED_TARGET_I`` (-23 LUFS integrated) so
  its level is deterministic regardless of source mastering.
- In the mux the bed runs at unity gain and is ducked by the narration via
  :func:`duck_filter` (sidechain compression, ~10-14 dB of gain reduction
  while speech is present), landing roughly 20 dB under the narration.
- The complete mix is dual-pass loudness-normalized to YouTube/EBU R128
  (-16 LUFS integrated, -2 dBTP, 7 LU) before the final mux.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from ai_video_factory.sanitization import sanitize_diagnostic


# Name of the environment variable carrying the operator-supplied music file.
MUSIC_BED_PATH_ENV = "MUSIC_BED_PATH"

# Integrated loudness target for the pre-normalized music bed. -23 LUFS keeps
# the bed clearly audible in narration gaps while leaving headroom for the
# final mix normalization and the ducking stage.
DEFAULT_BED_TARGET_I = -23.0

# Final mix targets (EBU R128 / YouTube): -16 LUFS integrated, -2 dBTP, 7 LU.
FINAL_MIX_TARGET_I = -16.0
FINAL_MIX_TARGET_TP = -2.0
FINAL_MIX_TARGET_LRA = 7.0


class MusicBedError(RuntimeError):
    """Raised when a music bed cannot be built."""


def resolve_music_track(music_bed_path: str | Path | None = None) -> Path | None:
    """Resolve the operator-supplied music track, if any.

    Checks the explicit ``music_bed_path`` first, then the
    ``MUSIC_BED_PATH`` environment variable. Returns the path when it points
    at an existing file, otherwise ``None`` (caller falls back to the
    procedural drone). A configured-but-missing path prints a warning and
    also resolves to ``None`` rather than failing the whole pipeline run.
    """
    candidates: list[Path] = []
    if music_bed_path:
        candidates.append(Path(music_bed_path))
    env_path = os.environ.get(MUSIC_BED_PATH_ENV, "").strip()
    if env_path:
        candidates.append(Path(env_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        print(
            f"[music_bed] WARNING: {MUSIC_BED_PATH_ENV}={candidate} does not "
            "exist; falling back to the procedural music bed."
        )
    return None


def _resolve_ffprobe(ffmpeg_bin: str) -> str:
    """Find ffprobe next to the ffmpeg binary, else rely on PATH."""
    ffmpeg_path = Path(ffmpeg_bin)
    if ffmpeg_path.name != ffmpeg_path.as_posix() or "/" in ffmpeg_bin:
        sibling = ffmpeg_path.parent / "ffprobe"
        if sibling.is_file():
            return str(sibling)
    return "ffprobe"


def _run(
    argv: list[str],
    *,
    name: str,
    cwd: Path,
    timeout: int,
    retries: int = 0,
) -> None:
    """Run a subprocess, retrying transient failures (mirrors the pipeline)."""
    last_error: str | None = None
    for attempt in range(retries + 1):
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
            last_error = sanitize_diagnostic(f"{name} could not run: {error}")
        else:
            if completed.returncode == 0:
                return
            detail = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"exit code {completed.returncode}"
            )
            last_error = sanitize_diagnostic(f"{name} failed: {detail}")
        if attempt < retries:
            print(f"[music_bed] {name}: attempt {attempt + 1} failed, retrying...")
            time.sleep(5)
    raise MusicBedError(last_error or f"{name} failed")


def probe_audio_duration(path: str | Path, *, ffprobe: str | None = None) -> float:
    """Return the duration of an audio file in seconds via ffprobe."""
    path = Path(path)
    ffprobe_bin = ffprobe or _resolve_ffprobe("ffmpeg")
    try:
        completed = subprocess.run(
            [
                ffprobe_bin,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MusicBedError(f"ffprobe could not run on {path}: {error}") from error
    if completed.returncode != 0:
        raise MusicBedError(
            f"ffprobe failed on {path}: "
            f"{sanitize_diagnostic(completed.stderr.strip() or 'unknown error')}"
        )
    try:
        return float(completed.stdout.strip())
    except ValueError as error:
        raise MusicBedError(f"ffprobe returned no duration for {path}") from error


def fit_track_to_duration(
    src: str | Path,
    duration_seconds: float,
    output: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str | None = None,
    fade_seconds: float = 2.0,
    timeout: int = 600,
) -> Path:
    """Fit a music track to exactly ``duration_seconds``.

    Shorter tracks are looped (``-stream_loop``) and longer tracks are
    trimmed; both get a short fade-in and fade-out so loop points and the
    video tail never click. Output is 48 kHz stereo PCM WAV.

    Note: loop seams are not crossfaded. For ambient/cinematic beds played
    quietly under ducked narration this is inaudible in practice; if a seam
    ever becomes audible, prefer a track whose own duration already covers
    the video.
    """
    src = Path(src)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if duration_seconds <= 0:
        raise MusicBedError(f"invalid target duration: {duration_seconds}")
    if not src.is_file():
        raise MusicBedError(f"music source not found: {src}")

    src_duration = probe_audio_duration(src, ffprobe=ffprobe)
    if src_duration <= 0:
        raise MusicBedError(f"music source has no measurable duration: {src}")

    fade = min(fade_seconds, duration_seconds / 4.0, src_duration / 4.0)
    fade_in = min(1.0, fade)
    fade_out_start = max(0.0, duration_seconds - fade)
    base_filter = (
        f"afade=t=in:st=0:d={fade_in:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade:.3f},"
        "aresample=48000,aformat=channel_layouts=stereo"
    )

    if src_duration >= duration_seconds:
        argv = [
            ffmpeg, "-y",
            "-i", str(src),
            "-af", f"atrim=0:{duration_seconds:.3f},{base_filter}",
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
            "-t", f"{duration_seconds:.3f}",
            str(output),
        ]
        name = "FFmpeg music bed trim"
    else:
        loops = math.ceil(duration_seconds / src_duration)
        argv = [
            ffmpeg, "-y",
            "-stream_loop", str(loops),
            "-i", str(src),
            "-af", base_filter,
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
            "-t", f"{duration_seconds:.3f}",
            str(output),
        ]
        name = "FFmpeg music bed loop"
    _run(argv, name=name, cwd=output.parent, timeout=timeout, retries=1)
    return output


def _parse_loudnorm_json(stderr: str) -> dict[str, str] | None:
    """Extract the loudnorm measurement JSON ffmpeg prints to stderr."""
    match = re.search(r"\{[^}]*\"input_i\"[^}]*\}", stderr, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    required = (
        "input_i", "input_tp", "input_lra", "input_thresh", "target_offset",
    )
    if not all(key in data for key in required):
        return None
    return {key: str(data[key]) for key in required}


def loudnorm_dual_pass(
    src: str | Path,
    dst: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    target_i: float = FINAL_MIX_TARGET_I,
    target_tp: float = FINAL_MIX_TARGET_TP,
    target_lra: float = FINAL_MIX_TARGET_LRA,
    timeout: int = 900,
    cwd: Path | None = None,
) -> Path:
    """Loudness-normalize audio with true dual-pass EBU R128.

    Pass 1 measures the input (``print_format=json`` to null); pass 2
    applies the measured values with ``linear=true``. If measurement parsing
    fails for any reason, falls back to single-pass loudnorm so the pipeline
    never breaks on an ffmpeg output-format quirk.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    workdir = cwd or dst.parent
    base = f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"

    measured: dict[str, str] | None = None
    try:
        probe = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(src), "-af",
             f"{base}:print_format=json", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if probe.returncode == 0:
            measured = _parse_loudnorm_json(probe.stderr)
    except (OSError, subprocess.TimeoutExpired):
        measured = None

    if measured is None:
        print("[music_bed] loudnorm measurement parse failed; "
              "falling back to single-pass loudnorm.")
        af = f"{base},aresample=48000"
    else:
        af = (
            f"{base}"
            f":measured_I={measured['input_i']}"
            f":measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}"
            f":measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}"
            ":linear=true,aresample=48000"
        )
    _run(
        [ffmpeg, "-y", "-i", str(src), "-af", af,
         "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(dst)],
        name="FFmpeg loudnorm",
        cwd=workdir,
        timeout=timeout,
        retries=1,
    )
    return dst


def prepare_music_bed(
    src: str | Path,
    duration_seconds: float,
    output: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str | None = None,
    target_i: float = DEFAULT_BED_TARGET_I,
    timeout: int = 900,
) -> Path:
    """Fit a track to the video duration and normalize its loudness.

    Convenience wrapper: :func:`fit_track_to_duration` then dual-pass
    loudnorm to ``target_i`` LUFS. Returns ``output`` (48 kHz stereo WAV)
    ready to be ducked under narration at unity gain.
    """
    output = Path(output)
    fitted = output.parent / f"{output.stem}.fitted.wav"
    try:
        fit_track_to_duration(
            src, duration_seconds, fitted,
            ffmpeg=ffmpeg, ffprobe=ffprobe, timeout=timeout,
        )
        loudnorm_dual_pass(
            fitted, output,
            ffmpeg=ffmpeg,
            target_i=target_i,
            target_tp=-2.0,
            target_lra=7.0,
            timeout=timeout,
            cwd=output.parent,
        )
    finally:
        fitted.unlink(missing_ok=True)
    return output


def duck_filter(music_label: str, narr_label: str, out_label: str) -> str:
    """Build the ffmpeg filter that ducks music under narration.

    ``music_label`` is the filter input for the (pre-normalized, unity-gain)
    music bed, ``narr_label`` the narration sidechain trigger. Uses
    ``sidechaincompress``: while speech is present the bed dips hard with
    a smooth 400 ms release, then returns to its full level in speech
    gaps (25 dB of reduction measured with -3 dBFS-peak gated narration;
    ~10-14 dB with quieter speech). With the bed pre-normalized to
    -23 LUFS, the ducked bed sits roughly 20 dB (or more) under active
    narration.
    """
    return (
        f"[{music_label}]volume=1.0[music_base];"
        f"[music_base][{narr_label}]"
        "sidechaincompress=threshold=0.02:ratio=12:attack=20:release=400"
        f"[{out_label}]"
    )
