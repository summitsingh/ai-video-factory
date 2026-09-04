from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any


CommandResult = tuple[int, str, str]
CommandRunner = Callable[[Sequence[str]], CommandResult]


class MediaProbeError(RuntimeError):
    """Raised when ffprobe cannot return usable metadata for a media file."""


@dataclass(frozen=True)
class MediaInfo:
    """The media attributes used by technical QC."""

    source_path: Path | None
    video_codec: str | None
    audio_codec: str | None
    width: int | None
    height: int | None
    frame_rate: Fraction | None
    duration_seconds: float | None
    decode_succeeded: bool | None = None
    decode_detail: str | None = None


def _subprocess_runner(argv: Sequence[str]) -> CommandResult:
    completed = subprocess.run(
        argv,
        shell=False,
        timeout=15,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _first_stream(payload: Mapping[str, Any], codec_type: str) -> Mapping[str, Any] | None:
    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        raise MediaProbeError("ffprobe streams must be a list")
    return next(
        (
            stream
            for stream in streams
            if isinstance(stream, Mapping) and stream.get("codec_type") == codec_type
        ),
        None,
    )


def _frame_rate(value: object) -> Fraction | None:
    if value in (None, "", "0/0"):
        return None
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise MediaProbeError(f"invalid ffprobe frame rate: {value!r}") from error


def _duration(payload: Mapping[str, Any]) -> float | None:
    format_data = payload.get("format", {})
    if not isinstance(format_data, Mapping):
        raise MediaProbeError("ffprobe format must be an object")
    value = format_data.get("duration")
    if value in (None, "", "N/A"):
        return None
    try:
        return float(str(value))
    except ValueError as error:
        raise MediaProbeError(f"invalid ffprobe duration: {value!r}") from error


def parse_ffprobe(payload: Mapping[str, Any]) -> MediaInfo:
    """Parse ffprobe's JSON response without losing rational frame-rate precision."""
    video = _first_stream(payload, "video")
    audio = _first_stream(payload, "audio")

    return MediaInfo(
        source_path=None,
        video_codec=str(video["codec_name"]) if video and video.get("codec_name") is not None else None,
        audio_codec=str(audio["codec_name"]) if audio and audio.get("codec_name") is not None else None,
        width=int(video["width"]) if video and video.get("width") is not None else None,
        height=int(video["height"]) if video and video.get("height") is not None else None,
        frame_rate=_frame_rate(video.get("avg_frame_rate")) if video else None,
        duration_seconds=_duration(payload),
    )


def _command_error(command: str, returncode: int, stdout: str, stderr: str) -> MediaProbeError:
    detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
    return MediaProbeError(f"{command} failed: {detail}")


def probe_media(path: Path, runner: CommandRunner = _subprocess_runner) -> MediaInfo:
    """Probe and fully decode one explicit local media path with fixed FFmpeg commands."""
    media_path = Path(path)
    ffprobe_argv = (
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(media_path),
    )
    returncode, stdout, stderr = runner(ffprobe_argv)
    if returncode != 0:
        raise _command_error("ffprobe", returncode, stdout, stderr)

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise MediaProbeError("ffprobe returned invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise MediaProbeError("ffprobe JSON root must be an object")

    media = replace(parse_ffprobe(payload), source_path=media_path)
    decode_argv = ("ffmpeg", "-v", "error", "-i", str(media_path), "-f", "null", "-")
    decode_returncode, decode_stdout, decode_stderr = runner(decode_argv)
    decode_detail = None
    if decode_returncode != 0:
        decode_detail = decode_stderr.strip() or decode_stdout.strip() or f"exit code {decode_returncode}"
    return replace(
        media,
        decode_succeeded=decode_returncode == 0,
        decode_detail=decode_detail,
    )
