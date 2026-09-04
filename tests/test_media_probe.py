import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from ai_video_factory.media_probe import MediaProbeError, parse_ffprobe, probe_media


def test_parse_ffprobe_preserves_rational_frame_rate() -> None:
    """A wrong rational parser would silently misreport non-integer media rates."""
    payload = json.loads(Path("tests/fixtures/ffprobe-video.json").read_text())

    media = parse_ffprobe(payload)

    assert media.video_codec == "h264"
    assert media.audio_codec == "aac"
    assert media.audio_sample_rate == 48_000
    assert media.audio_channels == 2
    assert media.audio_channel_layout == "stereo"
    assert (media.width, media.height) == (1280, 720)
    assert media.frame_rate == Fraction(30, 1)
    assert media.duration_seconds == 3.0


def test_probe_media_uses_fixed_ffprobe_and_decode_commands() -> None:
    """A changed command could let ffprobe parse untrusted options or skip decoding."""
    payload = Path("tests/fixtures/ffprobe-video.json").read_text()
    commands: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> tuple[int, str, str]:
        commands.append(argv)
        return (0, payload, "") if argv[0] == "ffprobe" else (0, "", "")

    media = probe_media(Path("/tmp/render.mp4"), runner)

    assert media.source_path == Path("/tmp/render.mp4")
    assert media.decode_succeeded is True
    assert commands == [
        (
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            "/tmp/render.mp4",
        ),
        ("ffmpeg", "-v", "error", "-i", "/tmp/render.mp4", "-f", "null", "-"),
    ]


def test_probe_media_rejects_a_nonzero_ffprobe_exit() -> None:
    """A failed ffprobe invocation must not be treated as valid media metadata."""

    def runner(argv: tuple[str, ...]) -> tuple[int, str, str]:
        return 1, "", "invalid input"

    with pytest.raises(MediaProbeError, match="ffprobe failed: invalid input"):
        probe_media(Path("/tmp/broken.mp4"), runner)


def test_probe_media_redacts_and_limits_subprocess_failure_detail() -> None:
    """ffprobe diagnostics must be safe before becoming a public pipeline error."""
    secret = "media-probe-secret"

    def runner(_argv: tuple[str, ...]) -> tuple[int, str, str]:
        return 1, "", f"token={secret}\n" + ("x" * 5_000)

    with pytest.raises(MediaProbeError) as caught:
        probe_media(Path("/tmp/broken.mp4"), runner)

    detail = str(caught.value)
    assert secret not in detail
    assert "[REDACTED]" in detail
    assert len(detail) <= 2_048


def test_probe_media_sanitizes_failed_full_decode_detail() -> None:
    """A QC report must not persist credentials emitted by FFmpeg decode."""
    payload = Path("tests/fixtures/ffprobe-video.json").read_text()

    def runner(argv: tuple[str, ...]) -> tuple[int, str, str]:
        if argv[0] == "ffprobe":
            return 0, payload, ""
        return 1, "", "https://user:decode-secret@example.test/file?api_key=query-secret"

    media = probe_media(Path("/tmp/broken.mp4"), runner)

    assert media.decode_succeeded is False
    assert "decode-secret" not in (media.decode_detail or "")
    assert "query-secret" not in (media.decode_detail or "")


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("ffprobe missing"),
        subprocess.TimeoutExpired(("ffprobe",), 15),
    ],
)
def test_probe_media_wraps_command_start_failures(error: Exception) -> None:
    """Missing and timed-out tools must use the stable media-probe error boundary."""
    def runner(_argv: tuple[str, ...]) -> tuple[int, str, str]:
        raise error

    with pytest.raises(MediaProbeError):
        probe_media(Path("/tmp/broken.mp4"), runner)
