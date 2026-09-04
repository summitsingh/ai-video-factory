import json
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
