"""Unit tests for the post-render final QC gate (ai_video_factory.qc_final)."""

import json
import shutil
from pathlib import Path

import pytest

from ai_video_factory.qc_final import (
    FinalQcReport,
    _parse_peak_db,
    _parse_silence_gaps,
    check_duration,
    check_resolution,
    run_final_qc,
)


def _payload(width: int, height: int, duration: float, audio: bool = True) -> dict:
    streams = [
        {
            "index": 0,
            "codec_type": "video",
            "width": width,
            "height": height,
            "duration": str(duration),
        }
    ]
    if audio:
        streams.append({"index": 1, "codec_type": "audio", "duration": str(duration)})
    return {"streams": streams, "format": {"duration": str(duration)}}


def test_resolution_exact_match_passes() -> None:
    check = check_resolution(_payload(1280, 720, 10.0), 1280, 720)
    assert check.passed
    assert check.value == "1280x720"


def test_resolution_letterbox_variant_passes() -> None:
    # The pipeline letterboxes 1280x720 to ~2.39:1 (1280x536).
    check = check_resolution(_payload(1280, 536, 10.0), 1280, 720)
    assert check.passed


def test_resolution_mismatch_fails() -> None:
    check = check_resolution(_payload(640, 480, 10.0), 1280, 720)
    assert not check.passed


def test_duration_within_tolerance_passes() -> None:
    check = check_duration(_payload(1280, 720, 10.4), 10.0)
    assert check.passed


def test_duration_outside_tolerance_fails() -> None:
    check = check_duration(_payload(1280, 720, 12.0), 10.0)
    assert not check.passed
    assert "20.00%" in check.detail


def test_parse_silence_gaps() -> None:
    stderr = (
        "[silencedetect @ 0x1] silence_start: 3.0\n"
        "[silencedetect @ 0x1] silence_end: 7.0 | silence_duration: 4.0\n"
    )
    assert _parse_silence_gaps(stderr, 10.0) == [4.0]


def test_parse_silence_gaps_trailing_silence() -> None:
    stderr = "[silencedetect @ 0x1] silence_start: 8.5\n"
    assert _parse_silence_gaps(stderr, 10.0) == [1.5]


def test_parse_silence_gaps_none() -> None:
    assert _parse_silence_gaps("nothing here\n", 10.0) == []


def test_parse_peak_db() -> None:
    assert _parse_peak_db("[Parsed_volumedetect] max_volume: -12.3 dB\n") == -12.3
    assert _parse_peak_db("[Parsed_volumedetect] max_volume: n/a dB\n") is None
    assert _parse_peak_db("no output\n") is None


def test_report_json_schema(tmp_path: Path) -> None:
    report = FinalQcReport(
        master="/tmp/master.mp4",
        target={"width": 1280, "height": 720, "duration_seconds": 10.0},
    )
    destination = report.write_json(tmp_path / "qc-report.json")
    payload = json.loads(destination.read_text())
    assert payload["schema_version"] == 1
    assert payload["tool"] == "qc_final"
    assert payload["status"] == "pass"  # no checks -> vacuously passing
    assert payload["target"]["width"] == 1280


ffmpeg_available = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not on PATH")
def test_run_final_qc_good_file_passes(tmp_path: Path) -> None:
    import subprocess

    master = tmp_path / "good.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4:sample_rate=48000",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", str(master),
        ],
        check=True,
        timeout=120,
    )
    report = run_final_qc(
        master,
        target_width=320,
        target_height=240,
        target_duration_seconds=4.0,
        report_path=tmp_path / "qc-report.json",
    )
    assert report.passed
    assert report.status == "pass"
    assert (tmp_path / "qc-report.json").exists()
    assert all(check.passed for check in report.checks)


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not on PATH")
def test_run_final_qc_black_file_fails_black_frames(tmp_path: Path) -> None:
    import subprocess

    master = tmp_path / "black.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:size=320x240:rate=15:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4:sample_rate=48000",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", str(master),
        ],
        check=True,
        timeout=120,
    )
    report = run_final_qc(
        master,
        target_width=320,
        target_height=240,
        target_duration_seconds=4.0,
        report_path=tmp_path / "qc-report.json",
    )
    assert not report.passed
    assert report.status == "fail"
    black = next(c for c in report.checks if c.name == "black-frames")
    assert not black.passed
