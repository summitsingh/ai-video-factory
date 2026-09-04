import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from ai_video_factory.edit_schema import load_edit
from ai_video_factory.media_probe import MediaInfo, parse_ffprobe
from ai_video_factory.qc import evaluate_qc, write_qc_reports


def _decoded_media() -> MediaInfo:
    payload = json.loads(Path("tests/fixtures/ffprobe-video.json").read_text())
    return replace(parse_ffprobe(payload), decode_succeeded=True)


def test_valid_synthetic_video_passes_qc() -> None:
    report = evaluate_qc(_decoded_media(), load_edit(Path("fixtures/synthetic-edit.json")))

    assert report.status == "pass"
    assert all(check.passed for check in report.checks)


def test_missing_audio_fails_qc() -> None:
    payload = json.loads(Path("tests/fixtures/ffprobe-video.json").read_text())
    payload["streams"] = [s for s in payload["streams"] if s["codec_type"] != "audio"]

    report = evaluate_qc(
        replace(parse_ffprobe(payload), decode_succeeded=True),
        load_edit(Path("fixtures/synthetic-edit.json")),
    )

    assert report.status == "fail"
    assert next(check for check in report.checks if check.name == "audio-stream").passed is False


def test_non_h264_video_fails_qc() -> None:
    report = evaluate_qc(
        replace(_decoded_media(), video_codec="vp9"),
        load_edit(Path("fixtures/synthetic-edit.json")),
    )

    assert report.status == "fail"
    assert next(check for check in report.checks if check.name == "video-codec").passed is False


def test_non_aac_audio_fails_qc() -> None:
    report = evaluate_qc(
        replace(_decoded_media(), audio_codec="opus"),
        load_edit(Path("fixtures/synthetic-edit.json")),
    )

    assert report.status == "fail"
    assert next(check for check in report.checks if check.name == "audio-codec").passed is False


def test_non_48_khz_audio_fails_qc() -> None:
    report = evaluate_qc(
        replace(_decoded_media(), audio_sample_rate=44_100),
        load_edit(Path("fixtures/synthetic-edit.json")),
    )

    assert report.status == "fail"
    assert next(check for check in report.checks if check.name == "audio-sample-rate").passed is False


def test_non_stereo_audio_fails_qc() -> None:
    report = evaluate_qc(
        replace(_decoded_media(), audio_channels=1, audio_channel_layout="mono"),
        load_edit(Path("fixtures/synthetic-edit.json")),
    )

    assert report.status == "fail"
    assert next(check for check in report.checks if check.name == "audio-layout").passed is False


def test_qc_reports_write_json_and_markdown(tmp_path: Path) -> None:
    """A report writer must leave both machine- and human-readable artifacts."""
    report = evaluate_qc(_decoded_media(), load_edit(Path("fixtures/synthetic-edit.json")))

    json_path, markdown_path = write_qc_reports(report, tmp_path)

    assert json_path.name == "qc_report.json"
    assert markdown_path.name == "qc_report.md"
    assert json.loads(json_path.read_text())["status"] == "pass"
    assert "# Technical QC: pass" in markdown_path.read_text()


def test_unexecuted_full_decode_fails_qc() -> None:
    """QC must not pass when no FFmpeg decode exit status was observed."""
    payload = json.loads(Path("tests/fixtures/ffprobe-video.json").read_text())

    report = evaluate_qc(parse_ffprobe(payload), load_edit(Path("fixtures/synthetic-edit.json")))

    assert report.status == "fail"
    assert next(check for check in report.checks if check.name == "full-decode").passed is False


def test_frame_rate_at_exact_tolerance_boundary_passes_qc() -> None:
    """A 0.01 fps delta is allowed, even when binary floats would round it upward."""
    report = evaluate_qc(
        replace(_decoded_media(), frame_rate=Fraction(3001, 100)),
        load_edit(Path("fixtures/synthetic-edit.json")),
    )

    assert report.status == "pass"
    assert next(check for check in report.checks if check.name == "frame-rate").passed is True


def test_frame_rate_beyond_tolerance_fails_qc() -> None:
    report = evaluate_qc(
        replace(_decoded_media(), frame_rate=Fraction(3001001, 100000)),
        load_edit(Path("fixtures/synthetic-edit.json")),
    )

    assert report.status == "fail"
    assert next(check for check in report.checks if check.name == "frame-rate").passed is False


def test_duration_at_exact_tolerance_boundary_passes_qc() -> None:
    """A 0.10 second delta is allowed, even when binary floats would round it upward."""
    report = evaluate_qc(
        replace(_decoded_media(), duration_seconds=3.1),
        load_edit(Path("fixtures/synthetic-edit.json")),
    )

    assert report.status == "pass"
    assert next(check for check in report.checks if check.name == "duration").passed is True


def test_duration_beyond_tolerance_fails_qc() -> None:
    report = evaluate_qc(
        replace(_decoded_media(), duration_seconds=3.1001),
        load_edit(Path("fixtures/synthetic-edit.json")),
    )

    assert report.status == "fail"
    assert next(check for check in report.checks if check.name == "duration").passed is False
