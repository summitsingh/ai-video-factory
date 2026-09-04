import json
from pathlib import Path

from ai_video_factory.edit_schema import load_edit
from ai_video_factory.media_probe import parse_ffprobe
from ai_video_factory.qc import evaluate_qc, write_qc_reports


def test_valid_synthetic_video_passes_qc() -> None:
    payload = json.loads(Path("tests/fixtures/ffprobe-video.json").read_text())
    media = parse_ffprobe(payload)

    report = evaluate_qc(media, load_edit(Path("fixtures/synthetic-edit.json")))

    assert report.status == "pass"
    assert all(check.passed for check in report.checks)


def test_missing_audio_fails_qc() -> None:
    payload = json.loads(Path("tests/fixtures/ffprobe-video.json").read_text())
    payload["streams"] = [s for s in payload["streams"] if s["codec_type"] != "audio"]

    report = evaluate_qc(parse_ffprobe(payload), load_edit(Path("fixtures/synthetic-edit.json")))

    assert report.status == "fail"
    assert next(check for check in report.checks if check.name == "audio-stream").passed is False


def test_qc_reports_write_json_and_markdown(tmp_path: Path) -> None:
    """A report writer must leave both machine- and human-readable artifacts."""
    payload = json.loads(Path("tests/fixtures/ffprobe-video.json").read_text())
    report = evaluate_qc(parse_ffprobe(payload), load_edit(Path("fixtures/synthetic-edit.json")))

    json_path, markdown_path = write_qc_reports(report, tmp_path)

    assert json_path.name == "qc_report.json"
    assert markdown_path.name == "qc_report.md"
    assert json.loads(json_path.read_text())["status"] == "pass"
    assert "# Technical QC: pass" in markdown_path.read_text()
