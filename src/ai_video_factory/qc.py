from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ai_video_factory.edit_schema import EditDocument
from ai_video_factory.media_probe import MediaInfo


class QcCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class QcReport(BaseModel):
    schema_version: Literal[1] = 1
    status: Literal["pass", "fail"]
    checks: list[QcCheck]


def _check(name: str, passed: bool, detail: str) -> QcCheck:
    return QcCheck(name=name, passed=passed, detail=detail)


def evaluate_qc(media: MediaInfo, edit: EditDocument) -> QcReport:
    """Compare probed media to an edit specification and aggregate technical checks."""
    dimensions_match = media.width == edit.width and media.height == edit.height
    frame_rate_match = (
        media.frame_rate is not None and abs(float(media.frame_rate) - edit.fps) <= 0.01
    )
    expected_duration = edit.duration_frames / edit.fps
    duration_match = (
        media.duration_seconds is not None
        and abs(media.duration_seconds - expected_duration) <= 0.10
    )
    decode_passed = media.decode_succeeded is not False

    checks = [
        _check("video-stream", media.video_codec is not None, "video stream detected"),
        _check("audio-stream", media.audio_codec is not None, "audio stream detected"),
        _check(
            "dimensions",
            dimensions_match,
            f"expected {edit.width}x{edit.height}; got {media.width}x{media.height}",
        ),
        _check(
            "frame-rate",
            frame_rate_match,
            f"expected {edit.fps:.2f} fps; got "
            f"{float(media.frame_rate):.6f} fps" if media.frame_rate is not None else "frame rate unavailable",
        ),
        _check(
            "duration",
            duration_match,
            f"expected {expected_duration:.3f}s; got "
            f"{media.duration_seconds:.3f}s"
            if media.duration_seconds is not None
            else "duration unavailable",
        ),
        _check(
            "full-decode",
            decode_passed,
            media.decode_detail
            or ("full decode passed" if media.decode_succeeded is True else "not run for parsed metadata"),
        ),
    ]
    status: Literal["pass", "fail"] = "pass" if all(check.passed for check in checks) else "fail"
    return QcReport(status=status, checks=checks)


def write_qc_reports(report: QcReport, output_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable JSON and a concise human-readable QC summary."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "qc_report.json"
    markdown_path = destination / "qc_report.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    rows = ["# Technical QC: " + report.status, "", "| Check | Status | Detail |", "| --- | --- | --- |"]
    rows.extend(
        f"| {check.name} | {'pass' if check.passed else 'fail'} | {check.detail.replace('|', '\\|')} |"
        for check in report.checks
    )
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, markdown_path
