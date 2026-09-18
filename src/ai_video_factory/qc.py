from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Literal, Mapping

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
    # The master is letterboxed to a cinematic ~2.39:1 band by _polish_master, so
    # accept either the original canvas dimensions or the cropped aspect ratio.
    def _aspect(w: int, h: int) -> float:
        return w / h if h else 0.0

    dimensions_match = (
        media.width == edit.width and media.height == edit.height
    ) or (
        media.width == edit.width
        and abs(_aspect(media.width, media.height) - 2.39) <= 0.02
    )
    expected_frame_rate = Fraction(edit.fps, 1)
    frame_rate_match = (
        media.frame_rate is not None
        and abs(media.frame_rate - expected_frame_rate) <= Fraction(1, 100)
    )
    expected_duration = Fraction(edit.duration_frames, edit.fps)
    duration_match = (
        media.duration_seconds is not None
        and abs(Fraction(str(media.duration_seconds)) - expected_duration) <= Fraction(1, 10)
    )
    decode_passed = media.decode_succeeded is True

    checks = [
        _check(
            "video-stream",
            media.video_codec is not None,
            "video stream detected" if media.video_codec is not None else "video stream missing",
        ),
        _check(
            "video-codec",
            media.video_codec == "h264",
            f"expected h264; got {media.video_codec or 'unavailable'}",
        ),
        _check(
            "audio-stream",
            media.audio_codec is not None,
            "audio stream detected" if media.audio_codec is not None else "audio stream missing",
        ),
        _check(
            "audio-codec",
            media.audio_codec == "aac",
            f"expected aac; got {media.audio_codec or 'unavailable'}",
        ),
        _check(
            "audio-sample-rate",
            media.audio_sample_rate == 48_000,
            f"expected 48000 Hz; got {media.audio_sample_rate or 'unavailable'}",
        ),
        _check(
            "audio-layout",
            media.audio_channels == 2 and media.audio_channel_layout == "stereo",
            "expected 2-channel stereo; got "
            f"{media.audio_channels if media.audio_channels is not None else 'unavailable'} "
            f"channels ({media.audio_channel_layout or 'layout unavailable'})",
        ),
        _check(
            "dimensions",
            dimensions_match,
            f"expected {edit.width}x{edit.height} or 2.39:1 letterbox; got {media.width}x{media.height}" + (" (2.39:1 letterbox OK)" if dimensions_match and (media.width, media.height) != (edit.width, edit.height) else ""),
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
            f"expected {float(expected_duration):.3f}s; got "
            f"{media.duration_seconds:.3f}s"
            if media.duration_seconds is not None
            else "duration unavailable",
        ),
        _check(
            "full-decode",
            decode_passed,
            media.decode_detail
            or ("full decode passed" if decode_passed else "full decode was not successful"),
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


def evaluate_content_qc(
    edit: EditDocument,
    *,
    target_minutes: float | None = None,
    artifacts: Mapping[str, str] | None = None,
    longform: bool = False,
) -> QcReport:
    """Content-level checks for long-form readiness.

    Verifies the edit actually delivers the promised runtime (within 5%),
    has enough scenes to sustain visual pacing, keeps narration on nearly
    every scene, carries the act structure for long-form, and produced the
    caption/chapter/thumbnail artifacts publishing needs.
    """
    actual_minutes = edit.duration_frames / edit.fps / 60 if edit.fps else 0.0
    checks: list[QcCheck] = []
    if target_minutes is not None and target_minutes > 0:
        within = abs(actual_minutes - target_minutes) / target_minutes <= 0.05
        checks.append(
            _check(
                "target-duration",
                within,
                f"target {target_minutes:.1f} min; edit is {actual_minutes:.1f} min",
            )
        )
    min_scenes = 10 if longform else 3
    checks.append(
        _check(
            "scene-count",
            len(edit.scenes) >= min_scenes,
            f"{len(edit.scenes)} scenes (minimum {min_scenes})",
        )
    )
    narrated = sum(1 for s in edit.scenes if s.narration and s.narration.strip())
    coverage = narrated / len(edit.scenes) if edit.scenes else 0.0
    checks.append(
        _check(
            "narration-coverage",
            coverage >= 0.9,
            f"{narrated}/{len(edit.scenes)} scenes carry narration",
        )
    )
    if longform:
        acts = {s.act for s in edit.scenes if s.act}
        checks.append(
            _check(
                "act-structure",
                len(acts) >= 5,
                f"{len(acts)} distinct act cards present",
            )
        )
    present = artifacts or {}
    for key, label in (
        ("srt", "captions"),
        ("chapters_json", "chapters"),
        ("thumbnail_1", "thumbnail"),
    ):
        found = bool(present.get(key))
        checks.append(
            _check(
                f"artifact-{key}",
                found,
                f"{label} {'present' if found else 'missing'}",
            )
        )
    status: Literal["pass", "fail"] = (
        "pass" if all(check.passed for check in checks) else "fail"
    )
    return QcReport(status=status, checks=checks)
