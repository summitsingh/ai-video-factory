"""Tests for content-level QC (long-form readiness)."""

import pytest

from ai_video_factory.edit_schema import EditDocument, EditScene
from ai_video_factory.qc import evaluate_content_qc


def _scene(i: int, *, narration: bool = True, act: str | None = None) -> EditScene:
    return EditScene(
        id=f"s{i}",
        from_frame=i * 1800,
        duration_frames=1800,
        title=f"Scene {i}",
        caption="caption",
        narration="some narration words here" if narration else None,
        act=act,
    )


def _doc(minutes: float, scenes: int, **kwargs) -> EditDocument:
    fps = 30
    total_frames = round(minutes * 60 * fps)
    per = total_frames // scenes
    built = []
    cursor = 0
    for i in range(scenes):
        span = per + (1 if i < total_frames % scenes else 0)
        scene = _scene(i, **kwargs)
        built.append(
            EditScene(
                id=scene.id, from_frame=cursor, duration_frames=span,
                title=scene.title, caption=scene.caption,
                narration=scene.narration, act=scene.act,
            )
        )
        cursor += span
    return EditDocument(
        schema_version=1, width=1280, height=720, fps=fps,
        duration_frames=cursor, scenes=built,
    )


def _artifacts() -> dict:
    return {"srt": "a.srt", "chapters_json": "c.json", "thumbnail_1": "t.png"}


def test_content_qc_passes_for_good_longform():
    scenes = 12
    doc = _doc(25.0, scenes, act=None)
    # Give 6 distinct act cards across the scenes.
    acts = ["COLD OPEN", "ACT I", "ACT II", "ACT II", "ACT II", "ACT III",
            "OUTRO", None, None, None, None, None]
    for scene, act in zip(doc.scenes, acts):
        scene.act = act
    report = evaluate_content_qc(
        doc, target_minutes=25.0, artifacts=_artifacts(), longform=True
    )
    assert report.status == "pass"
    assert {c.name for c in report.checks} >= {
        "target-duration", "scene-count", "narration-coverage",
        "act-structure", "artifact-srt", "artifact-chapters_json",
        "artifact-thumbnail_1",
    }


def test_content_qc_fails_short_runtime():
    doc = _doc(18.0, 12)
    report = evaluate_content_qc(
        doc, target_minutes=25.0, artifacts=_artifacts(), longform=True
    )
    assert report.status == "fail"
    failed = {c.name for c in report.checks if not c.passed}
    assert "target-duration" in failed


def test_content_qc_fails_missing_artifacts():
    doc = _doc(25.0, 12)
    report = evaluate_content_qc(doc, target_minutes=25.0, artifacts={}, longform=True)
    failed = {c.name for c in report.checks if not c.passed}
    assert failed >= {"artifact-srt", "artifact-chapters_json", "artifact-thumbnail_1"}


def test_content_qc_fails_thin_narration():
    doc = _doc(25.0, 12, narration=False)
    report = evaluate_content_qc(
        doc, target_minutes=25.0, artifacts=_artifacts(), longform=True
    )
    failed = {c.name for c in report.checks if not c.passed}
    assert "narration-coverage" in failed


def test_content_qc_shortform_defaults():
    doc = _doc(1.5, 4)
    report = evaluate_content_qc(doc, artifacts=_artifacts())
    assert report.status == "pass"
    assert not any(c.name == "act-structure" for c in report.checks)
