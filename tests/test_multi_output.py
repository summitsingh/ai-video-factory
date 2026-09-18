"""Tests for multi_output: one script -> longform + short90 + vertical shorts."""

from __future__ import annotations

import pytest

from ai_video_factory.edit_schema import EditDocument, EditScene
from ai_video_factory.multi_output import (
    beat_key_of,
    derive_short_edit,
    pick_shorts_segments,
    pick_short90_scenes,
    plan_outputs,
)

FPS = 30


def _scene(scene_id: str, seconds: float, title: str, narration_words: int = 30) -> EditScene:
    narration = " ".join(["word"] * narration_words)
    return EditScene(
        id=scene_id,
        from_frame=0,
        duration_frames=int(seconds * FPS),
        title=title,
        caption=narration,
        narration=narration,
        visual="a galaxy",
    )


def _doc() -> EditDocument:
    scenes = [
        _scene("cold_open-0", 8, "Hook", 40),
        _scene("act1_setup-0", 20, "Setup", 60),
        _scene("act2a_evidence-0", 25, "Evidence A", 80),
        _scene("act2a_evidence-1", 25, "Evidence B", 40),
        _scene("act2b_twist-0", 25, "Twist", 90),
        _scene("act2c_deepening-0", 25, "Deepening", 70),
        _scene("act3_climax-0", 20, "Climax", 85),
        _scene("outro-0", 12, "Outro", 30),
    ]
    cursor = 0
    for scene in scenes:
        scene.from_frame = cursor
        cursor += scene.duration_frames
    return EditDocument(
        schema_version=1,
        width=1280,
        height=720,
        fps=FPS,
        duration_frames=cursor,
        scenes=scenes,
        title="Test Doc",
        description="desc",
        sources=["https://example.com"],
    )


class _Script:
    title = "Test Script"
    description = "A test"
    sources = ["https://example.com"]


def test_plan_outputs_has_all_three_kinds():
    plan = plan_outputs(_Script(), _doc())
    assert set(plan["outputs"]) == {"longform", "short90", "shorts"}
    assert plan["outputs"]["longform"]["orientation"] == "landscape"
    assert plan["outputs"]["short90"]["target_seconds"] == 90.0
    assert plan["title"] == "Test Script"


def test_short90_keeps_hook_and_payoff_in_order():
    doc = _doc()
    picked = pick_short90_scenes(doc)
    ids = [s.id for s in picked]
    assert ids[0] == "cold_open-0"
    assert "act3_climax-0" in ids
    positions = [doc.scenes.index(s) for s in picked]
    assert positions == sorted(positions)


def test_short90_near_target_duration():
    picked = pick_short90_scenes(_doc())
    total = sum(s.duration_frames for s in picked) / FPS
    assert total <= 95.0
    assert total >= 30.0


def test_short90_falls_back_without_beat_ids():
    doc = _doc()
    for scene in doc.scenes:
        scene.id = f"scene-{scene.id}"
    picked = pick_short90_scenes(doc)
    assert picked, "fallback must still pick scenes"


def test_shorts_segments_are_20_to_40s_with_hooks():
    segments = pick_shorts_segments(_doc())
    assert segments, "expected at least one short"
    for seg in segments:
        assert 20.0 <= seg["duration_seconds"] <= 40.0
        assert seg["hook"], "every short needs a hook"
        assert seg["orientation"] == "vertical"
        assert seg["aspect"] == "9:16"


def test_derive_short_edit_retimes_contiguously():
    doc = _doc()
    keep = ["cold_open-0", "act2b_twist-0", "act3_climax-0"]
    short = derive_short_edit(doc, keep)
    assert [s.id for s in short.scenes] == keep
    assert short.scenes[0].from_frame == 0
    for prev, cur in zip(short.scenes, short.scenes[1:]):
        assert cur.from_frame == prev.from_frame + prev.duration_frames
    assert short.duration_frames == sum(s.duration_frames for s in short.scenes)


def test_derive_short_edit_bookends_intro_outro():
    doc = _doc()
    short = derive_short_edit(doc, ["act1_setup-0", "act2a_evidence-0"])
    assert short.scenes[0].kind == "intro"
    assert short.scenes[-1].kind == "outro"


def test_derive_short_edit_preserves_metadata():
    doc = _doc()
    short = derive_short_edit(doc, ["cold_open-0"])
    assert short.width == 1280 and short.height == 720 and short.fps == FPS
    assert short.title == "Test Doc"
    assert short.sources == ["https://example.com"]
    assert short.scenes[0].visual == "a galaxy"


def test_derive_short_edit_rejects_empty_and_unknown():
    doc = _doc()
    with pytest.raises(ValueError):
        derive_short_edit(doc, [])
    with pytest.raises(ValueError):
        derive_short_edit(doc, ["nope-0"])


def test_beat_key_of_handles_plain_ids():
    scene = _scene("plain", 5, "T")
    assert beat_key_of(scene) == ""
