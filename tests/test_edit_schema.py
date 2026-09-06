import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_video_factory.edit_schema import (
    EditDocument,
    EditScene,
    build_edit_document_from_scenes,
    frames_to_seconds,
    load_edit,
    scene_duration_frames,
    seconds_to_frames,
)


INVALID_CASES = json.loads(
    Path("tests/fixtures/edit-schema-invalid.json").read_text(encoding="utf-8")
)


def test_fixture_is_valid() -> None:
    edit = load_edit(Path("fixtures/synthetic-edit.json"))
    assert edit.schema_version == 1
    assert edit.width == 1280 and edit.height == 720
    assert edit.fps == 30 and edit.duration_frames == 90
    assert len(edit.scenes) == 1


def test_scene_cannot_exceed_composition() -> None:
    with pytest.raises(ValidationError):
        EditDocument.model_validate({
            "schema_version": 1, "width": 1280, "height": 720,
            "fps": 30, "duration_frames": 30,
            "scenes": [{"id": "title", "from_frame": 0, "duration_frames": 31,
                        "title": "Synthetic test", "caption": "Local render"}],
        })


@pytest.mark.parametrize("case", INVALID_CASES, ids=lambda case: case["name"])
def test_python_rejects_shared_strict_schema_corpus(case: dict[str, object]) -> None:
    """Python must reject the same coercions and unknown fields as TypeScript."""
    with pytest.raises(ValidationError):
        EditDocument.model_validate(case["document"])


# ========== Long-form helpers (Phase 3) ==========

def test_seconds_to_frames_rounds() -> None:
    assert seconds_to_frames(6.8, 30) == 204
    assert seconds_to_frames(1.0, 30) == 30
    assert seconds_to_frames(0.01, 30) == 1


def test_frames_to_seconds_roundtrip() -> None:
    assert frames_to_seconds(204, 30) == pytest.approx(6.8)


def test_build_edit_document_from_scenes_lays_out_timeline() -> None:
    scenes = [
        {"id": "scene-01", "duration_seconds": 6.8, "narration": "Cold open line.",
         "editorial_purpose": "hook", "claim_ids": ["claim-01"],
         "asset_strategy": "public_domain", "transition": "dissolve"},
        {"id": "scene-02", "duration_seconds": 8.4, "narration": "Evidence line.",
         "editorial_purpose": "evidence", "claim_ids": ["claim-02"],
         "asset_strategy": "map", "transition": "cut"},
    ]
    doc = build_edit_document_from_scenes(scenes)
    assert len(doc.scenes) == 2
    # Durations converted to frames at fps=30.
    assert doc.scenes[0].duration_frames == seconds_to_frames(6.8, 30)
    assert doc.scenes[1].duration_frames == seconds_to_frames(8.4, 30)
    # duration_frames is the sum of scene durations.
    assert doc.duration_frames == doc.scenes[0].duration_frames + doc.scenes[1].duration_frames
    # Long-form fields carried through.
    assert doc.scenes[0].editorial_purpose == "hook"
    assert doc.scenes[0].claim_ids == ["claim-01"]
    assert doc.scenes[1].asset_strategy == "map"
    assert doc.scenes[0].transition == "dissolve"


def test_build_edit_document_backwards_compatible() -> None:
    """Legacy short-form documents still validate unchanged."""
    legacy = {
        "schema_version": 1, "width": 1280, "height": 720, "fps": 30,
        "duration_frames": 90,
        "scenes": [{"id": "scene-01", "from_frame": 0, "duration_frames": 90,
                    "title": "Legacy scene", "caption": "No long-form fields"}],
    }
    doc = EditDocument.model_validate(legacy)
    assert doc.scenes[0].start_seconds == 0.0
    assert doc.scenes[0].claim_ids == []
    assert doc.scenes[0].editorial_purpose is None


def test_scene_duration_frames_prefers_explicit() -> None:
    scene = EditScene(id="scene-01", from_frame=0, duration_frames=30, title="t", caption="t")
    # No duration_seconds -> falls back to legacy duration_frames.
    assert scene_duration_frames(scene, 30) == 30
    scene.duration_seconds = 5.0
    assert scene_duration_frames(scene, 30) == seconds_to_frames(5.0, 30)


def test_build_edit_document_rejects_scene_out_of_bounds() -> None:
    scenes = [{"id": "scene-01", "duration_seconds": 6.8, "narration": "x"}]
    doc = build_edit_document_from_scenes(scenes)
    # duration_frames equals the single scene's frames; must not exceed itself.
    assert doc.duration_frames == doc.scenes[0].duration_frames
