import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_video_factory.edit_schema import EditDocument, load_edit


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


def test_seconds_frames_round_trip() -> None:
    """seconds_to_frames / frames_to_seconds are exact inverses on whole frames."""
    from ai_video_factory.edit_schema import frames_to_seconds, seconds_to_frames

    assert seconds_to_frames(4.0, 30) == 120
    assert seconds_to_frames(0.0, 30) == 1  # clamped to at least one frame
    assert frames_to_seconds(120, 30) == pytest.approx(4.0)


def test_build_edit_document_from_scenes_lays_out_back_to_back() -> None:
    """Restored builder lays scenes back-to-back on an exact frame cursor."""
    from ai_video_factory.edit_schema import build_edit_document_from_scenes

    doc = build_edit_document_from_scenes(
        [
            {"id": "scene-01", "duration_seconds": 2.0, "title": "A", "caption": "a", "narration": "one"},
            {"id": "scene-02", "duration_seconds": 1.5, "title": "B", "caption": "b", "narration": "two"},
        ],
        fps=30,
    )
    assert [s.from_frame for s in doc.scenes] == [0, 60]
    assert [s.duration_frames for s in doc.scenes] == [60, 45]
    assert doc.duration_frames == 105
    assert doc.scenes[0].narration == "one"
    assert doc.scenes[1].kind == "normal"


def test_build_edit_document_from_scenes_ignores_removed_longform_keys() -> None:
    """Keys for schema fields removed in the Sep 18 audit must not break the builder."""
    from ai_video_factory.edit_schema import build_edit_document_from_scenes

    doc = build_edit_document_from_scenes(
        [
            {
                "id": "s1", "duration_seconds": 2.0, "title": "T", "caption": "C",
                "claim_ids": ["c1"], "asset_queries": ["q"], "voice": "af_heart",
            }
        ],
        fps=30,
    )
    assert len(doc.scenes) == 1
    assert doc.scenes[0].duration_frames == 60
