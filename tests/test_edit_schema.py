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
