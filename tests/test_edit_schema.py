import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_video_factory.edit_schema import EditDocument, load_edit


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
