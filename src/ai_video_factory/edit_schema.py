from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EditScene(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str = Field(min_length=1)
    from_frame: int = Field(ge=0)
    duration_frames: int = Field(gt=0)
    title: str = Field(min_length=1)
    caption: str = Field(min_length=1)


class EditDocument(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)
    duration_frames: int = Field(gt=0)
    scenes: list[EditScene]

    @model_validator(mode="after")
    def validate_scenes(self) -> Self:
        scene_ids = [scene.id for scene in self.scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("scene IDs must be unique")
        if any(
            scene.from_frame + scene.duration_frames > self.duration_frames
            for scene in self.scenes
        ):
            raise ValueError("scene exceeds composition duration")
        return self


def load_edit(path: Path) -> EditDocument:
    return EditDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))
