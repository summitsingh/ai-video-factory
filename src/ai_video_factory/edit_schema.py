"""Extended edit schema for AI Video Factory video production."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EditScene(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str = Field(min_length=1)
    from_frame: int = Field(ge=0)
    duration_frames: int = Field(gt=0)
    title: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    kind: Literal["normal", "intro", "outro"] = Field(default="normal")
    
    # Extended fields for rich content
    visual: str | None = None
    narration: str | None = None
    background: str = Field(default="gradient", description="Background type or color")
    text_color: str = Field(default="#f8fafc", description="Text color in hex")
    accent_color: str = Field(default="#22d3ee", description="Accent color in hex")
    image: str | None = Field(default=None, description="Local still image path (Ken Burns)")
    clip: str | None = Field(default=None, description="Local video clip path (muted bed)")
    subtitle: str | None = Field(
        default=None,
        description="Burned-in caption text; falls back to caption when omitted",
    )
    pip: bool = Field(
        default=False,
        description="Show the scene's secondary asset as a picture-in-picture inset",
    )
    act: str | None = Field(
        default=None,
        description="Beat label rendered as an act title card (first scene of a beat)",
    )
    lower_third: str | None = Field(
        default=None,
        description="Name/date/location rendered as a lower-third overlay",
    )


class EditDocument(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)
    duration_frames: int = Field(gt=0)
    scenes: list[EditScene]
    
    # Extended metadata
    title: str | None = None
    description: str | None = None
    created_at: str | None = None
    sources: list[str] = Field(default_factory=list)
    # Chunk rendering metadata: when an edit is split into chunks, these
    # preserve the full-video scene numbering for on-screen chapter cards.
    total_scenes: int | None = None
    scene_start_index: int | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def reject_boolean_schema_version(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("schema_version must be the number 1, not a boolean")
        return value

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


def save_edit(document: EditDocument, path: Path) -> None:
    """Save an edit document to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.model_dump_json(indent=2))