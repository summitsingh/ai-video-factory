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
    # Styled captions (#9): when True, the Remotion render suppresses its
    # plain burned-in subtitles because karaoke word-highlight captions are
    # burned onto the final master instead.
    karaoke_captions: bool = False

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


def scene_asset_slots(scenes: list[EditScene]) -> dict[str, int]:
    """Map scene id -> per-scene asset slot index.

    Slots are positional among ``kind == "normal"`` scenes: the first normal
    scene is slot 0 (assets dir ``scene-00/``), the second is slot 1, and so
    on. This works for both the legacy ``scene-N`` ids and longform
    beat-prefixed ids (``discovery-0``, ``mechanism-3``, ...), which carry no
    numeric scene position. Intro/outro scenes get no slot and keep their
    title-card look.

    Every stage of the visual pipeline (NASA populate, stock-footage fetch,
    attach, AI-visual fallback, Remotion public staging) must use this one
    mapping so a scene's assets always land in the directory the renderer
    looks in.
    """
    slots: dict[str, int] = {}
    for scene in scenes:
        if getattr(scene, "kind", "normal") == "normal":
            slots[str(scene.id)] = len(slots)
    return slots


def load_edit(path: Path) -> EditDocument:
    return EditDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_edit(document: EditDocument, path: Path) -> None:
    """Save an edit document to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.model_dump_json(indent=2))


# ========== Seconds-keyed edit-document builder ==========

# Restored 2026-09-21: the Sep 18 audit fix (45204f7) removed these helpers
# from edit_schema.py while production.py still imported them, which broke
# test collection for test_production, test_daily_job, test_source_urls,
# and test_remotion_pixel_integration. The removed Phase-3 long-form *model*
# fields stay removed; the builder below only forwards fields the current
# EditScene schema supports.


def seconds_to_frames(seconds: float, fps: int) -> int:
    """Convert a duration in seconds to whole frames at the given fps."""
    return max(1, int(round(seconds * fps)))


def frames_to_seconds(frames: int, fps: int) -> float:
    """Convert whole frames back to seconds at the given fps."""
    return frames / fps


def build_edit_document_from_scenes(
    scenes_seconds: list[dict[str, Any]],
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    title: str | None = None,
    description: str | None = None,
    sources: list[str] | None = None,
) -> EditDocument:
    """Build an EditDocument from a list of per-scene dicts keyed by seconds.

    Each dict may carry ``id``, ``duration_seconds``, ``title``, ``caption``,
    ``kind``, ``narration``, ``visual``, ``subtitle``, and ``pip``. Scenes are
    laid out back-to-back on an exact integer-frame cursor;
    ``duration_frames`` is derived from seconds so the authoring path stays
    free of frame arithmetic. Keys for schema fields removed in the Sep 18
    audit (e.g. ``claim_ids``, ``asset_queries``, ``voice``) are ignored.
    """
    fps = int(fps)
    scenes: list[EditScene] = []
    cursor_frames = 0
    for index, raw in enumerate(scenes_seconds):
        scene_id = str(raw.get("id") or f"scene-{index:02d}")
        duration_seconds = float(raw["duration_seconds"]) if "duration_seconds" in raw else 1.0
        scene = EditScene(
            id=scene_id,
            from_frame=cursor_frames,
            duration_frames=seconds_to_frames(duration_seconds, fps),
            title=str(raw.get("title") or scene_id),
            caption=str(raw.get("caption") or scene_id),
            kind=str(raw.get("kind", "normal")),
            visual=raw.get("visual"),
            narration=raw.get("narration"),
            subtitle=raw.get("subtitle"),
            pip=bool(raw.get("pip", False)),
        )
        scenes.append(scene)
        cursor_frames += scene.duration_frames

    total_frames = cursor_frames or 1
    return EditDocument(
        schema_version=1,
        width=width,
        height=height,
        fps=fps,
        duration_frames=total_frames,
        scenes=scenes,
        title=title,
        description=description,
        sources=list(sources or []),
    )
