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
    voice: str | None = Field(
        default=None,
        description="Kokoro TTS voice id for this scene's narration (e.g. af_heart, am_michael)",
    )
    transition: str | None = Field(
        default=None,
        description="Transition style into this scene: 'dissolve' | 'cut' | 'fade'",
    )

    # Long-form fields (Phase 3). Optional with defaults so existing short-form
    # documents continue to validate. Durations are in seconds; the pipeline
    # converts to frames at render time using fps.
    start_seconds: float = Field(
        default=0.0, ge=0.0, description="Scene start time within the master (seconds)"
    )
    duration_seconds: float | None = Field(
        default=None, gt=0.0, description="Scene duration in seconds; derived from narration when omitted"
    )
    claim_ids: list[str] = Field(
        default_factory=list, description="Verified claim IDs this scene's narration references"
    )
    editorial_purpose: str | None = Field(
        default=None,
        description="hook | evidence | explanation | transition | payoff",
    )
    visual_brief: str | None = Field(
        default=None, description="Human-readable plan for the scene's visuals"
    )
    asset_queries: list[str] = Field(
        default_factory=list, description="Asset search queries for this scene"
    )
    asset_strategy: str | None = Field(
        default=None,
        description="licensed_clip | licensed_image | public_domain | map | chart | generated_visual | source_excerpt",
    )
    on_screen_text: str | None = Field(
        default=None, description="Transient on-screen text for this scene (null to omit)"
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


# ========== Long-form helpers (Phase 3) ==========

def seconds_to_frames(seconds: float, fps: int) -> int:
    """Convert a duration in seconds to whole frames at the given fps."""
    return max(1, int(round(seconds * fps)))


def frames_to_seconds(frames: int, fps: int) -> float:
    """Convert whole frames back to seconds at the given fps."""
    return frames / fps


def scene_duration_frames(scene: EditScene, fps: int) -> int:
    """Return a scene's duration in frames.

    Prefers an explicit ``duration_seconds`` when present; otherwise falls back to
    the legacy ``duration_frames`` field so short-form documents still work.
    """
    if scene.duration_seconds is not None:
        return seconds_to_frames(scene.duration_seconds, fps)
    return max(1, int(scene.duration_frames))


def start_frame(scene_id: str, start_seconds: float, fps: int) -> int:
    """Return a scene's start frame from ``start_seconds`` (or legacy from_frame)."""
    if start_seconds and start_seconds > 0:
        return seconds_to_frames(start_seconds, fps)
    return max(0, int(scene_id.split("-")[-1]) - 1 if scene_id.startswith("scene-") else 0)


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

    Each dict may contain any subset of the long-form fields (id, start_seconds,
    duration_seconds, narration, claim_ids, editorial_purpose, visual_brief,
    asset_queries, asset_strategy, on_screen_text, transition, voice). Scenes are
    laid out back-to-back; ``duration_frames`` is derived from total seconds. This
    keeps the long-form authoring path free of frame arithmetic.
    """
    fps = int(fps)
    scenes: list[EditScene] = []
    cursor_seconds = 0.0
    for index, raw in enumerate(scenes_seconds):
        scene_id = str(raw.get("id") or f"scene-{index:02d}")
        duration_seconds = float(raw["duration_seconds"]) if "duration_seconds" in raw else None
        start_seconds = float(raw.get("start_seconds", cursor_seconds))
        # If start_seconds is explicit, continue the timeline from it; otherwise
        # keep laying scenes back-to-back.
        if "start_seconds" in raw:
            cursor_seconds = start_seconds + (duration_seconds or 0.0)
        kind_value = str(raw.get("kind", "normal"))
        scene = EditScene(
            id=scene_id,
            from_frame=start_frame(scene_id, start_seconds, fps),
            duration_frames=seconds_to_frames(duration_seconds if duration_seconds is not None else 1.0, fps),
            title=str(raw.get("title") or scene_id),
            caption=str(raw.get("caption") or scene_id),
            kind=kind_value,  # type: ignore[arg-type] - validated downstream by EditDocument
            visual=raw.get("visual"),
            narration=raw.get("narration"),
            subtitle=raw.get("subtitle"),
            pip=bool(raw.get("pip", False)),
            voice=raw.get("voice"),
            transition=raw.get("transition"),
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            claim_ids=list(raw.get("claim_ids", [])),
            editorial_purpose=raw.get("editorial_purpose"),
            visual_brief=raw.get("visual_brief"),
            asset_queries=list(raw.get("asset_queries", [])),
            asset_strategy=raw.get("asset_strategy"),
            on_screen_text=raw.get("on_screen_text"),
        )
        scenes.append(scene)

    total_frames = sum(s.duration_frames for s in scenes) or 1
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