"""One script, many outputs.

Derives every distributable cut from a single long-form edit document:

- ``longform``: the full documentary, unchanged.
- ``short90``: a 90-second highest-retention subset (hook + key beats + payoff).
- ``shorts``: 20-40s vertical segments, each opening on a hook.

Pure data transformation. No rendering, no network, no LLM calls.
"""

from __future__ import annotations

from typing import Any

from ai_video_factory.edit_schema import EditDocument, EditScene

# Beat keys come from longform.LONGFORM_BEATS; scene ids are
# "{beat_key}-{scene_index}" per longform_to_edit_document.
_HOOK_BEATS = ("cold_open",)
_KEY_BEATS = ("act2a_evidence", "act2b_twist", "act2c_deepening")
_PAYOFF_BEATS = ("act3_climax",)
_OUTRO_BEATS = ("outro",)

_SHORT90_TARGET_SECONDS = 90.0
_SHORT_MIN_SECONDS = 20.0
_SHORT_MAX_SECONDS = 40.0


def beat_key_of(scene: EditScene) -> str:
    """Recover the beat key from a scene id like ``act2b_twist-3``."""
    if "-" in scene.id:
        return scene.id.rsplit("-", 1)[0]
    return ""


def scene_seconds(scene: EditScene, fps: int) -> float:
    return scene.duration_frames / fps if fps > 0 else 0.0


def _group_by_beat(doc: EditDocument) -> dict[str, list[EditScene]]:
    groups: dict[str, list[EditScene]] = {}
    for scene in doc.scenes:
        groups.setdefault(beat_key_of(scene), []).append(scene)
    return groups


def _densest_scene(scenes: list[EditScene]) -> EditScene:
    """The scene carrying the most narration (proxy for the beat's payload)."""
    return max(scenes, key=lambda s: len(s.narration or s.caption or ""))


def _total_seconds(scenes: list[EditScene], fps: int) -> float:
    return sum(scene_seconds(s, fps) for s in scenes)


def pick_short90_scenes(doc: EditDocument) -> list[EditScene]:
    """Hook + three key beats + payoff, in story order, capped near 90s.

    Always keeps the cold open (the hook) and at least one payoff scene.
    Trims from the middle (never the hook, never the payoff) when over
    the target.
    """
    groups = _group_by_beat(doc)
    picked: list[EditScene] = []

    def take(beat_keys: tuple[str, ...]) -> None:
        for key in beat_keys:
            scenes = groups.get(key)
            if scenes:
                candidate = _densest_scene(scenes)
                if candidate not in picked:
                    picked.append(candidate)

    take(_HOOK_BEATS)
    take(_KEY_BEATS)
    take(_PAYOFF_BEATS)
    if not picked:
        # Edit doc without beat-prefixed ids: fall back to even thirds.
        step = max(1, len(doc.scenes) // 3)
        picked = [doc.scenes[i] for i in range(0, len(doc.scenes), step)]

    # Restore story order.
    order = {id(scene): i for i, scene in enumerate(doc.scenes)}
    picked.sort(key=lambda s: order[id(s)])

    # Trim from the middle while over target; keep hook first, payoff last.
    while len(picked) > 2 and _total_seconds(picked, doc.fps) > _SHORT90_TARGET_SECONDS + 5:
        del picked[len(picked) // 2]
    return picked


def pick_shorts_segments(doc: EditDocument) -> list[dict[str, Any]]:
    """Split hook-heavy beats into standalone 20-40s vertical segments.

    Each segment opens on a scene from a high-retention beat and runs until
    it would exceed the cap. Segments carry their own hook line.
    """
    groups = _group_by_beat(doc)
    segments: list[dict[str, Any]] = []
    for key in (*_HOOK_BEATS, "act2b_twist", *_PAYOFF_BEATS):
        scenes = groups.get(key, [])
        current: list[EditScene] = []
        for scene in scenes:
            trial = current + [scene]
            if _total_seconds(trial, doc.fps) > _SHORT_MAX_SECONDS and current:
                segments.append(_make_segment(current, doc.fps, len(segments)))
                current = [scene]
            else:
                current = trial
        if current and _total_seconds(current, doc.fps) >= _SHORT_MIN_SECONDS:
            segments.append(_make_segment(current, doc.fps, len(segments)))
        elif current and segments:
            # Fold a runt tail into the previous segment when it still fits.
            last = segments[-1]
            merged = last["scenes"] + [s.id for s in current]
            if _total_seconds(
                [s for s in doc.scenes if s.id in merged], doc.fps
            ) <= _SHORT_MAX_SECONDS:
                last["scenes"] = merged
                last["duration_seconds"] = _total_seconds(
                    [s for s in doc.scenes if s.id in merged], doc.fps
                )
    return segments


def _make_segment(
    scenes: list[EditScene], fps: int, index: int
) -> dict[str, Any]:
    first = scenes[0]
    hook = first.title or (first.narration or first.caption or "")[:80]
    return {
        "index": index,
        "hook": hook,
        "scenes": [s.id for s in scenes],
        "duration_seconds": round(_total_seconds(scenes, fps), 1),
        "orientation": "vertical",
        "aspect": "9:16",
    }


def plan_outputs(script: Any, edit_doc: EditDocument) -> dict[str, Any]:
    """Map one script + edit doc to every output spec we will render.

    ``script`` is the LongformScript (only title/description/sources are
    read, so a duck-typed object works). Returns specs for the longform,
    the 90-second cut, and the vertical shorts.
    """
    short90_scenes = pick_short90_scenes(edit_doc)
    shorts = pick_shorts_segments(edit_doc)
    total = edit_doc.duration_frames / edit_doc.fps if edit_doc.fps else 0.0
    title = getattr(script, "title", None) or edit_doc.title or "Untitled"
    return {
        "title": title,
        "source": {
            "script_title": title,
            "scenes": len(edit_doc.scenes),
            "sources": list(getattr(script, "sources", []) or edit_doc.sources),
        },
        "outputs": {
            "longform": {
                "kind": "longform",
                "orientation": "landscape",
                "aspect": "16:9",
                "target_seconds": round(total, 1),
                "scene_ids": [s.id for s in edit_doc.scenes],
                "edit": "full",
            },
            "short90": {
                "kind": "short",
                "orientation": "landscape",
                "aspect": "16:9",
                "target_seconds": _SHORT90_TARGET_SECONDS,
                "scene_ids": [s.id for s in short90_scenes],
                "estimated_seconds": round(
                    _total_seconds(short90_scenes, edit_doc.fps), 1
                ),
            },
            "shorts": shorts,
        },
    }


def derive_short_edit(
    edit_doc: EditDocument, keep_scenes: list[str]
) -> EditDocument:
    """Build a valid subset edit document from kept scene ids.

    Scenes keep story order, are re-timed contiguously from frame 0, and the
    subset is bookended as intro/outro so the renderer treats it as a
    standalone video. Raises ValueError when nothing is kept or an id is
    unknown.
    """
    if not keep_scenes:
        raise ValueError("keep_scenes must name at least one scene")
    by_id = {s.id: s for s in edit_doc.scenes}
    unknown = [sid for sid in keep_scenes if sid not in by_id]
    if unknown:
        raise ValueError(f"unknown scene ids: {unknown}")

    kept = [by_id[sid] for sid in [s.id for s in edit_doc.scenes] if sid in set(keep_scenes)]

    new_scenes: list[EditScene] = []
    cursor = 0
    for position, scene in enumerate(kept):
        kind = "intro" if position == 0 else "outro" if position == len(kept) - 1 else "normal"
        new_scenes.append(
            EditScene(
                id=scene.id,
                from_frame=cursor,
                duration_frames=scene.duration_frames,
                title=scene.title,
                caption=scene.caption,
                kind=kind,  # type: ignore[arg-type]
                visual=scene.visual,
                narration=scene.narration,
                background=scene.background,
                text_color=scene.text_color,
                accent_color=scene.accent_color,
                image=scene.image,
                clip=scene.clip,
                subtitle=scene.subtitle,
                pip=scene.pip,
                act=scene.act,
                lower_third=scene.lower_third,
            )
        )
        cursor += scene.duration_frames

    return EditDocument(
        schema_version=1,
        width=edit_doc.width,
        height=edit_doc.height,
        fps=edit_doc.fps,
        duration_frames=cursor,
        scenes=new_scenes,
        title=edit_doc.title,
        description=edit_doc.description,
        created_at=edit_doc.created_at,
        sources=list(edit_doc.sources),
    )
