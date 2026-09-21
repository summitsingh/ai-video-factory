"""Regression: karaoke ASS captions must not appear during act-card windows.

Mirrors the Remotion renderer's ACT_CARD_TOTAL_FRAMES delay
(SyntheticVideo.tsx): on scenes carrying an act label, captions start only
after the card fades, so they never collide with the card.
"""
from __future__ import annotations

import re

from ai_video_factory.edit_schema import EditDocument, EditScene
from ai_video_factory.subtitle_export import (
    ACT_CARD_TOTAL_FRAMES,
    _cues_after_act_card,
    build_ass_karaoke,
)

FPS = 30
CARD_SECONDS = ACT_CARD_TOTAL_FRAMES / FPS  # 2.8


def _scene(
    scene_id: str,
    from_frame: int,
    duration_frames: int,
    kind: str = "normal",
    act: str | None = None,
) -> EditScene:
    return EditScene(
        id=scene_id,
        from_frame=from_frame,
        duration_frames=duration_frames,
        title="T",
        caption="C",
        kind=kind,
        act=act,
        narration=(
            "The first sentence of narration goes here. "
            "A second sentence continues the story. "
            "A third sentence wraps up the scene."
        ),
    )


def _doc() -> EditDocument:
    return EditDocument(
        schema_version=1,
        width=1280,
        height=720,
        fps=FPS,
        duration_frames=900,
        scenes=[
            _scene("cold_open-0", 0, 150),
            _scene("discovery-0", 150, 300, act="Act II"),
            _scene("mechanism-0", 450, 300),
        ],
    )


_DIALOGUE_RE = re.compile(
    r"Dialogue: 0,(\d+:\d\d:\d\d\.\d\d),(\d+:\d\d:\d\d\.\d\d),"
)


def _parse_ass_time(value: str) -> float:
    hours, minutes, rest = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def _dialogue_timings(ass: str) -> list[tuple[float, float]]:
    return [
        (_parse_ass_time(m.group(1)), _parse_ass_time(m.group(2)))
        for m in _DIALOGUE_RE.finditer(ass)
    ]


def test_act_card_total_frames_matches_tsx() -> None:
    # 12 (in) + 48 (hold) + 24 (out) in SyntheticVideo.tsx.
    assert ACT_CARD_TOTAL_FRAMES == 84


def test_cues_after_act_card_unit() -> None:
    assert _cues_after_act_card([(0.0, 1.0, "a"), (2.0, 2.8, "b")], 2.8) == []
    assert _cues_after_act_card([(1.0, 4.0, "c")], 2.8) == [(2.8, 4.0, "c")]
    assert _cues_after_act_card([(3.0, 5.0, "d")], 2.8) == [(3.0, 5.0, "d")]
    assert _cues_after_act_card([], 2.8) == []


def test_ass_karaoke_delays_captions_on_act_scenes() -> None:
    ass = build_ass_karaoke(_doc())
    timings = _dialogue_timings(ass)
    assert timings, "expected dialogue events"
    act_scene_start = 150 / FPS  # 5.0
    act_scene_end = act_scene_start + 300 / FPS
    boundary = act_scene_start + CARD_SECONDS  # 7.8
    act_cues = [t for t in timings if act_scene_start <= t[0] < act_scene_end]
    assert act_cues, "expected captions on the act scene"
    for start, end in act_cues:
        assert start >= boundary - 1e-6, (
            f"caption starts at {start}s, before the act card fades at {boundary}s"
        )
        assert start < end


def test_ass_karaoke_unchanged_on_scenes_without_act() -> None:
    ass = build_ass_karaoke(_doc())
    timings = _dialogue_timings(ass)
    first = min(timings, key=lambda t: t[0])
    assert first[0] == 0.0
    # The non-act scenes keep every cue: narration has 3 sentences per scene.
    non_act = [t for t in timings if t[0] < 150 / FPS or t[0] >= 450 / FPS]
    assert len(non_act) == 6
