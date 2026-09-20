"""Tests for the duration enforcement loop (item 3)."""

import json
import re

import pytest

from ai_video_factory.duration_enforcer import (
    _BEAT_CAP_MULTIPLE,
    enforce_duration,
)
from ai_video_factory.longform import (
    WORDS_PER_MINUTE,
    BeatSpec,
    LongformBeat,
    LongformError,
    LongformScene,
    LongformScript,
    beat_word_target,
)


def make_stub_script(target_minutes: float, fill: float = 0.6) -> LongformScript:
    """A script whose beats each hold ``fill`` of their word allocation."""
    from ai_video_factory.longform import LONGFORM_BEATS

    beats = []
    for spec in LONGFORM_BEATS:
        target = beat_word_target(spec, target_minutes)
        words = max(45, int(target * fill))
        scenes = [
            LongformScene(
                title=f"{spec.key} scene",
                narration=" ".join(["word"] * words),
                visual_direction="slow push-in",
            )
        ]
        beats.append(LongformBeat(spec=spec, scenes=scenes))
    return LongformScript(
        title="T",
        description="D",
        topic="The Fermi Paradox",
        target_minutes=target_minutes,
        beats=beats,
        sources=["https://www.nasa.gov"],
    )


def make_extension_chat():
    """Fake chat honoring the AT LEAST word counts in extension prompts."""

    def chat(messages, max_tokens):
        prompt = messages[-1]["content"]
        match = re.search(r"AT LEAST (\d+) (?:additional )?words", prompt)
        assert match, f"extension prompt missing word count:\n{prompt[:200]}"
        deficit = int(match.group(1))
        scene_match = re.search(r"EXACTLY (\d+) (?:NEW )?scene", prompt)
        scene_count = int(scene_match.group(1)) if scene_match else 1
        per_scene = max(25, -(-deficit // scene_count))  # ceil div
        scenes = [
            {
                "title": f"Extended scene {index + 1}",
                "narration": " ".join(["newword"] * per_scene),
                "visual_direction": "archival footage",
                "lower_third": None,
            }
            for index in range(scene_count)
        ]
        return json.dumps({"scenes": scenes})

    return chat


def test_enforce_duration_closes_gap_within_tolerance():
    script = make_stub_script(20.0, fill=0.6)
    before = script.total_words
    target = int(20.0 * WORDS_PER_MINUTE)
    assert before < target * 0.95  # genuinely short

    report = enforce_duration(script, 20.0, make_extension_chat())

    assert report.within_tolerance
    assert report.final_words >= target * 0.95
    assert len(report.iterations) >= 1
    assert len(report.iterations) <= 3
    # Every iteration logged which beats were extended + word counts.
    for iteration in report.iterations:
        assert iteration.total_words_before < iteration.total_words_after
        for ext in iteration.extensions:
            assert ext.words_added > 0
            assert ext.words_after == ext.words_before + ext.words_added
    # The most underweight beats (act2 fractions) were extended first.
    first_extended = {ext.beat_key for ext in report.iterations[0].extensions}
    assert first_extended <= {"act2a_evidence", "act2b_twist", "act2c_deepening"}


def test_enforce_duration_noop_when_on_target():
    script = make_stub_script(20.0, fill=1.0)
    calls = []

    def chat(messages, max_tokens):
        calls.append(messages)
        raise AssertionError("chat must not be called")

    report = enforce_duration(script, 20.0, chat)
    assert report.within_tolerance
    assert report.iterations == []
    assert calls == []


def test_enforce_duration_fails_loud_when_extension_impossible():
    script = make_stub_script(20.0, fill=0.6)

    def dead_chat(messages, max_tokens):
        return ""  # model returns nothing, twice, then gives up

    with pytest.raises(LongformError, match="duration enforcement failed"):
        enforce_duration(script, 20.0, dead_chat)


def test_enforce_duration_appends_encore_when_all_beats_at_cap():
    # One beat at 1.6x its allocation (at cap) covering only half the
    # runtime: no beat is extendable, so an encore beat must be appended.
    spec = BeatSpec(
        key="half", label="HALF", fraction=0.5, purpose="p", retention="r"
    )
    target_minutes = 20.0
    word_target = beat_word_target(spec, target_minutes)
    words = int(word_target * _BEAT_CAP_MULTIPLE)
    beat = LongformBeat(
        spec=spec,
        scenes=[
            LongformScene(title="s", narration=" ".join(["word"] * words),
                          visual_direction="x")
        ],
    )
    script = LongformScript(
        title="T", description="D", topic="T",
        target_minutes=target_minutes, beats=[beat],
    )
    target = int(target_minutes * WORDS_PER_MINUTE)
    assert script.total_words < target * 0.95

    report = enforce_duration(script, target_minutes, make_extension_chat())

    assert report.within_tolerance
    kinds = [ext.kind for it in report.iterations for ext in it.extensions]
    assert "encore" in kinds
    assert any(b.spec.key == "encore" for b in script.beats)


def test_enforce_duration_report_serializes():
    script = make_stub_script(20.0, fill=0.8)
    report = enforce_duration(script, 20.0, make_extension_chat())
    payload = report.to_dict()
    assert payload["target_minutes"] == 20.0
    assert payload["target_words"] == int(20.0 * WORDS_PER_MINUTE)
    assert payload["within_tolerance"] is True
    assert payload["final_words"] == script.total_words
    assert isinstance(payload["iterations"], list)
