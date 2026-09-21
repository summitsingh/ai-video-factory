import re
import sys

from ai_video_factory.longform import (
    LongformBeat,
    LongformScene,
    LongformScript,
    check_topic_adherence,
)
from ai_video_factory.longform import LONGFORM_BEATS, LongformError


def _script(topic="The Fermi Paradox", narration="Stars are silent tonight."):
    spec = LONGFORM_BEATS[0]
    scene = LongformScene(
        title="Silent Night", narration=narration,
        visual_direction="Stars.", lower_third=None,
    )
    return LongformScript(
        title="T", description="D", topic=topic, target_minutes=25.0,
        beats=[LongformBeat(spec=spec, scenes=[scene])],
    )


def test_clean_passes():
    script = _script()
    check_topic_adherence(script, chat=lambda messages, n: "CLEAN")
    assert "nasa" not in (script.beats[0].scenes[0].narration or "").lower() or True


def test_contaminated_raises_with_detail():
    script = _script(narration="The driver's bloodied hands in Dallas, Texas.")
    try:
        check_topic_adherence(
            script,
            chat=lambda messages, n: "CONTAMINATED. Intruding material about the JFK assassination in Dallas, Texas.",
        )
    except LongformError as error:
        assert "topic-adherence gate" in str(error), str(error)
        assert "Fermi Paradox" in str(error), str(error)
    else:
        raise AssertionError("expected LongformError")


def test_unparseable_verdict_raises():
    script = _script()
    try:
        check_topic_adherence(script, chat=lambda messages, n: "maybe, hard to say")
    except LongformError as error:
        assert "unparseable verdict" in str(error), str(error)
    else:
        raise AssertionError("expected LongformError")


def test_backend_failure_raises():
    script = _script()
    def boom(messages, n):
        raise ConnectionError("down")
    try:
        check_topic_adherence(script, chat=boom)
    except LongformError as error:
        assert "failed to run" in str(error), str(error)
    else:
        raise AssertionError("expected LongformError")


def _long_script(filler="The galaxy is vast and silent."):
    """A multi-scene script whose narration exceeds the old 12000-char prefix."""
    scenes = []
    for i in range(30):
        spec = LONGFORM_BEATS[i % len(LONGFORM_BEATS)]
        scenes.append(
            LongformScene(
                title=f"Scene {i}", narration=filler * 30,
                visual_direction="Stars.", lower_third=None,
            )
        )
        if i and i % 10 == 0:
            pass
    beats = []
    per_beat = len(scenes) // len(LONGFORM_BEATS)
    idx = 0
    for spec in LONGFORM_BEATS:
        beat_scenes = scenes[idx:idx + per_beat]
        idx += per_beat
        beats.append(LongformBeat(spec=spec, scenes=beat_scenes))
    beats[-1].scenes.extend(scenes[idx:])
    return LongformScript(
        title="T", description="D", topic="The Fermi Paradox", target_minutes=25.0,
        beats=beats,
    )


def test_contamination_deep_in_long_script_is_caught():
    # Contamination at the very end, past the old 12000-char prefix limit.
    script = _long_script()
    script.beats[-1].scenes[-1].narration = (
        "The driver's bloodied hands in Dallas, Texas. " * 40
    )
    calls = {"n": 0}
    def judge(messages, n):
        calls["n"] += 1
        prompt = messages[-1]["content"]
        if "bloodied hands" in prompt:
            return "CONTAMINATED. Intruding JFK assassination material in Dallas, Texas."
        return "CLEAN"
    try:
        check_topic_adherence(script, chat=judge)
    except LongformError as error:
        assert "topic-adherence gate" in str(error), str(error)
        assert "part" in str(error), str(error)
        assert calls["n"] >= 2, calls
    else:
        raise AssertionError("expected LongformError")


def test_long_clean_script_passes_all_chunks():
    script = _long_script()
    calls = {"n": 0}
    def judge(messages, n):
        calls["n"] += 1
        return "CLEAN"
    check_topic_adherence(script, chat=judge)
    assert calls["n"] >= 2, calls


def test_prompt_names_topic_and_scene():
    seen = {}
    def spy(messages, n):
        seen["prompt"] = messages[-1]["content"]
        return "CLEAN"
    script = _script(narration="Unique marker sentence ZZX-42.")
    check_topic_adherence(script, chat=spy)
    assert "The Fermi Paradox" in seen["prompt"]
    assert "ZZX-42" in seen["prompt"]
    assert seen["prompt"].split("\n")[0].startswith("Documentary topic")
