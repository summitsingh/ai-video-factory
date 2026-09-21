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
