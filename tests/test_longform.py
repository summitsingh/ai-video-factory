"""Tests for the long-form documentary script engine."""

import json
import re

import pytest

from ai_video_factory.edit_schema import EditDocument
from ai_video_factory.longform import (
    LONGFORM_BEATS,
    WORDS_PER_MINUTE,
    LongformError,
    beat_word_target,
    generate_longform_script,
    longform_to_edit_document,
)


def make_fake_chat():
    """A fake LM Studio chat that honors the word/scene counts in the prompt."""

    def fake_chat(messages, max_tokens):
        user = messages[-1]["content"]
        word_target = int(re.search(r"About (\d+) words", user).group(1))
        scene_count = int(re.search(r"Exactly (\d+) scenes", user).group(1))
        per_scene = max(45, word_target // scene_count)
        scenes = [
            {
                "title": f"Beat scene {index + 1}",
                "narration": " ".join(["word"] * per_scene),
                "visual_direction": "Slow push-in on archival footage",
                "lower_third": "Mars, 2026" if index == 0 else None,
            }
            for index in range(scene_count)
        ]
        return json.dumps({"scenes": scenes})

    return fake_chat


def test_beat_word_targets_cover_full_runtime():
    total = sum(beat_word_target(spec, 25.0) for spec in LONGFORM_BEATS)
    assert total == pytest.approx(25.0 * WORDS_PER_MINUTE, rel=0.02)


def test_beat_fractions_sum_to_one():
    assert sum(spec.fraction for spec in LONGFORM_BEATS) == pytest.approx(1.0)


def test_generate_longform_script_success():
    script = generate_longform_script(
        "Water on Mars",
        "Where Martian water is and why it matters",
        "https://example.com",
        target_minutes=20.0,
        chat_fn=make_fake_chat(),
    )
    assert len(script.beats) == 7
    assert script.total_words == pytest.approx(20.0 * WORDS_PER_MINUTE, rel=0.15)
    assert script.estimated_minutes == pytest.approx(20.0, rel=0.15)
    assert all(beat.scenes for beat in script.beats)


def test_generate_longform_script_rejects_out_of_range():
    with pytest.raises(LongformError, match="between 20 and 30"):
        generate_longform_script("T", "D", "https://example.com", target_minutes=10.0)
    with pytest.raises(LongformError, match="between 20 and 30"):
        generate_longform_script("T", "D", "https://example.com", target_minutes=45.0)


def test_generate_longform_thin_beat_fails():
    def thin_chat(messages, max_tokens):
        return json.dumps(
            {"scenes": [{"title": "T", "narration": "too short", "visual_direction": "x"}]}
        )

    with pytest.raises(LongformError, match="too thin"):
        generate_longform_script(
            "T", "D", "https://example.com", target_minutes=20.0, chat_fn=thin_chat
        )


def test_generate_longform_malformed_json_fails():
    def bad_chat(messages, max_tokens):
        return "this is not json at all"

    with pytest.raises(LongformError, match="no JSON object"):
        generate_longform_script(
            "T", "D", "https://example.com", target_minutes=20.0, chat_fn=bad_chat
        )


def test_generate_longform_chat_error_fails():
    def boom_chat(messages, max_tokens):
        raise ConnectionError("nope")

    with pytest.raises(LongformError, match="generation failed"):
        generate_longform_script(
            "T", "D", "https://example.com", target_minutes=20.0, chat_fn=boom_chat
        )


def test_generate_longform_writes_output(tmp_path):
    output = tmp_path / "script.json"
    script = generate_longform_script(
        "T", "D", "https://example.com",
        target_minutes=20.0, chat_fn=make_fake_chat(), output_path=output,
    )
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["total_words"] == script.total_words
    assert len(payload["beats"]) == 7


def _script_20min():
    return generate_longform_script(
        "Water on Mars", "Where it is", "https://example.com",
        target_minutes=20.0, chat_fn=make_fake_chat(),
    )


def test_longform_to_edit_document_timing():
    script = _script_20min()
    doc = longform_to_edit_document(script, fps=30)
    assert isinstance(doc, EditDocument)
    minutes = doc.duration_frames / doc.fps / 60
    assert minutes == pytest.approx(20.0, rel=0.15)
    ids = [scene.id for scene in doc.scenes]
    assert len(ids) == len(set(ids))


def test_longform_to_edit_document_structure():
    script = _script_20min()
    doc = longform_to_edit_document(script, fps=30)
    assert doc.scenes[0].kind == "intro"
    assert doc.scenes[-1].kind == "outro"
    assert all(scene.kind == "normal" for scene in doc.scenes[1:-1])
    acts = [scene.act for scene in doc.scenes if scene.act]
    assert acts == ["COLD OPEN", "ACT I", "ACT II", "ACT II", "ACT II", "ACT III", "OUTRO"]
    assert any(scene.lower_third == "Mars, 2026" for scene in doc.scenes)
    assert all(scene.narration for scene in doc.scenes)


def test_longform_scene_duration_matches_narration():
    script = _script_20min()
    doc = longform_to_edit_document(script, fps=30)
    first_beat_first_scene = script.beats[0].scenes[0]
    expected_frames = round(first_beat_first_scene.words / WORDS_PER_MINUTE * 60 * 30)
    assert doc.scenes[0].duration_frames == expected_frames
    # Timeline is contiguous with no gaps.
    cursor = 0
    for scene in doc.scenes:
        assert scene.from_frame == cursor
        cursor += scene.duration_frames
    assert cursor == doc.duration_frames


def test_generate_longform_retries_empty_thinking_output():
    """A reasoning model that returns empty content once must be retried."""
    calls: list[int] = []
    good = make_fake_chat()

    def flaky_chat(messages, max_tokens):
        calls.append(max_tokens)
        if len(calls) == 1:
            return ""  # budget burned thinking; finish_reason=length
        return good(messages, max_tokens)

    script = generate_longform_script(
        "T", "D", "https://example.com", target_minutes=20.0, chat_fn=flaky_chat
    )
    assert len(script.beats) == 7
    assert len(calls) == 8  # seven beats, first beat retried once
    assert calls[0] >= 8192
    assert calls[1] == calls[0] * 2  # retry doubles the token budget


def test_generate_longform_retries_truncated_json():
    good = make_fake_chat()
    calls: list[int] = []

    def trunc_chat(messages, max_tokens):
        calls.append(max_tokens)
        if len(calls) == 1:
            return '{"scenes": [{"title": "cut off mid-stream"'
        return good(messages, max_tokens)

    script = generate_longform_script(
        "T", "D", "https://example.com", target_minutes=20.0, chat_fn=trunc_chat
    )
    assert len(script.beats) == 7
    assert len(calls) == 8


def test_generate_longform_retry_exhaustion_fails_loud():
    def empty_chat(messages, max_tokens):
        return ""

    with pytest.raises(LongformError, match="after 2 attempts"):
        generate_longform_script(
            "T", "D", "https://example.com", target_minutes=20.0, chat_fn=empty_chat
        )
