"""Tests for the YouTube format presets and their pipeline wiring."""

from __future__ import annotations

import json

import pytest

from ai_video_factory.formats import (
    format_names,
    get_preset,
    list_presets,
)
from ai_video_factory.longform import (
    _beats_for_format,
    beat_word_target,
    generate_longform_script,
)
from ai_video_factory.packaging import (
    generate_description,
    generate_thumbnail_briefs,
    generate_title_variants,
    score_title,
)


def _fake_chat_factory(targets):
    calls = {"i": 0}

    def fake_chat(messages, max_tokens):
        target = targets[calls["i"]]
        calls["i"] += 1
        half = target // 2
        return json.dumps(
            {
                "scenes": [
                    {
                        "title": f"S{calls['i']}-1",
                        "narration": " ".join(["word"] * half),
                        "visual_direction": "archival footage",
                        "lower_third": None,
                    },
                    {
                        "title": f"S{calls['i']}-2",
                        "narration": " ".join(["word"] * (target - half)),
                        "visual_direction": "charts",
                        "lower_third": None,
                    },
                ]
            }
        )

    return fake_chat


def test_all_presets_registered():
    keys = format_names()
    for expected in (
        "business_autopsy",
        "systems_explainer",
        "history_reconstruction",
        "mystery_deep_dive",
        "science_doc",
        "armchair_true_crime",
        "horror_anthology",
    ):
        assert expected in keys


def test_tier1_first():
    ordered = list_presets()
    assert ordered[0].tier == 1
    assert ordered[0].key == "business_autopsy"


def test_beat_fractions_sum_to_one():
    for preset in list_presets():
        total = sum(b["fraction"] for b in preset.beats)
        assert total == pytest.approx(1.0), preset.key


def test_hook_has_three_beats():
    for preset in list_presets():
        assert preset.hook.beat1_0_10
        assert preset.hook.beat2_10_30
        assert preset.hook.beat3_30_60


def test_true_crime_needs_review():
    assert get_preset("armchair_true_crime").needs_human_review is True
    assert get_preset("business_autopsy").needs_human_review is False


def test_format_drives_beats_and_word_budget():
    preset = get_preset("business_autopsy")
    beats = _beats_for_format(preset)
    assert [b.key for b in beats] == [
        "cold_open", "rise", "throne", "turn", "fall",
        "verdict", "lesson", "outro",
    ]
    targets = [beat_word_target(b, 25.0) for b in beats]
    assert sum(targets) == pytest.approx(25.0 * 150, abs=10)
    # cold open carries the 3-beat hook plan in its retention device
    assert "0:00-0:10" in beats[0].retention


def test_format_script_end_to_end():
    preset = get_preset("business_autopsy")
    beats = _beats_for_format(preset)
    targets = [beat_word_target(b, 25.0) for b in beats]
    script = generate_longform_script(
        topic="BlackBerry",
        description="rise and fall",
        source_url="https://www.nasa.gov", verify_source_urls=False,
        target_minutes=25.0,
        chat_fn=_fake_chat_factory(targets),
        format_key="business_autopsy",
    )
    assert script.format_key == "business_autopsy"
    assert abs(script.estimated_minutes - 25.0) < 0.2
    payload = json.loads(script.to_json())
    assert payload["format_key"] == "business_autopsy"


def test_default_arc_unchanged_without_format():
    beats = _beats_for_format(None)
    assert [b.key for b in beats][0] == "cold_open"
    assert any(b.key == "act3_climax" for b in beats)


def test_packaging_titles_defensible():
    variants = generate_title_variants(
        "BlackBerry", "The Rise and Fall of BlackBerry",
        "From $83B to nothing", count=3,
    )
    assert len(variants) == 3
    assert all(v for v in variants)
    assert all(score_title(v) >= 0.0 for v in variants)


def test_thumbnail_briefs_new_info():
    briefs = generate_thumbnail_briefs(
        {"title": "The Rise and Fall of BlackBerry", "scenes": []}, count=3
    )
    assert len(briefs) == 3
    for brief in briefs:
        words = brief["text_overlay"].split()
        assert 1 <= len(words) <= 4, brief["text_overlay"]
        # overlay must not repeat the title's words wholesale
        title_words = set("the rise and fall of blackberry".split())
        overlay_words = set(brief["text_overlay"].lower().split())
        assert len(overlay_words - title_words) >= 1


def test_description_has_chapters_and_sources():
    desc = generate_description(
        "BlackBerry",
        {
            "title": "The Rise and Fall of BlackBerry",
            "scenes": [
                {"title": "Cold Open", "duration_frames": 1800},
                {"title": "The Rise", "duration_frames": 3600},
            ],
        },
        ["https://example.com/source"],
    )
    assert "CHAPTERS" in desc
    assert "SOURCES" in desc
    assert "https://example.com/source" in desc
