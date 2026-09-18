"""Tests for the YouTube packaging module (titles, thumbnails, descriptions)."""

from __future__ import annotations

from ai_video_factory.packaging import (
    generate_description,
    generate_thumbnail_briefs,
    generate_title_variants,
    score_title,
)


def _sample_script() -> dict:
    return {
        "title": "Where Is Everybody? The Fermi Paradox in 90 Seconds",
        "narration": "The galaxy should be full of life. So where is everybody?",
        "scenes": [
            {
                "title": "The Silent Sky",
                "narration": "We have searched the skies for decades and heard nothing.",
                "visual": "Slow push through a star field toward a distant galaxy",
                "duration_frames": 270,
            },
            {
                "title": "The Great Filter",
                "narration": "Maybe something stops civilizations before they spread.",
                "visual": "A planet engulfed in fire beside a black hole",
                "duration_frames": 270,
            },
        ],
        "sources": [
            "https://www.nasa.gov/image-detail/fermi-nebula",
            "https://news.google.com/rss/articles/fermi-123",
        ],
    }


def test_title_variants_returns_requested_count():
    variants = generate_title_variants(
        "The Fermi Paradox",
        "Where Is Everybody? The Fermi Paradox in 90 Seconds",
        "A documentary about why we see no aliens.",
        count=3,
    )
    assert len(variants) == 3


def test_title_variants_cover_multiple_formulas():
    variants = generate_title_variants(
        "The Fermi Paradox", "Fermi Paradox Short", "", count=5
    )
    assert len(set(variants)) == 5
    joined = " ".join(variants)
    assert "Nobody" in joined or "Wrong" in joined or "7" in joined


def test_title_variants_are_defensible_from_material():
    topic = "The Fermi Paradox"
    variants = generate_title_variants(topic, "Fermi Paradox Short", "", count=4)
    for variant in variants:
        lowered = variant.lower()
        assert "fermi" in lowered or "paradox" in lowered, variant


def test_no_em_dashes_in_titles():
    variants = generate_title_variants(
        "The Fermi Paradox", "Fermi Paradox - A Mystery", "", count=5
    )
    for variant in variants:
        assert "\u2014" not in variant
        assert "\u2013" not in variant


def test_score_title_rewards_good_titles():
    good = "The Fermi Paradox Mystery Nobody Talks About"  # 44 chars
    bad = "FERMI PARADOX ALIENS!!!"
    assert score_title(good) > score_title(bad)


def test_score_title_penalizes_em_dash_and_caps():
    assert score_title("The Truth \u2014 What They Hide") < score_title(
        "The Truth: What They Hide"
    )
    assert 0.0 <= score_title("A") <= 1.0
    assert score_title("") == 0.0


def test_score_title_prefers_40_to_60_chars():
    sweet = "x" * 50
    short = "x" * 10
    long_ = "x" * 100
    assert score_title(sweet) >= score_title(short)
    assert score_title(sweet) >= score_title(long_)


def test_thumbnail_briefs_have_all_keys():
    briefs = generate_thumbnail_briefs(_sample_script(), count=3)
    assert len(briefs) == 3
    for brief in briefs:
        for key in (
            "concept",
            "foreground_subject",
            "background",
            "text_overlay",
            "color_mood",
            "why_it_works",
        ):
            assert key in brief and brief[key], key


def test_thumbnail_overlay_max_four_words_uppercase():
    briefs = generate_thumbnail_briefs(_sample_script(), count=3)
    for brief in briefs:
        words = brief["text_overlay"].split()
        assert 1 <= len(words) <= 4
        assert brief["text_overlay"] == brief["text_overlay"].upper()
        assert "\u2014" not in brief["text_overlay"]


def test_thumbnail_briefs_use_script_visuals():
    briefs = generate_thumbnail_briefs(_sample_script(), count=3)
    joined = " ".join(
        b["foreground_subject"] + " " + b["background"] for b in briefs
    ).lower()
    assert "galaxy" in joined or "planet" in joined or "black hole" in joined


def test_thumbnail_briefs_survive_empty_script():
    briefs = generate_thumbnail_briefs({}, count=2)
    assert len(briefs) == 2
    for brief in briefs:
        assert brief["text_overlay"].split()


def test_description_has_hook_chapters_sources():
    script = _sample_script()
    desc = generate_description("The Fermi Paradox", script, script["sources"])
    assert "CHAPTERS" in desc
    assert "SOURCES" in desc
    assert "The Silent Sky" in desc
    assert "The Great Filter" in desc
    assert "00:00" in desc


def test_description_links_live_in_text_not_video():
    script = _sample_script()
    desc = generate_description("The Fermi Paradox", script, script["sources"])
    assert "https://www.nasa.gov/image-detail/fermi-nebula" in desc
    assert "https://news.google.com/rss/articles/fermi-123" in desc
    assert "NASA" in desc
    assert "Google News" in desc


def test_description_hook_uses_script_narration():
    script = _sample_script()
    desc = generate_description("The Fermi Paradox", script, script["sources"])
    assert "searched the skies" in desc
    assert "\u2014" not in desc
