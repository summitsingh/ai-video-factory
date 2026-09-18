"""Regression tests for the visual-quality fixes (2026-09-18 smoke test).

Covers: visual-direction NASA queries, press-conference photo rejection,
minimum still-image dimensions, and the true runtime in the script prompt.
"""

from __future__ import annotations

from pathlib import Path

from ai_video_factory.nasa_media import (
    _extract_visual_query,
    _image_is_usable,
    _record_is_event_photo,
)


def test_visual_query_pulls_concrete_nouns_from_direction():
    visual = (
        "Rapid montage: (1) a planet engulfed in fire, "
        "(2) a lone Earth-sized world in a dark void"
    )
    query = _extract_visual_query(visual, "Wild Explanations")
    assert "planet" in query
    assert "earth" in query
    assert "Wild" not in query


def test_visual_query_falls_back_to_title_without_anchors():
    query = _extract_visual_query(
        "Slow push through abstract light streaks", "The Cosmic Scale"
    )
    assert "Cosmic" in query or "Scale" in query


def test_visual_query_ignores_substring_matches():
    # "lab" must not fire on "elaborate".
    query = _extract_visual_query(
        "An elaborate dance of particles", "Particle Dance"
    )
    assert "lab" not in query.split()


def test_event_photo_blocklist():
    assert _record_is_event_photo("NASA press conference on Artemis")
    assert _record_is_event_photo("Award ceremony at Johnson Space Center")
    assert not _record_is_event_photo("Earth viewed from the ISS at night")
    assert not _record_is_event_photo("")


def test_image_is_usable_rejects_tiny_stills(tmp_path: Path):
    from PIL import Image

    tiny = tmp_path / "tiny.jpg"
    Image.new("RGB", (320, 200), "black").save(tiny)
    assert not _image_is_usable(tiny)

    big = tmp_path / "big.jpg"
    Image.new("RGB", (1280, 720), "black").save(big)
    assert _image_is_usable(big)

    missing = tmp_path / "nope.jpg"
    assert not _image_is_usable(missing)


def test_script_prompt_states_true_runtime():
    import inspect

    from ai_video_factory import script_generator

    source = inspect.getsource(
        script_generator.generate_script_with_lm_studio
    )
    assert "duration_seconds" in source
    assert "never invent a different length" in source
