"""Tests for ai_visuals: gap detection, theme routing, procedural renderer,
and diffusion dispatch (mocked, since no GPU model is installed).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from ai_video_factory import ai_visuals
from ai_video_factory.ai_visuals import (
    ACTIVE_BACKEND,
    detect_theme,
    generate_scene_visual,
    needs_generated_visual,
    render_procedural,
)


def test_active_backend_is_procedural_without_gpu_stack():
    assert ACTIVE_BACKEND == "procedural"


def test_needs_generated_visual_none_and_empty():
    assert needs_generated_visual(None) is True
    assert needs_generated_visual({}) is True
    assert needs_generated_visual({"image": None, "clip": None}) is True
    assert needs_generated_visual([]) is True


def test_needs_generated_visual_with_usable_asset(tmp_path: Path):
    good = tmp_path / "still.jpg"
    good.write_bytes(b"\xff\xd8fake")
    assert needs_generated_visual({"image": str(good), "clip": None}) is False
    assert needs_generated_visual([str(good)]) is False


def test_needs_generated_visual_missing_file_counts_as_failed(tmp_path: Path):
    missing = tmp_path / "nope.jpg"
    assert needs_generated_visual({"image": str(missing)}) is True
    assert needs_generated_visual([str(missing)]) is True


def test_needs_generated_visual_qc_failed_flag():
    assert needs_generated_visual({"image": "/tmp/x.jpg", "qc_failed": True}) is True


def test_needs_generated_visual_duck_typed_scene(tmp_path: Path):
    class FakeScene:
        image = None
        clip = None

    assert needs_generated_visual(FakeScene()) is True
    good = tmp_path / "clip.mp4"
    good.write_bytes(b"fake")
    scene = FakeScene()
    scene.clip = str(good)
    assert needs_generated_visual(scene) is False


def test_detect_theme_keywords():
    assert detect_theme("Slow push through a spiral galaxy") == "space"
    assert detect_theme("Earth viewed from orbit at night") == "earth"
    assert detect_theme("A planet engulfed in fire") == "fire"
    assert detect_theme("Deep ocean trench, dark water") == "ocean"
    assert detect_theme("Inside a research laboratory") == "tech"
    assert detect_theme("Abstract light streaks") == "abstract"
    assert detect_theme(None) == "abstract"


def test_render_procedural_produces_720p_still(tmp_path: Path):
    out = tmp_path / "scene" / "visual.jpg"
    result = render_procedural(
        "Rapid montage: a planet engulfed in fire",
        "Narration about stellar death.",
        out,
    )
    assert result == out
    assert out.is_file()
    with Image.open(out) as img:
        assert img.size == (1280, 720)
        assert img.mode == "RGB"


def test_render_procedural_is_deterministic(tmp_path: Path):
    a = render_procedural("Nebula clouds drift", "Space narration",
                          tmp_path / "a.jpg")
    b = render_procedural("Nebula clouds drift", "Space narration",
                          tmp_path / "b.jpg")
    assert a.read_bytes() == b.read_bytes()


def test_render_procedural_varies_by_scene(tmp_path: Path):
    a = render_procedural("Nebula clouds drift", None, tmp_path / "a.jpg")
    b = render_procedural("Volcano eruption at night", None, tmp_path / "b.jpg")
    assert a.read_bytes() != b.read_bytes()


def test_render_procedural_creates_parent_dirs(tmp_path: Path):
    out = tmp_path / "deep" / "nested" / "v.jpg"
    render_procedural("Ocean waves", None, out)
    assert out.is_file()


def test_generate_scene_visual_falls_back_to_procedural(tmp_path: Path):
    out = tmp_path / "gen.jpg"
    result = generate_scene_visual(
        "Telescope array under starry sky",
        "Astronomers listen for signals.",
        out,
    )
    assert isinstance(result, Path)
    assert out.is_file()
    with Image.open(out) as img:
        assert img.size == (1280, 720)


def test_generate_scene_visual_dispatches_to_diffusion_when_unlocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = []

    def fake_diffusion(direction, narration, target, style):
        calls.append((direction, narration, str(target), style))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-diffusion-image")
        return target

    monkeypatch.setattr(ai_visuals, "_diffusion_available", lambda: True)
    monkeypatch.setattr(ai_visuals, "_diffusion_model_present", lambda: True)
    monkeypatch.setattr(ai_visuals, "_generate_with_diffusion", fake_diffusion)

    out = tmp_path / "d.jpg"
    result = generate_scene_visual("Galaxy", "Narration", out,
                                   style="epic sci-fi")
    assert result == out
    assert out.read_bytes() == b"fake-diffusion-image"
    assert calls[0][0] == "Galaxy"
    assert calls[0][3] == "epic sci-fi"


def test_diffusion_not_used_when_model_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ai_visuals, "_diffusion_available", lambda: True)
    monkeypatch.setattr(ai_visuals, "_diffusion_model_present", lambda: False)
    out = tmp_path / "p.jpg"
    generate_scene_visual("Galaxy", None, out)
    with Image.open(out) as img:
        assert img.size == (1280, 720)


def test_module_has_no_em_dashes():
    src = Path(ai_visuals.__file__).read_text(encoding="utf-8")
    assert "\u2014" not in src
