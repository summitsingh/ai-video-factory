"""Tests for src/ai_video_factory/output_sidecar.py (L5)."""

from __future__ import annotations

import json

from ai_video_factory import output_sidecar


def _settings(**over):
    base = dict(
        prompt="a mountain lake at dawn",
        seed=42,
        model="ltx2.5-22b-distilled",
        quant="int8_convrot",
        width=768,
        height=512,
        num_frames=33,
        fps=24.0,
        steps=8,
    )
    base.update(over)
    return base


def test_sidecar_path():
    assert output_sidecar.sidecar_path("/a/b/clip.mp4").name == "clip.json"
    assert output_sidecar.sidecar_path("clip.mp4").name == "clip.json"


def test_build_sidecar_required_fields():
    s = output_sidecar.build_sidecar(media_name="clip.mp4", **_settings())
    assert s["sidecar_version"] == 1
    assert s["media"] == "clip.mp4"
    assert s["prompt"] == "a mountain lake at dawn"
    assert s["seed"] == 42
    assert s["model"] == "ltx2.5-22b-distilled"
    assert s["quant"] == "int8_convrot"
    assert (s["width"], s["height"], s["num_frames"]) == (768, 512, 33)
    assert s["steps"] == 8
    assert "creation_date" in s and "creation_timestamp" in s
    # optional fields absent when not given
    for key in ("variant", "guidance", "negative_prompt", "timings", "backend", "extra"):
        assert key not in s


def test_build_sidecar_optional_fields():
    s = output_sidecar.build_sidecar(
        media_name="clip.mp4",
        variant="ltx2.5",
        guidance=4.0,
        negative_prompt="shaky",
        timings={"wall_sec": 620.5, "gen_sec": 251.0},
        backend="generate_ltx2.py",
        extra={"raw_prompt": "lake"},
        **_settings(),
    )
    assert s["variant"] == "ltx2.5"
    assert s["guidance"] == 4.0
    assert s["negative_prompt"] == "shaky"
    assert s["timings"] == {"wall_sec": 620.5, "gen_sec": 251.0}
    assert s["backend"] == "generate_ltx2.py"
    assert s["extra"] == {"raw_prompt": "lake"}


def test_write_and_read_roundtrip(tmp_path):
    media = tmp_path / "hero-20261018-s42.mp4"
    media.write_bytes(b"fake-video")
    settings = output_sidecar.build_sidecar(media_name=media.name, **_settings())
    path = output_sidecar.write_sidecar(media, settings)
    assert path == tmp_path / "hero-20261018-s42.json"
    assert path.exists()
    back = output_sidecar.read_sidecar(media)
    assert back == settings
    # valid JSON on disk
    assert json.loads(path.read_text(encoding="utf-8"))["seed"] == 42


def test_read_sidecar_missing_returns_none(tmp_path):
    assert output_sidecar.read_sidecar(tmp_path / "nope.mp4") is None


def test_record_never_raises(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(output_sidecar, "write_sidecar", boom)
    assert output_sidecar.record(media, **_settings()) is None


def test_record_writes_next_to_media(tmp_path):
    media = tmp_path / "sub" / "clip.mp4"
    media.parent.mkdir()
    media.write_bytes(b"x")
    path = output_sidecar.record(media, **_settings())
    assert path == tmp_path / "sub" / "clip.json"
