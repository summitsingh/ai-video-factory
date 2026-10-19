"""Tests for LTX variant/layout plumbing in src/ai_video_factory/hero_video.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_video_factory import hero_video


def _make_ltx25_flat(root: Path) -> Path:
    d = root / "ltx25"
    d.mkdir()
    (d / hero_video.LTX25_TRANSFORMER).write_bytes(b"fake")
    te = d / "gemma4-12b-ltx-v1"
    te.mkdir()
    (te / "gemma4-12b-ltx-v1_int8_convrot.safetensors").write_bytes(b"fake")
    return d


def _make_ltx2_diffusers(root: Path) -> Path:
    d = root / "ltx2"
    t = d / "transformer"
    t.mkdir(parents=True)
    (t / "diffusion_pytorch_model.safetensors").write_bytes(b"fake")
    return d


def test_detect_layout_ltx25_flat(tmp_path):
    assert hero_video.detect_layout(_make_ltx25_flat(tmp_path)) == "ltx25-flat"


def test_detect_layout_ltx2_diffusers(tmp_path):
    assert hero_video.detect_layout(_make_ltx2_diffusers(tmp_path)) == "ltx2-diffusers"


def test_detect_layout_empty(tmp_path):
    assert hero_video.detect_layout(tmp_path) is None


def test_resolve_variant_auto_prefers_ltx25(tmp_path, monkeypatch):
    flat = _make_ltx25_flat(tmp_path)
    legacy = _make_ltx2_diffusers(tmp_path)
    monkeypatch.setattr(hero_video, "LTX25_MODEL_DIR", flat)
    monkeypatch.setattr(hero_video, "LTX2_MODEL_DIR", legacy)
    variant, model_dir = hero_video._resolve_variant("auto")
    assert (variant, model_dir) == ("ltx2.5", flat)


def test_resolve_variant_auto_falls_back_to_legacy(tmp_path, monkeypatch):
    legacy = _make_ltx2_diffusers(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(hero_video, "LTX25_MODEL_DIR", empty)
    monkeypatch.setattr(hero_video, "LTX2_MODEL_DIR", legacy)
    variant, model_dir = hero_video._resolve_variant("auto")
    assert (variant, model_dir) == ("ltx2", legacy)


def test_resolve_variant_explicit_missing_raises(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(hero_video, "LTX25_MODEL_DIR", empty)
    with pytest.raises(hero_video.HeroVideoError, match="no ltx2.5 weights found"):
        hero_video._resolve_variant("ltx2.5")


def test_resolve_variant_unknown_raises():
    with pytest.raises(hero_video.HeroVideoError, match="unknown variant"):
        hero_video._resolve_variant("sora")


def test_default_steps():
    assert hero_video.default_steps("ltx2.5") == hero_video.LTX25_DEFAULT_STEPS == 8
    assert hero_video.default_steps("ltx2") == hero_video.LTX2_DEFAULT_STEPS == 40


def _fake_venv(root: Path) -> Path:
    p = root / "venv" / "bin" / "python"
    p.parent.mkdir(parents=True)
    p.write_text("#!/bin/sh\n")
    return root / "venv"


def test_backend_available_ok(tmp_path, monkeypatch):
    flat = _make_ltx25_flat(tmp_path)
    worker = tmp_path / "generate_ltx2.py"
    worker.write_text("# worker\n")
    monkeypatch.setattr(hero_video, "BACKEND_VENV", _fake_venv(tmp_path))
    monkeypatch.setattr(hero_video, "WORKER", worker)
    monkeypatch.setattr(hero_video, "LTX25_MODEL_DIR", flat)
    ok, detail = hero_video.backend_available("ltx2.5")
    assert ok, detail
    assert "variant=ltx2.5" in detail


def test_backend_available_no_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(hero_video, "BACKEND_VENV", tmp_path / "nope")
    ok, detail = hero_video.backend_available("ltx2.5")
    assert not ok
    assert "backend venv python not found" in detail


def test_backend_available_no_weights(tmp_path, monkeypatch):
    worker = tmp_path / "generate_ltx2.py"
    worker.write_text("# worker\n")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(hero_video, "BACKEND_VENV", _fake_venv(tmp_path))
    monkeypatch.setattr(hero_video, "WORKER", worker)
    monkeypatch.setattr(hero_video, "LTX25_MODEL_DIR", empty)
    monkeypatch.setattr(hero_video, "LTX2_MODEL_DIR", empty)
    ok, detail = hero_video.backend_available("auto")
    assert not ok
    assert "download_ltx2.py" in detail


def test_qc_hero_clip_uses_hard_fail_policy(tmp_path, monkeypatch):
    import ai_video_factory.asset_qc as asset_qc

    seen = {}

    def fake_verify(path, work_dir=None, *, cut_policy="advisory"):
        seen["cut_policy"] = cut_policy
        seen["path"] = path
        return {"ok": True, "reasons": [], "scores": {}}

    monkeypatch.setattr(asset_qc, "verify_video_asset", fake_verify)
    clip = tmp_path / "hero.mp4"
    clip.write_bytes(b"x")
    verdict = hero_video.qc_hero_clip(clip)
    assert verdict["ok"] is True
    assert seen["cut_policy"] == "hard_fail"
    assert seen["path"] == clip


def test_qc_hero_clip_raises_on_mid_shot_cuts(tmp_path, monkeypatch):
    import ai_video_factory.asset_qc as asset_qc

    def fake_verify(path, work_dir=None, *, cut_policy="advisory"):
        return {
            "ok": False,
            "reasons": ["mid_shot_cuts"],
            "scores": {"cut_count": 2.0},
        }

    monkeypatch.setattr(asset_qc, "verify_video_asset", fake_verify)
    clip = tmp_path / "hero.mp4"
    clip.write_bytes(b"x")
    with pytest.raises(hero_video.HeroVideoError, match="mid_shot_cuts"):
        hero_video.qc_hero_clip(clip)
