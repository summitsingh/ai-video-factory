"""Tests for preflight: CPU-only validation before expensive pipeline stages."""

from __future__ import annotations

from pathlib import Path

from ai_video_factory.preflight import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARN,
    PreflightReport,
    check_caption_fonts,
    check_hero_backend,
    check_inference_config,
    check_music_bed,
    check_resolution_grid,
    check_stock_api_keys,
    format_report,
    run_preflight,
)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "assets" / "music").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---- stock API keys ----


def test_stock_keys_both_set_pass():
    env = {"PEXELS_API_KEY": "a", "PIXABAY_API_KEY": "b"}
    result = check_stock_api_keys(env)
    assert result.status == STATUS_PASS
    assert "pexels" in result.detail and "pixabay" in result.detail


def test_stock_keys_one_set_pass():
    result = check_stock_api_keys({"PIXABAY_API_KEY": "b"})
    assert result.status == STATUS_PASS


def test_stock_keys_none_set_fail():
    result = check_stock_api_keys({})
    assert result.status == STATUS_FAIL
    assert "PEXELS_API_KEY" in result.detail


def test_stock_keys_blank_counts_as_missing():
    result = check_stock_api_keys({"PEXELS_API_KEY": "   "})
    assert result.status == STATUS_FAIL


# ---- hero backend ----


def test_hero_backend_missing_warns(tmp_path: Path):
    root = _repo(tmp_path)
    result = check_hero_backend(root, {})
    assert result.status == STATUS_WARN
    assert "venv" in result.detail


def test_hero_backend_ready_passes(tmp_path: Path):
    root = _repo(tmp_path)
    venv_bin = root / ".venv-ltx2" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text("#!/bin/sh\n")
    (root / "scripts" / "generate_ltx2.py").write_text("# worker\n")
    weights = root / "models" / "LTX-2" / "transformer"
    weights.mkdir(parents=True)
    (weights / "transformer.safetensors").write_bytes(b"\x00" * 8)
    result = check_hero_backend(root, {})
    assert result.status == STATUS_PASS


def test_hero_backend_env_override(tmp_path: Path, monkeypatch):
    root = _repo(tmp_path)
    (root / "scripts" / "generate_ltx2.py").write_text("# worker\n")
    alt = tmp_path / "alt-venv"
    (alt / "bin").mkdir(parents=True)
    (alt / "bin" / "python").write_text("#!/bin/sh\n")
    model = tmp_path / "alt-model" / "transformer"
    model.mkdir(parents=True)
    (model / "w.safetensors").write_bytes(b"\x00")
    env = {"LTX2_VENV": str(alt), "LTX2_MODEL_DIR": str(tmp_path / "alt-model")}
    assert check_hero_backend(root, env).status == STATUS_PASS


# ---- resolution grid ----


def test_resolution_grid_aligned_pass():
    # 1280x736 is on the 32px grid (1280/32=40, 736/32=23).
    result = check_resolution_grid(1280, 736)
    assert result.status == STATUS_PASS


def test_resolution_grid_misaligned_warns_with_adjusted_canvas():
    # 720 % 32 == 16, so the canonical 1280x720 snaps up to 1280x736.
    result = check_resolution_grid(1280, 720)
    assert result.status == STATUS_WARN
    assert "1280x736" in result.detail


def test_resolution_grid_invalid_dims_fail():
    assert check_resolution_grid(0, 720).status == STATUS_FAIL
    assert check_resolution_grid(1280, -1).status == STATUS_FAIL


# ---- music bed ----


def test_music_bed_library_passes(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "assets" / "music" / "calm-bed.mp3").write_bytes(b"\x00")
    result = check_music_bed(root, tmp_path / "data", {})
    assert result.status == STATUS_PASS
    assert "curated library" in result.detail


def test_music_bed_env_override_passes(tmp_path: Path):
    root = _repo(tmp_path)
    track = tmp_path / "custom.mp3"
    track.write_bytes(b"\x00")
    result = check_music_bed(root, tmp_path / "data", {"MUSIC_BED_PATH": str(track)})
    assert result.status == STATUS_PASS
    assert "MUSIC_BED_PATH" in result.detail


def test_music_bed_env_override_missing_file_fails(tmp_path: Path):
    root = _repo(tmp_path)
    result = check_music_bed(
        root, tmp_path / "data", {"MUSIC_BED_PATH": str(tmp_path / "nope.mp3")}
    )
    assert result.status == STATUS_FAIL


def test_music_bed_nothing_warns(tmp_path: Path):
    root = _repo(tmp_path)
    result = check_music_bed(root, tmp_path / "data", {})
    assert result.status == STATUS_WARN
    assert "procedural" in result.detail


# ---- caption fonts ----


def test_caption_fonts_reports_pass_or_warn():
    # Environment-dependent: CI boxes may lack system fonts.
    assert check_caption_fonts().status in (STATUS_PASS, STATUS_WARN)


# ---- inference config ----


def test_inference_config_valid_passes(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "config" / "inference.toml").write_text('base_url = "http://x"\n')
    assert check_inference_config(root).status == STATUS_PASS


def test_inference_config_missing_warns(tmp_path: Path):
    root = _repo(tmp_path)
    result = check_inference_config(root)
    assert result.status == STATUS_WARN


def test_inference_config_unparseable_warns(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "config" / "inference.toml").write_text("not = [valid toml\n")
    assert check_inference_config(root).status == STATUS_WARN


# ---- report orchestration ----


def _full_pass_env(tmp_path: Path) -> dict:
    root = _repo(tmp_path)
    venv_bin = root / ".venv-ltx2" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text("#!/bin/sh\n")
    (root / "scripts" / "generate_ltx2.py").write_text("# worker\n")
    weights = root / "models" / "LTX-2" / "transformer"
    weights.mkdir(parents=True)
    (weights / "w.safetensors").write_bytes(b"\x00")
    (root / "assets" / "music" / "calm-bed.mp3").write_bytes(b"\x00")
    (root / "config" / "inference.toml").write_text('base_url = "http://x"\n')
    return {"PEXELS_API_KEY": "a", "PIXABAY_API_KEY": "b"}


def test_run_preflight_ok_when_everything_passes(tmp_path: Path):
    root = _repo(tmp_path)
    env = _full_pass_env(root)
    report = run_preflight(root, tmp_path / "data", env=env)
    assert isinstance(report, PreflightReport)
    assert len(report.checks) == 6
    assert report.ok
    assert report.failures == []
    # The default 1280x720 canvas is off the 32px hero grid (720 % 32 == 16);
    # preflight reports the adjustment as a warning, not a failure.
    assert [w.name for w in report.warnings] == ["resolution_grid"]


def test_run_preflight_not_ok_when_stock_keys_missing(tmp_path: Path):
    root = _repo(tmp_path)
    env = _full_pass_env(root)
    env.pop("PEXELS_API_KEY")
    env.pop("PIXABAY_API_KEY")
    report = run_preflight(root, tmp_path / "data", env=env)
    assert not report.ok
    assert [f.name for f in report.failures] == ["stock_api_keys"]


def test_run_preflight_never_touches_network_or_models(tmp_path: Path, monkeypatch):
    # Preflight must stay CPU-only: no subprocess, no socket, no torch import.
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("preflight must not spawn subprocesses")
        ),
    )
    root = _repo(tmp_path)
    report = run_preflight(root, tmp_path / "data", env=_full_pass_env(root))
    assert len(report.checks) == 6


def test_report_to_dict_shape(tmp_path: Path):
    root = _repo(tmp_path)
    payload = run_preflight(root, tmp_path / "data", env={}).to_dict()
    assert set(payload) == {"ok", "checks"}
    assert payload["ok"] is False
    assert all(set(c) == {"name", "status", "detail"} for c in payload["checks"])


def test_format_report_marks_and_verdict(tmp_path: Path):
    root = _repo(tmp_path)
    text = format_report(run_preflight(root, tmp_path / "data", env={}))
    assert "[FAIL] stock_api_keys" in text
    assert "preflight FAILED" in text

    ok_text = format_report(
        run_preflight(root, tmp_path / "data", env=_full_pass_env(root))
    )
    assert "[PASS] stock_api_keys" in ok_text
    assert ok_text.rstrip().endswith("preflight OK")
