#!/usr/bin/env python3
"""Dry-run test for ITEM 1: stock provider wiring (Pexels/Pixabay/NASA).

Usage (from the repo root):
    STOCK_CACHE_DIR=/tmp/stock-test-cache \
    .venv/bin/python \
    scripts/test_stock_wiring.py

Exercises:
  1. fetch_clip for 3 sample scenes (space, nature, tech); asserts the
     cached MP4s are valid (ffprobe duration > 0, width >= 640).
  2. _populate_stock_clips end-to-end with a synthetic EditDocument:
     a scene that already has a NASA clip is skipped, the other gets a
     stock-clip.mp4 copy in its scene dir.
  3. _load_repo_dotenv loads PEXELS_API_KEY/PIXABAY_API_KEY from the
     repo-root .env without printing their values.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCENES = [
    ("space", "slow drift through a purple nebula", 3.0),
    ("nature", "aerial over ocean waves at sunset", 3.0),
    ("tech", "circuit board macro, shallow depth of field", 3.0),
]


def ffprobe_info(path: Path) -> tuple[float, int]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr[:300]}")
    data = json.loads(out.stdout)
    stream = (data.get("streams") or [{}])[0]
    return float(stream.get("duration") or 0), int(stream.get("width") or 0)


def main() -> int:
    failures: list[str] = []

    # --- Part 1: direct fetch_clip checks ---
    from ai_video_factory.stock_providers import StockFootageProvider

    provider = StockFootageProvider()
    print(f"pexels available: {provider.pexels.available}, "
          f"pixabay available: {provider.pixabay.available}, "
          f"nasa available: {provider.nasa.available}")
    for label, description, min_dur in SCENES:
        try:
            asset = provider.fetch_clip(description, min_duration_sec=min_dur)
        except Exception as error:  # network hiccups are informative, not fatal
            print(f"[{label}] fetch raised {type(error).__name__}: {str(error)[:120]}")
            failures.append(f"{label}: fetch raised")
            continue
        if asset is None:
            print(f"[{label}] no stock clip found (fallbacks unchanged)")
            continue
        path = Path(asset.path)
        if not (path.is_file() and path.stat().st_size > 50 * 1024):
            print(f"[{label}] FAIL: cached file missing/too small: {path}")
            failures.append(f"{label}: bad cache file")
            continue
        duration, width = ffprobe_info(path)
        status = "OK" if (duration > 0 and width >= 640) else "FAIL"
        print(f"[{label}] {status}: {path.name} provider-derived "
              f"duration={duration:.1f}s width={width} "
              f"license={asset.license!r} query={asset.query!r}")
        if status == "FAIL":
            failures.append(f"{label}: ffprobe check failed")

    # --- Part 2: _populate_stock_clips wiring ---
    from ai_video_factory.video_pipeline import (
        _load_repo_dotenv,
        _populate_stock_clips,
        attach_scene_assets,
    )
    from ai_video_factory.edit_schema import EditDocument, EditScene

    _load_repo_dotenv()
    for var in ("PEXELS_API_KEY", "PIXABAY_API_KEY"):
        print(f"{var} loaded from .env: {bool(os.environ.get(var))}")

    with tempfile.TemporaryDirectory() as tmp:
        assets_dir = Path(tmp)
        (assets_dir / "scene-00").mkdir()
        # Pretend the NASA pass already found a clip for scene-00.
        (assets_dir / "scene-00" / "nasa-clip-0.mp4").write_bytes(b"placeholder")
        edit = EditDocument(
            schema_version=1, width=1280, height=720, fps=24, duration_frames=240,
            scenes=[
                EditScene(id="scene-0", from_frame=0, duration_frames=120,
                          title="Nebula drift", caption="A nebula drifts by.",
                          visual="slow drift through a purple nebula"),
                EditScene(id="scene-1", from_frame=120, duration_frames=120,
                          title="Ocean sunset", caption="Waves at sunset.",
                          visual="aerial over ocean waves at sunset"),
                EditScene(id="intro", from_frame=0, duration_frames=60,
                          title="Intro", caption="Intro.", kind="intro"),
            ],
        )
        assets = _populate_stock_clips(edit, assets_dir)
        skip_ok = (assets_dir / "scene-00" / "nasa-clip-0.mp4").is_file() and not (
            assets_dir / "scene-00" / "stock-clip.mp4").exists()
        stock_copy = assets_dir / "scene-01" / "stock-clip.mp4"
        copy_ok = stock_copy.is_file() and stock_copy.stat().st_size > 50 * 1024
        attach = attach_scene_assets(edit, assets_dir)
        scene1 = next(s for s in attach.scenes if s.id == "scene-1")
        print(f"[wiring] scene-00 untouched (skipped, has clip): {skip_ok}")
        print(f"[wiring] scene-01 stock-clip.mp4 copied: {copy_ok} "
              f"({len(assets)} assets for attribution)")
        print(f"[wiring] attach bound scene-1 clip: {scene1.clip!r}")
        if not (skip_ok and copy_ok and scene1.clip):
            failures.append("wiring: _populate_stock_clips/attach check failed")

    if failures:
        print("FAILURES:", failures)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
