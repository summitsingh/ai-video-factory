"""Tests for asset_qc: pre-render quality checks for downloaded assets."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from ai_video_factory.asset_qc import (
    _has_content,
    _mean_brightness,
    verify_image_asset,
    verify_video_asset,
)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")


def _big_font(size: int):
    return ImageFont.truetype(FONT, size)


def make_black(path: Path) -> Path:
    Image.new("RGB", (1280, 720), (0, 0, 0)).save(path)
    return path


def make_starfield(path: Path) -> Path:
    import random
    random.seed(11)
    img = Image.new("RGB", (1280, 720), (2, 2, 6))
    draw = ImageDraw.Draw(img)
    for _ in range(500):
        x, y = random.randrange(1280), random.randrange(720)
        b = random.randrange(140, 255)
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(b, b, b))
    img.save(path)
    return path


def make_natural(path: Path) -> Path:
    import random
    random.seed(5)
    img = Image.new("RGB", (1280, 720), (70, 80, 110))
    draw = ImageDraw.Draw(img)
    for _ in range(250):
        x, y = random.randrange(1280), random.randrange(720)
        r = random.randrange(30, 130)
        c = random.randrange(50, 150)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(c // 2, c, c + 10))
    img.save(path)
    return path


def make_slate(path: Path) -> Path:
    img = Image.new("RGB", (1280, 720), (8, 12, 40))
    draw = ImageDraw.Draw(img)
    draw.text((80, 70), "ROCKET LAB PREFIRE-1 LAUNCH", font=_big_font(72), fill=(255, 255, 255))
    draw.text((80, 620), "CONTACT: NASA TV   www.nasa.gov", font=_big_font(36), fill=(170, 180, 200))
    img.save(path)
    return path


def make_video(path: Path, source: str, duration: int = 3) -> Path:
    """Make a test clip. `source` is a lavfi source name (testsrc, color)."""
    if source == "testsrc":
        spec = f"testsrc=size=320x180:rate=10:duration={duration}"
    else:
        spec = f"{source}:s=320x180:r=10:d={duration}"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi",
         "-i", spec, "-pix_fmt", "yuv420p", str(path)],
        check=True, timeout=120,
    )
    return path


# -- image checks --------------------------------------------------------


def test_black_image_rejected(tmp_path: Path):
    verdict = verify_image_asset(make_black(tmp_path / "black.jpg"))
    assert verdict["ok"] is False
    assert "black_frame" in verdict["reasons"]


def test_starfield_not_mistaken_for_black(tmp_path: Path):
    verdict = verify_image_asset(make_starfield(tmp_path / "stars.jpg"))
    assert "black_frame" not in verdict["reasons"]
    assert verdict["scores"]["has_content"] == 1.0


def test_natural_image_accepted(tmp_path: Path):
    verdict = verify_image_asset(make_natural(tmp_path / "natural.jpg"))
    assert verdict["ok"] is True, verdict["reasons"]


def test_title_slate_rejected_for_text(tmp_path: Path):
    verdict = verify_image_asset(make_slate(tmp_path / "slate.jpg"))
    assert verdict["ok"] is False
    assert "heavy_text_overlay" in verdict["reasons"]


def test_small_image_rejected(tmp_path: Path):
    tiny = tmp_path / "tiny.jpg"
    Image.new("RGB", (320, 200), (90, 100, 120)).save(tiny)
    verdict = verify_image_asset(tiny)
    assert verdict["ok"] is False
    assert "too_small" in verdict["reasons"]


def test_missing_image_rejected(tmp_path: Path):
    verdict = verify_image_asset(tmp_path / "nope.jpg")
    assert verdict["ok"] is False
    assert verdict["reasons"] == ["unreadable"]


def test_verdict_shape(tmp_path: Path):
    verdict = verify_image_asset(make_natural(tmp_path / "n.jpg"))
    assert set(verdict.keys()) == {"ok", "reasons", "scores"}
    assert isinstance(verdict["ok"], bool)
    assert isinstance(verdict["reasons"], list)
    assert "brightness" in verdict["scores"]


def test_has_content_separates_black_from_stars(tmp_path: Path):
    black = Image.open(make_black(tmp_path / "b.jpg"))
    stars = Image.open(make_starfield(tmp_path / "s.jpg"))
    assert _has_content(black) is False
    assert _has_content(stars) is True
    assert _mean_brightness(stars) < 12  # dark but not empty


# -- video checks --------------------------------------------------------


@needs_ffmpeg
def test_black_video_rejected(tmp_path: Path):
    verdict = verify_video_asset(make_video(tmp_path / "black.mp4", "color=c=black"))
    assert verdict["ok"] is False
    assert "black_frame" in verdict["reasons"]


@needs_ffmpeg
def test_normal_video_accepted(tmp_path: Path):
    verdict = verify_video_asset(make_video(tmp_path / "src.mp4", "testsrc"))
    assert verdict["ok"] is True, verdict["reasons"]


@needs_ffmpeg
def test_static_video_rejected_as_slate(tmp_path: Path):
    verdict = verify_video_asset(make_video(tmp_path / "static.mp4", "color=c=navy", duration=4))
    assert verdict["ok"] is False
    assert "static_slate" in verdict["reasons"]


@needs_ffmpeg
def test_missing_video_rejected(tmp_path: Path):
    verdict = verify_video_asset(tmp_path / "nope.mp4")
    assert verdict["ok"] is False
    assert verdict["reasons"] == ["unreadable"]


@needs_ffmpeg
def test_video_verdict_reports_samples(tmp_path: Path):
    verdict = verify_video_asset(make_video(tmp_path / "src.mp4", "testsrc"))
    assert verdict["scores"]["frames_sampled"] >= 2
