"""Pre-render quality checks for downloaded video and image assets.

Catches the bad-asset classes seen in production renders: black or empty
frames at clip starts, NASA TV title slates and broadcast graphics with
heavy burned-in text, and static slates (near-duplicate frames). These
checks run on the cheap downloaded files BEFORE the expensive Remotion
render, so a bad asset can be skipped and replaced while it is still cheap.

Dependency-light by design: ffmpeg for frame extraction, Pillow for pixel
statistics. Tesseract OCR is used when available, otherwise an
edge-density heuristic in text-band regions stands in for text detection.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageStat

# ---------------------------------------------------------------------------
# Config knobs (tune here; see INTEGRATION.md)
# ---------------------------------------------------------------------------

#: Mean grayscale brightness (0-255) below which a frame may be black.
BLACK_MEAN_THRESHOLD = 12.0

#: Mean brightness of the brightest 1% of pixels below which a dark frame
#: counts as truly empty (separates empty black from starfields).
BLACK_CONTENT_THRESHOLD = 15.0

#: Mean brightness below which a frame counts as suspiciously dark.
DARK_MEAN_THRESHOLD = 28.0

#: Band edge density (0-1) at/above which a frame is clearly text-heavy.
TEXT_EDGE_THRESHOLD = 0.04

#: For dark frames: concentration ratio (max band / mean band) that marks
#: text-like edge clustering.
TEXT_CONCENTRATION_THRESHOLD = 1.4

#: Minimum band edge density for the dark-frame text rule.
TEXT_DARK_EDGE_FLOOR = 0.016

#: Frame-to-frame similarity (0-1) above which frames count as identical.
STATIC_SIMILARITY_THRESHOLD = 0.995

#: OCR word count at/above which a frame is text-heavy (tesseract only).
OCR_WORD_THRESHOLD = 8

#: Minimum acceptable still-image dimensions.
MIN_IMAGE_WIDTH = 640
MIN_IMAGE_HEIGHT = 360

#: Size for brightness/similarity statistics (fast).
THUMBNAIL_SIZE = (320, 180)

#: Size for text-band analysis (finer edge detail).
TEXT_ANALYSIS_SIZE = (640, 360)

#: Timestamps (fractions of duration) sampled from video assets.
SAMPLE_FRACTIONS = (0.02, 0.5, 0.95)

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> Image.Image | None:
    """Open an image file, or return None when it cannot be decoded."""
    try:
        with Image.open(path) as handle:
            return handle.convert("RGB")
    except Exception:
        return None


def _fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Downscale to fit inside size, centered on a black canvas of size."""
    work = img.copy()
    work.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(work, ((size[0] - work.width) // 2, (size[1] - work.height) // 2))
    return canvas


def _mean_brightness(img: Image.Image) -> float:
    """Mean grayscale brightness on a 0-255 scale."""
    return ImageStat.Stat(img.convert("L")).mean[0]


def _has_content(img: Image.Image) -> bool:
    """True when the frame holds real bright content, not just dark noise.

    Uses the mean of the brightest 1% of pixels (from the histogram, so it
    is fast). Separates an empty black frame (top 1% near zero) from a
    starfield or night scene (top 1% holds stars, lights, text).
    """
    hist = img.convert("L").histogram()
    total = sum(hist)
    if total == 0:
        return False
    need = total // 100  # brightest 1%
    acc = 0
    weighted = 0
    for level in range(255, -1, -1):
        take = min(hist[level], need - acc)
        weighted += take * level
        acc += take
        if acc >= need:
            break
    return (weighted / max(acc, 1)) >= BLACK_CONTENT_THRESHOLD


def _edge_density(img: Image.Image, box: tuple[int, int, int, int] | None = None) -> float:
    """Fraction of edge energy in a region, 0-1.

    Uses Pillow's FIND_EDGES on grayscale; bright output pixels mark edges.
    """
    gray = img.convert("L")
    if box is not None:
        gray = gray.crop(box)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return ImageStat.Stat(edges).mean[0] / 255.0


def _text_bands(img: Image.Image) -> tuple[float, float]:
    """Return (max band edge density, concentration ratio).

    Measures edge density in six narrow horizontal bands at
    text-analysis resolution. Slates, lower-thirds, and broadcast graphics
    cluster sharp text edges in one or two bands; natural footage spreads
    edges evenly. Concentration is max band over mean band.
    """
    work = _fit(img, TEXT_ANALYSIS_SIZE)
    w, h = work.size
    bands = [
        (0, int(h * i / 6), w, int(h * (i + 1) / 6))
        for i in range(6)
    ]
    vals = [_edge_density(work, box) for box in bands]
    peak = max(vals)
    mean = sum(vals) / len(vals)
    concentration = peak / mean if mean > 0 else 0.0
    return peak, concentration


def _looks_text_heavy(img: Image.Image, brightness: float) -> bool:
    """Heuristic text detection for one frame."""
    peak, concentration = _text_bands(img)
    if peak >= TEXT_EDGE_THRESHOLD:
        return True
    if (
        brightness < DARK_MEAN_THRESHOLD
        and concentration >= TEXT_CONCENTRATION_THRESHOLD
        and peak >= TEXT_DARK_EDGE_FLOOR
    ):
        return True
    if (
        brightness < BLACK_MEAN_THRESHOLD
        and _has_content(img)
        and concentration >= 2.0
    ):
        return True
    return False


def _frame_similarity(a: Image.Image, b: Image.Image) -> float:
    """Similarity of two frames, 0-1 (1 means pixel-identical)."""
    diff = ImageChops.difference(a.convert("L"), b.convert("L"))
    mean_diff = ImageStat.Stat(diff).mean[0]
    return 1.0 - (mean_diff / 255.0)


def _ocr_word_count(img: Image.Image) -> int | None:
    """Confident OCR word count, or None when tesseract is unavailable."""
    try:
        import pytesseract  # type: ignore[import]
    except ImportError:
        return None
    if shutil.which("tesseract") is None:
        return None
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception:
        return None
    words = 0
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        try:
            if text.strip() and float(conf) > 30:
                words += 1
        except (ValueError, TypeError):
            continue
    return words


def _video_duration(path: Path) -> float | None:
    """Duration in seconds via ffprobe, or None when unavailable."""
    if not _FFPROBE:
        return None
    try:
        proc = subprocess.run(
            [_FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _sample_video_frames(video: Path, dest_dir: Path) -> list[Image.Image]:
    """Extract first/middle/last frames as PIL images (empty on failure)."""
    if not _FFMPEG:
        return []
    duration = _video_duration(video) or 0.0
    frames: list[Image.Image] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i, frac in enumerate(SAMPLE_FRACTIONS):
        stamp = max(0.1, duration * frac)
        out = dest_dir / f"sample-{i}.png"
        try:
            subprocess.run(
                [_FFMPEG, "-v", "error", "-y", "-ss", f"{stamp:.2f}",
                 "-i", str(video), "-frames:v", "1", str(out)],
                capture_output=True, timeout=60,
            )
        except (subprocess.SubprocessError, OSError):
            continue
        img = _load_image(out)
        if img is not None:
            frames.append(img)
    return frames


def _frame_problems(img: Image.Image) -> tuple[list[str], dict[str, float]]:
    """Per-frame checks shared by image and video paths."""
    reasons: list[str] = []
    scores: dict[str, float] = {}
    small = _fit(img, THUMBNAIL_SIZE)
    brightness = _mean_brightness(small)
    scores["brightness"] = brightness
    content = _has_content(img)
    scores["has_content"] = 1.0 if content else 0.0
    if brightness < BLACK_MEAN_THRESHOLD and not content:
        reasons.append("black_frame")
    elif brightness < DARK_MEAN_THRESHOLD and not content:
        reasons.append("dark_frame")

    ocr_words = _ocr_word_count(small)
    if ocr_words is not None:
        scores["ocr_words"] = float(ocr_words)
        if ocr_words >= OCR_WORD_THRESHOLD:
            reasons.append("heavy_text_overlay")
    elif _looks_text_heavy(img, brightness):
        peak, concentration = _text_bands(img)
        scores["text_edge_density"] = peak
        scores["text_concentration"] = concentration
        reasons.append("heavy_text_overlay")
    return reasons, scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_image_asset(path: str | Path) -> dict:
    """Check a downloaded still image. Returns a verdict dict.

    Verdict shape: {"ok": bool, "reasons": [str], "scores": {str: float}}.
    Reasons are machine-readable slugs: unreadable, too_small, black_frame,
    dark_frame, heavy_text_overlay.
    """
    file_path = Path(path)
    img = _load_image(file_path)
    if img is None:
        return {"ok": False, "reasons": ["unreadable"], "scores": {}}

    reasons: list[str] = []
    scores: dict[str, float] = {
        "width": float(img.width),
        "height": float(img.height),
    }
    if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
        reasons.append("too_small")

    frame_reasons, frame_scores = _frame_problems(img)
    reasons.extend(frame_reasons)
    scores.update(frame_scores)
    return {"ok": not reasons, "reasons": reasons, "scores": scores}


def verify_video_asset(path: str | Path, work_dir: str | Path | None = None) -> dict:
    """Check a downloaded video clip. Returns a verdict dict.

    Samples first/middle/last frames and rejects black or empty frames,
    heavy burned-in text (slates, broadcast graphics), and static slates
    (near-duplicate frames). Same verdict shape as verify_image_asset,
    with extra reasons: dark_frames, static_slate, probe_unavailable.
    """
    file_path = Path(path)
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return {"ok": False, "reasons": ["unreadable"], "scores": {}}
    if not _FFMPEG:
        return {"ok": False, "reasons": ["probe_unavailable"], "scores": {"probe": 0.0}}

    scratch = Path(work_dir) if work_dir else file_path.parent / ".asset_qc"
    frames = _sample_video_frames(file_path, scratch)
    if not frames:
        return {"ok": False, "reasons": ["unreadable"], "scores": {}}

    reasons: list[str] = []
    scores: dict[str, float] = {"frames_sampled": float(len(frames))}
    for frame in frames:
        frame_reasons, frame_scores = _frame_problems(frame)
        for reason in frame_reasons:
            # Per-frame "dark_frame" rolls up to the clip-level "dark_frames".
            canonical = "dark_frames" if reason == "dark_frame" else reason
            if canonical not in reasons:
                reasons.append(canonical)
        for key, value in frame_scores.items():
            if key in ("brightness",):
                scores.setdefault("min_brightness", value)
                scores["min_brightness"] = min(scores["min_brightness"], value)
            elif key not in scores:
                scores[key] = value

    if len(frames) >= 2:
        thumbs = [_fit(f, THUMBNAIL_SIZE) for f in frames]
        sims = [
            _frame_similarity(thumbs[i], thumbs[i + 1])
            for i in range(len(thumbs) - 1)
        ]
        scores["min_frame_similarity"] = min(sims)
        if all(s >= STATIC_SIMILARITY_THRESHOLD for s in sims):
            reasons.append("static_slate")

    try:
        for leftover in scratch.glob("sample-*.png"):
            leftover.unlink()
    except OSError:
        pass

    return {"ok": not reasons, "reasons": reasons, "scores": scores}
