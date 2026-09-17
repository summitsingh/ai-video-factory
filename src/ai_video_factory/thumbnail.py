"""Generate YouTube thumbnails from a finished master video.

A documentary is only as clickable as its thumbnail, so this module produces a
set of production-grade stills for the finished master:

1. Extract an iconic frame from the master using ffmpeg's ``peakcontrast`` +
   ``select`` filters (a high-contrast, visually interesting moment rather than
   a bland mid-shot). Falls back to a single mid-point frame if extraction fails.
2. Composite bold, high-contrast YouTube-style title text over that frame:
   short hook copy, large weight, with an outline + drop shadow for legibility
   at thumbnail size.
3. Emit several variants (different text placements / color accents) so a
   creator can A/B test them before publishing.

Everything is written to disk; nothing is uploaded here.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from ai_video_factory.edit_schema import EditDocument
from ai_video_factory.sanitization import sanitize_diagnostic


# A few web-safe fallbacks for a heavy sans; the first that loads wins.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def _load_font(size: int) -> Any:
    """Load the first available bold-ish TTF at ``size`` pixels."""
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    # Last resort: Pillow's bundled bitmap font.
    return ImageFont.load_default()


def _load_font_bold(size: int) -> Any:
    """Load a bold TTF if one is available, else fall back to :func:`_load_font`."""
    for path in _FONT_CANDIDATES:
        if "Bold" in path or "DejaVuSans-Bold" in path:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
    return _load_font(size)


def _truncate_text(text: str, max_chars: int = 6) -> str:
    """YouTube thumbnails read best with very short copy.

    Collapse whitespace and cap the length so text stays large and legible.
    """
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed.upper()
    # Prefer word-boundary truncation, then hard-cut as a last resort.
    words = collapsed.split(" ")
    truncated = " ".join(words[:max_chars])
    return (truncated + "…").upper() if len(collapsed) > max_chars else collapsed.upper()


def _generate_hook(
    doc: EditDocument, title: str, power_words: set[str] | None = None
) -> str:
    """Build a short, punchy hook for the thumbnail.

    Repeating the full documentary title on a thumbnail is amateur - viewers
    read it in the video page anyway. Instead we distill the headline into a
    high-impact phrase (e.g. "THE GOLDEN EYE", "BEYOND EARTH") that teases the
    topic without spoiling it, maximizing curiosity and click-through.

    We prefer an actual scene title from the edit doc when one is short and
    punchy ("The Oldest Galaxy", "Signatures of Life"), since those are already
    written to be intriguing; only fall back to keyword extraction for a long
    or empty document title.

    ``power_words`` defaults to the original space-documentary set; pass a
    theme's list for per-topic theming.
    """
    # Prefer a compelling, concise scene title as the hook.
    best_scene = None
    for scene in doc.scenes:
        if scene.kind in ("intro", "outro"):
            continue
        name = (scene.title or "").strip()
        words = len(name.split())
        # 2-4 words reads well on a thumbnail; skip anything too long.
        if 2 <= words <= 4 and len(name) <= 30:
            best_scene = name
            break

    if best_scene:
        return best_scene.upper()

    # Fallback: distill the document title into power-word hooks.
    power_words = (
        set(power_words)
        if power_words is not None
        else {
            "eyes", "golden eye", "golden", "deepest", "oldest", "invisible",
            "beyond", "frontier", "origins", "cosmic", "universe", "life",
            "secrets", "hidden", "first light", "telescope", "jwst", "webb",
        }
    )

    words = re.split(r"[^a-zA-Z]+", title.lower())
    chosen = [w for w in words if w in power_words and len(w) >= 3]

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique = []
    for w in chosen:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    chosen = unique

    if len(chosen) >= 2:
        # Prefer a two-word hook; fall back to the first power word.
        return " ".join(chosen[:2]).upper()
    if chosen:
        return chosen[0].upper()

    # Fallback: take the first meaningful words of the title.
    fallback = [w for w in re.split(r"[^a-zA-Z]+", title) if len(w) >= 4]
    hook = " ".join(fallback[:3])
    return hook.upper() if hook else "EXPLORE"


def _score_cleanliness(frame_path: Path) -> float:
    """Score how free of on-screen text a frame is, 0.0 (busy) to 1.0 (clean).

    Burned-in captions/lower-thirds appear as bright strokes against the dark
    background. We downscale moderately and count connected components of bright
    pixels: more components means more on-screen text. The moderate scale keeps
    letter strokes thick enough to survive downsampling, unlike a hard 60x34
    resize which erased thin white text entirely (a bug that previously scored
    text-heavy frames as perfectly clean).
    """
    try:
        from PIL import Image as _Image

        small = _Image.open(frame_path).convert("L").resize((120, 68))
        raw = small.tobytes()
        width, height = small.size

        # Bright pixel mask (potential text strokes): bytearray of 0/1 per byte.
        bright = bytearray(1 if b > 140 else 0 for b in raw)

        def _flood(x: int, y: int) -> None:
            """Mark a connected component of bright pixels as consumed."""
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if cx < 0 or cy < 0 or cx >= width or cy >= height:
                    continue
                idx = cy * width + cx
                if not bright[idx]:
                    continue
                bright[idx] = 0
                stack.append((cx + 1, cy))
                stack.append((cx - 1, cy))
                stack.append((cx, cy + 1))
                stack.append((cx, cy - 1))

        components = 0
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                if bright[idx]:
                    components += 1
                    _flood(x, y)

        # Fewer text blobs -> cleaner. A clean photo has ~0-5 blobs; a title card
        # with several words has many more. Map to [0, 1].
        return max(0.0, min(1.0, 1.0 - components / 60.0))
    except Exception:
        return 0.5


def _score_visual_richness(frame_path: Path) -> float:
    """Score a frame for color variety and saturation (for visual interest).

    Used as the primary selection metric when choosing among frames, so the
    thumbnail background looks vivid rather than flat. Higher is better.
    """
    try:
        from PIL import Image as _Image

        img = _Image.open(frame_path).convert("RGB")
        raw = list(img.tobytes())
        ncolors = len(set(tuple(raw[i : i + 3]) for i in range(0, len(raw), 3)))
        sat = sum(
            1
            for i in range(0, len(raw) - 2, 3)
            if abs(raw[i] - raw[i + 1]) + abs(raw[i + 1] - raw[i + 2]) + abs(raw[i + 2] - raw[i]) > 40
        ) / max(1, len(raw) // 3)
        # Normalize: ~6000 colors and ~0.6 saturation are "good" for this content.
        color_score = min(1.0, ncolors / 7000.0)
        return (color_score * 0.5 + min(1.0, sat / 0.7) * 0.5)
    except Exception:
        return 0.0


def _extract_iconic_frame(
    master: Path, frame_path: Path, *, fps: int = 30
) -> bool:
    """Extract the most visually rich frame from ``master``.

    ffmpeg extracts one frame every ~2 seconds across the whole video (avoiding
    intro/outro title cards by starting a few seconds in); we then score each
    candidate for visual richness (color variety + saturation) and pick the most
    vivid one, so the thumbnail background looks interesting rather than flat.

    For content that is entirely dark slides with burned-in text there may be no
    "clean" frame to find, so we optimize for vividness instead and rely on a
    gradient scrim (see :func:`_apply_scrim`) to make the headline dominate.
    Returns False if extraction fails, in which case callers fall back to a
    mid-point grab.
    """
    master = Path(master)
    frame_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = frame_path.parent / ".thumb_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Extract one frame every 2 seconds at full master resolution (avoiding
        # the blur that came from upscaling a small extracted frame).
        cmd = [
            "ffmpeg", "-y",
            "-i", str(master),
            "-vf", "fps=1/2,scale=-1:720",
            "-vsync", "0",
            str(tmp_dir / "frame-%04d.png"),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=False)
        candidates = sorted(tmp_dir.glob("frame-*.png"))
        if not candidates:
            return False
        # Pick the most visually rich candidate.
        best = max(candidates, key=_score_visual_richness)
        shutil.copyfile(best, frame_path)
        return True
    except Exception:
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _extract_midpoint_frame(master: Path, frame_path: Path) -> None:
    """Fallback: grab a single frame at the video's midpoint."""
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(master),
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=False)
    try:
        duration = float(result.stdout.strip())
    except (ValueError, AttributeError):
        duration = 30.0
    seek_at = max(1.0, min(duration / 2, duration - 1.0))

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek_at),
        "-i", str(master),
        "-frames:v", "1",
        str(frame_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=False)


def _draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: str = "#ffffff",
    outline_color: str = "#000000",
    outline_width: int = 4,
    shadow_offset: int = 3,
    align: str = "center",
) -> tuple[int, int]:
    """Draw ``text`` with an outline + drop shadow; return the box's (w, h)."""
    bbox = draw.textbbox(xy, text, font=font)
    w = int(bbox[2] - bbox[0])
    h = int(bbox[3] - bbox[1])

    # Drop shadow.
    for dx in range(-shadow_offset, shadow_offset + 1):
        for dy in range(-shadow_offset, shadow_offset + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text(
                (xy[0] + dx, xy[1] + dy), text, font=font, fill="#000000"
            )

    # Outline via repeated offset draws.
    for ox in (-outline_width, outline_width):
        for oy in (-outline_width, outline_width):
            if ox == 0 and oy == 0:
                continue
            draw.text(
                (xy[0] + ox, xy[1] + oy),
                text,
                font=font,
                fill=outline_color,
            )

    # Fill.
    draw.text(xy, text, font=font, fill=fill)
    return w, h


def _normalize_frame(frame: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize/pad ``frame`` to an exact ``(width, height)`` (default 16:9).

    The master is letterboxed (~2.39:1), so we pad with a dark background to
    reach YouTube's standard 1280x720 canvas rather than emitting a tall, thin
    thumbnail that looks wrong in the player sidebar.
    """
    target_w, target_h = target_size
    current_w, current_h = frame.size

    if (current_w, current_h) == target_size:
        return frame

    # Scale to cover the target, then center-crop/pad as needed.
    scale = max(target_w / current_w, target_h / current_h)
    new_w = int(current_w * scale)
    new_h = int(current_h * scale)
    frame = frame.resize((new_w, new_h), Image.LANCZOS)

    # Pad (not crop) so no image content is lost; fill with near-black.
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(frame, offset)
    return canvas


def _apply_scrim(frame: Image.Image, *, strength: float = 0.55) -> Image.Image:
    """Darken the whole frame so an overlaid headline dominates busy backgrounds.

    This documentary's scenes are dark slides with burned-in text throughout, so
    there is no "clean" frame to find. A uniform scrim tames that clutter and
    gives the single headline a calm, high-contrast stage without hiding the
    underlying imagery entirely (``strength`` in [0, 1]).
    """
    if strength <= 0:
        return frame
    overlay = Image.new("L", frame.size, int(255 * (1.0 - strength)))
    return Image.composite(frame, Image.new("RGB", frame.size, (0, 0, 0)), overlay)


def _measure_text(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Return ``(width, height)`` of ``text`` under ``font`` without drawing."""
    bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _composite_title(
    frame: Image.Image,
    title: str,
    *,
    accent: str = "#ffd200",
    placement: str = "bottom",
    target_size: tuple[int, int] = (1280, 720),
) -> Image.Image:
    """Composite a bold YouTube-style title over ``frame``."""
    draw = ImageDraw.Draw(frame)
    width, height = frame.size

    hook = _truncate_text(title, max_chars=6)
    if not hook:
        return frame

    # Scale font to roughly 12% of frame height, capped for very tall frames.
    base_size = int(height * 0.12)
    font = _load_font_bold(max(24, base_size))

    text_w, text_h = _measure_text(hook, font)

    # Position: bottom-center with padding, or top-center for a variant.
    pad = int(height * 0.06)
    if placement == "bottom":
        x = width // 2 - text_w // 2
        y = height - pad - text_h
    elif placement == "top":
        x = width // 2 - text_w // 2
        y = pad
    else:  # center
        x = width // 2 - text_w // 2
        y = height // 2 - text_h // 2

    # Semi-transparent dark band behind the text for guaranteed legibility.
    band_height = text_h + int(pad * 1.5)
    draw.rectangle(
        [int(x) - pad, int(y) - pad // 2, int(x) + text_w + pad, int(y) + band_height],
        fill=(0, 0, 0, 160),
    )

    _draw_text_box(draw, (x, y), hook, font, fill=accent)
    return frame


def build_thumbnails(
    doc: EditDocument,
    master: Path,
    output_dir: Path,
    *,
    count: int = 3,
    target_size: tuple[int, int] = (1280, 720),
    power_words: set[str] | list[str] | None = None,
) -> dict[str, Path]:
    """Generate ``count`` thumbnail variants from ``master``.

    Returns a mapping of variant name to written path. The primary title is the
    document title (falling back to the first scene's title). Output is normalized
    to ``target_size`` (default YouTube 16:9, 1280x720). ``power_words``
    overrides the default hook vocabulary for per-topic theming.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose the headline copy: doc title, else first non-intro/outro scene.
    title = (doc.title or "").strip()
    if not title:
        for scene in doc.scenes:
            if scene.kind not in ("intro", "outro"):
                title = scene.title or ""
                break

    # Distill the headline into a short, punchy hook rather than repeating the
    # full documentary title on the thumbnail.
    hook = _generate_hook(doc, title, power_words)

    # Extract the base frame once; fall back to a mid-point grab.
    frame_path = output_dir / ".thumb_base.png"
    ok = _extract_iconic_frame(master, frame_path, fps=doc.fps)
    if not ok:
        _extract_midpoint_frame(master, frame_path)

    try:
        base = Image.open(frame_path).convert("RGB")
    except Exception as error:  # noqa: BLE001 - thumbnails are best-effort
        raise RuntimeError(
            f"could not open extracted thumbnail frame {frame_path}: "
            + sanitize_diagnostic(error)
        ) from error

    accent_variants = ["#ffd200", "#ff3b6b", "#22d3ee"]
    placement_variants = ["bottom", "top", "center"]
    thumbnails: dict[str, Path] = {}
    for i in range(count):
        accent = accent_variants[i % len(accent_variants)]
        placement = placement_variants[i % len(placement_variants)]
        normalized = _normalize_frame(base, target_size)
        scrimmed = _apply_scrim(normalized, strength=0.55)
        variant = _composite_title(
            scrimmed, hook, accent=accent, placement=placement, target_size=target_size
        )
        out_path = output_dir / f"thumbnail-{i + 1}.jpg"
        variant.save(out_path, quality=92)
        thumbnails[f"thumbnail_{i + 1}"] = out_path

    # Write a small manifest describing the variants.
    manifest_path = output_dir / "thumbnails.json"
    manifest_path.write_text(
        json.dumps({"title": title, "variants": list(thumbnails)}, indent=2),
        encoding="utf-8",
    )
    thumbnails["manifest"] = manifest_path

    return thumbnails


def validate_thumbnails(doc: EditDocument) -> bool:
    """Return True if there is any usable headline copy for a thumbnail."""
    if (doc.title or "").strip():
        return True
    return any(
        scene.kind not in ("intro", "outro") and (scene.title or "").strip()
        for scene in doc.scenes
    )
