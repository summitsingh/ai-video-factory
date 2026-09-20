"""AI-generated cinematic visuals for scenes where stock assets failed.

When the NASA/stock search comes back empty (or every candidate fails QC),
the pipeline used to fall back to a cheap phone-mockup card with raw text.
This module replaces that fallback: it generates a cinematic still for the
scene instead.

Two backends:

* ``procedural`` (ACTIVE TODAY): a deterministic Pillow/numpy renderer that
  paints a themed 1280x720 cinematic canvas (gradient, nebula clouds,
  starfield, film grain, vignette) keyed off the scene's visual direction.
  No downloads, no GPU, runs in under a second.
* ``diffusion`` (DORMANT): calls a local Stable Diffusion XL Turbo pipeline
  when torch + diffusers + the checkpoint are installed. See INTEGRATION.md
  for the exact unlock steps.

``generate_scene_visual`` picks diffusion automatically when it is available
and falls back to procedural otherwise. ``ACTIVE_BACKEND`` reports which
path was selected at import time.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

WIDTH = 1280
HEIGHT = 720

# Diffusion unlock: SDXL-Turbo, few-step model that fits an iGPU.
DIFFUSION_MODEL_ID = "stabilityai/sdxl-turbo"
_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/ai_video_factory/ -> repo root
DIFFUSION_MODEL_DIR = Path(
    os.environ.get("AI_VISUALS_MODEL_DIR", _REPO_ROOT / "models" / "sdxl-turbo")
)

_THEMES: dict[str, dict] = {
    "earth": {
        "keywords": (
            "earth", "globe", "world", "atmosphere", "continent",
            "blue marble",
        ),
        "top": (45, 95, 200),
        "bottom": (15, 35, 90),
        "accents": [(59, 130, 246), (34, 211, 238), (16, 122, 180)],
        "stars": 500,
        "nebula_blobs": 4,
        "planet": True,
    },
    "fire": {
        "keywords": (
            "fire", "flame", "explosion", "lava", "ember", "inferno",
            "engulfed", "burning", "blast",
        ),
        "top": (180, 65, 35),
        "bottom": (55, 22, 12),
        "accents": [(249, 115, 22), (239, 68, 68), (250, 204, 21)],
        "stars": 120,
        "nebula_blobs": 7,
        "embers": True,
    },
    "ocean": {
        "keywords": (
            "ocean", "sea", "water", "wave", "tide", "abyss",
        ),
        "top": (35, 140, 210),
        "bottom": (10, 55, 110),
        "accents": [(34, 211, 238), (59, 130, 246), (103, 232, 249)],
        "stars": 60,
        "nebula_blobs": 5,
        "rays": True,
    },
    "tech": {
        "keywords": (
            "lab", "laboratory", "computer", "data", "circuit", "satellite",
            "rocket", "robot", "signal", "telescope array", "control room",
        ),
        "top": (55, 75, 130),
        "bottom": (22, 32, 65),
        "accents": [(34, 211, 238), (129, 140, 248), (52, 211, 153)],
        "stars": 150,
        "nebula_blobs": 3,
        "grid": True,
    },
    "space": {
        "keywords": (
            "galaxy", "nebula", "star", "cosmos", "universe", "planet",
            "orbit", "astronaut", "telescope", "moon", "mars", "black hole",
            "void", "solar", "interstellar", "constellation",
        ),
        "top": (70, 90, 180),
        "bottom": (25, 38, 95),
        "accents": [(124, 93, 250), (56, 189, 248), (45, 212, 191)],
        "stars": 900,
        "nebula_blobs": 8,
    },
}


def detect_theme(scene_visual_direction: str | None, narration_snippet: str | None = None) -> str:
    """Pick a procedural theme from keywords in the direction/narration.

    Scores each theme by keyword hit count; ties break toward the more
    specific theme (dict order: earth, fire, ocean, tech, then space).
    """
    text = f"{scene_visual_direction or ''} {narration_snippet or ''}".lower()
    best_theme = "abstract"
    best_score = 0
    for theme, spec in _THEMES.items():
        score = sum(1 for kw in spec["keywords"] if kw in text)
        if score > best_score:
            best_score = score
            best_theme = theme
    return best_theme


def _path_usable(path: str | Path | None) -> bool:
    if not path:
        return False
    try:
        p = Path(path)
    except (TypeError, ValueError):
        return False
    return p.is_file() and p.stat().st_size > 0


def needs_generated_visual(scene_assets) -> bool:
    """Return True when stock assets failed and a generated visual is needed.

    Accepts a dict (``{"image": ..., "clip": ..., "qc_failed": ...}``), a
    sequence of asset paths, an object with ``image``/``clip`` attributes
    (e.g. ``EditScene``), or None. Conservative: any doubt means generate.
    """
    if scene_assets is None:
        return True
    if isinstance(scene_assets, dict):
        if scene_assets.get("qc_failed"):
            return True
        candidates = [scene_assets.get("image"), scene_assets.get("clip")]
        candidates = [c for c in candidates if c]
        if not candidates:
            return True
        return not any(_path_usable(c) for c in candidates)
    if isinstance(scene_assets, (list, tuple)):
        candidates = [c for c in scene_assets if c]
        if not candidates:
            return True
        return not any(_path_usable(c) for c in candidates)
    image = getattr(scene_assets, "image", None)
    clip = getattr(scene_assets, "clip", None)
    candidates = [c for c in (image, clip) if c]
    if not candidates:
        return True
    return not any(_path_usable(c) for c in candidates)


def _seed_for(direction: str | None, narration: str | None) -> int:
    digest = hashlib.sha256(
        f"{direction or ''}\n{narration or ''}".encode("utf-8")
    ).hexdigest()
    return int(digest, 16) % (2**32)


def _vertical_gradient(h: int, w: int, top: tuple, bottom: tuple, rng) -> "object":
    import numpy as np

    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top_arr = np.array(top, dtype=np.float32)[None, None, :]
    bottom_arr = np.array(bottom, dtype=np.float32)[None, None, :]
    base = top_arr * (1.0 - t) + bottom_arr * t
    return np.broadcast_to(base, (h, w, 3)).copy()


def _add_nebula(canvas, rng, accents: list[tuple], count: int) -> None:
    import numpy as np

    h, w, _ = canvas.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(count):
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        radius = rng.uniform(h * 0.25, h * 0.75)
        color = np.array(accents[rng.integers(0, len(accents))], dtype=np.float32)
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        falloff = np.exp(-dist2 / (2.0 * (radius ** 2)))
        strength = rng.uniform(0.08, 0.20)
        canvas += (falloff[..., None] * color[None, None, :] * strength)


def _add_stars(canvas, rng, count: int) -> None:
    import numpy as np

    h, w, _ = canvas.shape
    xs = rng.integers(0, w, size=count)
    ys = rng.integers(0, h, size=count)
    brightness = rng.uniform(0.35, 1.0, size=count).astype(np.float32)
    tint = rng.uniform(0.85, 1.0, size=(count, 3)).astype(np.float32)
    canvas[ys, xs] += (brightness[:, None] * 255.0 * tint).astype(np.float32)
    big = rng.random(count) > 0.93
    for x, y, b in zip(xs[big], ys[big], brightness[big]):
        if 1 <= x < w - 1 and 1 <= y < h - 1:
            canvas[y - 1 : y + 2, x] += b * 90.0
            canvas[y, x - 1 : x + 2] += b * 90.0


def _add_vignette(canvas) -> None:
    import numpy as np

    h, w, _ = canvas.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((xx - cx) / (w * 0.62)) ** 2 + ((yy - cy) / (h * 0.62)) ** 2)
    mask = np.clip(1.0 - dist * 0.55, 0.35, 1.0).astype(np.float32)
    canvas *= mask[..., None]


def _add_grain(canvas, rng, sigma: float = 4.0) -> None:
    import numpy as np

    canvas += rng.normal(0.0, sigma, size=canvas.shape).astype(np.float32)


def render_procedural(
    scene_visual_direction: str | None,
    narration_snippet: str | None = None,
    out_path: str | Path | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> Path:
    """Render a deterministic cinematic still with Pillow/numpy.

    Seeded from the direction + narration text, so the same scene always
    produces the same visual. No text is burned into the frame; the
    renderer composites captions separately.
    """
    import numpy as np
    from PIL import Image

    theme = detect_theme(scene_visual_direction, narration_snippet)
    spec = _THEMES.get(theme, {})
    rng = np.random.default_rng(_seed_for(scene_visual_direction, narration_snippet))

    canvas = _vertical_gradient(
        height, width,
        spec.get("top", (28, 32, 68)),
        spec.get("bottom", (25, 35, 80)),
        rng,
    )
    _add_nebula(canvas, rng, spec.get("accents", [(90, 110, 200)]),
                spec.get("nebula_blobs", 5))
    _add_stars(canvas, rng, spec.get("stars", 300))

    if spec.get("planet"):
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        cx, cy, r = width * 0.68, height * 0.52, height * 0.34
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        disc = np.clip(1.0 - dist / r, 0.0, 1.0)
        glow = np.clip(1.0 - np.abs(dist - r) / (r * 0.25), 0.0, 1.0)
        blue = np.array([70, 140, 230], dtype=np.float32)
        canvas += disc[..., None] * blue[None, None, :] * 0.55
        canvas += glow[..., None] * np.array([120, 200, 255], dtype=np.float32)[None, None, :] * 0.35

    if spec.get("embers"):
        for _ in range(160):
            x = rng.integers(0, width)
            y = rng.integers(0, height)
            length = rng.integers(6, 26)
            heat = rng.uniform(0.4, 1.0)
            y0 = max(0, y - length)
            canvas[y0:y, x] += np.array([255, 140, 40], dtype=np.float32) * heat * 0.8

    if spec.get("rays"):
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        bands = np.sin((xx * 0.7 + yy) * 0.02 + rng.uniform(0, 6.28))
        bands = np.clip(bands, 0.0, 1.0) ** 3
        canvas += bands[..., None] * np.array([150, 220, 255], dtype=np.float32)[None, None, :] * 0.12

    if spec.get("grid"):
        step_x, step_y = width // 16, height // 9
        canvas[:, ::step_x] += 14.0
        canvas[::step_y, :] += 14.0
        for _ in range(14):
            x = rng.integers(0, width)
            y = rng.integers(0, height)
            canvas[max(0, y - 3):y + 4, max(0, x - 3):x + 4] += np.array(
                [40, 200, 220], dtype=np.float32) * 0.5

    # Cinematic depth: layered terrain silhouettes (Higgsfield-style foreground/midground/background)
    yy2, xx2 = np.mgrid[0:height, 0:width].astype(np.float32)
    for layer in range(3):
        base_y = height * (0.72 + layer * 0.10)
        amp = height * (0.08 - layer * 0.02)
        freq = 0.008 + layer * 0.004
        phase = rng.uniform(0, 6.28)
        ridge = base_y + amp * np.sin(xx2 * freq + phase) + amp * 0.5 * np.sin(xx2 * freq * 2.7 + phase * 1.3)
        mask = (yy2 > ridge).astype(np.float32)
        shade = 12 + layer * 18
        canvas = canvas * (1 - mask[..., None]*0.7) + shade * mask[..., None]*0.7
    # Directional key light from upper-left
    light_x, light_y = width * 0.25, height * 0.15
    dist_light = np.sqrt((xx2 - light_x) ** 2 + (yy2 - light_y) ** 2)
    light_falloff = np.exp(-dist_light / (width * 0.9))
    canvas += light_falloff[..., None] * np.array([255, 230, 190], dtype=np.float32)[None, None, :] * 0.15
    _add_vignette(canvas)
    _add_grain(canvas, rng)
    np.clip(canvas, 0, 255, out=canvas)

    if out_path is None:
        out_path = Path(f"/tmp/ai-visual-{_seed_for(scene_visual_direction, narration_snippet):08x}.jpg")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.astype("uint8"), mode="RGB").save(out_path, quality=92)
    log.info("procedural visual rendered", extra={"theme": theme, "path": str(out_path)})
    return out_path


def _diffusion_available() -> bool:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _diffusion_model_present() -> bool:
    d = DIFFUSION_MODEL_DIR
    if not d.is_dir():
        return False
    return (d / "model_index.json").exists() or any(d.glob("*.safetensors"))


_pipe = None  # cached diffusion pipeline, created on first use


def _generate_with_diffusion(
    scene_visual_direction: str | None,
    narration_snippet: str | None,
    out_path: Path,
    style: str,
) -> Path:
    """Run the local SDXL-Turbo pipeline. Only called when available."""
    import torch
    from diffusers import AutoPipelineForText2Image

    global _pipe
    if _pipe is None:
        _pipe = AutoPipelineForText2Image.from_pretrained(
            str(DIFFUSION_MODEL_DIR) if _diffusion_model_present() else DIFFUSION_MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _pipe = _pipe.to(device)
    prompt = (
        f"{style} cinematic still frame, {scene_visual_direction or 'abstract scene'}. "
        f"Story context: {narration_snippet or 'documentary narration'}. "
        "No text, no watermark, no logo, no people in closeup, photorealistic, "
        "dramatic lighting, film still."
    )
    image = _pipe(
        prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        width=WIDTH,
        height=HEIGHT,
        generator=torch.Generator().manual_seed(
            _seed_for(scene_visual_direction, narration_snippet)
        ),
    ).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    log.info("diffusion visual rendered", extra={"path": str(out_path)})
    return out_path


def _active_backend() -> str:
    if _diffusion_available() and _diffusion_model_present():
        return "diffusion"
    return "procedural"


#: Which generation path is active. "procedural" until the diffusion
#: unlock in INTEGRATION.md is completed.
ACTIVE_BACKEND = _active_backend()


def generate_scene_visual(
    scene_visual_direction: str | None,
    narration_snippet: str | None,
    out_path: str | Path,
    style: str = "cinematic documentary",
) -> Path:
    """Generate a cinematic still for a scene whose stock assets failed.

    Uses the local diffusion model when installed, otherwise the
    deterministic procedural renderer. Always returns the output Path.
    """
    target = Path(out_path)
    if _diffusion_available() and _diffusion_model_present():
        return _generate_with_diffusion(
            scene_visual_direction, narration_snippet, target, style
        )
    return render_procedural(scene_visual_direction, narration_snippet, target)
