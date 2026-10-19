"""Local hero-shot video generation with LTX-2.5 (and legacy LTX-2).

Produces custom AI-generated video clips for scenes where real stock
footage cannot deliver: impossible camera moves, alien worlds, abstract
concepts, exact brand-matched visuals. Complements
:mod:`ai_video_factory.stock_providers` (Pexels/Pixabay/NASA): stock wins
when real camera footage exists; this module wins when it does not.

Supported config (default): LTX-2.5 22B Distilled, int8_convrot transformer
+ int8 Gemma4 text encoder + int8 connectors + bf16 VAEs/vocoder, from the
``DeepBeepMeep/LTX-2`` mirror (ungated, no HF token). Benchmarked on
summit-amd 2026-10-18: 768x512, 33f @24fps, 8 steps, ~10-12 min/clip.
See ``docs/wangp-ltx25-benchmark.md``.

Backend: the heavy lifting runs in a dedicated virtualenv (TheRock ROCm
torch build). The main package never imports torch/diffusers itself;
generation is delegated to a worker script (``scripts/generate_ltx2.py``)
executed with the backend venv's python, so machines without the backend
get a clear error instead of an ImportError. NOTE: direct inference of the
LTX-2.5 flat int8 checkpoints is the L1(b) follow-up (WanGP headless
backend); until then the worker validates the weight set and fails fast
with a clear message.

SETUP (LTX-2.5)
---------------
1. Create the backend venv and install the diffusion stack (see
   docs/wangp-ltx25-benchmark.md for the proven summit-amd recipe)::

     python3.12 -m venv .venv-ltx2   # under the repo root (gitignored)

2. Get the weights (mirror is NOT gated, no HF token needed)::

     python scripts/download_ltx2.py        # -> models/LTX-2.5/

   ...or reuse the benchmark weights read-only::

     LTX25_MODEL_DIR=/home/summit/wangp/ckpts python -m ...

LEGACY: LTX-2 (2.0) in diffusers layout under models/LTX-2
(``LTX2_MODEL_DIR``) is still accepted via ``variant="ltx2"``.

MODEL LOCATION
--------------
LTX-2.5 flat checkpoints: ``models/LTX-2.5`` under the repo root (override
with ``LTX25_MODEL_DIR``). Legacy diffusers layout: ``models/LTX-2``
(override with ``LTX2_MODEL_DIR``).
Backend venv: ``.venv-ltx2`` under the repo root (override with ``LTX2_VENV``).
Test clips: ``output/hero-clips/`` under the repo root (override with
``HERO_CLIP_DIR``); ``output/`` is gitignored so clips never get committed.

SIDECARS
--------
Every generated clip gets a ``<name>.json`` sidecar next to it (prompt,
seed, model, quant, timings) via :mod:`ai_video_factory.output_sidecar`,
so a bad render can be debugged and regenerated without manifest
archaeology.

KNOWN ISSUES
------------
* ``width``/``height`` must be multiples of 32 (pipeline hard-requires it);
  this module rounds UP to the next multiple and center-crops back to the
  requested size before writing the MP4.
* ``num_frames`` should be 8n+1; the module snaps ``duration*fps`` to that.
* Sequential CPU offload is used for safety on the APU; it costs speed.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from ai_video_factory.output_sidecar import record as record_sidecar

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]  # src/ai_video_factory/hero_video.py -> repo root

BACKEND_VENV = Path(os.environ.get("LTX2_VENV", _REPO_ROOT / ".venv-ltx2"))
# LTX-2.5 flat checkpoints (DeepBeepMeep/LTX-2 mirror layout). Point this at
# /home/summit/wangp/ckpts to reuse the benchmark weights read-only.
LTX25_MODEL_DIR = Path(
    os.environ.get("LTX25_MODEL_DIR", _REPO_ROOT / "models" / "LTX-2.5")
)
# Legacy LTX-2 (2.0) diffusers layout.
LTX2_MODEL_DIR = Path(
    os.environ.get("LTX2_MODEL_DIR", _REPO_ROOT / "models" / "LTX-2")
)
WORKER = _REPO_ROOT / "scripts" / "generate_ltx2.py"
OUTPUT_DIR = Path(os.environ.get("HERO_CLIP_DIR", _REPO_ROOT / "output" / "hero-clips"))

FPS = 24.0

# The supported LTX-2.5 config, as benchmarked on summit-amd.
LTX25_MODEL_ID = "ltx2.5-22b-distilled"
LTX25_QUANT = "int8_convrot"
LTX25_TRANSFORMER = "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
LTX25_TEXT_ENCODER = "gemma4-12b-ltx-v1/gemma4-12b-ltx-v1_int8_convrot.safetensors"
# Key files that must exist for the 2.5 layout to count as present. The
# worker (scripts/generate_ltx2.py) validates the full manifest at run time.
_LTX25_KEY_FILES = (LTX25_TRANSFORMER, LTX25_TEXT_ENCODER)

LTX25_DEFAULT_STEPS = int(os.environ.get("LTX25_STEPS", "8"))  # distilled: 8 steps
LTX2_DEFAULT_STEPS = int(os.environ.get("LTX2_STEPS", "40"))

VARIANT_LTX25 = "ltx2.5"
VARIANT_LTX2 = "ltx2"


class HeroVideoError(RuntimeError):
    """Raised when the LTX backend is missing, broken, or a run fails."""


def detect_layout(model_dir: Path) -> str | None:
    """Return "ltx25-flat", "ltx2-diffusers", or None for ``model_dir``."""
    if (model_dir / LTX25_TRANSFORMER).is_file():
        return "ltx25-flat"
    transformer_dir = model_dir / "transformer"
    if transformer_dir.is_dir() and any(transformer_dir.glob("*.safetensors")):
        return "ltx2-diffusers"
    return None


def _resolve_variant(variant: str) -> tuple[str, Path]:
    """Resolve ``variant`` ("auto", "ltx2.5", "ltx2") to (variant, model_dir).

    "auto" prefers the LTX-2.5 flat layout, then the legacy diffusers layout.
    Raises HeroVideoError when nothing usable is found.
    """
    candidates: list[tuple[str, Path]]
    if variant == VARIANT_LTX25:
        candidates = [(VARIANT_LTX25, LTX25_MODEL_DIR)]
    elif variant == VARIANT_LTX2:
        candidates = [(VARIANT_LTX2, LTX2_MODEL_DIR)]
    elif variant == "auto":
        candidates = [(VARIANT_LTX25, LTX25_MODEL_DIR), (VARIANT_LTX2, LTX2_MODEL_DIR)]
    else:
        raise HeroVideoError(f"unknown variant {variant!r} (want 'auto', 'ltx2.5', 'ltx2')")

    tried = []
    for name, model_dir in candidates:
        layout = detect_layout(model_dir)
        if layout == "ltx25-flat" and name == VARIANT_LTX25:
            return name, model_dir
        if layout == "ltx2-diffusers" and name == VARIANT_LTX2:
            return name, model_dir
        tried.append(str(model_dir))
    hint = (
        "fetch the LTX-2.5 set: python scripts/download_ltx2.py "
        f"(or set LTX25_MODEL_DIR, e.g. /home/summit/wangp/ckpts)"
        if variant in ("auto", VARIANT_LTX25)
        else "fetch the legacy set or switch variant to 'ltx2.5'"
    )
    raise HeroVideoError(
        f"no {variant} weights found (tried: {', '.join(tried)}). {hint}"
    )


def default_steps(variant: str) -> int:
    """Variant-default inference steps (distilled 2.5 needs only 8)."""
    return LTX25_DEFAULT_STEPS if variant == VARIANT_LTX25 else LTX2_DEFAULT_STEPS


def backend_available(variant: str = "auto") -> tuple[bool, str]:
    """Check the LTX backend. Returns (ok, human-readable detail)."""
    venv_python = BACKEND_VENV / "bin" / "python"
    if not venv_python.exists():
        return False, (
            f"backend venv python not found: {venv_python}. "
            "Install per the module docstring: TheRock torch for gfx1151 + "
            f"diffusers into {BACKEND_VENV}"
        )
    if not WORKER.exists():
        return False, f"worker script missing: {WORKER}"
    try:
        resolved, model_dir = _resolve_variant(variant)
    except HeroVideoError as exc:
        return False, str(exc)
    return True, f"backend OK (variant={resolved}, venv={BACKEND_VENV}, model={model_dir})"


def _snap32(n: int) -> int:
    return (n + 31) // 32 * 32


def qc_hero_clip(path: Path) -> dict:
    """Run asset QC on a hero clip with ``cut_policy="hard_fail"``.

    Hero/T2V clips must be one continuous shot: any mid-shot cut (or a
    cut-detection failure) fails the verdict. Raises HeroVideoError on
    failure; returns the verdict dict otherwise.
    """
    from ai_video_factory.asset_qc import verify_video_asset

    verdict = verify_video_asset(path, cut_policy="hard_fail")
    if not verdict.get("ok", False):
        reasons = verdict.get("reasons", [])
        raise HeroVideoError(
            f"hero clip QC failed for {path}: "
            f"{', '.join(reasons) if reasons else 'unknown reason'}"
        )
    return verdict


def generate_hero_clip(
    prompt: str,
    duration_sec: float = 8,
    width: int = 1280,
    height: int = 720,
    seed: int | None = None,
    num_inference_steps: int | None = None,
    fps: float = FPS,
    variant: str = "auto",
    guidance_scale: float = 4.0,
    run_qc: bool = True,
) -> Path:
    """Generate a hero video clip and return the MP4 path.

    ``variant``: "auto" (prefer LTX-2.5, fall back to legacy LTX-2),
    "ltx2.5", or "ltx2". ``num_inference_steps=None`` uses the variant
    default (8 for distilled LTX-2.5, 40 for legacy LTX-2).

    The clip is rendered at the next multiple-of-32 above ``width``/``height``
    (a pipeline requirement) and center-cropped back to the requested size,
    so the returned file is exactly ``width``x``height``.

    Unless ``run_qc`` is False, the finished clip is checked with
    :func:`qc_hero_clip` (asset QC, ``cut_policy="hard_fail"``): mid-shot
    cuts fail the render, consistent with the pipeline's fail-hard
    philosophy.

    A settings sidecar (``<name>.json``: prompt, seed, model, quant,
    timings, QC verdict) is written next to the clip; see
    :mod:`ai_video_factory.output_sidecar`.

    Raises:
        HeroVideoError: if the backend is not installed, generation fails,
            or (with run_qc) the clip fails QC.
    """
    resolved, model_dir = _resolve_variant(variant)
    ok, detail = backend_available(resolved)
    if not ok:
        raise HeroVideoError(f"LTX backend unavailable: {detail}")
    steps = num_inference_steps if num_inference_steps is not None else default_steps(resolved)

    gen_w, gen_h = _snap32(width), _snap32(height)
    num_frames = int(round(duration_sec * fps))
    num_frames = (num_frames // 8) * 8 + 1  # snap to 8n+1
    num_frames = max(num_frames, 9)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = OUTPUT_DIR / f"hero-{stamp}-s{seed if seed is not None else 'rand'}.mp4"

    cmd = [
        str(BACKEND_VENV / "bin" / "python"),
        str(WORKER),
        "--variant", resolved,
        "--model-dir", str(model_dir),
        "--prompt", prompt,
        "--width", str(gen_w),
        "--height", str(gen_h),
        "--num-frames", str(num_frames),
        "--fps", str(fps),
        "--steps", str(steps),
        "--guidance", str(guidance_scale),
        "--out", str(out),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if (gen_w, gen_h) != (width, height):
        cmd += ["--crop-width", str(width), "--crop-height", str(height)]

    log.info(
        "LTX (%s) generating %ss %dx%d (%d frames, %d steps) -> %s",
        resolved, duration_sec, width, height, num_frames, steps, out,
    )
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=6 * 3600,
            preexec_fn=lambda: os.nice(10),
        )
    except subprocess.TimeoutExpired as exc:
        raise HeroVideoError(f"LTX generation timed out after 6h: {exc}") from exc

    result = None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith('{"path"'):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                pass
    if proc.returncode != 0 or result is None:
        tail = (proc.stderr or "")[-2000:]
        raise HeroVideoError(
            f"LTX generation failed (rc={proc.returncode}). stderr tail:\n{tail}"
        )

    path = Path(result["path"])
    if not path.exists() or path.stat().st_size == 0:
        raise HeroVideoError(f"LTX worker reported success but {path} is missing/empty")
    wall_total = time.time() - t0
    timings: dict[str, float] = {"wall_sec": round(wall_total, 1)}
    for key, dest_key in (("gen_sec", "gen_sec"), ("wall_sec", "worker_wall_sec")):
        value = result.get(key)
        if isinstance(value, (int, float)):
            timings[dest_key] = value
    qc_extra: dict | None = None
    if run_qc:
        verdict = qc_hero_clip(path)
        qc_extra = {
            "qc": {
                "ok": verdict.get("ok", False),
                "reasons": list(verdict.get("reasons", [])),
                "scores": dict(verdict.get("scores", {})),
                "cut_policy": "hard_fail",
            }
        }
        log.info("hero clip QC ok: %s", path)
    record_sidecar(
        path,
        prompt=prompt,
        seed=seed,
        model=LTX25_MODEL_ID if resolved == VARIANT_LTX25 else "ltx2",
        quant=LTX25_QUANT if resolved == VARIANT_LTX25 else "bf16",
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        steps=steps,
        variant=resolved,
        guidance=guidance_scale,
        timings=timings,
        backend=f"generate_ltx2.py via {BACKEND_VENV}",
        extra=qc_extra,
    )
    log.info("LTX clip done in %.1fs: %s", wall_total, path)
    return path
