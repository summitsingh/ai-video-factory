"""Local hero-shot video generation with LTX-2 (Lightricks).

Produces custom AI-generated video clips for scenes where real stock
footage cannot deliver: impossible camera moves, alien worlds, abstract
concepts, exact brand-matched visuals. Complements
:mod:`ai_video_factory.stock_providers` (Pexels/Pixabay/NASA): stock wins
when real camera footage exists; this module wins when it does not.

Backend: diffusers ``LTX2Pipeline`` running on the local AMD GPU via a
dedicated virtualenv (TheRock ROCm torch build). The main package never
imports torch/diffusers itself; generation is delegated to a worker script
(``scripts/generate_ltx2.py``) executed with the backend venv's python, so
machines without the backend get a clear error instead of an ImportError.

SETUP (verified 2026-09-20 on summit-amd, Strix Halo gfx1151, ROCm 7.2.4)
-----------------------------------------------------------------------
1. Create a python3.12 venv (NOT the repo venv, NOT system python 3.14)::

     python3.12 -m venv /home/summit/avf-work/item11-ltx2/.venv-ltx2

2. Install TheRock nightly torch for gfx1151 (self-contained wheel, installs
   its own ROCm 7.13 runtime into the venv, does not touch /opt/rocm)::

     .venv-ltx2/bin/pip install --pre \\
       --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \\
       torch torchaudio torchvision

   Installed torch 2.12.0a0+rocm7.13.0a20260411. ``torch.cuda.is_available()``
   is True and reports "Radeon 8060S Graphics", capability (11, 5): native
   gfx1151, no HSA_OVERRIDE_GFX_VERSION needed.

3. Install the diffusion stack::

     .venv-ltx2/bin/pip install diffusers transformers accelerate safetensors \\
       sentencepiece huggingface_hub imageio imageio-ffmpeg

   (diffusers 0.40.0: first release line with LTX2Pipeline.)

4. Download the weights (repo is NOT gated, no HF token needed)::

     python scripts/download_ltx2.py

   Fetches the diffusers-layout subset of ``Lightricks/LTX-2`` into
   ``models/LTX-2/`` (transformer/, text_encoder/, vae/, tokenizer/,
   scheduler/, audio_vae/, connectors/, vocoder/, latent_upsampler/).
   Skips the giant single-file variants at the repo root.

MODEL LOCATION
--------------
Default: ``/home/summit/avf-work/item11-ltx2/models/LTX-2`` (override with
``LTX2_MODEL_DIR``). Backend venv: ``/home/summit/avf-work/item11-ltx2/.venv-ltx2``
(override with ``LTX2_VENV``). Test clips: ``/home/summit/avf-work/item11-ltx2-tests/``
(override with ``HERO_CLIP_DIR``) - deliberately OUTSIDE the repo.

MEASURED PERFORMANCE (summit-amd, 2026-09-20)
--------------------------------------------
TBD - filled in after the first successful runs. Prior research estimate:
~7-12 min per 720p 8s clip.

VRAM/RAM: the box reports a 512MB VRAM carve-out but 137GB GTT; torch on this
APU allocates through GTT (unified system memory), so the 19B bf16
transformer (~38GB) + Gemma3 text encoder fit comfortably in the 122GB RAM.

KNOWN ISSUES
------------
* ``width``/``height`` must be multiples of 32 (pipeline hard-requires it);
  this module rounds UP to the next multiple and center-crops back to the
  requested size before writing the MP4.
* ``num_frames`` should be 8n+1; the module snaps ``duration*fps`` to that.
* Sequential CPU offload is used for safety on the APU; it costs speed.
* The Fermi Paradox CPU render may be running on this box - generation is
  launched niced; check ``uptime`` first.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]  # src/ai_video_factory/hero_video.py -> repo root

BACKEND_VENV = Path(os.environ.get("LTX2_VENV", "/home/summit/avf-work/item11-ltx2/.venv-ltx2"))
MODEL_DIR = Path(os.environ.get("LTX2_MODEL_DIR", "/home/summit/avf-work/item11-ltx2/models/LTX-2"))
WORKER = _REPO_ROOT / "scripts" / "generate_ltx2.py"
OUTPUT_DIR = Path(os.environ.get("HERO_CLIP_DIR", "/home/summit/avf-work/item11-ltx2-tests"))

FPS = 24.0
DEFAULT_STEPS = int(os.environ.get("LTX2_STEPS", "40"))


class HeroVideoError(RuntimeError):
    """Raised when the LTX-2 backend is missing, broken, or a run fails."""


def backend_available() -> tuple[bool, str]:
    """Check the LTX-2 backend. Returns (ok, human-readable detail)."""
    venv_python = BACKEND_VENV / "bin" / "python"
    if not venv_python.exists():
        return False, (
            f"backend venv python not found: {venv_python}. "
            "Install per the module docstring: TheRock torch for gfx1151 + "
            f"diffusers into {BACKEND_VENV}"
        )
    if not WORKER.exists():
        return False, f"worker script missing: {WORKER}"
    transformer_dir = MODEL_DIR / "transformer"
    if not transformer_dir.is_dir() or not any(transformer_dir.glob("*.safetensors")):
        return False, (
            f"LTX-2 weights not found under {MODEL_DIR}/transformer/. "
            "Run: python scripts/download_ltx2.py"
        )
    return True, f"backend OK (venv={BACKEND_VENV}, model={MODEL_DIR})"


def _snap32(n: int) -> int:
    return (n + 31) // 32 * 32


def generate_hero_clip(
    prompt: str,
    duration_sec: float = 8,
    width: int = 1280,
    height: int = 720,
    seed: int | None = None,
    num_inference_steps: int = DEFAULT_STEPS,
    fps: float = FPS,
) -> Path:
    """Generate a hero video clip with LTX-2 and return the MP4 path.

    The clip is rendered at the next multiple-of-32 above ``width``/``height``
    (a pipeline requirement) and center-cropped back to the requested size,
    so the returned file is exactly ``width``x``height``.

    Raises:
        HeroVideoError: if the backend is not installed or generation fails.
    """
    ok, detail = backend_available()
    if not ok:
        raise HeroVideoError(f"LTX-2 backend unavailable: {detail}")

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
        "--model-dir", str(MODEL_DIR),
        "--prompt", prompt,
        "--width", str(gen_w),
        "--height", str(gen_h),
        "--num-frames", str(num_frames),
        "--fps", str(fps),
        "--steps", str(num_inference_steps),
        "--out", str(out),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if (gen_w, gen_h) != (width, height):
        cmd += ["--crop-width", str(width), "--crop-height", str(height)]

    log.info(
        "LTX-2 generating %ss %dx%d (%d frames, %d steps) -> %s",
        duration_sec, width, height, num_frames, num_inference_steps, out,
    )
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=6 * 3600,
            preexec_fn=lambda: os.nice(10),
        )
    except subprocess.TimeoutExpired as exc:
        raise HeroVideoError(f"LTX-2 generation timed out after 6h: {exc}") from exc

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
            f"LTX-2 generation failed (rc={proc.returncode}). stderr tail:\n{tail}"
        )

    path = Path(result["path"])
    if not path.exists() or path.stat().st_size == 0:
        raise HeroVideoError(f"LTX-2 worker reported success but {path} is missing/empty")
    log.info("LTX-2 clip done in %.1fs: %s", time.time() - t0, path)
    return path
