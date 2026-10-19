"""Worker: generate one LTX hero clip.

Runs inside the .venv-ltx2 environment (TheRock torch + diffusers).

Variants:
  ltx2.5 (default) -- LTX-2.5 22B distilled int8_convrot flat checkpoints
      from the DeepBeepMeep/LTX-2 mirror (see scripts/download_ltx2.py).
      This worker validates the weight set and reports the resolved
      manifest; actual inference of these flat int8 checkpoints runs
      through the WanGP-headless backend (L1b follow-up), so a generation
      attempt currently exits with a clear error instead of a cryptic
      stack trace. --validate-only exits 0 after validation (no GPU).
  ltx2 -- legacy LTX-2 (2.0) diffusers layout via diffusers LTX2Pipeline.

On success prints a single JSON object to stdout:
  {"path": ..., "wall_sec": ..., "gen_sec": ..., "num_frames": ...,
   "width": ..., "height": ..., "variant": ..., "quant": ...}
Anything else on stdout/stderr is progress noise; the caller parses the
line starting with '{"path"'.
"""
import argparse
import json
import sys
import time
from pathlib import Path

# scripts/ is not a package; import the download manifest from next door.
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_NEGATIVE = (
    "shaky, glitchy, low quality, worst quality, deformed, distorted, "
    "disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, "
    "weird hand, ugly, transition, static."
)

L1B_NOTE = (
    "direct diffusers inference of the LTX-2.5 flat int8_convrot checkpoints "
    "is not supported (22B arch + Gemma4 text encoder + int8_convrot need the "
    "WanGP loader); the execution path is the WanGP-headless backend "
    "(L1b follow-up). Weight files validated OK; nothing was generated."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["ltx2.5", "ltx2"], default="ltx2.5",
                   help="weight layout to use")
    p.add_argument("--quant", default=None,
                   help="quant label recorded in output (default per variant)")
    p.add_argument("--validate-only", action="store_true",
                   help="ltx2.5: validate the weight set and exit 0 (no GPU)")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    p.add_argument("--width", type=int, required=True)   # multiple of 32
    p.add_argument("--height", type=int, required=True)  # multiple of 32
    p.add_argument("--num-frames", type=int, required=True)  # 8n+1
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--crop-width", type=int, default=None)
    p.add_argument("--crop-height", type=int, default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _validate_ltx25(model_dir: Path) -> dict:
    """Check every required mirror file exists. Returns the manifest dict."""
    import download_ltx2

    missing = [
        rel for rel in download_ltx2.REQUIRED_FILES
        if not (model_dir / rel).is_file()
    ]
    manifest = {
        "variant": "ltx2.5",
        "quant": "int8_convrot",
        "model_dir": str(model_dir),
        "required": list(download_ltx2.REQUIRED_FILES),
        "missing": missing,
    }
    print(json.dumps({"manifest": manifest}), flush=True)
    return manifest


def _run_ltx25(args) -> int:
    model_dir = Path(args.model_dir)
    manifest = _validate_ltx25(model_dir)
    if manifest["missing"]:
        print(
            "[ltx2.5] missing weight files in "
            f"{model_dir}: {', '.join(manifest['missing'])}; "
            "run: python scripts/download_ltx2.py",
            file=sys.stderr,
        )
        return 2
    if args.validate_only:
        print("[ltx2.5] weight set OK", flush=True)
        return 0
    print(f"[ltx2.5] {L1B_NOTE}", file=sys.stderr)
    return 3


def _run_ltx2(args):
    from diffusers.pipelines.ltx2 import LTX2Pipeline
    from diffusers.utils import export_to_video

    import torch

    t0 = time.time()
    print(f"[ltx2] loading pipeline from {args.model_dir} ...", flush=True)
    pipe = LTX2Pipeline.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    # VAE tiling/slicing: decode of long 720p clips OOMs without it.
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.enable_sequential_cpu_offload(device="cuda")
    print(f"[ltx2] loaded in {time.time()-t0:.1f}s, generating "
          f"{args.num_frames}f {args.width}x{args.height} ...", flush=True)

    generator = None
    if args.seed is not None:
        generator = torch.Generator("cuda").manual_seed(args.seed)

    t1 = time.time()
    video, _audio = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        frame_rate=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        output_type="np",
        return_dict=False,
    )
    gen_sec = time.time() - t1
    frames = video[0]  # (F, H, W, 3) float in [0, 1]

    if args.crop_width and args.crop_height:
        _, h, w, _ = frames.shape
        cw, ch = args.crop_width, args.crop_height
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        frames = frames[:, y0:y0 + ch, x0:x0 + cw, :]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out), fps=args.fps)
    wall = time.time() - t0
    print(json.dumps({
        "path": str(out),
        "wall_sec": round(wall, 1),
        "gen_sec": round(gen_sec, 1),
        "num_frames": int(frames.shape[0]),
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "variant": "ltx2",
        "quant": args.quant or "bf16",
    }), flush=True)


def main() -> int:
    args = parse_args()
    if args.variant == "ltx2.5":
        return _run_ltx25(args)
    _run_ltx2(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
