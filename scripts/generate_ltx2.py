"""Worker: generate one LTX-2 text-to-video clip.

Runs inside the .venv-ltx2 environment (TheRock torch + diffusers).
Prints a single JSON object to stdout on success:
  {"path": ..., "wall_sec": ..., "num_frames": ..., "width": ..., "height": ...}
Anything else on stdout/stderr is progress noise; the caller parses the
last line starting with "{\\"path\\"".
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

DEFAULT_NEGATIVE = (
    "shaky, glitchy, low quality, worst quality, deformed, distorted, "
    "disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, "
    "weird hand, ugly, transition, static."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    p.add_argument("--width", type=int, required=True)   # multiple of 32
    p.add_argument("--height", type=int, required=True)  # multiple of 32
    p.add_argument("--num-frames", type=int, required=True)  # 8n+1
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--crop-width", type=int, default=None)
    p.add_argument("--crop-height", type=int, default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    from diffusers.pipelines.ltx2 import LTX2Pipeline
    from diffusers.utils import export_to_video

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
    }), flush=True)


if __name__ == "__main__":
    main()
