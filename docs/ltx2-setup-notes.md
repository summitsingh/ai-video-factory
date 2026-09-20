# LTX-2 local video generation - setup notes (item 11)

Date: 2026-09-20. Machine: summit-amd (AMD Strix Halo, gfx1151 / Radeon 8060S,
122GB RAM, ROCm 7.2.4 under /opt/rocm). Branch: `work/item11-ltx2`.

## What works

**PyTorch on gfx1151 via TheRock nightlies.** The official PyTorch ROCm
wheels do not target gfx1151, but AMD's TheRock project publishes nightly
wheels built for it:

```
python3.12 -m venv .venv-ltx2
.venv-ltx2/bin/pip install --pre \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  torch torchaudio torchvision
```

Result: torch `2.12.0a0+rocm7.13.0a20260411`. `torch.cuda.is_available()` is
True, device name "Radeon 8060S Graphics", capability (11, 5). A bf16
4096x4096 matmul runs in ~0.3s. **No `HSA_OVERRIDE_GFX_VERSION` needed.**
The wheel is self-contained (installs its own ROCm 7.13 runtime into the
venv) and does not touch the host `/opt/rocm`.

Then: `.venv-ltx2/bin/pip install diffusers transformers accelerate safetensors
sentencepiece huggingface_hub imageio imageio-ffmpeg`
(diffusers 0.40.0 has `LTX2Pipeline`).

**Model:** `Lightricks/LTX-2` on HuggingFace is NOT gated (no token needed).
`scripts/download_ltx2.py` fetches the diffusers-layout subset
(transformer/, text_encoder/, vae/, tokenizer/, scheduler/, audio_vae/,
connectors/, vocoder/, model_index.json) into `models/LTX-2/`. It
deliberately skips: the giant single-file variants at the repo root,
the x2 upscalers, `latent_upsampler/` (2-stage pipeline only), and the
redundant `text_encoder/diffusion_pytorch_model-*` shard set (model_index
assigns the text encoder to transformers.Gemma3ForConditionalGeneration,
which only reads the `model-*` shards).

**Memory model on this APU:** `rocm-smi` reports a 512MB VRAM carve-out but
137GB GTT. torch allocates through GTT (unified system memory), so the 19B
bf16 transformer (~38GB) plus the Gemma3 text encoder fit in the 122GB RAM.

## Usage

```python
import sys
sys.path.insert(0, "/home/summit/avf-work/item11-ltx2/src")
from ai_video_factory.hero_video import generate_hero_clip
path = generate_hero_clip("a red hot air balloon over green hills at sunrise",
                          duration_sec=8, width=1280, height=720, seed=7)
```

or `python scripts/test_hero_clips.py [smoke|full]`. Test clips land in
`/home/summit/avf-work/item11-ltx2-tests/` (outside the repo, per plan).

`generate_hero_clip` raises `HeroVideoError` (a RuntimeError) with install
instructions when the venv or the weights are missing.

## Constraints / quirks baked into the code

* `width`/`height` MUST be multiples of 32 (pipeline raises otherwise).
  `generate_hero_clip` rounds up and center-crops back to the requested size.
* `num_frames` should be 8n+1; the module snaps `duration*fps` to that.
* The worker enables `pipe.vae.enable_tiling()` + `enable_slicing()` and
  `enable_sequential_cpu_offload("cuda")` for safety on the APU.
* Generation subprocesses are launched with `nice(10)` so the Fermi Paradox
  CPU render (if running) is not starved.

## Lessons learned (for the next attempt)

* Never run long `amd-ssh` jobs as plain background `exec`: when the SSH
  session drops, the remote process dies with it. Always
  `setsid nohup ... > log 2>&1 < /dev/null &` and poll the log file.
* `python3 -m pip` may exist where `bin/pip` does not (ai-video-factory venv).
  System python3 (3.14) has no pip at all on this box.
* stdin redirection through `amd-ssh` works, but `< file` binds to the LAST
  command in an `&&` chain - group it or run it alone.
* Unauthenticated HF downloads are rate-limited but usable (~0.5-1 GB/min
  observed); the full subset is ~60-65GB, so budget ~1h for the download.

## Status / resume steps

[updated at commit time]
