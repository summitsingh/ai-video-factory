# WanGP LTX-2.5 Benchmark (summit-amd)

Date: 2026-10-18. Machine: summit-amd (AMD Ryzen AI Max+ 395, Strix Halo gfx1151 RDNA 3.5 iGPU, 30GB RAM, Ubuntu).

## Install

- WanGP (deepbeepmeep/WanGP) cloned to `/home/summit/wangp` at commit `22d1f261`.
- Installed via the Linux AMD path in `setup_config.json` (the README's `docs/AMD-INSTALLATION.md` is Windows-only): miniconda Python 3.12.15 env, `torch 2.13.0+rocm10.0.0` targeting `device-gfx1151` from TheRock's stable wheel index. System Python is 3.14 with no python3.12 in apt, so conda was needed.
- Verified: `torch.cuda.is_available()` = True, device "AMD Radeon 8060S Graphics", `wgp.py --help` launches cleanly. No reboots, no GPU resets; GNOME/Hermes untouched.

## Model sourcing

- No Hugging Face token required. All LTX-2.5 files come from the public mirror `DeepBeepMeep/LTX-2` (not the gated Lightricks repos), downloaded anonymously. The previously 401'd HF values were never used.

## Benchmark results

- Model: LTX-2.5 22B Distilled, INT8 ConvRot checkpoint (19.5GB) + Gemma4-12b INT8 text encoder (12.9GB) + bf16 VAEs/vocoder/connectors, ~52GB total in `ckpts/`. INT8 backend auto-resolved to Comfy Kitchen HIP.
- Params: 768x512, 33 frames @ 24fps (1.38s), 8 steps, seed 42, Profile 4.
- Wall time: queue total 14m34s (included in-flight downloads + model load). Pure compute: denoising 8 steps ~4m11s (~31s/step); VAE decode 33 frames ~4m55s.
- RAM peaked ~23GB of 30GB; no OOM, no crash.
- Output: `docs/assets/ltx25-bench-768x512-33f.mp4` (687KB). Visual sanity verified: coherent mountain-lake-at-dawn scene matching the prompt, real inter-frame motion (mean abs diff 11.0), brightness 83-91 (not black). Genuine generation, not a still.

## Deviations from the planned spec

- `euler_a` is not exposed in WanGP's LTX-2.5 distilled path; it uses the native distilled scheduler.
- Manual CFG 1.0 is not exposed; the distilled pipeline runs its tuned guidance.
- The existing `/home/summit/video-bench` `LTX-2.5-Distilled-Q4_K_M.gguf` is NOT usable: WanGP's LTX-2.5 handler does not load GGUF checkpoints. The INT8 ConvRot safetensors is the functional equivalent and is now in place.

## Viability

Workable as the pipeline's short-clip local-model fallback tier: ~10-12 min per 1.4s clip at 768x512 with models cached, scaling roughly linearly with length. Pause LM Studio serving during generation; RAM headroom is thin (23/30GB). Not suitable for bulk throughput.

See also: full step-by-step install log in the project workspace (`goals/ai-video-factory-pipeline-rebuild/hidden_files/wangp-install-log.md`).
