# L3: Memory-profile streaming experiment (measure first)

Date: 2026-10-18. Machine: summit-amd (Ryzen AI Max 395 / Radeon 8060S,
30 GB RAM, unified memory: torch "cuda" allocates via GTT).
Script: `scripts/measure_memory_profile.py` (rerunnable, read-only toward
the checkpoint and the machine).

## Question

WanGP's mmgp Profile 4 streams model layers GPU<->RAM instead of the
diffusers sequential CPU offload ai-video-factory's hero worker uses
(`scripts/generate_ltx2.py`: `LTX2Pipeline.enable_sequential_cpu_offload`).
For a ~20 GB transformer, does pipelined layer streaming clearly win?
Measure before adopting.

## What was actually measured

The real checkpoint (`/home/summit/wangp/ckpts/`,
`ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors`) is
**19.48 GB, 48 transformer blocks x 388.1 MB**, int8_convrot (mmgp
quantized format). It is **not diffusers-loadable**, so no real-model
forward pass was possible. Two-part measurement instead:

- **Phase 1 (real file, read-only):** stream all 48 blocks' tensors
  host->device from the memory-mapped safetensors, timed. Cold run:
  **19.48 GB in 59.4 s = 0.33 GB/s** (includes mmap page-in and tensor
  materialization; an earlier page-cache-warm pass hit 0.93 GB/s).
  This is the honest "model-load via streaming" number.
- **Phase 2 (proxy forward, real matmuls):** 8 synthetic transformer
  blocks sized to match (402.7 MB each, 12x Linear(4096) + GELU, bf16),
  5 timed denoising-style steps after warmup, on the 8060S:
  - **A. sequential offload** (current approach, faithfully modeled:
    per block per step: `.to(device)` -> forward -> `.to("cpu")`,
    fully synchronous): **1.227 s/step**, peak device 0.49 GB.
  - **B. Profile-4-style pipelined streaming** (own reimplementation:
    two GPU-resident slot modules; block i+1's parameters streamed
    host->device on a separate CUDA stream while block i computes;
    weights never copied back since inference is read-only):
    **0.083 s/step**, peak device 0.89 GB.
  - **Speedup: 14.8x.** Host RSS identical (18.37 GB, proxy weights
    live on CPU in both).

## Why streaming wins here

Sequential offload pays **two** synchronous copies per block per step
(H2D before, D2H write-back after) plus zero overlap. Streaming
eliminates the write-back entirely (weights are read-only during
inference) and overlaps the H2D copy of block i+1 with the compute of
block i. On this unified-memory APU the H2D copy is just a fast memcpy,
so the overlapped cost nearly disappears; the D2H write-back in the
sequential path is pure waste.

## Decision: document, do NOT implement in the hero worker now

The win is real and large, but there is no integration point yet:

1. The hero worker (`hero_video.py` / `scripts/generate_ltx2.py`) runs
   diffusers `LTX2Pipeline` on **LTX-2 2.0 bf16** weights (not the 2.5
   int8 checkpoint measured here). Adopting streaming means a custom
   denoising loop around the transformer instead of
   `pipeline.enable_sequential_cpu_offload` - that work belongs to L1
   (hero backend / weight migration), not a standalone change now.
2. On this unified-memory APU an even simpler option may dominate both:
   torch "cuda" allocations are GTT-backed, so the full 19.5 GB model
   might just stay resident - no offload engineering at all. Untested;
   flagged as the first thing to try in L1 before building streaming.
3. The single cheapest improvement, independent of L1: never copy
   weights back to CPU during inference (the D2H write-back is the
   biggest term in the sequential path).

**Recommendation for L1:** when the hero worker moves to LTX-2.5, skip
naive sequential CPU offload. First try full GTT-resident; if memory
pressure forces offload, use pipelined layer streaming per the pattern
in `scripts/measure_memory_profile.py` (two-slot prefetch, no
write-back), not accelerate's sequential offload.

## Caveats

- Proxy uses bf16 dense matmuls; real denoising adds attention,
  int8->bf16 dequantization, text encoder, and VAE (not modeled).
  The 14.8x is a mechanism ratio, not a promised end-to-end speedup.
- 8 proxy blocks, not 48; per-step time should scale ~linearly.
- No other heavy GPU workload was running concurrently; LM Studio
  serving state was not verified during the run.
- Phase-1 cold number (0.33 GB/s) is pessimistic; warm is ~3x faster.

## Raw results

`/tmp/l3-measure.json` on summit-amd (not committed); rerun with:

```
/home/summit/miniconda3/envs/wangp/bin/python \
    scripts/measure_memory_profile.py --blocks 8 --steps 5
```
