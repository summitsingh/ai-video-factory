"""Measure memory-profile strategies for large transformer inference (L3).

Compares, on small cheap workloads (no full generation):

A. ``sequential`` - the current ai-video-factory approach: diffusers-style
   ``enable_sequential_cpu_offload`` semantics, reimplemented plainly: each
   transformer block is moved GPU -> forward -> back to CPU, synchronously,
   one block at a time.

B. ``streaming`` - WanGP mmgp Profile-4 style layer streaming, reimplemented:
   a two-slot GPU ring buffer; block i+1 is prefetched host->device on a
   separate CUDA stream while block i computes; weights are never copied
   back (inference is read-only).

Phase 1 streams the REAL LTX-2.5 int8 distilled checkpoint at
``/home/summit/wangp/ckpts/`` (read-only) host->device to measure actual
weight-transfer throughput (no forward pass: the mmgp int8_convrot format
is not diffusers-loadable, so no real-model inference is possible here).

Phase 2 runs a synthetic proxy forward pass: transformer blocks sized to
match the real checkpoint (~388 MB each, 48 blocks in the real model; the
proxy uses fewer blocks to stay quick) with real matmuls, measuring
per-step time and peak memory for both strategies.

Metrics: model-load/stream time, peak device memory, peak host RSS,
per-denoising-step time. Prints a table and writes a JSON sidecar.

Usage (needs torch with a CUDA/ROCm device, e.g. the WanGP conda env)::

    /home/summit/miniconda3/envs/wangp/bin/python \\
        scripts/measure_memory_profile.py --blocks 8 --steps 5

Read-only with respect to the checkpoint and the machine: no model files
are written, no processes are killed, nothing is rebooted.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import resource
import struct
import sys
import time
from pathlib import Path

CKPT_PATH = Path(
    "/home/summit/wangp/ckpts/"
    "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
)
BLOCK_KEY_RE = re.compile(r"transformer_blocks\.(\d+)\.")

# Proxy block shape: 12 x Linear(4096, 4096) in bf16 ~= 403 MB, close to the
# real 388 MB int8 block (transfer size is what matters for the comparison).
PROXY_HIDDEN = 4096
PROXY_LINEARS = 12
PROXY_SEQ = 256
PROXY_BATCH = 1


def _bytes_of(entry) -> int:
    n = 1
    for dim in entry["shape"]:
        n *= dim
    bpe = {"I8": 1, "U8": 1, "F16": 2, "BF16": 2, "F32": 4, "I32": 4}
    return n * bpe.get(entry["dtype"], 0)


def read_ckpt_header(path: Path) -> dict:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size))


def phase1_real_checkpoint_stream(device: str) -> dict:
    """Stream the real checkpoint's block tensors host->device, timed."""
    import torch
    from safetensors import safe_open

    header = read_ckpt_header(CKPT_PATH)
    items = [(k, v) for k, v in header.items() if k != "__metadata__"]
    blocks: dict[int, list[tuple[str, dict]]] = {}
    for key, val in items:
        match = BLOCK_KEY_RE.search(key)
        if match:
            blocks.setdefault(int(match.group(1)), []).append((key, val))
    block_ids = sorted(blocks)
    total_bytes = sum(_bytes_of(v) for _, v in items)
    block_bytes = sum(_bytes_of(v) for _, v in blocks[block_ids[0]])

    stream = torch.cuda.Stream(device=device)
    gpu_tensors = []
    torch.cuda.synchronize(device=device)
    started = time.perf_counter()
    with safe_open(CKPT_PATH, framework="pt", device="cpu") as handle:
        for block_id in block_ids:
            with torch.cuda.stream(stream):
                for key, _ in blocks[block_id]:
                    tensor = handle.get_tensor(key)
                    gpu_tensors.append(tensor.to(device, non_blocking=True))
            # keep only a sliding window resident to bound memory
            if len(gpu_tensors) > 400:
                del gpu_tensors[:200]
    torch.cuda.synchronize(device=device)
    elapsed = time.perf_counter() - started
    del gpu_tensors
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "checkpoint": str(CKPT_PATH),
        "total_gb": round(total_bytes / 1e9, 2),
        "num_blocks": len(block_ids),
        "mb_per_block": round(block_bytes / 1e6, 1),
        "stream_all_blocks_s": round(elapsed, 2),
        "effective_gbps": round(total_bytes / 1e9 / elapsed, 2),
    }


def make_proxy_block():
    import torch.nn as nn

    layers = []
    for _ in range(PROXY_LINEARS):
        layers.append(nn.Linear(PROXY_HIDDEN, PROXY_HIDDEN, bias=False))
        layers.append(nn.GELU())
    block = nn.Sequential(*layers).to(dtype=__import__("torch").bfloat16).eval()
    for param in block.parameters():
        param.requires_grad_(False)
    return block


def proxy_block_mb() -> float:
    import torch

    block = make_proxy_block()
    total = sum(p.numel() * torch.finfo(p.dtype).bits // 8 for p in block.parameters())
    return total / 1e6


def _copy_params_to_slot(cpu_block, gpu_slot) -> None:
    """Stream one block's parameters host->device into a GPU-resident slot.

    Must be called inside the desired ``torch.cuda.stream()`` context;
    the caller synchronizes before the slot is used.
    """
    for dst, src in zip(gpu_slot.parameters(), cpu_block.parameters()):
        dst.copy_(src, non_blocking=True)


def _host_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def phase2_proxy_benchmark(
    device: str, num_blocks: int, steps: int, warmup: int = 2
) -> dict:
    import torch

    results: dict = {
        "proxy_mb_per_block": round(proxy_block_mb(), 1),
        "num_blocks": num_blocks,
        "steps": steps,
        "device": torch.cuda.get_device_name(device),
    }
    activation_shape = (PROXY_BATCH, PROXY_SEQ, PROXY_HIDDEN)

    def fresh_blocks():
        return [make_proxy_block() for _ in range(num_blocks)]

    def fresh_activation():
        return torch.randn(
            *activation_shape, dtype=torch.bfloat16, device=device
        )

    # -- Strategy A: sequential CPU offload (current approach) --
    torch.cuda.reset_peak_memory_stats(device)
    blocks = fresh_blocks()
    step_times: list[float] = []
    torch.cuda.synchronize(device)
    for _ in range(warmup + steps):
        x = fresh_activation()
        started = time.perf_counter()
        with torch.inference_mode():
            for block in blocks:
                block.to(device)
                x = block(x)
                block.to("cpu")
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - started)
    results["sequential"] = {
        "step_s_mean": round(sum(step_times[warmup:]) / steps, 3),
        "step_s_min": round(min(step_times[warmup:]), 3),
        "peak_device_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
        "peak_host_rss_gb": round(_host_rss_gb(), 2),
    }
    del blocks, x
    torch.cuda.empty_cache()
    gc.collect()

    # -- Strategy B: Profile-4 style pipelined streaming --
    # Two GPU-resident slot modules; each step streams every block's
    # parameters host->device into a slot (the CPU blocks are never
    # mutated). Block i+1 is prefetched on a separate stream while block i
    # computes: transfer and compute overlap.
    torch.cuda.reset_peak_memory_stats(device)
    blocks = fresh_blocks()
    compute_stream = torch.cuda.current_stream(device)
    prefetch_stream = torch.cuda.Stream(device=device)
    slots = [make_proxy_block().to(device).eval() for _ in range(2)]
    step_times = []
    torch.cuda.synchronize(device)
    for _ in range(warmup + steps):
        x = fresh_activation()
        started = time.perf_counter()
        with torch.inference_mode():
            # prime: block 0 into slot 0, synchronously
            with torch.cuda.stream(prefetch_stream):
                _copy_params_to_slot(blocks[0], slots[0])
            compute_stream.wait_stream(prefetch_stream)
            for i in range(num_blocks):
                if i > 0:
                    # block i was prefetched during iteration i-1
                    compute_stream.wait_stream(prefetch_stream)
                if i + 1 < num_blocks:
                    # prefetch the next block while block i computes
                    with torch.cuda.stream(prefetch_stream):
                        _copy_params_to_slot(blocks[i + 1], slots[(i + 1) % 2])
                x = slots[i % 2](x)
            # slots are reused across steps; weights stay on the CPU blocks
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - started)
    results["streaming"] = {
        "step_s_mean": round(sum(step_times[warmup:]) / steps, 3),
        "step_s_min": round(min(step_times[warmup:]), 3),
        "peak_device_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
        "peak_host_rss_gb": round(_host_rss_gb(), 2),
    }
    del blocks, x, slots
    torch.cuda.empty_cache()
    gc.collect()

    seq = results["sequential"]["step_s_mean"]
    stream = results["streaming"]["step_s_mean"]
    results["speedup_streaming_vs_sequential"] = (
        round(seq / stream, 2) if stream > 0 else None
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: torch is required for measurement", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: no CUDA/ROCm device visible", file=sys.stderr)
        return 2
    try:
        import safetensors  # noqa: F401
    except ImportError:
        print("ERROR: safetensors is required for phase 1", file=sys.stderr)
        return 2

    device = "cuda"
    report: dict = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
    }
    print(f"device: {report['device']} (torch {report['torch']})", flush=True)

    if not args.skip_real:
        print("phase 1: streaming real LTX-2.5 int8 checkpoint (read-only)...", flush=True)
        report["phase1_real_checkpoint"] = phase1_real_checkpoint_stream(device)
        p1 = report["phase1_real_checkpoint"]
        print(
            f"  streamed {p1['total_gb']} GB / {p1['num_blocks']} blocks "
            f"({p1['mb_per_block']} MB/block) in {p1['stream_all_blocks_s']}s "
            f"-> {p1['effective_gbps']} GB/s",
            flush=True,
        )
    else:
        print("phase 1: skipped", flush=True)

    print(
        f"phase 2: proxy forward benchmark "
        f"({args.blocks} blocks, {args.steps} steps)...",
        flush=True,
    )
    report["phase2_proxy"] = phase2_proxy_benchmark(device, args.blocks, args.steps)
    p2 = report["phase2_proxy"]
    print(f"  proxy block size: {p2['proxy_mb_per_block']} MB", flush=True)
    for name in ("sequential", "streaming"):
        r = p2[name]
        print(
            f"  {name}: step {r['step_s_mean']}s "
            f"(peak device {r['peak_device_gb']} GB, host RSS {r['peak_host_rss_gb']} GB)",
            flush=True,
        )
    print(
        f"  streaming speedup vs sequential: "
        f"{p2['speedup_streaming_vs_sequential']}x",
        flush=True,
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
