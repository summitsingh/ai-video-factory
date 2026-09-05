# AI Video Factory — System Audit

**Audited:** 2026-09-04; LM Studio follow-up 2026-09-05
**Scope:** host inspection plus a controlled, temporary local model lifecycle;
no packages, drivers, runtimes, model files, credentials, or external services
were installed, downloaded, selected, updated, or changed.

## Summary

This workstation has the CPU, unified memory, storage, installed editing
prerequisites, and an LM Studio-bundled AMD runtime family needed for controlled
local-inference verification. The AMD display driver is active, and the user
belongs to the `render` and `video` groups. This does not establish that the
host's separate system ROCm stack works: `rocminfo` and HIP tooling remain
absent and unchanged. Do not alter that GPU stack until a supported ROCm path
for this exact host is confirmed.

## Hardware

| Component | Observed state |
|---|---|
| CPU | AMD Ryzen AI Max+ 395, 16 cores / 32 threads |
| GPU | AMD Strix Halo / Radeon 8060S-class, PCI ID `1002:1586` |
| Kernel driver | `amdgpu` loaded and in use |
| Memory | 122 GiB total; 115 GiB available during audit |
| Swap | 8 GiB total; 6.9 GiB free during audit |
| System disk | 1.8 TB NVMe; approximately 457 GiB free on `/` during the LM Studio audit |

## Operating system and development tools

| Component | Observed state |
|---|---|
| OS | Ubuntu 26.04.1 LTS |
| Kernel | 7.0.0-30-generic |
| Git | 2.53.0 |
| Python | 3.14.4 |
| Node / npm | 22.23.1 / 10.9.8 |
| CMake | 4.2.3 |
| GCC | 15.2.0 |
| FFmpeg | 8.0.1 |
| Chrome | 152.0.7977.64 |
| Docker / Compose | not installed |
| Ninja | not installed |

## GPU runtime status

- `rocm-smi` sees device `0x1586`, but reports an internal `map::at` exception and does not provide a reliable Strix Halo memory reading.
- `rocminfo` is absent. HIP runtime libraries and compiler tooling were not found. ROCm compute is therefore **not verified**.
- The installed AMD package set includes packages labelled `24.04` although the host is Ubuntu 26.04. This mixed state is a change-control risk.
- Mesa Vulkan tools and the Radeon ICD are installed. The automated audit session has no `/dev/dri` device and no X server, so its Vulkan probe cannot establish desktop GPU compatibility. This is an environment limitation of the audit session, not evidence that the interactive desktop stack is broken.
- FFmpeg advertises `vaapi`, `drm`, `opencl`, and `vulkan` acceleration plus H.264, HEVC, AV1 VAAPI/Vulkan encoders. Actual encode behavior remains unverified until a controlled local test can access `/dev/dri`.

## LM Studio inference audit

| Item | Audited state |
|---|---|
| CLI | canonical `/home/summit/.lmstudio/bin/lms`; authoritative commit `07b7252` |
| Bundled runtime families | AMD ROCm AVX2 is installed through `2.31.2`; Vulkan AVX2 is installed through `2.33.0` and was selected during the passing capability run; no runtime selection or update was performed |
| Server boundary | OpenAI-compatible API bound only to `127.0.0.1:1234` (`http://127.0.0.1:1234/v1`); persisted config safety fields are port `1234`, `networkInterface` `127.0.0.1`, and CORS disabled |
| Bundled survey | 85.67 GiB GPU-accessible memory and 122.69 GiB system RAM |
| Primary existing model | `qwen3.6-35b-a3b-udt-mtp`; primary GGUF 17,741,611,328 bytes plus direct `mmproj` companion 899,283,584 bytes, exactly matching LM Studio's 18,640,894,912-byte resource total; stable API identifier `avf-qwen36-executor` |
| Primary load estimate | 17.36 GiB total at 65,536 context, maximum GPU offload, parallelism 1; LM Studio confidence `LOW` |
| Existing fallback | `gemma-4-26b-a4b-it`; estimated 17.49 GiB total; not selected or loaded by Phase 2A |
| Existing LM Studio storage | Approximately 922 GiB of models; no model was downloaded, moved, converted, edited, or deleted |
| Free filesystem capacity | Approximately 457 GiB during the LM Studio audit |

LM Studio's runtime-family names describe bundled backends. They are not
evidence that the separate host ROCm installation is valid. System ROCm remains
unverified and was not repaired, installed, removed, or otherwise changed.

### Controlled capability verification

The hardened v4 capability run `ab73ed9f596e467cae6718161b59d1a4` passed ordinary
generation, strict structured output, and an exact `record_scene` tool call
through `http://127.0.0.1:1234/v1`. Its integrity-checked report is stored at
`data/projects/system/runs/ab73ed9f596e467cae6718161b59d1a4/inference_report.json`.
The measured probe latencies were 2,219.804 ms, 5,043.842 ms, and 2,823.761 ms,
respectively. Its fingerprint includes the canonical CLI path, commit, runtime,
survey, strict server-config path, primary-file digest, and exact package-size
relationship but persists no prompts, responses, reasoning, or raw HTTP bodies.

The loaded-model identifier set was empty before the controlled start and empty
again after the targeted unload. Only `avf-qwen36-executor` was loaded and
unloaded; the shared loopback server remained running. The existing factory
doctor and benchmark exited successfully, and the synthetic pipeline returned
`status: pass`. The doctor continues to report the separate system ROCm probe as
not ready, as expected and unchanged.

This passing, provenance-verified report satisfies the local-inference
capability prerequisite for Phase 2B on the audited host. Phase 2B must still
revalidate report provenance before use. Hermes was not installed or configured
as part of this work.

## Compatibility decision

1. Treat Vulkan as the first backend to test because the host has Mesa/Radeon Vulkan packages and it does not require completing ROCm.
2. Do not install or repair ROCm yet. First capture a direct desktop `vulkaninfo --summary` and `/dev/dri` result, then confirm a supported ROCm release for Ubuntu 26.04 and this GPU.
3. Build llama.cpp and whisper.cpp only after the relevant backend passes a small smoke test. Their upstream documentation supports both Vulkan and HIP builds.

## Capacity budget

Keep at least 120 GB free for the operating system and ordinary user work. With approximately 457 GiB available during the LM Studio audit, use the remaining space conservatively:

| Use | Initial budget |
|---|---:|
| Source, project environments, logs, tests | 5–10 GB |
| One pilot GGUF model | 20–30 GB maximum |
| Local TTS / transcription models | 1–5 GB |
| Image-generation models and cache | 10–30 GB |
| Active video project's assets and renders | 30–80 GB |
| Optional video-generation experiment | 30–80 GB, only when separately approved |

Do not store several large language models or video-generation checkpoints concurrently in the first release.

## Changes requiring explicit approval

- Installing, removing, or changing AMD GPU, Mesa, ROCm, kernel, or firmware packages.
- Any command requiring `sudo`.
- Downloading a large model or media checkpoint (over 5 GB).
- Installing Docker, exposing a service on the network, or adding persistent system services.
- Entering OAuth/API credentials, downloading external assets, uploading, or publishing.
