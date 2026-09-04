# AI Video Factory — System Audit

**Audited:** 2026-09-04  
**Scope:** read-only inspection; no packages, drivers, models, credentials, or external services were changed.

## Summary

This workstation has the CPU, unified memory, storage, and installed editing prerequisites for a local AI-video workflow. The AMD display driver is active, and the user belongs to the `render` and `video` groups. The inference runtime is not yet ready: `rocminfo` and HIP tooling are absent, while AMD packages built for Ubuntu 24.04 are installed on an Ubuntu 26.04 host. Do not alter the GPU stack until a supported ROCm path for this exact host is confirmed.

## Hardware

| Component | Observed state |
|---|---|
| CPU | AMD Ryzen AI Max+ 395, 16 cores / 32 threads |
| GPU | AMD Strix Halo / Radeon 8060S-class, PCI ID `1002:1586` |
| Kernel driver | `amdgpu` loaded and in use |
| Memory | 122 GiB total; 115 GiB available during audit |
| Swap | 8 GiB total; 6.9 GiB free during audit |
| System disk | 1.8 TB NVMe; 458 GiB free on `/` |

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

## Compatibility decision

1. Treat Vulkan as the first backend to test because the host has Mesa/Radeon Vulkan packages and it does not require completing ROCm.
2. Do not install or repair ROCm yet. First capture a direct desktop `vulkaninfo --summary` and `/dev/dri` result, then confirm a supported ROCm release for Ubuntu 26.04 and this GPU.
3. Build llama.cpp and whisper.cpp only after the relevant backend passes a small smoke test. Their upstream documentation supports both Vulkan and HIP builds.

## Capacity budget

Keep at least 120 GB free for the operating system and ordinary user work. With approximately 458 GB currently available, use the remaining space conservatively:

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

