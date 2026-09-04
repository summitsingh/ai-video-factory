# AI Video Factory Phase 1 Design

## Goal

Create a local, deterministic foundation for an AI-assisted video-production pipeline on the AMD Strix Halo workstation. Codex performs the initial setup; Hermes becomes the long-running local orchestrator after the stages are verified. Phase 1 makes no system changes, model downloads, credential changes, external uploads, or publications.

## Scope

Phase 1 creates a Git-managed project that can report its environment and safely exercise a synthetic video workflow. It does not acquire external content, invoke local generative models, automate a browser, or interact with YouTube.

## Architecture

The project is a Python command-line application with a small TypeScript/Remotion composition. Hermes invokes the same stable CLI that a human or Codex uses; it does not bypass validation or directly manipulate internal files. Every command creates a run directory containing an immutable manifest, structured JSONL event log, and stage output. Stages use content-addressed input fingerprints so a successful stage may be resumed only when its declared inputs are unchanged.

The initial execution path is deliberately narrow:

```text
doctor → synthetic project manifest → Remotion render → FFmpeg validation → QC report
```

No language, speech, image, vision, or video-generation model is required for this path. Future services consume and emit schema-validated artifacts under `data/projects/<project-id>/`.

## Components

| Component | Responsibility | Inputs | Outputs |
|---|---|---|---|
| `doctor` | Collect safe system and tool status | host commands | JSON and Markdown diagnostic report |
| `run` core | Create/locate resumable runs and emit structured events | stage name, inputs | manifest and JSONL log |
| schemas | Validate stage manifests and edit decisions | JSON | typed Python objects/errors |
| Remotion fixture | Deterministic short video with captions and title card | fixture JSON | MP4 render |
| FFmpeg validator | Verify decode, streams, duration, dimensions | MP4 | JSON validation result |
| QC fixture | Aggregate technical checks; never publishes | validation output | `qc_report.json` and Markdown |

## Automation ownership

- Codex bootstraps, tests, and documents the installation.
- Hermes runs the finished local workflow through versioned CLI commands and machine-readable responses.
- Conventional tools perform deterministic work: FFmpeg processes media, Remotion composes frames, schemas validate artifacts, and SQLite indexes state.
- Local models are replaceable workers behind explicit adapters. Hermes selects a role or stage, not a hard-coded model file.
- Every externally consequential action remains behind a policy gate. Initial publication is absent; later YouTube support must progress from dry run to private upload and requires explicit approval before public publishing.

## Data and provenance rules

- Store source inputs, generated assets, commands, tool versions, checksums, and timestamps in each run manifest.
- Store non-secret configuration in `config/`; keep secrets only in a local `.env` ignored by Git.
- Do not treat a web URL as reusable media without licensing/provenance capture. Asset acquisition is out of scope for Phase 1.
- Never write credentials, absolute secrets, OAuth data, or tokens to logs.

## Exact initial stack

- Python 3.12 project runtime via `uv` or an isolated virtual environment; do not rely on system Python 3.14 compatibility for future ML packages.
- Python: `typer`, `pydantic`, `jsonschema`, `pytest`, and a JSONL logger.
- Node 22 with a pinned Remotion release and TypeScript.
- Existing FFmpeg 8 for probing and software rendering validation. Hardware encoding is a later opt-in benchmark.
- SQLite for future project/run indexes; no Docker in Phase 1.
- llama.cpp/whisper.cpp are not installed or built until Vulkan or ROCm passes a direct-device compatibility gate.

## Runtime strategy

Vulkan is the first candidate backend. Its viability must be verified in an interactive environment that can access `/dev/dri`, not only inside the restricted audit session. HIP/ROCm is a second candidate and remains blocked on a support and package-coherence review. If neither GPU route passes, the core pipeline remains functional with CPU rendering and CPU inference only.

## Failure and safety behavior

- A failed stage records an error event and leaves successful upstream outputs intact.
- Resume is denied if input checksums or configuration fingerprints differ.
- Doctor marks unavailable tools as `not_ready`; it does not attempt installation or repair.
- QC failures block any later publishing stage by contract. There is no publishing command in Phase 1.

## Verification

- Unit tests cover run identity, resume eligibility, schema rejection, and FFmpeg-report parsing.
- `doctor` runs without elevated privileges and produces a machine-readable report.
- The synthetic fixture renders a short MP4 deterministically using Remotion and validates it with FFmpeg.
- `benchmark` reports only local tool/backend availability in Phase 1; it does not download models.
- A Hermes-facing command contract returns structured status, run ID, artifact paths, and retryability without requiring Hermes to parse human-oriented terminal output.

## Non-goals

- ROCm installation or driver changes.
- Downloading models or large media assets.
- ComfyUI, TTS, Whisper, local LLM inference, research, asset acquisition, browser automation, OAuth, upload, or public release.
