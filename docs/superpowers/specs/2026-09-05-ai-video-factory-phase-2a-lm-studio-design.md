# AI Video Factory Phase 2A: LM Studio Inference Design

**Date:** 2026-09-05

## Goal

Add a safe, deterministic local-inference boundary to the existing AI Video
Factory. LM Studio owns model serving and GPU execution; the factory owns
validation, lifecycle commands, structured results, provenance, and the stable
contract that Hermes will call in Phase 2B.

## Confirmed host state

- LM Studio CLI is installed at `/home/summit/.lmstudio/bin/lms`.
- The local server uses `127.0.0.1:1234` and exposes OpenAI-compatible APIs.
- The newest installed LM Studio engine is
  `llama.cpp-linux-x86_64-amd-rocm-avx2@2.31.2`; older installed versions are
  also present. The CLI did not mark an explicit selection during the audit,
  so the adapter records the engine actually used after model load without
  changing LM Studio's runtime selection.
- LM Studio's bundled runtime survey detects `85.67 GiB` of GPU-accessible
  memory on the AMD Strix Halo and `122.69 GiB` of system RAM. This does not
  establish compatibility for a separate system ROCm installation.
- No model was loaded when this design was prepared.
- Existing LM Studio models occupy about `922 GiB`; the filesystem has about
  `457 GiB` free.
- Both selected candidates are already present. No model download is needed.

## Model selection

The primary execution model is the installed
`qwen3.6-35b-a3b-udt-mtp` GGUF. LM Studio reports that it is trained for tool
use, supports vision, and has a maximum context length of 262,144 tokens. Its
current model file occupies about 18.64 GB.

The installed `gemma-4-26b-a4b-it` GGUF is the fallback and future
multimodal/director candidate. It remains unloaded unless an operator selects
it in a later approved change.

Initial Qwen load settings are:

- stable API identifier: `avf-qwen36-executor`
- GPU offload: `max`
- context length: `65,536`
- parallel requests: `1`
- idle TTL: `3,600` seconds

LM Studio's low-confidence estimate at these settings is `17.36 GiB` total,
compared with `17.49 GiB` for Gemma. The factory must run `--estimate-only`
before every first load after a configuration or runtime change. It must refuse
to load if LM Studio rejects the estimate or if available system memory is
below 40 GiB. It records the estimate but does not pretend it is an exact
allocation measurement.

MTP speculative decoding is disabled for the baseline. It may be enabled only
after ordinary generation and tool calling are correct and a separate benchmark
shows a reproducible benefit.

## Architecture

`LmStudioBackend` is an adapter behind a backend-neutral inference service.
It invokes the installed `lms` binary with fixed argument arrays and calls only
loopback HTTP endpoints. The rest of the factory consumes typed request and
result models and never imports LM Studio internals or manipulates model files.

Checked-in `config/inference.toml` defines the backend, loopback base URL, model
key, identifier, context, offload, parallelism, TTL, and safety thresholds.
Python's standard `tomllib` reads it, so no YAML or OpenAI SDK dependency is
introduced. Runtime state and reports remain beneath the existing project data
and log roots; model weights remain in LM Studio's existing model directory.

The factory adds these machine-readable commands:

- `inference doctor` inspects the CLI, selected engine, Vulkan survey, server,
  configured model presence, and loaded-model state. It never loads a model.
- `inference estimate` runs LM Studio's estimate-only operation and applies the
  memory gate. It never loads a model.
- `inference start` verifies the estimate, starts the loopback server if
  necessary, and loads only the configured model under the stable identifier.
- `inference status` reports server and configured-model state without changing
  model residency.
- `inference benchmark` requires the configured model to be loaded, then tests
  ordinary generation, strict JSON output, and one deterministic tool call.
- `inference stop` unloads only `avf-qwen36-executor`. It never uses
  `lms unload --all` and never stops the shared LM Studio server.

Every command returns versioned JSON with status, retryability, backend,
model identifier, runtime identity, timings, sanitized errors, and relevant
artifacts. Human-readable output is optional and derived from the same result.

## Data flow

1. The operator or Hermes invokes a supported factory CLI command.
2. The factory validates `inference.toml` and resolves the exact `lms` binary.
3. Read-only commands inspect LM Studio and the loopback API.
4. `start` runs the memory gate, then performs bounded server/model lifecycle
   operations with timeouts and postcondition checks.
5. `benchmark` sends deterministic requests to
   `http://127.0.0.1:1234/v1`, validates the response schema, and writes a
   structured report with sanitized diagnostics.
6. A successful benchmark produces a backend capability record that Phase 2B
   can require before configuring Hermes.

## Provenance and resumability

The capability fingerprint includes the factory configuration, LM Studio CLI
version, selected runtime version, AMD device identity, model key, resolved
model path metadata, model size, and a cached SHA-256 computed without modifying
the model. The cached digest is reused only while path, size, and modification
time match.

Benchmark results are reusable only when the capability fingerprint and test
request corpus are unchanged. Failed or interrupted lifecycle operations write
sanitized structured events and remain retryable where safe. No credential,
prompt response, or unsanitized subprocess output crosses a persistence
boundary.

## Safety boundaries

- No CUDA, ROCm, Mesa, Vulkan, kernel, firmware, or Ubuntu package changes.
- No llama.cpp or whisper.cpp installation.
- No model download, conversion, deletion, relocation, or modification.
- No change to LM Studio's selected runtime.
- Bind and call only `127.0.0.1`; never enable LAN serving or CORS.
- Never enter, request, or persist an API key for the loopback endpoint.
- Never unload unrelated models or stop the shared LM Studio service.
- Never start inference implicitly from `doctor`, the synthetic pipeline, or
  import-time code.
- Hermes installation/configuration, TTS, Whisper, image generation, external
  research, browser automation, credentials, uploads, and publication remain
  outside Phase 2A.

## Failure handling

- Missing or incompatible `lms`, model, runtime, AMD device, or loopback API
  produces `not_ready`, not an attempted repair.
- Estimate or free-memory guard failure prevents loading and reports the exact
  safe next action.
- Startup must verify both the stable model identifier and API visibility; a
  partially completed load is not success.
- Timeouts and subprocess/API errors are sanitized through the existing shared
  diagnostic boundary.
- `stop` is idempotent for the configured identifier and cannot target a model
  selected only by a fuzzy name.
- Existing loaded user models are treated as external state and left unchanged.

## Verification

Automated tests use fake subprocess and loopback clients to cover configuration,
fixed argv, timeouts, malformed JSON, diagnostic redaction, memory gates,
idempotent lifecycle behavior, unrelated-model preservation, response parsing,
capability fingerprints, and resume invalidation.

The controlled live verification order is:

1. `inference doctor`
2. `inference estimate`
3. `inference start`
4. `inference status`
5. `inference benchmark`
6. existing `doctor`, `benchmark`, and `test-pipeline --json`
7. `inference stop`
8. verify no configured model remains loaded and unrelated models are unchanged

The live benchmark passes only if LM Studio reports the AMD GPU device, the
configured Qwen model answers a deterministic prompt, strict JSON validates,
and a declared tool is selected with schema-valid arguments.

## Phase 2B handoff

Phase 2B may install Hermes locally only after Phase 2A emits a passing
capability record. Hermes will use `http://127.0.0.1:1234/v1` as a custom
OpenAI-compatible provider, omit API credentials, select
`avf-qwen36-executor`, and invoke only the factory's supported CLI contract.
Hermes will not receive direct permission to download models, alter LM Studio
runtimes, manage unrelated models, upload content, or publish videos.
