# AI Video Factory Phase 2A LM Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guarded, resumable LM Studio inference backend using the existing Qwen 3.6 35B-A3B model and expose deterministic status, lifecycle, and capability-benchmark commands for Hermes.

**Architecture:** A strict TOML configuration and typed public result contract feed an `LmStudioBackend` that owns fixed-argument CLI and loopback HTTP operations. A separate service applies memory and lifecycle policy, computes cached model provenance, and persists response-free benchmark evidence through the existing `RunStore`; Typer exposes only the supported boundary.

**Tech Stack:** Python 3.12, Pydantic 2, Typer, standard-library `tomllib` and `urllib`, LM Studio `lms`, existing `RunStore`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-ai-video-factory-phase-2a-lm-studio-design.md`

## Global Constraints

- Work directly on `main` with repository-local Git identity `Summit Singh Thakur <400621+summitsingh@users.noreply.github.com>`.
- Use the installed `qwen3.6-35b-a3b-udt-mtp` model; do not download, move, convert, edit, or delete model weights.
- Use LM Studio's existing runtime selection; do not install or update llama.cpp, LM Studio runtimes, ROCm, CUDA, Mesa, Vulkan, kernel, firmware, or Ubuntu packages.
- Bind and call only `127.0.0.1:1234`; never enable LAN serving or CORS.
- Never use `lms unload --all`, stop the shared LM Studio server, or alter unrelated loaded models.
- Do not install Hermes, Whisper, TTS, image generation, browser automation, credentials, upload support, or publishing in Phase 2A.
- Every subprocess uses `shell=False`, an exact argument vector, a timeout, bounded sanitized diagnostics, and an injected test seam.
- Public and persisted output must not contain prompts, model response text, credentials, or unsanitized subprocess/API bodies.
- Each task follows red-green-refactor, runs focused tests, runs the full Python suite, and commits independently.

---

### Task 1: Strict inference configuration and public contracts

**Files:**
- Create: `config/inference.toml`
- Create: `src/ai_video_factory/inference_config.py`
- Create: `src/ai_video_factory/inference_models.py`
- Create: `tests/test_inference_config.py`
- Create: `tests/test_inference_models.py`

**Interfaces:**
- Produces: `load_inference_config(path: Path) -> InferenceConfig`
- Produces: `InferenceCheck`, `InferenceResult`, `MemoryEstimate`, and `ModelIdentity`
- Consumes later: Tasks 2-6 import these types and configuration fields without redefining them.

- [ ] **Step 1: Add failing configuration tests**

```python
def test_loads_checked_in_lm_studio_configuration(project_root: Path) -> None:
    config = load_inference_config(project_root / "config" / "inference.toml")
    assert config.model_key == "qwen3.6-35b-a3b-udt-mtp"
    assert config.identifier == "avf-qwen36-executor"
    assert config.base_url == "http://127.0.0.1:1234/v1"
    assert config.context_length == 65_536
    assert config.gpu == "max"
    assert config.parallel == 1
    assert config.ttl_seconds == 3_600
    assert config.minimum_available_memory_gib == 40


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:1234/v1",
        "http://localhost:1234/v1",
        "http://0.0.0.0:1234/v1",
        "http://127.0.0.1:1234/v1?token=secret",
        "http://user:secret@127.0.0.1:1234/v1",
    ],
)
def test_rejects_noncanonical_or_credentialed_endpoint(base_url: str) -> None:
    with pytest.raises(ValidationError):
        InferenceConfig.model_validate(valid_config(base_url=base_url))
```

- [ ] **Step 2: Run the configuration tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest tests/test_inference_config.py tests/test_inference_models.py -q -p no:cacheprovider`

Expected: collection fails because the new modules do not exist.

- [ ] **Step 3: Implement strict configuration and result types**

Create `config/inference.toml` with these exact values:

```toml
schema_version = 1
backend = "lm_studio"
base_url = "http://127.0.0.1:1234/v1"
lms_binary = "lms"
model_key = "qwen3.6-35b-a3b-udt-mtp"
identifier = "avf-qwen36-executor"
context_length = 65536
gpu = "max"
parallel = 1
ttl_seconds = 3600
minimum_available_memory_gib = 40
models_directory = "/home/summit/.lmstudio/models"
```

Implement `InferenceConfig` with `ConfigDict(strict=True, extra="forbid")`,
positive bounds for numeric fields, `Literal[1]`, `Literal["lm_studio"]`, and
`Literal["max"]`. A `mode="after"` validator must require the base URL to have
exactly scheme `http`, hostname `127.0.0.1`, port `1234`, path `/v1`, and no
username, password, query, or fragment. Validate `identifier` against
`^[a-z0-9][a-z0-9-]{2,63}$` and require an absolute `models_directory`.

Implement the public models with strict/forbid settings:

```python
class InferenceCheck(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    status: Literal["ready", "not_ready"]
    detail: str | int | float | bool | None


class MemoryEstimate(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    gpu_gib: float
    total_gib: float
    confidence: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    allowed: bool


class ModelIdentity(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    model_key: str
    identifier: str
    relative_path: str
    size_bytes: int
    sha256: str | None = None


class InferenceResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_version: Literal[1] = 1
    command: Literal["doctor", "estimate", "start", "status", "benchmark", "stop"]
    status: Literal["pass", "fail", "not_ready"]
    retryable: bool
    backend: Literal["lm_studio"] = "lm_studio"
    model_identifier: str
    checks: dict[str, InferenceCheck]
    metrics: dict[str, int | float | str | bool | None]
    artifacts: dict[str, str]
    error: str | None
```

- [ ] **Step 4: Run focused and full tests**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest tests/test_inference_config.py tests/test_inference_models.py -q -p no:cacheprovider`

Expected: PASS.

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest -q -p no:cacheprovider`

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add config/inference.toml src/ai_video_factory/inference_config.py src/ai_video_factory/inference_models.py tests/test_inference_config.py tests/test_inference_models.py
git commit -m "feat: define local inference contract"
```

---

### Task 2: Read-only LM Studio discovery and status adapter

**Files:**
- Create: `src/ai_video_factory/lm_studio.py`
- Create: `tests/test_lm_studio.py`

**Interfaces:**
- Consumes: `InferenceConfig`, `InferenceCheck`, `ModelIdentity`
- Produces: `ProcessResult`, `LmStudioModel`, `LmStudioSnapshot`, `LmStudioBackend.snapshot()`
- Produces: injectable `CommandRunner` and `HttpTransport` protocols used by Tasks 3 and 5.

- [ ] **Step 1: Add failing discovery tests**

```python
def test_snapshot_uses_only_read_only_fixed_commands(config: InferenceConfig) -> None:
    runner = RecordingRunner(
        outputs={
            ("lms", "--help"): ok("lms is LM Studio's CLI utility (v0.0.47)"),
            ("lms", "runtime", "ls"): ok("llama.cpp-linux-x86_64-amd-rocm-avx2@2.31.2"),
            ("lms", "runtime", "survey"): ok("GPU: 85.67 GiB\nRAM: 122.69 GiB"),
            ("lms", "server", "status"): ok("The server is running on port 1234."),
            ("lms", "ls", "--json"): ok(model_inventory_json()),
            ("lms", "ps", "--json"): ok("[]"),
        }
    )
    snapshot = LmStudioBackend(config, runner=runner).snapshot()
    assert snapshot.configured_model.model_key == config.model_key
    assert snapshot.configured_model_loaded is False
    assert runner.calls == EXPECTED_READ_ONLY_COMMANDS


def test_snapshot_rejects_model_path_outside_configured_root(config: InferenceConfig) -> None:
    runner = inventory_runner(path="../../outside.gguf")
    with pytest.raises(LmStudioError, match="contained"):
        LmStudioBackend(config, runner=runner).snapshot()
```

Also cover missing executable, timeout, malformed JSON, duplicate model keys,
missing configured model, server stopped, sanitized/capped stderr, and a loaded
unrelated model that remains visible but untouched.

- [ ] **Step 2: Run the adapter tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest tests/test_lm_studio.py -q -p no:cacheprovider`

Expected: FAIL because `ai_video_factory.lm_studio` does not exist.

- [ ] **Step 3: Implement fixed-command discovery**

Use these exact types and seams:

```python
@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], float], ProcessResult]
HttpTransport = Callable[
    [str, str, dict[str, object] | None, Mapping[str, str], float],
    dict[str, object],
]


class LmStudioBackend:
    """Typed boundary around fixed LM Studio CLI and loopback HTTP calls."""
```

The constructor signature is `LmStudioBackend(config: InferenceConfig, *,
runner: CommandRunner = run_process, http: HttpTransport = request_json)` and
the read-only entry point is `snapshot(self) -> LmStudioSnapshot`.

`run_process()` resolves `lms` once with `shutil.which`, verifies it is an
executable regular file, then calls `subprocess.run(argv, shell=False,
capture_output=True, text=True, check=False, timeout=timeout)`. Snapshot uses
only the six fixed vectors listed in the test. Parse `lms ls --json` and `lms
ps --json` with Pydantic strict models. Resolve inventory paths beneath
`models_directory` and reject traversal or ambiguous duplicate configured keys.
Sanitize every raised diagnostic with `sanitize_diagnostic()`.

Implement `request_json()` with `urllib.request`, a JSON content-type, bounded
2 MiB response reads, explicit timeout, no redirect handler, and a preflight
assertion that every URL is the configured loopback origin. Do not send an
Authorization header. The only request headers supplied by the backend are
`Content-Type: application/json` and `Accept: application/json`.

- [ ] **Step 4: Run focused and full tests**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest tests/test_lm_studio.py -q -p no:cacheprovider`

Expected: PASS.

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest -q -p no:cacheprovider`

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/ai_video_factory/lm_studio.py tests/test_lm_studio.py
git commit -m "feat: inspect LM Studio safely"
```

---

### Task 3: Memory-gated model lifecycle

**Files:**
- Create: `src/ai_video_factory/inference_service.py`
- Create: `tests/test_inference_service.py`
- Modify: `src/ai_video_factory/lm_studio.py`
- Modify: `tests/test_lm_studio.py`

**Interfaces:**
- Consumes: `LmStudioBackend.snapshot()`, `InferenceConfig`, `InferenceResult`, `MemoryEstimate`
- Produces: `LmStudioBackend.estimate()`, `.start()`, `.stop()`
- Produces: `InferenceService.doctor()`, `.estimate()`, `.start()`, `.status()`, `.stop()`

- [ ] **Step 1: Add failing estimate and lifecycle tests**

```python
def test_estimate_uses_exact_nonloading_vector(config: InferenceConfig) -> None:
    backend, runner = backend_with_estimate(
        "Estimated GPU Memory: 17.36 GiB\n"
        "Estimated Total Memory: 17.36 GiB\nConfidence: LOW\n"
    )
    estimate = backend.estimate()
    assert estimate.total_gib == 17.36
    assert runner.calls[-1] == (
        "lms", "load", config.model_key, "--gpu", "max", "--context-length",
        "65536", "--no-speculative-draft-mtp", "--estimate-only", "-y",
    )


def test_start_refuses_below_free_memory_gate(config: InferenceConfig) -> None:
    service = service_fixture(available_memory_gib=39.99)
    result = service.start()
    assert result.status == "not_ready"
    assert not service.backend.mutating_calls


def test_start_loads_only_configured_model_with_exact_vector(config: InferenceConfig) -> None:
    service = service_fixture(available_memory_gib=100, server_running=True)
    result = service.start()
    assert result.status == "pass"
    assert service.backend.load_calls == [(
        "lms", "load", config.model_key, "--gpu", "max", "--context-length",
        "65536", "--parallel", "1", "--ttl", "3600",
        "--no-speculative-draft-mtp", "--identifier", config.identifier, "-y",
    )]


def test_stop_never_unloads_unrelated_models(config: InferenceConfig) -> None:
    service = service_fixture(loaded=[config.identifier, "user-model"])
    result = service.stop()
    assert result.status == "pass"
    assert service.backend.unload_calls == [("lms", "unload", config.identifier)]
```

Cover start idempotence, stop idempotence, server-start vector
`("lms", "server", "start", "--port", "1234", "--bind", "127.0.0.1")`,
postcondition failure, estimate parse failure, low-confidence recording, timeout,
and the absence of `get`, `runtime select`, `runtime update`, `unload --all`, and
`server stop` in every call history.

- [ ] **Step 2: Run the lifecycle tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest tests/test_inference_service.py tests/test_lm_studio.py -q -p no:cacheprovider`

Expected: FAIL because lifecycle methods are missing.

- [ ] **Step 3: Implement estimate parsing and lifecycle policy**

Parse the three labeled estimate lines with anchored regular expressions and
reject missing, duplicate, negative, NaN, or infinite values. `estimate()` uses
a 60-second timeout and does not call `load` without `--estimate-only`.

Implement `available_memory_gib(path: Path = Path("/proc/meminfo")) -> float`
from `MemAvailable`, using 1024-based GiB and rejecting malformed input. `start()`
must run estimate and the 40 GiB free-memory check before any mutation. It starts
the server only if stopped, loads only the exact configured key, and polls
`snapshot()` with a bounded deadline until the exact stable identifier appears.
Use a 600-second model-load timeout and injectable clock/sleeper seams.

`stop()` looks up the exact stable identifier; when absent it returns pass
without a subprocess call. When present it invokes only `lms unload
avf-qwen36-executor`, then verifies that identifier is absent while preserving
the before/after set of unrelated identifiers.

After the CLI reports the configured identifier loaded, `start()` must also
call `GET /v1/models` and require that exact identifier in the API model list.
The postcondition fails safely if CLI residency and API visibility disagree.

All service methods convert expected failures into `InferenceResult`; unexpected
errors are sanitized and capped without tracebacks in the public result.

- [ ] **Step 4: Run focused and full tests**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest tests/test_inference_service.py tests/test_lm_studio.py -q -p no:cacheprovider`

Expected: PASS.

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest -q -p no:cacheprovider`

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/ai_video_factory/inference_service.py src/ai_video_factory/lm_studio.py tests/test_inference_service.py tests/test_lm_studio.py
git commit -m "feat: guard LM Studio model lifecycle"
```

---

### Task 4: Contained cached model provenance

**Files:**
- Create: `src/ai_video_factory/model_provenance.py`
- Create: `tests/test_model_provenance.py`
- Modify: `src/ai_video_factory/inference_service.py`
- Modify: `tests/test_inference_service.py`

**Interfaces:**
- Consumes: contained configured model path from `LmStudioSnapshot`
- Produces: `ModelDigestCache.identity(model: LmStudioModel) -> ModelIdentity`
- Produces: `capability_inputs(config, snapshot, identity, corpus_version) -> dict[str, object]`

- [ ] **Step 1: Add failing containment and cache tests**

```python
def test_hashes_model_and_atomically_caches_identity(tmp_path: Path) -> None:
    model = tmp_path / "models" / "publisher" / "model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model-bytes")
    cache = ModelDigestCache(tmp_path / "cache", model_root=tmp_path / "models")
    identity = cache.identity(model_record(model))
    assert identity.sha256 == hashlib.sha256(b"model-bytes").hexdigest()
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_reuses_digest_only_when_path_size_and_mtime_match(tmp_path: Path) -> None:
    cache, model, hasher = cache_fixture(tmp_path)
    cache.identity(model_record(model))
    cache.identity(model_record(model))
    assert hasher.calls == [model]
    model.write_bytes(b"changed")
    cache.identity(model_record(model))
    assert hasher.calls == [model, model]


def test_rejects_symlink_or_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contained"):
        cache_fixture(tmp_path).cache.identity(model_record(Path("../../secret")))
```

Also cover size disagreement with LM Studio inventory, malformed cache JSON,
wrong cached digest length, concurrent-safe atomic replacement, and a
deterministic capability-input fingerprint.

- [ ] **Step 2: Run provenance tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest tests/test_model_provenance.py -q -p no:cacheprovider`

Expected: FAIL because `model_provenance` does not exist.

- [ ] **Step 3: Implement contained SHA-256 caching**

Resolve the model path beneath `models_directory`, require a regular non-symlink
file, verify its size against the inventory's `sizeBytes`, and hash in 1 MiB
blocks. Store this strict record beneath `data/system/model-digests/`:

```json
{
  "schema_version": 1,
  "relative_path": "AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF/Qwen3.6-35B-A3B-UDT-Q3_K_XL_MTP.gguf",
  "size_bytes": 18640894912,
  "mtime_ns": 0,
  "sha256": "64 lowercase hexadecimal characters"
}
```

Use the real `mtime_ns`; zero above is only the schema example. Write via a
same-directory `.tmp` followed by `Path.replace()`. Cache reuse requires exact
path, size, and `mtime_ns`. Do not modify, open writable, or relocate the model.

`capability_inputs()` must include the full inference config, sanitized runtime
and AMD survey identity, model identity, and literal corpus version
`lm-studio-capability-v1`; pass the result through existing
`fingerprint_inputs()`.

- [ ] **Step 4: Run focused and full tests**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest tests/test_model_provenance.py tests/test_inference_service.py -q -p no:cacheprovider`

Expected: PASS.

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest -q -p no:cacheprovider`

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/ai_video_factory/model_provenance.py src/ai_video_factory/inference_service.py tests/test_model_provenance.py tests/test_inference_service.py
git commit -m "feat: fingerprint local inference capability"
```

---

### Task 5: Deterministic OpenAI-compatible capability benchmark

**Files:**
- Create: `src/ai_video_factory/inference_benchmark.py`
- Create: `tests/test_inference_benchmark.py`
- Modify: `src/ai_video_factory/inference_service.py`
- Modify: `tests/test_inference_service.py`

**Interfaces:**
- Consumes: `HttpTransport`, configured model identifier, capability inputs, existing `RunStore`
- Produces: `Clock = Callable[[], float]`
- Produces: `run_capability_benchmark(service: InferenceService, store: RunStore, data_root: Path, clock: Clock = time.monotonic) -> InferenceResult`
- Produces artifact: `data/projects/system/runs/<run-id>/inference_report.json`

- [ ] **Step 1: Add failing request and persistence tests**

```python
def test_benchmark_sends_three_deterministic_requests_without_credentials() -> None:
    transport = RecordingHttpTransport(responses=passing_responses())
    service, store, data_root = benchmark_fixture(http=transport)
    result = run_capability_benchmark(service, store, data_root)
    assert result.status == "pass"
    assert [call.path for call in transport.calls] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert all("Authorization" not in call.headers for call in transport.calls)
    assert transport.calls[0].body["temperature"] == 0
    assert transport.calls[0].body["seed"] == 7


def test_report_persists_metrics_but_not_prompts_or_response_text(tmp_path: Path) -> None:
    secret_response = "MODEL_RESPONSE_MUST_NOT_PERSIST"
    service, store, data_root = benchmark_fixture(
        tmp_path=tmp_path,
        response_text=secret_response,
    )
    result = run_capability_benchmark(service, store, data_root)
    report = Path(result.artifacts["report"]).read_text()
    assert secret_response not in report
    assert "Reply exactly LOCAL_OK" not in report
    assert json.loads(report)["checks"]["ordinary_generation"] == "pass"


def test_resumes_only_matching_verified_report(tmp_path: Path) -> None:
    service, store, data_root = benchmark_fixture(tmp_path=tmp_path)
    first = run_capability_benchmark(service, store, data_root)
    second = run_capability_benchmark(service, store, data_root)
    assert first.metrics["resumed"] is False
    assert second.metrics["resumed"] is True
```

Also cover model not loaded, wrong model identifier, HTTP timeout, non-2xx,
oversized response, malformed JSON, missing choices, invalid structured output,
wrong tool name, invalid JSON tool arguments, tool-schema mismatch, sanitized
error persistence, artifact tampering invalidation, and duration measurements.

- [ ] **Step 2: Run benchmark tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest tests/test_inference_benchmark.py -q -p no:cacheprovider`

Expected: FAIL because the benchmark module does not exist.

- [ ] **Step 3: Implement the three capability probes**

All calls use `POST /v1/chat/completions`, `model=avf-qwen36-executor`,
`temperature=0`, `seed=7`, `stream=false`, and bounded token counts.

Probe 1 asks `Reply exactly LOCAL_OK.` and passes only when normalized assistant
content equals `LOCAL_OK`.

Probe 2 supplies this strict response format and validates decoded content with
a strict Pydantic model:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "local_capability",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {"status": {"type": "string", "enum": ["LOCAL_OK"]}},
      "required": ["status"],
      "additionalProperties": false
    }
  }
}
```

Probe 3 provides exactly one function named `record_scene` with required
`scene_id: string` and `duration_seconds: integer`, `additionalProperties:
false`, and `tool_choice` forcing that function. Pass only when
`choices[0].message.tool_calls` contains exactly one matching call whose decoded
arguments equal `{"scene_id": "intro", "duration_seconds": 3}`.

Persist only pass/fail check names, latency milliseconds, usage token counts,
runtime/model provenance, run ID, and resume state. Never persist message
content, reasoning content, tool-call IDs, raw arguments, prompts, or raw HTTP
bodies. Complete the `RunStore` stage `lm-studio-capability` with the JSON report
as its sole expected artifact.

- [ ] **Step 4: Run focused and full tests**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest tests/test_inference_benchmark.py tests/test_inference_service.py -q -p no:cacheprovider`

Expected: PASS.

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest -q -p no:cacheprovider`

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/ai_video_factory/inference_benchmark.py src/ai_video_factory/inference_service.py tests/test_inference_benchmark.py tests/test_inference_service.py
git commit -m "feat: benchmark local model capabilities"
```

---

### Task 6: CLI contract, operator documentation, and live verification

**Files:**
- Modify: `src/ai_video_factory/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `docs/hermes-command-contract.md`
- Modify: `system_report.md`

**Interfaces:**
- Consumes: `InferenceService` and `run_capability_benchmark()`
- Produces: `ai-video-factory inference doctor|estimate|start|status|benchmark|stop`
- Produces: Phase 2B readiness statement only after a live passing capability report.

- [ ] **Step 1: Add failing CLI contract tests**

```python
def test_cli_exposes_inference_command_group() -> None:
    result = CliRunner().invoke(cli.app, ["inference", "--help"])
    assert result.exit_code == 0
    for command in ("doctor", "estimate", "start", "status", "benchmark", "stop"):
        assert command in result.stdout


@pytest.mark.parametrize(
    "command",
    ["doctor", "estimate", "start", "status", "benchmark", "stop"],
)
def test_inference_commands_emit_one_json_document(monkeypatch, command: str) -> None:
    monkeypatch.setattr(cli, "build_inference_service", service_for(command))
    result = CliRunner().invoke(cli.app, ["inference", command])
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["command"] == command
    assert "Traceback" not in result.stdout


def test_inference_failure_is_sanitized_and_exits_two(monkeypatch) -> None:
    monkeypatch.setattr(cli, "build_inference_service", broken_service("token=secret"))
    result = CliRunner().invoke(cli.app, ["inference", "start"])
    assert result.exit_code == 2
    assert "secret" not in result.stdout
    assert json.loads(result.stdout)["status"] != "pass"
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest tests/test_cli.py -q -p no:cacheprovider`

Expected: FAIL because the inference group is absent.

- [ ] **Step 3: Implement the Typer inference group and docs**

Create `inference_app = typer.Typer(no_args_is_help=True)` and register it with
`app.add_typer(inference_app, name="inference")`. Each subcommand constructs the
service from repository-root `config/inference.toml`, emits exactly one
`InferenceResult.model_dump_json()` document, and exits 2 unless status is
`pass`. Catch unexpected exceptions, sanitize them, and emit the same contract
without a traceback.

Update README with the exact safe sequence and state effects. Update the Hermes
contract to say Phase 2B may consume only a previously passing capability report
and the stable loopback identifier; do not claim Hermes is installed. Update
`system_report.md` with the audited LM Studio CLI path, installed runtime family,
loopback bind, 85.67 GiB survey result, selected model sizes, estimate values,
922 GiB existing model storage, 457 GiB free disk, and the fact that system ROCm
remains unverified and unchanged.

- [ ] **Step 4: Run all automated verification before touching live model state**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest -q -p no:cacheprovider`

Expected: all Python tests PASS.

Run: `cd remotion && npm test && npm run build`

Expected: 8 Remotion tests PASS and TypeScript build succeeds.

Run: `git diff --check`

Expected: no output and exit 0.

- [ ] **Step 5: Execute the controlled live lifecycle in order**

Run each command separately from the repository root and capture its JSON result:

```bash
uv run ai-video-factory inference doctor
uv run ai-video-factory inference estimate
uv run ai-video-factory inference start
uv run ai-video-factory inference status
uv run ai-video-factory inference benchmark
uv run ai-video-factory doctor
uv run ai-video-factory benchmark
uv run ai-video-factory test-pipeline --json
uv run ai-video-factory inference stop
uv run ai-video-factory inference status
lms ps --json
```

Required outcomes: inference doctor and estimate pass; start loads only
`avf-qwen36-executor`; status sees that exact identifier; all three model probes
pass; the existing synthetic pipeline still passes; stop unloads only the
configured identifier; final `lms ps --json` contains no configured identifier
and preserves any unrelated identifiers seen before start. If any inference
probe fails, stop the configured model safely, persist the sanitized failure,
and do not claim Phase 2B readiness.

- [ ] **Step 6: Re-run final tests and commit Task 6**

Run: `UV_CACHE_DIR=/tmp/ai-video-factory-uv-cache uv run pytest -q -p no:cacheprovider`

Expected: all Python tests PASS.

Run: `cd remotion && npm test && npm run build`

Expected: tests and build PASS.

Run: `git diff --check && git status --short`

Expected: only Task 6 source, test, and documentation files are modified;
ignored runtime/model/report data is absent.

```bash
git add src/ai_video_factory/cli.py tests/test_cli.py README.md docs/hermes-command-contract.md system_report.md
git commit -m "feat: expose LM Studio inference workflow"
```

---

## Completion gate

Phase 2A is complete only when all automated tests pass, the TypeScript fixture
still builds, independent code review reports no unresolved Important issue,
the live Qwen capability benchmark passes through the loopback OpenAI-compatible
API, the configured model is safely unloaded afterward, unrelated LM Studio
models are unchanged, `main` is clean, and no model/runtime/system download or
installation occurred.
