# AI Video Factory Phase 2B Hermes Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep Hermes's existing Nous free model as the parent orchestrator while routing production work and verified vision to local Qwen 3.6 in LM Studio, with one explicit Nous-free fallback after local failure.

**Architecture:** A strict project configuration and read-only Hermes adapter prove the installed Hermes identity and resolved non-secret routes. The existing LM Studio capability layer gains companion-file provenance and a deterministic vision probe. Repository-scoped Hermes instructions bind the cloud parent to local-first native delegation, while factory smoke commands produce sanitized, resumable route evidence and test the explicit parent fallback state machine.

**Tech Stack:** Python 3.12, Pydantic 2, Typer, pytest, Hermes Agent 0.21.0, LM Studio OpenAI-compatible API, Qwen 3.6 35B-A3B, Git, existing RunStore.

**Spec:** `docs/superpowers/specs/2026-09-05-ai-video-factory-phase-2b-hermes-orchestration-design.md`

## Global Constraints

- Work directly on `main`; each task ends in a focused, revertible commit.
- Preserve Hermes parent route `nous / stepfun/step-3.7-flash:free`.
- Route local workers only to `http://127.0.0.1:1234/v1` model `avf-qwen36-executor`.
- Use one local worker at a time; recursive delegation and subagent auto-approval stay disabled.
- The factory alone may start or stop the configured model; Hermes must not manage LM Studio directly.
- Web search and extraction are allowed; local model inference is not an offline guarantee.
- Permit exactly one Nous-free parent fallback after a classified local failure.
- Do not install or upgrade Hermes, download models, change GPU drivers/runtimes, enter credentials, upload, message externally, or publish.
- LM Studio sensitive local logging remains enabled under the operator's explicit waiver; do not change or delete its logs.
- Never print, copy, fixture, report, or commit `~/.hermes/config.yaml`, `~/.hermes/.env`, auth material, raw prompts, raw responses, reasoning, tool arguments, or image bytes.

---

## Planned file structure

- Create `config/hermes.toml`: checked-in, non-secret expected Hermes and routing contract.
- Delete `config.example.toml`: remove the redundant host-specific copy identified in the Phase 2A final review.
- Create `src/ai_video_factory/hermes_config.py`: strict TOML configuration model.
- Create `src/ai_video_factory/hermes_models.py`: versioned result, attempt, and snapshot contracts.
- Create `src/ai_video_factory/hermes_backend.py`: fixed-vector Hermes subprocess adapter and resolved-route inspection.
- Create `src/ai_video_factory/hermes_service.py`: doctor, configuration transaction, smoke orchestration, validation, and fallback policy.
- Create `src/ai_video_factory/vision_benchmark.py`: deterministic multimodal request, validation, persistence, and resume logic.
- Create `assets/vision/capability-probe.png`: deterministic local-only vision fixture.
- Create `.hermes.md`: repository-scoped parent/delegate policy.
- Modify `src/ai_video_factory/model_provenance.py`: contained companion digest identity.
- Modify `src/ai_video_factory/inference_service.py`: vision capability provenance builder.
- Modify `src/ai_video_factory/cli.py`: `hermes` command group.
- Modify `README.md`, `system_report.md`, and `docs/hermes-command-contract.md`: operator workflow and accepted privacy/fallback disclosures.
- Create tests matching each new module and extend existing inference/CLI tests.

---

### Task 1: Strict Hermes configuration and public contracts

**Files:**
- Create: `config/hermes.toml`
- Delete: `config.example.toml`
- Create: `src/ai_video_factory/hermes_config.py`
- Create: `src/ai_video_factory/hermes_models.py`
- Create: `tests/test_hermes_config.py`
- Create: `tests/test_hermes_models.py`

**Interfaces:**
- Produces: `HermesConfig`, `load_hermes_config(path: Path) -> HermesConfig`.
- Produces: `HermesAttempt`, `HermesCheck`, `HermesResult`, and `HermesSnapshot`.
- Consumes: no Phase 2B code; only Pydantic and the existing strict-config conventions.

- [ ] **Step 1: Add failing strict-config tests**

```python
def test_checked_in_hermes_config_is_exact() -> None:
    config = load_hermes_config(PROJECT_ROOT / "config" / "hermes.toml")
    assert config.model_dump() == {
        "schema_version": 1,
        "hermes_binary": "hermes",
        "required_version": "0.21.0",
        "required_commit": "b0ab2e16",
        "profile": "default",
        "parent_provider": "nous",
        "parent_model": "stepfun/step-3.7-flash:free",
        "delegation_base_url": "http://127.0.0.1:1234/v1",
        "delegation_model": "avf-qwen36-executor",
        "delegation_api_mode": "chat_completions",
        "local_api_key_placeholder": "no-key-required",
        "max_concurrent_children": 1,
        "max_iterations": 50,
        "vision_fixture": "assets/vision/capability-probe.png",
        "fallback_provider": "nous",
        "fallback_model": "stepfun/step-3.7-flash:free",
    }
```

Add parameterized failures for booleans in numeric fields, extras, relative
binary paths masquerading as endpoint data, any host other than `127.0.0.1`,
credentials in the URL, a non-`:free` fallback, parent/fallback mismatch,
parallelism above one, and invalid relative fixture traversal.

- [ ] **Step 2: Run the new tests and verify RED**

Run: `uv run pytest tests/test_hermes_config.py tests/test_hermes_models.py -q`

Expected: collection fails because both modules are absent.

- [ ] **Step 3: Implement the strict configuration**

```python
class HermesConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_version: Literal[1]
    hermes_binary: str = Field(min_length=1)
    required_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    required_commit: str = Field(pattern=r"^[0-9a-f]{8}$")
    profile: Literal["default"]
    parent_provider: Literal["nous"]
    parent_model: str = Field(pattern=r"^[A-Za-z0-9._/-]+:free$")
    delegation_base_url: str
    delegation_model: Literal["avf-qwen36-executor"]
    delegation_api_mode: Literal["chat_completions"]
    local_api_key_placeholder: Literal["no-key-required"]
    max_concurrent_children: Literal[1]
    max_iterations: int = Field(ge=1, le=100)
    vision_fixture: str
    fallback_provider: Literal["nous"]
    fallback_model: str

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if self.delegation_base_url != "http://127.0.0.1:1234/v1":
            raise ValueError("delegation endpoint must be the exact LM Studio loopback URL")
        if self.fallback_model != self.parent_model:
            raise ValueError("fallback model must be the unchanged parent model")
        fixture = PurePosixPath(self.vision_fixture)
        if fixture.is_absolute() or ".." in fixture.parts:
            raise ValueError("vision fixture must be a contained project-relative path")
        return self
```

Write the exact TOML values asserted by the test. Remove `config.example.toml`;
`config/hermes.toml` and `config/inference.toml` are the only authoritative
checked-in configuration contracts.

- [ ] **Step 4: Implement versioned result models**

```python
class HermesAttempt(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    backend: Literal["local_lmstudio", "nous_free_fallback"]
    outcome: Literal["pass", "fail", "not_attempted"]
    retryable: bool
    reason_code: str | None

class HermesResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_version: Literal[1] = 1
    command: Literal["doctor", "smoke-text", "smoke-vision", "smoke-fallback"]
    status: Literal["pass", "fail", "not_ready"]
    retryable: bool
    parent_provider: Literal["nous"]
    parent_model: str
    attempts: list[HermesAttempt]
    checks: dict[str, HermesCheck]
    metrics: dict[str, int | float | str | bool | None]
    artifacts: dict[str, str]
    error: str | None
```

`HermesSnapshot` contains only canonical Hermes path, version, commit, config
path, profile, parent provider/model, non-secret delegation fields, concurrency,
iteration count, recursion/approval flags, and auxiliary route summaries. It
must have no generic raw-config field.

- [ ] **Step 5: Run focused and full tests**

Run: `uv run pytest tests/test_hermes_config.py tests/test_hermes_models.py -q`

Expected: all pass.

Run: `uv run pytest -q`

Expected: existing 296 tests plus new tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add config/hermes.toml config.example.toml src/ai_video_factory/hermes_config.py src/ai_video_factory/hermes_models.py tests/test_hermes_config.py tests/test_hermes_models.py
git commit -m "feat: define Hermes orchestration contract"
```

---

### Task 2: Fixed-vector Hermes adapter and read-only doctor

**Files:**
- Create: `src/ai_video_factory/hermes_backend.py`
- Create: `src/ai_video_factory/hermes_service.py`
- Create: `tests/test_hermes_backend.py`
- Create: `tests/test_hermes_service.py`

**Interfaces:**
- Consumes: `HermesConfig`, `HermesSnapshot`, `HermesResult`.
- Produces: `HermesBackend.snapshot() -> HermesSnapshot`.
- Produces: `HermesService.doctor() -> HermesResult`.
- Produces: injected `HermesCommandRunner` and `HermesFileOps` seams used by Task 5.

- [ ] **Step 1: Write subprocess-boundary tests**

Test that the default runner accepts only these read-only canonical vectors:

```python
READ_ONLY = {
    ("hermes", "--version"),
    ("hermes", "config", "path"),
    ("hermes", "config", "check"),
    ("hermes", "config", "get", "model.provider"),
    ("hermes", "config", "get", "model.default"),
    ("hermes", "config", "get", "model.base_url"),
    ("hermes", "config", "get", "delegation.model"),
    ("hermes", "config", "get", "delegation.base_url"),
    ("hermes", "config", "get", "delegation.api_mode"),
    ("hermes", "config", "get", "delegation.max_iterations"),
    ("hermes", "config", "get", "delegation.max_concurrent_children"),
    ("hermes", "config", "get", "delegation.max_spawn_depth"),
    ("hermes", "config", "get", "delegation.orchestrator_enabled"),
    ("hermes", "config", "get", "delegation.subagent_auto_approve"),
    ("hermes", "config", "get", "delegation.inherit_mcp_toolsets"),
}
```

Reject shell execution, extra flags, environment overrides, alternate config
keys, timeouts above the fixed bound, non-zero exits, output above 64 KiB, a
missing executable, symlink swaps, and malformed version text. Prove the
canonical resolved executable is the one passed to `subprocess.run`.

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `uv run pytest tests/test_hermes_backend.py tests/test_hermes_service.py -q`

Expected: imports fail.

- [ ] **Step 3: Implement the command runner and version parser**

```python
_VERSION = re.compile(
    r"^Hermes Agent v(?P<version>\d+\.\d+\.\d+) \([^\n]+\) · upstream (?P<commit>[0-9a-f]{8})$",
    re.MULTILINE,
)

def run_process(argv: tuple[str, ...], *, timeout: float) -> ProcessResult:
    if argv not in ALLOWED_COMMANDS:
        raise HermesError("Hermes command is not allowlisted")
    executable = _resolved_hermes()
    completed = subprocess.run(
        (executable, *argv[1:]), shell=False, capture_output=True,
        text=True, timeout=timeout, check=False,
    )
    return _bounded_result(completed)
```

Resolve `~/.local/bin/hermes` through `shutil.which`, require a
canonical absolute regular executable, and retain its canonical path in the
snapshot. Never include subprocess environment contents.

- [ ] **Step 4: Implement strict non-secret snapshot parsing**

Use individual `hermes config get` calls. Do not run `hermes config show` and
do not read `config.yaml` into a report. Missing delegation keys are represented
as `None` before setup; conflicting, multi-line, control-character, or secret-
looking values fail closed.

`doctor()` passes only when installed path/version/commit, active default
profile, unchanged parent route, exact local delegation route, concurrency one,
flat delegation, no subagent auto-approval, and current text capability report
all validate. Before Task 5 configuration, a structurally sound but unconfigured
delegation route returns `not_ready`, never `fail`.

- [ ] **Step 5: Test sanitization and route rejection**

```python
result = service.doctor()
assert result.status == "not_ready"
assert "token=" not in result.model_dump_json()
assert "api_key" not in result.model_dump_json().casefold()
```

Add wrong-provider, paid-model, remote-delegation, recursion-enabled,
auto-approval-enabled, stale text capability, and unexpected-exception cases.

- [ ] **Step 6: Run focused and full tests, then commit**

Run: `uv run pytest tests/test_hermes_backend.py tests/test_hermes_service.py -q`

Run: `uv run pytest -q`

Expected: all pass.

```bash
git add src/ai_video_factory/hermes_backend.py src/ai_video_factory/hermes_service.py tests/test_hermes_backend.py tests/test_hermes_service.py
git commit -m "feat: inspect Hermes routes safely"
```

---

### Task 3: Companion provenance and deterministic local vision gate

**Files:**
- Create: `assets/vision/capability-probe.png`
- Create: `src/ai_video_factory/vision_benchmark.py`
- Create: `tests/test_vision_benchmark.py`
- Modify: `src/ai_video_factory/model_provenance.py`
- Modify: `src/ai_video_factory/inference_service.py`
- Modify: `src/ai_video_factory/inference_models.py`
- Modify: `tests/test_model_provenance.py`
- Modify: `tests/test_inference_service.py`

**Interfaces:**
- Produces: `CompanionIdentity(relative_path, size_bytes, mtime_ns, sha256)`.
- Produces: `InferenceService.vision_capability_inputs(data_root, fixture_path)`.
- Produces: `run_vision_benchmark(service, store, data_root, fixture_path) -> InferenceResult` with command `vision-benchmark`.
- Consumes: exact model containment and `ModelDigestCache` patterns from Phase 2A.

- [ ] **Step 1: Add a deterministic checked-in PNG fixture**

Generate once from a reviewable standard-library script: 512×256 RGB, white
background, a red square in the left third, a blue circle in the right third,
and a large black block numeral `7` in the center. Encode scanlines with
`zlib.compress`, write PNG chunks with `struct.pack` and CRC32, then commit only
the PNG. Record its SHA-256 in the test.

```python
assert fixture.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
assert hashlib.sha256(fixture.read_bytes()).hexdigest() == EXPECTED_FIXTURE_SHA256
```

- [ ] **Step 2: Write failing companion-identity tests**

Cover exactly one direct `mmproj*.gguf` companion, regular-file and non-symlink
requirements, canonical containment, ambiguity, missing companion, size drift,
mtime drift, cached digest reuse, digest invalidation, and atomic cache writes.

- [ ] **Step 3: Implement companion identity without weakening text provenance**

```python
class CompanionIdentity(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    relative_path: str
    size_bytes: int = Field(gt=0)
    mtime_ns: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
```

Add `ModelDigestCache.companion_identity(model: LmStudioModel)` using the same
contained, non-symlink, regular-file and cache rules as the primary. Keep the
existing text capability fingerprint stable; only the new vision fingerprint
contains the companion digest.

- [ ] **Step 4: Write failing vision request and validator tests**

Require one POST to `/chat/completions` with:

```python
{
    "model": "avf-qwen36-executor",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + encoded_fixture,
            "detail": "high",
        }},
    ]}],
    "response_format": VISION_RESPONSE_FORMAT,
    "temperature": 0,
    "seed": 7,
    "stream": False,
    "max_tokens": 512,
}
```

The strict response is
`{"left_shape":"red_square","center_numeral":7,"right_shape":"blue_circle"}`.
Test wrong colors/order/shape/number, extra keys, malformed completion/usage,
wrong model, timeout, HTTP failure, and raw content leakage.

- [ ] **Step 5: Implement versioned vision persistence and resume**

Use stage `lm-studio-vision-capability` and corpus
`lm-studio-vision-capability-v1`. The persisted report contains only pass/fail,
latency, token usage, and provenance. It must not contain `VISION_PROMPT`,
base64, image bytes, response text, reasoning, completion IDs, or tool data.

On resume, revalidate manifest integrity and exact current provenance; invalidate
tampered or stale reports and create one fresh run.

- [ ] **Step 6: Run focused and full tests, then commit**

Run: `uv run pytest tests/test_model_provenance.py tests/test_inference_service.py tests/test_vision_benchmark.py -q`

Run: `uv run pytest -q`

Expected: all pass.

```bash
git add assets/vision/capability-probe.png src/ai_video_factory/vision_benchmark.py src/ai_video_factory/model_provenance.py src/ai_video_factory/inference_service.py src/ai_video_factory/inference_models.py tests/test_vision_benchmark.py tests/test_model_provenance.py tests/test_inference_service.py
git commit -m "feat: verify local vision capability"
```

---

### Task 4: Local-first Hermes smoke runner and explicit fallback state machine

**Files:**
- Modify: `src/ai_video_factory/hermes_backend.py`
- Modify: `src/ai_video_factory/hermes_service.py`
- Modify: `tests/test_hermes_backend.py`
- Modify: `tests/test_hermes_service.py`

**Interfaces:**
- Produces: `HermesService.smoke_text()`, `.smoke_vision()`, and `.smoke_fallback()`.
- Consumes: guarded `InferenceService`, current text/vision reports, fixed Hermes routes, and `RunStore`.
- Produces: sanitized attempt records using `local_lmstudio` and `nous_free_fallback` only.

- [ ] **Step 1: Add fixed smoke-process vectors and fake-runner tests**

Build query files inside the active run directory with mode `0o600`; pass their
paths via `--query-file` so no prompt is placed in argv. Allow only exact Hermes
vectors assembled from canonical constants:

```python
("hermes", "chat", "--query-file", contained_query_path, "--oneshot", "--quiet",
 "--toolsets", "delegation,web", "--in", project_root, "--source", "tool",
 "--run-budget", "900")
```

The text smoke tells the local child to retrieve `https://example.com/` with
the inherited web tool and return its exact `Example Domain` title plus the
synthetic output marker. This proves real web research access while keeping the
fixture public and non-sensitive. Treat external network unavailability as a
retryable smoke failure, not proof that local inference is broken.

Vision uses `delegation,vision`; it passes the contained fixture path in the
synthetic query rather than attaching the image to the cloud parent. Reject any
query path outside the run root, symlink, alternate toolset, `--yolo`,
`--accept-hooks`, worktree flag, environment override, or unbounded execution.
Delete query files in `finally` after the subprocess returns.

- [ ] **Step 2: Add local-success and validation tests**

Inject fakes for inference lifecycle, capability reports, clock, Hermes runner,
and RunStore. A local text result must contain the exact synthetic output
envelope and route marker supplied by the delegated child. A vision result must
contain the exact three fixture facts. Validate that start occurs before Hermes,
stop occurs in `finally`, and unrelated model identifiers are unchanged.

- [ ] **Step 3: Implement classified fallback policy**

```python
class LocalFailure(str, Enum):
    endpoint_unavailable = "endpoint_unavailable"
    timeout = "timeout"
    transport = "transport"
    tool_call = "tool_call"
    invalid_output = "invalid_output"
    quality = "quality"

def may_fallback(reason: LocalFailure | PolicyFailure) -> bool:
    return isinstance(reason, LocalFailure)
```

For allowed failures, accept exactly one subsequent parent-produced result from
the same Hermes invocation and record backend `nous_free_fallback`. Reject a
second fallback, a provider/model different from the snapshot, or any parent
route not equal to the approved Nous `:free` route.

Policy, credential, provenance, upload, publication, and QC gate failures return
`not_ready` or `fail` without cloud fallback.

- [ ] **Step 4: Prove report non-persistence**

The run manifest, events, and result may contain only route class, reason code,
outcome, retryability, latency/token counters, report paths, and validator
status. Seed fake prompts/responses/reasoning/API keys and assert none occur in
any file below the test run directory.

- [ ] **Step 5: Test resumability and cleanup**

Successful smoke evidence resumes only when Hermes identity, resolved routes,
text/vision capability provenance, workspace-policy digest, fixture digest, and
smoke-corpus version match. A changed parent model, local route, policy file,
fixture, or capability report invalidates resume.

Ensure targeted model stop runs after success, local failure, cloud failure,
validator failure, timeout, and `KeyboardInterrupt`-compatible cancellation.

- [ ] **Step 6: Run focused and full tests, then commit**

Run: `uv run pytest tests/test_hermes_backend.py tests/test_hermes_service.py -q`

Run: `uv run pytest -q`

Expected: all pass.

```bash
git add src/ai_video_factory/hermes_backend.py src/ai_video_factory/hermes_service.py tests/test_hermes_backend.py tests/test_hermes_service.py
git commit -m "feat: orchestrate local workers with cloud fallback"
```

---

### Task 5: Transactional default-profile setup and repository policy

**Files:**
- Create: `.hermes.md`
- Modify: `src/ai_video_factory/hermes_backend.py`
- Modify: `src/ai_video_factory/hermes_service.py`
- Modify: `tests/test_hermes_backend.py`
- Modify: `tests/test_hermes_service.py`

**Interfaces:**
- Produces: `HermesService.configure_default_profile() -> HermesSnapshot` for controlled setup use, not a model-facing public command.
- Consumes: `HermesFileOps.copy2`, fixed `hermes config set` vectors, `snapshot()`, and `doctor()`.
- Produces: permission-preserving backup path confined to the external Hermes directory and a restore instruction.

- [ ] **Step 1: Write failing transaction tests**

Use a temporary fake Hermes directory and injected command runner. Assert:

- config path is canonical, regular, non-symlink, owner-readable, and below the
  canonical Hermes home;
- backup is created beside the source with mode no broader than the source;
- no backup content is read into a result or test log;
- only approved keys are written;
- any failed set, config check, doctor, or read-back restores with `copy2`;
- successful setup preserves the backup and returns its path only;
- restore refuses paths outside the Hermes directory and symlinks.

- [ ] **Step 2: Define the exact mutation set**

```python
EXPECTED_SETTINGS = {
    "delegation.model": "avf-qwen36-executor",
    "delegation.base_url": "http://127.0.0.1:1234/v1",
    "delegation.api_key": "no-key-required",
    "delegation.api_mode": "chat_completions",
    "delegation.max_iterations": "50",
    "delegation.max_concurrent_children": "1",
    "delegation.max_spawn_depth": "1",
    "delegation.orchestrator_enabled": "false",
    "delegation.subagent_auto_approve": "false",
    "delegation.inherit_mcp_toolsets": "false",
    "auxiliary.vision.base_url": "http://127.0.0.1:1234/v1",
    "auxiliary.vision.api_key": "no-key-required",
    "auxiliary.vision.model": "avf-qwen36-executor",
    "auxiliary.web_extract.base_url": "http://127.0.0.1:1234/v1",
    "auxiliary.web_extract.api_key": "no-key-required",
    "auxiliary.web_extract.model": "avf-qwen36-executor",
    "auxiliary.compression.base_url": "http://127.0.0.1:1234/v1",
    "auxiliary.compression.api_key": "no-key-required",
    "auxiliary.compression.model": "avf-qwen36-executor",
    "auxiliary.title_generation.enabled": "false",
    "auxiliary.background_review.enabled": "false",
}
```

Before mutation, assert `model.provider`, `model.default`, and `model.base_url`
match the audited parent. After mutation, assert those three values are byte-for-
byte unchanged through individual `config get` calls. Do not touch fallback,
auth, `.env`, gateway, session, memory, skill, or unrelated tool keys.

- [ ] **Step 3: Write the binding `.hermes.md` policy**

The document must instruct the parent to run factory doctor/start/status and
current capability gates; delegate substantive production work; use the local
worker's complete context/output contract; classify failure; allow one current
Nous-free parent fallback; record sanitized route evidence; always targeted-
stop; and require user approval for credentials, uploads, external messages,
and publication. Explicitly forbid direct `lms`, npm, FFmpeg, internal Python,
runtime changes, model downloads, recursive delegation, `--yolo`, and unrelated
model unloads.

- [ ] **Step 4: Test policy completeness and digest binding**

Assert exact required commands, routes, forbidden actions, fallback model,
approval gates, and privacy exception are present. Include the SHA-256 of
`.hermes.md` in every Hermes smoke input fingerprint.

- [ ] **Step 5: Run focused and full tests, then commit**

Run: `uv run pytest tests/test_hermes_backend.py tests/test_hermes_service.py -q`

Run: `uv run pytest -q`

Expected: all pass without touching the real Hermes configuration.

```bash
git add .hermes.md src/ai_video_factory/hermes_backend.py src/ai_video_factory/hermes_service.py tests/test_hermes_backend.py tests/test_hermes_service.py
git commit -m "feat: bind Hermes to local-first production policy"
```

---

### Task 6: Public CLI, documentation, and automated verification

**Files:**
- Modify: `src/ai_video_factory/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `system_report.md`
- Modify: `docs/hermes-command-contract.md`

**Interfaces:**
- Consumes: `HermesService` and `run_vision_benchmark`.
- Produces: `ai-video-factory hermes doctor|smoke-text|smoke-vision|smoke-fallback`.
- Produces: `ai-video-factory inference vision-benchmark`.

- [ ] **Step 1: Write failing CLI-contract tests**

```python
@pytest.mark.parametrize("command", ["doctor", "smoke-text", "smoke-vision", "smoke-fallback"])
def test_hermes_commands_emit_one_json_document(monkeypatch, command: str) -> None:
    result = CliRunner().invoke(cli.app, ["hermes", command])
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["command"] == command
    assert result.stdout.count("\n") == 1
    assert "Traceback" not in result.stdout
```

Assert exit 0 only for pass, exit 2 for fail/not-ready, repository-root path
resolution from another CWD, sanitized unexpected errors, and no secret/raw
content in stdout. Add the same coverage for `inference vision-benchmark`.

- [ ] **Step 2: Implement the CLI group**

```python
hermes_app = typer.Typer(no_args_is_help=True)
app.add_typer(hermes_app, name="hermes")

@hermes_app.command("doctor")
def hermes_doctor() -> None:
    _emit_hermes(build_hermes_service().doctor())
```

Implement the remaining three commands through one `_run_hermes_command`
dispatcher. Build all paths from `_PROJECT_ROOT`; never accept arbitrary model,
endpoint, config, query, image, or output path options.

- [ ] **Step 3: Update operator documentation**

Document:

- Hermes is preinstalled and is not upgraded;
- cloud parent versus local worker responsibilities;
- exact local-first lifecycle;
- web egress and Nous fallback disclosure;
- vision gate and fallback disclosure for image data;
- explicit LM Studio sensitive-logging waiver;
- profile backup and exact restore instruction;
- no credentials/uploads/publication;
- expected final unloaded-model state;
- next phase remains full production-stage implementation.

Correct any Phase 2A statement that still claims end-to-end non-persistence.

- [ ] **Step 4: Run all automated project checks**

Run:

```bash
uv run pytest -q
(cd remotion && npm test)
(cd remotion && npm run build)
git diff --check
```

Expected: all Python and Remotion tests pass, TypeScript builds, and diff check
has no output.

- [ ] **Step 5: Commit Task 6**

```bash
git add src/ai_video_factory/cli.py tests/test_cli.py README.md system_report.md docs/hermes-command-contract.md
git commit -m "feat: expose Hermes orchestration workflow"
```

---

### Task 7: Controlled live configuration and end-to-end verification

**Files:**
- Modify only if live compatibility requires a fail-closed adapter correction:
  the smallest Phase 2B source/test/doc set implicated by the evidence.
- Do not commit external Hermes configuration or generated run data.

**Interfaces:**
- Consumes every Phase 2B command and the transactional configuration method.
- Produces live, sanitized capability and smoke artifacts below ignored
  `data/projects/system/` plus a permission-preserving Hermes config backup.

- [ ] **Step 1: Capture safe preconditions**

Run read-only checks and record only non-secret summaries:

```bash
git status --short
hermes --version
hermes config get model.provider
hermes config get model.default
hermes config get model.base_url
~/.lmstudio/bin/lms ps --json
uv run ai-video-factory inference doctor
```

Require clean Git, exact Hermes version/commit, exact Nous-free parent, safe LM
Studio runtime/server, and an explicit initial loaded-model identifier set.

- [ ] **Step 2: Apply the transactional Hermes configuration**

Invoke the tested configuration method from the project environment. Preserve
the returned backup path privately for rollback. Then run `hermes config check`,
`hermes doctor` without `--fix` or `--live`, and
`uv run ai-video-factory hermes doctor`.

If any check fails, restore immediately and stop. Do not run `hermes setup`,
`hermes model`, `hermes update`, `hermes doctor --fix`, or any login flow.

- [ ] **Step 3: Run local text and direct vision capability gates**

```bash
uv run ai-video-factory inference estimate
uv run ai-video-factory inference start
uv run ai-video-factory inference benchmark
uv run ai-video-factory inference vision-benchmark
```

Require exact Qwen identity, 3/3 text checks, the deterministic vision facts,
current primary/companion/fixture provenance, and no unrelated-model change.

- [ ] **Step 4: Run Hermes text and vision smokes**

```bash
uv run ai-video-factory hermes smoke-text
uv run ai-video-factory hermes smoke-vision
```

These calls use synthetic content only. Require one local attempt, no cloud
fallback, validated output, current Hermes and capability provenance, and
observable local model activity.

- [ ] **Step 5: Run the safe fallback smoke**

Targeted-stop the exact Qwen identifier, verify it is absent, then run:

```bash
uv run ai-video-factory hermes smoke-fallback
```

Require one failed local attempt followed by exactly one validated
`nous_free_fallback` attempt on `stepfun/step-3.7-flash:free`. The fixture is
synthetic and contains no user content. Confirm the fallback does not load a
model, change provider configuration, or invoke another cloud route.

- [ ] **Step 6: Final cleanup and regression verification**

```bash
uv run ai-video-factory inference stop
~/.lmstudio/bin/lms ps --json
uv run ai-video-factory doctor
uv run ai-video-factory benchmark
uv run ai-video-factory test-pipeline --json
uv run pytest -q
(cd remotion && npm test)
(cd remotion && npm run build)
git status --short
```

Require the final LM Studio identifier set to equal the initial unrelated set,
all tests/builds green, the expected root-doctor degradation only for absent
system ROCm, and a clean Git tree. The shared LM Studio server remains running.

- [ ] **Step 7: Commit only verified compatibility corrections**

If no correction was needed, create no empty commit. If a correction was
required, first reproduce it with a focused failing test, implement the narrow
fail-closed fix, rerun every automated/live check above, and commit:

```bash
git add config/hermes.toml .hermes.md assets/vision/capability-probe.png src/ai_video_factory/hermes_config.py src/ai_video_factory/hermes_models.py src/ai_video_factory/hermes_backend.py src/ai_video_factory/hermes_service.py src/ai_video_factory/vision_benchmark.py src/ai_video_factory/model_provenance.py src/ai_video_factory/inference_service.py src/ai_video_factory/inference_models.py src/ai_video_factory/cli.py tests/test_hermes_config.py tests/test_hermes_models.py tests/test_hermes_backend.py tests/test_hermes_service.py tests/test_vision_benchmark.py tests/test_model_provenance.py tests/test_inference_service.py tests/test_cli.py README.md system_report.md docs/hermes-command-contract.md
git commit -m "fix: align Hermes live orchestration"
```

Record test counts, run IDs, capability fingerprints, fallback route, backup
location (path only), final loaded identifiers, and residual limitations in the
SDD task report. Never record config contents, prompts, responses, reasoning,
tool arguments, credentials, or image bytes.

---

## Final review gate

Review the complete Phase 2B range against the spec. Treat as release blockers:
parent-route mutation, paid or non-Nous fallback, cloud use on local success,
unverified vision, more than one local child, direct Hermes LM Studio model
management, unsafe endpoint/config handling, missing targeted cleanup, secret or
raw model-content persistence in factory artifacts, or any path to upload or
publication without explicit approval.

The accepted LM Studio sensitive-log waiver is not a Phase 2B failure, but all
operator-facing documentation must disclose it accurately.
