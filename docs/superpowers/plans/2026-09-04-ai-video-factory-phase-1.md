# AI Video Factory Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, testable foundation that Hermes can drive through stable commands to diagnose the workstation, create resumable runs, render a synthetic Remotion video, validate it with FFmpeg, and produce a QC report.

**Architecture:** A Python 3.12 package owns schemas, manifests, structured logs, diagnostics, validation, and the Hermes-facing CLI. A pinned TypeScript/Remotion package consumes a validated fixture and produces deterministic frames; FFmpeg/ffprobe perform media validation. All mutable run data lives below `data/projects/`, and no Phase 1 command installs system software, downloads models, accesses credentials, or publishes content.

**Tech Stack:** uv 0.11.32, Python 3.12, Typer, Pydantic 2, pytest, Node 22.23.1, npm 10.9.8, Remotion 4.0.520, React 19.2.8, TypeScript, FFmpeg 8.0.1, SQLite reserved for later indexing.

**Spec:** `docs/superpowers/specs/2026-09-04-ai-video-factory-phase-1-design.md`

## Global Constraints

- Codex performs bootstrap and verification; Hermes invokes the resulting versioned CLI.
- Use Python 3.12 in a project-local `.venv`; do not depend on system Python 3.14 for future ML compatibility.
- Do not use `sudo`, alter drivers/kernel/firmware, install Docker, or expose a network service.
- Do not download model weights or media assets; dependency and browser-runtime downloads must be disclosed before execution.
- Do not read or write credentials, upload content, or implement publication.
- Every command that mutates project data must emit JSONL events and preserve successful upstream stage outputs.
- A run may resume only when its canonical input fingerprint matches.
- Keep `.env`, caches, generated videos, run data, and model directories out of Git.
- Remotion, `@remotion/cli`, and `@remotion/renderer` must all be exactly `4.0.520`; React and React DOM must be exactly `19.2.8`.

## File map

```text
pyproject.toml                         Python package and command entry point
.python-version                       Python 3.12 runtime selection
.gitignore                            Secret, environment, cache, model, and output exclusions
src/ai_video_factory/cli.py           Hermes/human command contract
src/ai_video_factory/models.py        Typed run, diagnostic, benchmark, and QC records
src/ai_video_factory/run_store.py     Fingerprints, manifests, resume logic, JSONL events
src/ai_video_factory/doctor.py        Read-only host diagnostics
src/ai_video_factory/benchmark.py     No-model backend/tool probes
src/ai_video_factory/media_probe.py   ffprobe execution and parsing
src/ai_video_factory/qc.py            Technical QC aggregation and reports
src/ai_video_factory/pipeline.py      Synthetic render/validate orchestration
tests/                                Unit and end-to-end tests
remotion/                             Pinned deterministic composition
fixtures/synthetic-edit.json          Validated, local-only edit fixture
docs/hermes-command-contract.md        Machine-readable invocation rules
README.md                              Operator quick start and safety boundaries
```

---

### Task 1: Python package and safe CLI shell

**Files:**
- Create: `.python-version`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/ai_video_factory/__init__.py`
- Create: `src/ai_video_factory/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: uv 0.11.32 and project-local Python 3.12.
- Produces: console command `ai-video-factory` and `ai_video_factory.cli:app`.

- [ ] **Step 1: Add the failing CLI test**

```python
from typer.testing import CliRunner

from ai_video_factory.cli import app


def test_cli_exposes_phase_one_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "benchmark", "test-pipeline"):
        assert command in result.stdout
```

- [ ] **Step 2: Run the test before creating the package**

Run: `uv run pytest tests/test_cli.py -v`  
Expected: FAIL because `ai_video_factory` does not exist.

- [ ] **Step 3: Add the package metadata and CLI shell**

Use `.python-version` containing `3.12`. Configure `pyproject.toml` with `requires-python = ">=3.12,<3.13"`, a `src` layout, and the script entry point `ai-video-factory = "ai_video_factory.cli:app"`. Add dependencies `typer>=0.27,<0.28` and `pydantic>=2.12,<3`; add development dependency `pytest>=8,<10`.

```python
import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def doctor() -> None:
    """Report readiness without modifying the host."""


@app.command()
def benchmark() -> None:
    """Probe installed tools without downloading models."""


@app.command("test-pipeline")
def test_pipeline() -> None:
    """Run the synthetic local video fixture."""
```

Ignore `.env`, `.venv/`, `.pytest_cache/`, `__pycache__/`, `node_modules/`, `models/`, `cache/`, `temp/`, `output/`, `logs/`, and `data/projects/`, retaining required empty directories with `.gitkeep` only when needed.

- [ ] **Step 4: Resolve and test the local Python environment**

Run: `uv sync --dev && uv run pytest tests/test_cli.py -v`  
Expected: PASS; `.venv` uses Python 3.12 and `uv.lock` is created.

- [ ] **Step 5: Commit the CLI shell**

```bash
git add .python-version .gitignore pyproject.toml uv.lock src tests/test_cli.py
git commit -m "build: scaffold phase 1 Python CLI"
```

---

### Task 2: Typed run store, fingerprints, and structured logs

**Files:**
- Create: `src/ai_video_factory/models.py`
- Create: `src/ai_video_factory/run_store.py`
- Create: `tests/test_run_store.py`

**Interfaces:**
- Consumes: `pathlib.Path`, JSON-compatible input dictionaries.
- Produces: `StageStatus`, `RunManifest`, `fingerprint_inputs(inputs) -> str`, `RunStore.start(stage, inputs) -> RunManifest`, `RunStore.event(run_id, event, fields) -> None`, and `RunStore.complete(run_id, artifacts) -> RunManifest`.

- [ ] **Step 1: Add failing fingerprint and resume tests**

```python
from pathlib import Path

import pytest

from ai_video_factory.run_store import FingerprintMismatch, RunStore, fingerprint_inputs


def test_fingerprint_is_order_independent() -> None:
    assert fingerprint_inputs({"b": 2, "a": 1}) == fingerprint_inputs({"a": 1, "b": 2})


def test_completed_run_resumes_with_identical_inputs(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    first = store.start("render", {"fixture": "v1"})
    store.complete(first.run_id, {"video": "output/master.mp4"})
    resumed = store.start("render", {"fixture": "v1"})
    assert resumed.run_id == first.run_id
    assert resumed.resumed is True


def test_explicit_resume_rejects_changed_inputs(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    first = store.start("render", {"fixture": "v1"})
    with pytest.raises(FingerprintMismatch):
        store.resume(first.run_id, {"fixture": "v2"})
```

- [ ] **Step 2: Confirm the tests fail**

Run: `uv run pytest tests/test_run_store.py -v`  
Expected: FAIL because the run-store interfaces do not exist.

- [ ] **Step 3: Implement canonical manifests and append-only events**

Define `StageStatus` as `pending | running | completed | failed`; define `RunManifest` fields `schema_version`, `run_id`, `stage`, `input_fingerprint`, `status`, `resumed`, `created_at`, `updated_at`, `inputs`, `artifacts`, and `error`. Compute SHA-256 over UTF-8 JSON rendered with sorted keys and compact separators. Store each run at `<root>/<stage>/<run_id>/manifest.json` and events at `events.jsonl`. Write JSON atomically through a sibling `.tmp` file followed by `Path.replace()`.

`start()` must reuse the newest completed run with the same stage and fingerprint. `resume()` must raise `FingerprintMismatch` before changing files when the supplied fingerprint differs. Each event line must contain `timestamp`, `run_id`, `stage`, `event`, and `fields`.

- [ ] **Step 4: Run the focused and full Python tests**

Run: `uv run pytest tests/test_run_store.py -v && uv run pytest -v`  
Expected: all tests PASS.

- [ ] **Step 5: Commit run persistence**

```bash
git add src/ai_video_factory/models.py src/ai_video_factory/run_store.py tests/test_run_store.py
git commit -m "feat: add resumable run store and structured logs"
```

---

### Task 3: Schema-validated synthetic edit fixture

**Files:**
- Create: `src/ai_video_factory/edit_schema.py`
- Create: `fixtures/synthetic-edit.json`
- Create: `tests/test_edit_schema.py`

**Interfaces:**
- Consumes: JSON edit documents.
- Produces: `EditDocument`, `EditScene`, and `load_edit(path: Path) -> EditDocument`.

- [ ] **Step 1: Add failing validation tests**

```python
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_video_factory.edit_schema import EditDocument, load_edit


def test_fixture_is_valid() -> None:
    edit = load_edit(Path("fixtures/synthetic-edit.json"))
    assert edit.schema_version == 1
    assert edit.width == 1280 and edit.height == 720
    assert edit.fps == 30 and edit.duration_frames == 90
    assert len(edit.scenes) == 1


def test_scene_cannot_exceed_composition() -> None:
    with pytest.raises(ValidationError):
        EditDocument.model_validate({
            "schema_version": 1, "width": 1280, "height": 720,
            "fps": 30, "duration_frames": 30,
            "scenes": [{"id": "title", "from_frame": 0, "duration_frames": 31,
                        "title": "Synthetic test", "caption": "Local render"}],
        })
```

- [ ] **Step 2: Confirm the tests fail**

Run: `uv run pytest tests/test_edit_schema.py -v`  
Expected: FAIL because `edit_schema` does not exist.

- [ ] **Step 3: Implement the edit contract and fixture**

Define positive integer `width`, `height`, `fps`, and `duration_frames`. Define each scene with non-empty `id`, zero-or-greater `from_frame`, positive `duration_frames`, non-empty `title`, and non-empty `caption`. Add an `after` model validator that rejects scenes where `from_frame + duration_frames > document.duration_frames` and duplicate scene IDs.

Create a version-1 fixture at 1280×720, 30 fps, 90 frames, with one 90-frame scene: ID `phase-1-title`, title `AI Video Factory`, caption `Synthetic local pipeline test`.

- [ ] **Step 4: Run schema tests**

Run: `uv run pytest tests/test_edit_schema.py -v`  
Expected: both tests PASS.

- [ ] **Step 5: Commit the schema contract**

```bash
git add src/ai_video_factory/edit_schema.py fixtures/synthetic-edit.json tests/test_edit_schema.py
git commit -m "feat: validate edit decision documents"
```

---

### Task 4: Read-only doctor command

**Files:**
- Create: `src/ai_video_factory/doctor.py`
- Modify: `src/ai_video_factory/cli.py`
- Create: `tests/test_doctor.py`

**Interfaces:**
- Consumes: a dependency-injected `CommandRunner` callable.
- Produces: `ToolCheck(name, status, version, detail)`, `DoctorReport(schema_version, overall_status, checks)`, and `collect_doctor_report(runner) -> DoctorReport`; CLI JSON response on stdout.

- [ ] **Step 1: Add failing diagnostic tests**

```python
from ai_video_factory.doctor import collect_doctor_report


def test_missing_optional_gpu_tools_are_not_ready() -> None:
    results = {
        ("ffmpeg", "-version"): (0, "ffmpeg version 8.0.1", ""),
        ("node", "--version"): (0, "v22.23.1", ""),
        ("vulkaninfo", "--summary"): (1, "", "no display or DRM device"),
        ("rocminfo",): (127, "", "not found"),
    }

    report = collect_doctor_report(lambda argv: results[tuple(argv)])
    by_name = {check.name: check for check in report.checks}
    assert by_name["ffmpeg"].status == "ready"
    assert by_name["vulkan"].status == "not_ready"
    assert by_name["rocm"].status == "not_ready"
    assert report.overall_status == "degraded"
```

- [ ] **Step 2: Confirm the test fails**

Run: `uv run pytest tests/test_doctor.py -v`  
Expected: FAIL because doctor collection is missing.

- [ ] **Step 3: Implement safe probes and JSON output**

Probe only fixed argv lists: `ffmpeg -version`, `ffprobe -version`, `node --version`, `npm --version`, `git --version`, `vulkaninfo --summary`, `rocminfo`, and `rocm-smi --showproductname`. Use `subprocess.run(..., shell=False, timeout=15, capture_output=True, text=True)`. Convert missing executables, timeouts, permission failures, and nonzero exits to `not_ready` records without raising. Redact line values whose keys contain `TOKEN`, `SECRET`, `PASSWORD`, or `KEY`. The CLI must print only `report.model_dump_json(indent=2)` to stdout and exit 0 for `ready` or `degraded`.

- [ ] **Step 4: Test and perform the real read-only probe**

Run: `uv run pytest tests/test_doctor.py -v && uv run ai-video-factory doctor > /tmp/ai-video-factory-doctor.json`  
Expected: test PASS; JSON parses; FFmpeg and Node are ready; ROCm remains not ready; no installation is attempted.

- [ ] **Step 5: Commit doctor**

```bash
git add src/ai_video_factory/doctor.py src/ai_video_factory/cli.py tests/test_doctor.py
git commit -m "feat: add non-destructive system doctor"
```

---

### Task 5: No-model backend benchmark

**Files:**
- Create: `src/ai_video_factory/benchmark.py`
- Modify: `src/ai_video_factory/cli.py`
- Create: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: injected command runner and monotonic clock.
- Produces: `BenchmarkProbe(name, status, duration_ms, detail)` and `run_benchmarks(runner, clock) -> list[BenchmarkProbe]`; CLI JSON object with `schema_version` and `probes`.

- [ ] **Step 1: Add failing deterministic benchmark test**

```python
from ai_video_factory.benchmark import run_benchmarks


def test_benchmark_records_success_and_failure_duration() -> None:
    times = iter([1.0, 1.125, 2.0, 2.250])
    results = {
        ("ffmpeg", "-version"): (0, "ffmpeg version 8", ""),
        ("vulkaninfo", "--summary"): (1, "", "device unavailable"),
    }
    probes = run_benchmarks(lambda argv: results[tuple(argv)], lambda: next(times))
    assert [(p.name, p.status, p.duration_ms) for p in probes] == [
        ("ffmpeg-startup", "ready", 125),
        ("vulkan-enumeration", "not_ready", 250),
    ]
```

- [ ] **Step 2: Confirm the test fails**

Run: `uv run pytest tests/test_benchmark.py -v`  
Expected: FAIL because benchmark interfaces are missing.

- [ ] **Step 3: Implement bounded probes**

Run only `ffmpeg -version` and `vulkaninfo --summary`, with the same safe subprocess rules as doctor. Report elapsed milliseconds, exit state, and a single sanitized detail line. Do not download a test model, compile llama.cpp, allocate a large buffer, or alter GPU power state.

- [ ] **Step 4: Run tests and the host benchmark**

Run: `uv run pytest tests/test_benchmark.py -v && uv run ai-video-factory benchmark`  
Expected: test PASS and valid JSON; Vulkan may legitimately be `not_ready` in the restricted session.

- [ ] **Step 5: Commit benchmark**

```bash
git add src/ai_video_factory/benchmark.py src/ai_video_factory/cli.py tests/test_benchmark.py
git commit -m "feat: add safe backend readiness benchmark"
```

---

### Task 6: FFmpeg media probe and technical QC

**Files:**
- Create: `src/ai_video_factory/media_probe.py`
- Create: `src/ai_video_factory/qc.py`
- Create: `tests/fixtures/ffprobe-video.json`
- Create: `tests/test_media_probe.py`
- Create: `tests/test_qc.py`

**Interfaces:**
- Consumes: ffprobe JSON and an expected `EditDocument`.
- Produces: `MediaInfo`, `probe_media(path, runner) -> MediaInfo`, `QcCheck`, `QcReport`, and `evaluate_qc(media, edit) -> QcReport`.

- [ ] **Step 1: Add failing probe and QC tests**

```python
import json
from pathlib import Path

from ai_video_factory.edit_schema import load_edit
from ai_video_factory.media_probe import parse_ffprobe
from ai_video_factory.qc import evaluate_qc


def test_valid_synthetic_video_passes_qc() -> None:
    payload = json.loads(Path("tests/fixtures/ffprobe-video.json").read_text())
    media = parse_ffprobe(payload)
    report = evaluate_qc(media, load_edit(Path("fixtures/synthetic-edit.json")))
    assert report.status == "pass"
    assert all(check.passed for check in report.checks)


def test_missing_audio_fails_qc() -> None:
    payload = json.loads(Path("tests/fixtures/ffprobe-video.json").read_text())
    payload["streams"] = [s for s in payload["streams"] if s["codec_type"] != "audio"]
    report = evaluate_qc(parse_ffprobe(payload), load_edit(Path("fixtures/synthetic-edit.json")))
    assert report.status == "fail"
    assert next(c for c in report.checks if c.name == "audio-stream").passed is False
```

- [ ] **Step 2: Confirm the tests fail**

Run: `uv run pytest tests/test_media_probe.py tests/test_qc.py -v`  
Expected: FAIL because probe and QC modules do not exist.

- [ ] **Step 3: Implement parsing, probing, and QC**

The fixture must describe H.264 video at 1280×720 and 30 fps, AAC audio, and 3.0-second format duration. Execute ffprobe with fixed arguments `-v error -show_streams -show_format -of json <path>` and `shell=False`. Parse rational frame rates exactly with `fractions.Fraction`.

QC checks must cover: video stream exists; audio stream exists; width/height equal the edit; frame rate is within 0.01 fps; absolute duration difference is no more than 0.10 seconds; and a full decode command `ffmpeg -v error -i <path> -f null -` exits 0. Return overall `pass` only when all checks pass. Write JSON and Markdown reports via explicit functions `write_qc_reports(report, output_dir)`.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_media_probe.py tests/test_qc.py -v`  
Expected: all tests PASS.

- [ ] **Step 5: Commit media validation**

```bash
git add src/ai_video_factory/media_probe.py src/ai_video_factory/qc.py tests/fixtures/ffprobe-video.json tests/test_media_probe.py tests/test_qc.py
git commit -m "feat: validate rendered media and emit QC reports"
```

---

### Task 7: Deterministic Remotion synthetic composition

**Files:**
- Create: `remotion/package.json`
- Create: `remotion/package-lock.json`
- Create: `remotion/tsconfig.json`
- Create: `remotion/src/index.ts`
- Create: `remotion/src/Root.tsx`
- Create: `remotion/src/SyntheticVideo.tsx`
- Create: `remotion/src/schema.ts`
- Create: `remotion/tests/schema.test.ts`

**Interfaces:**
- Consumes: `fixtures/synthetic-edit.json` copied or passed as input props.
- Produces: composition ID `SyntheticVideo` and command `npm run render -- --props ../fixtures/synthetic-edit.json ../output/synthetic.mp4`.

- [ ] **Step 1: Create a failing TypeScript schema test**

```typescript
import {describe, expect, test} from 'vitest';
import {parseEditDocument} from '../src/schema';

describe('parseEditDocument', () => {
  test('rejects a scene beyond the composition', () => {
    expect(() => parseEditDocument({
      schema_version: 1, width: 1280, height: 720, fps: 30, duration_frames: 30,
      scenes: [{id: 'title', from_frame: 0, duration_frames: 31,
        title: 'Synthetic test', caption: 'Local render'}],
    })).toThrow('scene exceeds composition');
  });
});
```

- [ ] **Step 2: Install pinned project-local packages and confirm failure**

Use exact production dependencies `remotion@4.0.520`, `@remotion/cli@4.0.520`, `@remotion/renderer@4.0.520`, `react@19.2.8`, `react-dom@19.2.8`; use development dependencies `typescript` and `vitest`. Run: `cd remotion && npm install && npm test`  
Expected: FAIL because `src/schema.ts` is missing. `package-lock.json` records the full dependency graph.

- [ ] **Step 3: Implement schema parsing and the composition**

Parse all fields defensively and enforce the same constraints as Python. Register one composition using fixture width, height, fps, and duration. Render a navy-to-black background, centered white title, cyan accent rule, and bottom caption safe within 10% horizontal/vertical margins. Use only local CSS and React elements—no remote fonts, images, audio, or network calls.

Render video only in Remotion. Task 8 must add the required silent stereo audio track deterministically with FFmpeg `anullsrc=r=48000:cl=stereo`, encoded as AAC at 192 kbps; the final artifact must contain both video and audio streams.

- [ ] **Step 4: Test TypeScript and render locally**

Run: `cd remotion && npm test && npm run render -- --props ../fixtures/synthetic-edit.json ../output/synthetic.mp4`  
Expected: tests PASS; a 1280×720, 30 fps, approximately 3-second MP4 is created without any external asset request.

- [ ] **Step 5: Commit the deterministic editor fixture**

```bash
git add remotion fixtures/synthetic-edit.json
git commit -m "feat: add deterministic Remotion smoke composition"
```

---

### Task 8: Resumable synthetic pipeline and Hermes command contract

**Files:**
- Create: `src/ai_video_factory/pipeline.py`
- Modify: `src/ai_video_factory/cli.py`
- Create: `tests/test_pipeline.py`
- Create: `docs/hermes-command-contract.md`
- Create: `README.md`

**Interfaces:**
- Consumes: `RunStore`, `EditDocument`, local Remotion CLI, FFmpeg, and ffprobe.
- Produces: `run_synthetic_pipeline(project_root: Path, data_root: Path) -> PipelineResult`; CLI `ai-video-factory test-pipeline --json`; exit code 0 on QC pass and 2 on failure.

- [ ] **Step 1: Add a failing orchestration test with injected stages**

```python
from pathlib import Path

from ai_video_factory.pipeline import run_synthetic_pipeline


def test_pipeline_resumes_completed_render(tmp_path: Path) -> None:
    calls: list[str] = []

    def render(_edit: Path, output: Path) -> None:
        calls.append("render")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"synthetic")

    def validate(_video: Path, output_dir: Path) -> dict[str, str]:
        calls.append("validate")
        return {"status": "pass", "report": str(output_dir / "qc_report.json")}

    first = run_synthetic_pipeline(Path.cwd(), tmp_path, render=render, validate=validate)
    second = run_synthetic_pipeline(Path.cwd(), tmp_path, render=render, validate=validate)
    assert first.status == second.status == "pass"
    assert calls == ["render", "validate"]
    assert second.resumed is True
```

- [ ] **Step 2: Confirm the orchestration test fails**

Run: `uv run pytest tests/test_pipeline.py -v`  
Expected: FAIL because pipeline orchestration is missing.

- [ ] **Step 3: Implement orchestration and stable JSON response**

Use stage names `synthetic-render` and `synthetic-qc`. Fingerprint the fixture bytes, Remotion lockfile bytes, and command configuration. Render video to a temporary file inside `data/projects/synthetic/runs/<run-id>/`, then always mux `anullsrc=r=48000:cl=stereo` as AAC at 192 kbps into `master.mp4` with `-shortest`; probe and decode the finished file; write `qc_report.json` and `qc_report.md`. Return fields `schema_version`, `command`, `status`, `run_id`, `resumed`, `retryable`, `artifacts`, and `error`. On exceptions, record a failed event, keep prior artifacts, set `retryable` by exception class, and never print a traceback in JSON mode.

Document for Hermes:

```text
ai-video-factory doctor
ai-video-factory benchmark
ai-video-factory test-pipeline --json
```

Hermes must parse stdout as JSON, retain `run_id`, use `retryable` to decide whether to retry, and stop on QC failure. It must not invoke npm, FFmpeg, or internal Python modules directly.

- [ ] **Step 4: Run the complete Phase 1 verification**

Run:

```bash
uv run pytest -v
cd remotion && npm test
cd .. && uv run ai-video-factory doctor
uv run ai-video-factory benchmark
uv run ai-video-factory test-pipeline --json
ffprobe -v error -show_streams -show_format -of json data/projects/synthetic/runs/*/master.mp4
git status --short
```

Expected: Python and TypeScript tests PASS; doctor and benchmark return valid JSON; synthetic pipeline returns `status: "pass"`; ffprobe shows 1280×720 video, 30 fps, audio, and approximately 3 seconds; Git status contains only ignored generated data.

- [ ] **Step 5: Commit the verified Phase 1 pipeline**

```bash
git add src/ai_video_factory/pipeline.py src/ai_video_factory/cli.py tests/test_pipeline.py docs/hermes-command-contract.md README.md
git commit -m "feat: complete Hermes-ready phase 1 pipeline"
```

---

## Phase 1 completion gate

Phase 1 is complete only when every test above passes from a clean checkout, the synthetic MP4 passes technical QC, the doctor reports unavailable GPU runtimes without attempting repair, and `git status --short` is clean. The next phase must begin with a direct-device Vulkan probe and a written ROCm package-coherence proposal; it must not download an LLM until the user approves the exact file, size, license, and backend.
