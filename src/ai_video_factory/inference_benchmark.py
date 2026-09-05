"""Deterministic, persistence-safe probes for OpenAI-compatible inference."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ai_video_factory.inference_models import InferenceCheck, InferenceResult
from ai_video_factory.inference_service import InferenceService
from ai_video_factory.lm_studio import LmStudioError
from ai_video_factory.run_store import RunStore


Clock = Callable[[], float]

_STAGE = "lm-studio-capability"
_MODEL_IDENTIFIER = "avf-qwen36-executor"
_HTTP_TIMEOUT_SECONDS = 60.0
_CHECK_NAMES = (
    "ordinary_generation",
    "structured_output",
    "tool_calling",
)


class _BenchmarkClockError(ValueError):
    """Raised when the injected monotonic clock violates its contract."""


_RESPONSE_FORMAT: dict[str, object] = {
    "type": "json_schema",
    "json_schema": {
        "name": "local_capability",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["LOCAL_OK"]}},
            "required": ["status"],
            "additionalProperties": False,
        },
    },
}

_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "record_scene",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string"},
                    "duration_seconds": {"type": "integer"},
                },
                "required": ["scene_id", "duration_seconds"],
                "additionalProperties": False,
            },
        },
    }
]


class _Usage(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> _Usage:
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total token count does not match its components")
        return self


class _FunctionCall(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    name: str
    arguments: str


class _ToolCall(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    id: str
    type: Literal["function"]
    function: _FunctionCall


class _Message(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    role: Literal["assistant"]
    content: str | None
    tool_calls: list[_ToolCall] | None = None


class _Choice(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    index: int
    message: _Message
    finish_reason: str | None


class _Completion(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[_Choice] = Field(min_length=1, max_length=1)
    usage: _Usage


class _LocalCapability(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    status: Literal["LOCAL_OK"]


class _SceneArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    scene_id: str
    duration_seconds: int


class _Report(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1]
    run_id: str
    resumed: bool
    checks: dict[str, Literal["pass", "fail"]]
    latency_ms: dict[str, float]
    usage: dict[str, _Usage]
    provenance: dict[str, object]

    @model_validator(mode="after")
    def validate_probe_keys(self) -> _Report:
        expected = set(_CHECK_NAMES)
        if set(self.checks) != expected or set(self.latency_ms) != expected:
            raise ValueError("capability report probe keys are invalid")
        if not set(self.usage).issubset(expected):
            raise ValueError("capability report usage keys are invalid")
        if any(not math.isfinite(value) or value < 0 for value in self.latency_ms.values()):
            raise ValueError("capability report latency is invalid")
        return self


def _common_request(prompt: str) -> dict[str, object]:
    return {
        "model": _MODEL_IDENTIFIER,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 7,
        "stream": False,
        "max_tokens": 512,
    }


def _requests() -> tuple[dict[str, object], ...]:
    ordinary = _common_request("Reply exactly LOCAL_OK.")
    structured = _common_request(
        'Return a JSON object whose status is exactly "LOCAL_OK".'
    )
    structured["response_format"] = _RESPONSE_FORMAT
    tool = _common_request(
        "Call record_scene for scene intro lasting 3 seconds."
    )
    tool["tools"] = _TOOLS
    tool["tool_choice"] = "required"
    return ordinary, structured, tool


def _completion(response: dict[str, object]) -> _Completion | None:
    try:
        parsed = _Completion.model_validate(response, strict=True)
    except ValidationError:
        return None
    if parsed.model != _MODEL_IDENTIFIER:
        return None
    return parsed


def _ordinary_passes(completion: _Completion) -> bool:
    content = completion.choices[0].message.content
    return content is not None and content.strip() == "LOCAL_OK"


def _structured_passes(completion: _Completion) -> bool:
    content = completion.choices[0].message.content
    if content is None:
        return False
    try:
        _LocalCapability.model_validate_json(content, strict=True)
    except (ValidationError, ValueError):
        return False
    return True


def _tool_passes(completion: _Completion) -> bool:
    tool_calls = completion.choices[0].message.tool_calls
    if tool_calls is None or len(tool_calls) != 1:
        return False
    function = tool_calls[0].function
    if function.name != "record_scene":
        return False
    try:
        arguments = _SceneArguments.model_validate_json(function.arguments, strict=True)
    except (ValidationError, ValueError):
        return False
    return arguments.model_dump() == {
        "scene_id": "intro",
        "duration_seconds": 3,
    }


def _duration_ms(start: float, end: float) -> float:
    if not all(math.isfinite(value) for value in (start, end)) or end < start:
        raise _BenchmarkClockError("capability benchmark clock returned invalid values")
    return round((end - start) * 1000, 3)


def _metrics(
    report: _Report, *, resumed: bool
) -> dict[str, int | float | str | bool | None]:
    metrics: dict[str, int | float | str | bool | None] = {
        "run_id": report.run_id,
        "resumed": resumed,
    }
    for name in _CHECK_NAMES:
        metrics[f"{name}_latency_ms"] = report.latency_ms[name]
        usage = report.usage.get(name)
        if usage is not None:
            metrics[f"{name}_prompt_tokens"] = usage.prompt_tokens
            metrics[f"{name}_completion_tokens"] = usage.completion_tokens
            metrics[f"{name}_total_tokens"] = usage.total_tokens
    return metrics


def _result_from_report(
    report: _Report, report_path: Path, *, resumed: bool
) -> InferenceResult:
    passed = all(value == "pass" for value in report.checks.values())
    return InferenceResult(
        command="benchmark",
        status="pass" if passed else "fail",
        retryable=False,
        model_identifier=_MODEL_IDENTIFIER,
        checks={
            name: InferenceCheck(
                status="ready" if status == "pass" else "not_ready", detail=None
            )
            for name, status in report.checks.items()
        },
        metrics=_metrics(report, resumed=resumed),
        artifacts={"report": str(report_path)},
        error=None if passed else "one or more local capability checks failed",
    )


def _safe_error(error: BaseException) -> tuple[str, bool]:
    detail = str(error).casefold()
    if isinstance(error, _BenchmarkClockError):
        return "capability benchmark clock returned invalid values", False
    if "identifier does not match benchmark" in detail:
        return "configured model identifier does not match benchmark", True
    if isinstance(error, TimeoutError) or "timed out" in detail or "timeout" in detail:
        return "LM Studio capability request timed out", True
    if "status" in detail:
        return "LM Studio capability request failed with an HTTP status", True
    if "exceed" in detail or "oversized" in detail or "too large" in detail:
        return "LM Studio capability response exceeded the size limit", True
    if "json" in detail:
        return "LM Studio capability response was invalid", True
    if isinstance(error, LmStudioError):
        if "not loaded" in detail or "not running" in detail:
            return "configured LM Studio model is not ready", True
        return "LM Studio capability request failed", True
    return "local capability benchmark failed", False


def _failure_result(
    service: InferenceService,
    error: BaseException,
    *,
    run_id: str | None,
    resumed: bool,
) -> InferenceResult:
    safe_error, retryable = _safe_error(error)
    return InferenceResult(
        command="benchmark",
        status="not_ready" if retryable else "fail",
        retryable=retryable,
        model_identifier=service.config.identifier,
        checks={
            "operation": InferenceCheck(status="not_ready", detail=None),
        },
        metrics={"run_id": run_id, "resumed": resumed},
        artifacts={},
        error=safe_error,
    )


def _write_report(path: Path, report: _Report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as destination:
            destination.write(report.model_dump_json(indent=2))
            destination.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_verified_report(
    path: Path, *, run_id: str, inputs: dict[str, object]
) -> _Report:
    try:
        report = _Report.model_validate_json(path.read_text(encoding="utf-8"), strict=True)
    except (OSError, ValidationError, ValueError) as error:
        raise LmStudioError("verified capability report JSON is invalid") from error
    if report.run_id != run_id or report.provenance != inputs:
        raise LmStudioError("verified capability report provenance does not match")
    return report


def current_text_capability(
    service: InferenceService, store: RunStore, data_root: Path,
) -> bool:
    """Read the current, integrity-checked three-probe report without resuming it."""
    try:
        inputs = service.current_capability_inputs(Path(data_root))
        run = store.completed_read_only(_STAGE, inputs)
        if run is None:
            return False
        report_path = Path(str(run.artifacts["report"]))
        report = _load_verified_report(report_path, run_id=run.run_id, inputs=inputs)
    except (KeyError, OSError, ValueError, LmStudioError):
        return False
    return report.checks == {
        "ordinary_generation": "pass",
        "structured_output": "pass",
        "tool_calling": "pass",
    }


def run_capability_benchmark(
    service: InferenceService,
    store: RunStore,
    data_root: Path,
    clock: Clock = time.monotonic,
) -> InferenceResult:
    """Run or safely resume the fixed three-probe capability benchmark."""
    run = None
    try:
        if service.config.identifier != _MODEL_IDENTIFIER:
            raise LmStudioError("configured model identifier does not match benchmark")
        inputs = service.capability_inputs(Path(data_root))
        while True:
            run = store.start(_STAGE, inputs)
            report_path = (
                Path(data_root)
                / "projects"
                / "system"
                / "runs"
                / run.run_id
                / "inference_report.json"
            )
            if not run.resumed:
                break
            recorded_path = Path(str(run.artifacts.get("report", report_path)))
            try:
                report = _load_verified_report(
                    recorded_path, run_id=run.run_id, inputs=inputs
                )
            except LmStudioError:
                store.invalidate_completed(
                    run.run_id, inputs, "capability report validation failed"
                )
                run = None
                continue
            return _result_from_report(report, recorded_path, resumed=True)

        checks: dict[str, Literal["pass", "fail"]] = {}
        latency_ms: dict[str, float] = {}
        usage: dict[str, _Usage] = {}
        validators = (_ordinary_passes, _structured_passes, _tool_passes)
        for name, request, validator in zip(
            _CHECK_NAMES, _requests(), validators, strict=True
        ):
            started = clock()
            response = service.chat_completion(request, timeout=_HTTP_TIMEOUT_SECONDS)
            finished = clock()
            latency_ms[name] = _duration_ms(started, finished)
            completion = _completion(response)
            checks[name] = (
                "pass" if completion is not None and validator(completion) else "fail"
            )
            if completion is not None:
                usage[name] = completion.usage

        report = _Report(
            schema_version=1,
            run_id=run.run_id,
            resumed=False,
            checks=checks,
            latency_ms=latency_ms,
            usage=usage,
            provenance=inputs,
        )
        _write_report(report_path, report)
        store.complete(
            run.run_id,
            {"report": str(report_path)},
            expected_artifacts={"report": report_path},
        )
        return _result_from_report(report, report_path, resumed=False)
    except Exception as error:
        safe_error, _ = _safe_error(error)
        if run is not None and not run.resumed:
            try:
                store.fail(run.run_id, safe_error)
            except Exception:
                pass
        return _failure_result(
            service,
            error,
            run_id=None if run is None else run.run_id,
            resumed=bool(run and run.resumed),
        )
