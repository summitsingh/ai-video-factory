from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ai_video_factory.inference_benchmark import run_capability_benchmark
from ai_video_factory.inference_config import InferenceConfig
from ai_video_factory.inference_service import InferenceService
from ai_video_factory.lm_studio import LmStudioError, LmStudioModel
from ai_video_factory.run_store import RunStore


@dataclass(frozen=True)
class HttpCall:
    method: str
    url: str
    body: dict[str, object] | None
    headers: Mapping[str, str]
    timeout: float

    @property
    def path(self) -> str:
        return self.url.removeprefix("http://127.0.0.1:1234")


class RecordingHttpTransport:
    def __init__(
        self,
        responses: list[dict[str, object]] | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[HttpCall] = []

    def __call__(
        self,
        method: str,
        url: str,
        body: dict[str, object] | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> dict[str, object]:
        self.calls.append(HttpCall(method, url, body, headers, timeout))
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


class BenchmarkBackend:
    def __init__(
        self,
        config: InferenceConfig,
        http: RecordingHttpTransport,
        *,
        loaded_identifier: str | None = "avf-qwen36-executor",
    ) -> None:
        self.config = config
        self.http = http
        self.loaded_identifier = loaded_identifier

    def snapshot(self) -> object:
        model_path = Path(self.config.models_directory) / "publisher" / "model.gguf"
        loaded = () if self.loaded_identifier is None else (self.loaded_identifier,)
        return type(
            "Snapshot",
            (),
            {
                "cli_help": "lms 0.0.47",
                "runtimes": "llama.cpp-linux-x86_64-amd-rocm-avx2 1.66.0",
                "runtime_survey": "AMD Radeon 8060S",
                "server_running": True,
                "loaded_identifiers": loaded,
                "configured_model": LmStudioModel(
                    model_key=self.config.model_key,
                    path=model_path,
                    relative_path="publisher/model.gguf",
                    size_bytes=model_path.stat().st_size,
                ),
                "configured_model_loaded": self.config.identifier in loaded,
                "checks": {},
            },
        )()


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def _response(
    content: str | None,
    *,
    model: str = "avf-qwen36-executor",
    tool_calls: list[dict[str, object]] | None = None,
    usage: dict[str, object] | None = None,
    reasoning_content: str = "REASONING_MUST_NOT_PERSIST",
) -> dict[str, object]:
    message: dict[str, object] = {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning_content,
    }
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-secret-id",
        "object": "chat.completion",
        "created": 1_777_777_777,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage
        or {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
    }


def passing_responses(*, response_text: str = "LOCAL_OK") -> list[dict[str, object]]:
    return [
        _response("LOCAL_OK", reasoning_content=response_text),
        _response('{"status":"LOCAL_OK"}'),
        _response(
            None,
            tool_calls=[
                {
                    "id": "call-secret-id",
                    "type": "function",
                    "function": {
                        "name": "record_scene",
                        "arguments": '{"scene_id":"intro","duration_seconds":3}',
                    },
                }
            ],
        ),
    ]


def benchmark_fixture(
    tmp_path: Path,
    *,
    http: RecordingHttpTransport | None = None,
    response_text: str = "LOCAL_OK",
    loaded_identifier: str | None = "avf-qwen36-executor",
    configured_identifier: str = "avf-qwen36-executor",
) -> tuple[InferenceService, RunStore, Path, RecordingHttpTransport]:
    data_root = tmp_path / "data"
    model_path = tmp_path / "models" / "publisher" / "model.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"deterministic-model")
    config = InferenceConfig.model_validate(
        {
            "schema_version": 1,
            "backend": "lm_studio",
            "base_url": "http://127.0.0.1:1234/v1",
            "lms_binary": "lms",
            "model_key": "qwen3.6-35b-a3b-udt-mtp",
            "identifier": configured_identifier,
            "context_length": 65_536,
            "gpu": "max",
            "parallel": 1,
            "ttl_seconds": 3_600,
            "minimum_available_memory_gib": 40,
            "models_directory": str(tmp_path / "models"),
        }
    )
    transport = http or RecordingHttpTransport(
        passing_responses(response_text=response_text)
    )
    backend = BenchmarkBackend(
        config, transport, loaded_identifier=loaded_identifier
    )
    service = InferenceService(config, backend=backend)  # type: ignore[arg-type]
    artifact_root = data_root / "projects" / "system" / "runs"
    store = RunStore(
        data_root / "projects" / "system" / "state", artifact_root=artifact_root
    )
    return service, store, data_root, transport


def _seed_completed_report(
    service: InferenceService,
    store: RunStore,
    data_root: Path,
    *,
    invalid_field: str,
) -> str:
    inputs = service.capability_inputs(data_root)
    run = store.start("lm-studio-capability", inputs)
    report_path = (
        data_root
        / "projects"
        / "system"
        / "runs"
        / run.run_id
        / "inference_report.json"
    )
    report = {
        "schema_version": 1,
        "run_id": run.run_id,
        "resumed": False,
        "checks": {
            "ordinary_generation": "pass",
            "structured_output": "pass",
            "tool_calling": "pass",
        },
        "latency_ms": {
            "ordinary_generation": 1.0,
            "structured_output": 1.0,
            "tool_calling": 1.0,
        },
        "usage": {
            name: {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }
            for name in (
                "ordinary_generation",
                "structured_output",
                "tool_calling",
            )
        },
        "provenance": inputs,
    }
    if invalid_field == "schema":
        report["checks"] = {"ordinary_generation": "pass"}
    elif invalid_field == "run_id":
        report["run_id"] = "0" * 32
    elif invalid_field == "provenance":
        report["provenance"] = {"stale": True}
    else:
        raise AssertionError(f"unknown invalid field: {invalid_field}")
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    store.complete(
        run.run_id,
        {"report": str(report_path)},
        expected_artifacts={"report": report_path},
    )
    return run.run_id


def _read_report(result: Any) -> tuple[Path, dict[str, object]]:
    report_path = Path(result.artifacts["report"])
    return report_path, json.loads(report_path.read_text(encoding="utf-8"))


def test_benchmark_sends_three_deterministic_requests_without_credentials(
    tmp_path: Path,
) -> None:
    service, store, data_root, transport = benchmark_fixture(tmp_path)

    result = run_capability_benchmark(service, store, data_root)

    assert result.status == "pass"
    assert [call.path for call in transport.calls] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert all(call.method == "POST" for call in transport.calls)
    assert all("Authorization" not in call.headers for call in transport.calls)
    assert all(
        call.headers
        == {"Content-Type": "application/json", "Accept": "application/json"}
        for call in transport.calls
    )
    assert all(call.body is not None for call in transport.calls)
    assert all(
        call.body["model"] == "avf-qwen36-executor"
        for call in transport.calls
        if call.body
    )
    assert all(call.body["temperature"] == 0 for call in transport.calls if call.body)
    assert all(call.body["seed"] == 7 for call in transport.calls if call.body)
    assert all(call.body["stream"] is False for call in transport.calls if call.body)
    assert all(
        call.body["max_tokens"] == 512
        for call in transport.calls
        if call.body
    )
    serialized_requests = json.dumps([call.body for call in transport.calls])
    for thinking_control in ("/no_think", "<|think_off|>", "chat_template_kwargs"):
        assert thinking_control not in serialized_requests
    assert [call.body["messages"] for call in transport.calls] == [
        [{"role": "user", "content": "Reply exactly LOCAL_OK."}],
        [
            {
                "role": "user",
                "content": 'Return a JSON object whose status is exactly "LOCAL_OK".',
            }
        ],
        [
            {
                "role": "user",
                "content": "Call record_scene for scene intro lasting 3 seconds.",
            }
        ],
    ]
    assert transport.calls[0].body == {
        "model": "avf-qwen36-executor",
        "messages": [{"role": "user", "content": "Reply exactly LOCAL_OK."}],
        "temperature": 0,
        "seed": 7,
        "stream": False,
        "max_tokens": 512,
    }
    assert transport.calls[1].body["response_format"] == {
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
    assert transport.calls[2].body["tools"] == [
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
    assert len(transport.calls[2].body["tools"]) == 1
    assert transport.calls[2].body["tool_choice"] == "required"


def test_report_persists_metrics_but_not_prompts_or_response_text(
    tmp_path: Path,
) -> None:
    secret_response = "MODEL_RESPONSE_MUST_NOT_PERSIST"
    service, store, data_root, _transport = benchmark_fixture(
        tmp_path, response_text=secret_response
    )

    result = run_capability_benchmark(service, store, data_root)

    _report_path, report = _read_report(result)
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in data_root.rglob("*")
        if path.is_file()
    )
    assert secret_response not in persisted
    assert "chat_template_kwargs" not in persisted
    assert "Reply exactly LOCAL_OK" not in persisted
    assert "call-secret-id" not in persisted
    assert "chatcmpl-secret-id" not in persisted
    assert "REASONING_MUST_NOT_PERSIST" not in persisted
    assert '"arguments"' not in persisted
    assert report["checks"]["ordinary_generation"] == "pass"
    assert set(report) == {
        "schema_version",
        "run_id",
        "resumed",
        "checks",
        "latency_ms",
        "usage",
        "provenance",
    }


def test_resumes_only_matching_verified_report(tmp_path: Path) -> None:
    service, store, data_root, transport = benchmark_fixture(tmp_path)
    first = run_capability_benchmark(service, store, data_root)

    second = run_capability_benchmark(service, store, data_root)

    assert first.metrics["resumed"] is False
    assert second.metrics["resumed"] is True
    assert second.metrics["run_id"] == first.metrics["run_id"]
    assert len(transport.calls) == 3


def test_tampered_report_is_invalidated_and_reexecuted(tmp_path: Path) -> None:
    service, store, data_root, transport = benchmark_fixture(tmp_path)
    first = run_capability_benchmark(service, store, data_root)
    Path(first.artifacts["report"]).write_text("{}\n", encoding="utf-8")
    transport.responses.extend(passing_responses())

    second = run_capability_benchmark(service, store, data_root)

    assert second.metrics["resumed"] is False
    assert second.metrics["run_id"] != first.metrics["run_id"]
    assert len(transport.calls) == 6


@pytest.mark.parametrize("invalid_field", ["schema", "run_id", "provenance"])
def test_invalid_resumed_report_is_invalidated_then_reexecuted_once(
    tmp_path: Path, invalid_field: str
) -> None:
    service, store, data_root, transport = benchmark_fixture(tmp_path)
    stale_run_id = _seed_completed_report(
        service, store, data_root, invalid_field=invalid_field
    )

    replacement = run_capability_benchmark(service, store, data_root)
    resumed = run_capability_benchmark(service, store, data_root)

    assert replacement.status == "pass"
    assert replacement.metrics["resumed"] is False
    assert replacement.metrics["run_id"] != stale_run_id
    assert resumed.status == "pass"
    assert resumed.metrics["resumed"] is True
    assert resumed.metrics["run_id"] == replacement.metrics["run_id"]
    assert len(transport.calls) == 3
    stale_manifest = json.loads(
        (
            data_root
            / "projects"
            / "system"
            / "state"
            / "lm-studio-capability"
            / stale_run_id
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert stale_manifest["status"] == "failed"
    assert stale_manifest["error"] == "capability report validation failed"


def test_duration_and_usage_metrics_are_recorded_per_probe(tmp_path: Path) -> None:
    service, store, data_root, _transport = benchmark_fixture(tmp_path)
    clock = FakeClock([1.0, 1.012, 2.0, 2.025, 3.0, 3.125])

    result = run_capability_benchmark(service, store, data_root, clock=clock)

    _path, report = _read_report(result)
    assert report["latency_ms"] == {
        "ordinary_generation": 12.0,
        "structured_output": 25.0,
        "tool_calling": 125.0,
    }
    assert report["usage"] == {
        probe: {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}
        for probe in ("ordinary_generation", "structured_output", "tool_calling")
    }


@pytest.mark.parametrize(
    "clock_values",
    [
        [math.nan, 1.0],
        [1.0, math.inf],
        [2.0, 1.0],
    ],
    ids=["non-finite-start", "non-finite-end", "regressing"],
)
def test_invalid_clock_values_fail_safely_without_report(
    tmp_path: Path, clock_values: list[float]
) -> None:
    service, store, data_root, transport = benchmark_fixture(tmp_path)

    result = run_capability_benchmark(
        service, store, data_root, clock=FakeClock(clock_values)
    )

    assert result.status == "fail"
    assert result.retryable is False
    assert result.error == "capability benchmark clock returned invalid values"
    assert len(transport.calls) == 1
    assert result.artifacts == {}
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in data_root.rglob("*")
        if path.is_file()
    )
    assert "nan" not in persisted.casefold()
    assert "inf" not in persisted.casefold()


def test_model_not_loaded_is_not_ready_without_http_requests(tmp_path: Path) -> None:
    service, store, data_root, transport = benchmark_fixture(
        tmp_path, loaded_identifier=None
    )

    result = run_capability_benchmark(service, store, data_root)

    assert result.status == "not_ready"
    assert result.retryable is True
    assert transport.calls == []


def test_wrong_response_model_identifier_fails_all_checks(tmp_path: Path) -> None:
    responses = [
        _response("LOCAL_OK", model="other-model"),
        _response('{"status":"LOCAL_OK"}', model="other-model"),
        _response(
            None,
            model="other-model",
            tool_calls=[
                {
                    "id": "call-id",
                    "type": "function",
                    "function": {
                        "name": "record_scene",
                        "arguments": '{"scene_id":"intro","duration_seconds":3}',
                    },
                }
            ],
        ),
    ]
    transport = RecordingHttpTransport(responses)
    service, store, data_root, _ = benchmark_fixture(tmp_path, http=transport)

    result = run_capability_benchmark(service, store, data_root)

    assert result.status == "fail"
    _path, report = _read_report(result)
    assert set(report["checks"].values()) == {"fail"}


def test_configured_identifier_mismatch_fails_before_http_or_persistence(
    tmp_path: Path,
) -> None:
    service, store, data_root, transport = benchmark_fixture(
        tmp_path,
        configured_identifier="other-stable-model",
        loaded_identifier="other-stable-model",
    )

    result = run_capability_benchmark(service, store, data_root)

    assert result.status == "not_ready"
    assert result.retryable is True
    assert result.error == "configured model identifier does not match benchmark"
    assert result.model_identifier == "other-stable-model"
    assert transport.calls == []
    assert not (data_root / "projects" / "system" / "state").exists()


@pytest.mark.parametrize(
    ("error", "expected_error"),
    [
        (TimeoutError("token=timeout-secret"), "timed out"),
        (
            LmStudioError("HTTP status 503 body MODEL_RESPONSE_MUST_NOT_PERSIST"),
            "HTTP status",
        ),
        (
            LmStudioError(
                "response exceeds 2 MiB limit MODEL_RESPONSE_MUST_NOT_PERSIST"
            ),
            "size limit",
        ),
        (
            LmStudioError(
                "response was not valid JSON MODEL_RESPONSE_MUST_NOT_PERSIST"
            ),
            "invalid",
        ),
    ],
)
def test_transport_failures_persist_only_sanitized_bounded_diagnostics(
    tmp_path: Path, error: BaseException, expected_error: str
) -> None:
    transport = RecordingHttpTransport(error=error)
    service, store, data_root, _ = benchmark_fixture(tmp_path, http=transport)

    result = run_capability_benchmark(service, store, data_root)

    assert result.status == "not_ready"
    assert expected_error in (result.error or "")
    state_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (data_root / "projects" / "system" / "state").rglob("*")
        if path.is_file()
    )
    assert "timeout-secret" not in state_text
    assert "MODEL_RESPONSE_MUST_NOT_PERSIST" not in state_text
    assert "Authorization" not in state_text
    assert not list((data_root / "projects" / "system" / "runs").rglob("*.json"))


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda responses: responses.__setitem__(0, _response("NOT_OK")), "ordinary_generation"),
        (lambda responses: responses.__setitem__(0, {"model": "avf-qwen36-executor", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}), "ordinary_generation"),
        (lambda responses: responses.__setitem__(1, _response("not-json")), "structured_output"),
        (lambda responses: responses.__setitem__(1, _response('{"status":"LOCAL_OK","extra":true}')), "structured_output"),
        (lambda responses: responses.__setitem__(2, _response(None, tool_calls=[])), "tool_calling"),
        (lambda responses: responses.__setitem__(2, _response(None, tool_calls=[{"id": "id", "type": "function", "function": {"name": "wrong_tool", "arguments": '{"scene_id":"intro","duration_seconds":3}'}}])), "tool_calling"),
        (lambda responses: responses.__setitem__(2, _response(None, tool_calls=[{"id": "id", "type": "function", "function": {"name": "record_scene", "arguments": "not-json"}}])), "tool_calling"),
        (lambda responses: responses.__setitem__(2, _response(None, tool_calls=[{"id": "id", "type": "function", "function": {"name": "record_scene", "arguments": '{"scene_id":"intro","duration_seconds":"3"}'}}])), "tool_calling"),
    ],
)
def test_invalid_probe_outputs_fail_only_the_affected_check(
    tmp_path: Path,
    mutation: Callable[[list[dict[str, object]]], None],
    failed_check: str,
) -> None:
    responses = passing_responses()
    mutation(responses)
    transport = RecordingHttpTransport(responses)
    service, store, data_root, _ = benchmark_fixture(tmp_path, http=transport)

    result = run_capability_benchmark(service, store, data_root)

    assert result.status == "fail"
    _path, report = _read_report(result)
    assert report["checks"][failed_check] == "fail"
    assert len(transport.calls) == 3


def _assert_invalid_tool_calls_do_not_persist(
    tmp_path: Path,
    tool_calls: list[dict[str, object]],
    forbidden_value: str,
) -> None:
    responses = passing_responses()
    responses[2] = _response(None, tool_calls=tool_calls)
    service, store, data_root, transport = benchmark_fixture(
        tmp_path, http=RecordingHttpTransport(responses)
    )

    result = run_capability_benchmark(service, store, data_root)

    assert result.status == "fail"
    _report_path, report = _read_report(result)
    assert report["checks"]["tool_calling"] == "fail"
    assert len(transport.calls) == 3
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in data_root.rglob("*")
        if path.is_file()
    )
    assert forbidden_value not in persisted


def test_multiple_tool_calls_fail_without_persisting_arguments(tmp_path: Path) -> None:
    _assert_invalid_tool_calls_do_not_persist(
        tmp_path,
        [
            {
                "id": "first-secret-id",
                "type": "function",
                "function": {
                    "name": "record_scene",
                    "arguments": '{"scene_id":"first-secret","duration_seconds":3}',
                },
            },
            {
                "id": "second-secret-id",
                "type": "function",
                "function": {
                    "name": "record_scene",
                    "arguments": '{"scene_id":"second-secret","duration_seconds":3}',
                },
            },
        ],
        "first-secret",
    )


def test_wrong_scene_id_fails_without_persisting_arguments(tmp_path: Path) -> None:
    _assert_invalid_tool_calls_do_not_persist(
        tmp_path,
        [
            {
                "id": "scene-secret-id",
                "type": "function",
                "function": {
                    "name": "record_scene",
                    "arguments": '{"scene_id":"wrong-secret-scene","duration_seconds":3}',
                },
            }
        ],
        "wrong-secret-scene",
    )


def test_wrong_duration_fails_without_persisting_arguments(tmp_path: Path) -> None:
    _assert_invalid_tool_calls_do_not_persist(
        tmp_path,
        [
            {
                "id": "duration-secret-id",
                "type": "function",
                "function": {
                    "name": "record_scene",
                    "arguments": (
                        '{"scene_id":"intro","duration_seconds":987654321}'
                    ),
                },
            }
        ],
        "987654321",
    )


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3},
        {"prompt_tokens": -1, "completion_tokens": 2, "total_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": 2},
    ],
)
def test_invalid_usage_fails_the_affected_probe(
    tmp_path: Path, usage: dict[str, object]
) -> None:
    responses = passing_responses()
    responses[0] = _response("LOCAL_OK", usage=usage)
    service, store, data_root, _ = benchmark_fixture(
        tmp_path, http=RecordingHttpTransport(responses)
    )

    result = run_capability_benchmark(service, store, data_root)

    assert result.status == "fail"
    _path, report = _read_report(result)
    assert report["checks"]["ordinary_generation"] == "fail"
    assert "ordinary_generation" not in report["usage"]
