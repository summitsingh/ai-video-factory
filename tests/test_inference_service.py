from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_video_factory.inference_config import InferenceConfig
from ai_video_factory.inference_models import MemoryEstimate
from ai_video_factory.inference_service import InferenceService, available_memory_gib
from ai_video_factory.lm_studio import LmStudioError, LmStudioModel
from ai_video_factory.run_store import fingerprint_inputs
from ai_video_factory.sanitization import MAX_DIAGNOSTIC_CHARS


ESTIMATE_COMMAND = (
    "lms", "load", "tiel-coder-35b-a3b-mtp", "--gpu", "max",
    "--context-length", "65536", "--no-speculative-draft-mtp",
    "--estimate-only", "-y",
)
LOAD_COMMAND = (
    "lms", "load", "tiel-coder-35b-a3b-mtp", "--gpu", "max",
    "--context-length", "65536", "--parallel", "1", "--ttl", "3600",
    "--no-speculative-draft-mtp", "--identifier", "avf-tiel-coder-executor", "-y",
)
SERVER_START_COMMAND = (
    "lms", "server", "start", "--port", "1234", "--bind", "127.0.0.1",
)
UNLOAD_COMMAND = ("lms", "unload", "avf-tiel-coder-executor")
FORBIDDEN_FRAGMENTS = (
    ("get",),
    ("runtime", "select"),
    ("runtime", "update"),
    ("unload", "--all"),
    ("server", "stop"),
)


@pytest.fixture
def config(tmp_path: Path) -> InferenceConfig:
    return InferenceConfig.model_validate(
        {
            "schema_version": 1,
            "backend": "lm_studio",
            "base_url": "http://127.0.0.1:1234/v1",
            "lms_binary": "lms",
            "model_key": "tiel-coder-35b-a3b-mtp",
            "identifier": "avf-tiel-coder-executor",
            "context_length": 65_536,
            "gpu": "max",
            "parallel": 1,
            "ttl_seconds": 3_600,
            "minimum_available_memory_gib": 40,
            "models_directory": str(tmp_path / "models"),
            "server_config_path": str(tmp_path / "http-server-config.json"),
        }
    )


class FakeBackend:
    def __init__(
        self,
        config: InferenceConfig,
        *,
        loaded: list[str] | None = None,
        server_running: bool = True,
        cli_load_visible: bool = True,
        api_visible: bool = True,
        remove_unrelated_on_stop: bool = False,
        remove_unrelated_on_server_start: bool = False,
        estimate_error: BaseException | None = None,
        snapshot_error: BaseException | None = None,
        unsafe_after_server_start: bool = False,
    ) -> None:
        self.config = config
        self.loaded = list(loaded or [])
        self.server_running = server_running
        self.cli_load_visible = cli_load_visible
        self.api_visible = api_visible
        self.remove_unrelated_on_stop = remove_unrelated_on_stop
        self.remove_unrelated_on_server_start = remove_unrelated_on_server_start
        self.estimate_error = estimate_error
        self.snapshot_error = snapshot_error
        self.unsafe_after_server_start = unsafe_after_server_start
        self.server_was_started = False
        self.calls: list[tuple[str, ...]] = []
        self.load_calls: list[tuple[str, ...]] = []
        self.unload_calls: list[tuple[str, ...]] = []
        self.mutating_calls: list[tuple[str, ...]] = []
        self.estimate_calls = 0
        self.snapshot_calls = 0
        self.api_calls = 0

    def estimate(self) -> MemoryEstimate:
        self.estimate_calls += 1
        self.calls.append(ESTIMATE_COMMAND)
        if self.estimate_error is not None:
            raise self.estimate_error
        return MemoryEstimate(
            gpu_gib=17.36,
            total_gib=17.36,
            confidence="LOW",
            allowed=True,
        )

    def snapshot(self) -> SimpleNamespace:
        self.snapshot_calls += 1
        if self.snapshot_error is not None:
            raise self.snapshot_error
        if self.unsafe_after_server_start and self.server_was_started:
            raise LmStudioError("LM Studio server configuration is unsafe")
        model_path = Path(self.config.models_directory) / "publisher" / "model.gguf"
        return SimpleNamespace(
            cli_help="lms 0.0.47",
            cli_path="/safe/lms",
            cli_commit="07b7252",
            runtimes="llama.cpp-linux-x86_64-amd-rocm-avx2@2.31.2 ✓ GGUF",
            selected_runtime="llama.cpp-linux-x86_64-amd-rocm-avx2@2.31.2",
            runtime_survey=(
                "Survey by llama.cpp-linux-x86_64-amd-rocm-avx2 (2.31.2)\n"
                "AMD Radeon Graphics 85.67 GiB"
            ),
            server_running=self.server_running,
            loaded_identifiers=tuple(self.loaded),
            configured_model=LmStudioModel(
                model_key=self.config.model_key,
                path=model_path,
                relative_path="publisher/model.gguf",
                size_bytes=model_path.stat().st_size if model_path.exists() else 0,
            ),
            configured_model_loaded=self.config.identifier in self.loaded,
            checks={},
        )

    def start_server(self) -> None:
        self.calls.append(SERVER_START_COMMAND)
        self.mutating_calls.append(SERVER_START_COMMAND)
        self.server_running = True
        self.server_was_started = True
        if self.remove_unrelated_on_server_start:
            self.loaded = [
                value for value in self.loaded if value == self.config.identifier
            ]

    def start(self) -> None:
        self.calls.append(LOAD_COMMAND)
        self.load_calls.append(LOAD_COMMAND)
        self.mutating_calls.append(LOAD_COMMAND)
        if self.cli_load_visible and self.config.identifier not in self.loaded:
            self.loaded.append(self.config.identifier)

    def stop(self) -> None:
        self.calls.append(UNLOAD_COMMAND)
        self.unload_calls.append(UNLOAD_COMMAND)
        self.mutating_calls.append(UNLOAD_COMMAND)
        self.loaded = [value for value in self.loaded if value != self.config.identifier]
        if self.remove_unrelated_on_stop:
            self.loaded.clear()

    def api_model_identifiers(self) -> tuple[str, ...]:
        self.api_calls += 1
        if self.api_visible and self.config.identifier in self.loaded:
            return tuple(self.loaded)
        return tuple(value for value in self.loaded if value != self.config.identifier)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(seconds, 300.0)


def service_fixture(
    config: InferenceConfig,
    *,
    available_memory: float = 100.0,
    loaded: list[str] | None = None,
    server_running: bool = True,
    cli_load_visible: bool = True,
    api_visible: bool = True,
    remove_unrelated_on_stop: bool = False,
    remove_unrelated_on_server_start: bool = False,
    estimate_error: BaseException | None = None,
    snapshot_error: BaseException | None = None,
    unsafe_after_server_start: bool = False,
) -> tuple[InferenceService, FakeBackend, FakeClock]:
    backend = FakeBackend(
        config,
        loaded=loaded,
        server_running=server_running,
        cli_load_visible=cli_load_visible,
        api_visible=api_visible,
        remove_unrelated_on_stop=remove_unrelated_on_stop,
        remove_unrelated_on_server_start=remove_unrelated_on_server_start,
        estimate_error=estimate_error,
        snapshot_error=snapshot_error,
        unsafe_after_server_start=unsafe_after_server_start,
    )
    clock = FakeClock()
    service = InferenceService(
        config,
        backend=backend,
        memory_reader=lambda: available_memory,
        clock=clock,
        sleeper=clock.sleep,
    )
    return service, backend, clock


def assert_no_forbidden_commands(calls: list[tuple[str, ...]]) -> None:
    for call in calls:
        for fragment in FORBIDDEN_FRAGMENTS:
            assert not all(value in call for value in fragment)


def test_available_memory_gib_reads_memavailable_using_1024_based_gib(
    tmp_path: Path,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 999999 kB\nMemAvailable: 41943040 kB\n")

    assert available_memory_gib(meminfo) == 40.0


@pytest.mark.parametrize(
    "contents",
    [
        "MemTotal: 999999 kB\n",
        "MemAvailable: unknown kB\n",
        "MemAvailable: -1 kB\n",
        "MemAvailable: 1 MB\n",
        "MemAvailable: 1 kB\nMemAvailable: 2 kB\n",
    ],
)
def test_available_memory_gib_rejects_malformed_input(
    tmp_path: Path, contents: str,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(contents)

    with pytest.raises(LmStudioError, match="MemAvailable"):
        available_memory_gib(meminfo)


def test_start_refuses_below_free_memory_gate(config: InferenceConfig) -> None:
    service, backend, _clock = service_fixture(config, available_memory=39.99)

    result = service.start()

    assert result.status == "not_ready"
    assert result.metrics["available_memory_gib"] == 39.99
    assert result.metrics["minimum_available_memory_gib"] == 40
    assert backend.estimate_calls == 1
    assert not backend.mutating_calls
    assert_no_forbidden_commands(backend.calls)


def test_start_loads_only_configured_model_with_exact_vector(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(config, available_memory=100)

    result = service.start()

    assert result.status == "pass"
    assert backend.load_calls == [LOAD_COMMAND]
    assert backend.api_calls == 1
    assert_no_forbidden_commands(backend.calls)


def test_start_uses_exact_loopback_server_vector_only_when_stopped(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(
        config, available_memory=100, server_running=False,
    )

    result = service.start()

    assert result.status == "pass"
    assert backend.mutating_calls == [SERVER_START_COMMAND, LOAD_COMMAND]
    assert_no_forbidden_commands(backend.calls)


def test_start_is_idempotent_when_exact_identifier_is_already_visible(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(
        config, loaded=[config.identifier, "user-model"], available_memory=100,
    )

    result = service.start()

    assert result.status == "pass"
    assert not backend.mutating_calls
    assert backend.estimate_calls == 0
    assert backend.api_calls == 1
    assert backend.loaded == [config.identifier, "user-model"]
    assert_no_forbidden_commands(backend.calls)


def test_start_exact_resident_model_bypasses_low_memory_gate(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(
        config, loaded=[config.identifier], available_memory=0,
    )

    result = service.start()

    assert result.status == "pass"
    assert backend.estimate_calls == 0
    assert not backend.mutating_calls


def test_start_rechecks_server_safety_immediately_after_starting_server(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(
        config,
        server_running=False,
        unsafe_after_server_start=True,
    )

    result = service.start()

    assert result.status == "not_ready"
    assert "server configuration" in (result.error or "")
    assert backend.mutating_calls == [SERVER_START_COMMAND]
    assert backend.load_calls == []


@pytest.mark.parametrize("operation", ["start", "stop"])
def test_lifecycle_never_mutates_when_snapshot_identity_is_unsafe(
    config: InferenceConfig, operation: str,
) -> None:
    service, backend, _clock = service_fixture(
        config,
        snapshot_error=LmStudioError(
            "configured LM Studio identifier is bound to a different model"
        ),
    )

    result = getattr(service, operation)()

    assert result.status == "not_ready"
    assert "different model" in (result.error or "")
    assert not backend.mutating_calls
    assert backend.estimate_calls == 0


def test_start_verifies_unrelated_models_after_starting_shared_server(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(
        config,
        loaded=[config.identifier, "user-model"],
        server_running=False,
        remove_unrelated_on_server_start=True,
    )

    result = service.start()

    assert result.status == "not_ready"
    assert "unrelated" in (result.error or "")
    assert backend.mutating_calls == [SERVER_START_COMMAND]
    assert_no_forbidden_commands(backend.calls)


def test_start_fails_when_cli_postcondition_never_becomes_true(
    config: InferenceConfig,
) -> None:
    service, backend, clock = service_fixture(
        config, available_memory=100, cli_load_visible=False,
    )

    result = service.start()

    assert result.status == "not_ready"
    assert result.retryable is True
    assert "timed out" in (result.error or "")
    assert backend.load_calls == [LOAD_COMMAND]
    assert backend.api_calls == 0
    assert clock.now >= 600
    assert_no_forbidden_commands(backend.calls)


def test_start_fails_when_cli_and_api_visibility_disagree(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(
        config, available_memory=100, api_visible=False,
    )

    result = service.start()

    assert result.status == "not_ready"
    assert result.retryable is True
    assert "API" in (result.error or "")
    assert backend.loaded == [config.identifier]
    assert_no_forbidden_commands(backend.calls)


def test_estimate_records_low_confidence_and_memory_gate(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(config, available_memory=100)

    result = service.estimate()

    assert result.status == "pass"
    assert result.metrics == {
        "estimated_gpu_gib": 17.36,
        "estimated_total_gib": 17.36,
        "estimate_confidence": "LOW",
        "available_memory_gib": 100.0,
        "minimum_available_memory_gib": 40,
        "allowed": True,
    }
    assert not backend.mutating_calls


def test_estimate_parse_failure_becomes_safe_public_result(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(
        config,
        estimate_error=LmStudioError("LM Studio estimate output is invalid"),
    )

    result = service.estimate()

    assert result.status == "not_ready"
    assert result.retryable is True
    assert result.error == "LM Studio estimate output is invalid"
    assert not backend.mutating_calls


def test_stop_never_unloads_unrelated_models(config: InferenceConfig) -> None:
    service, backend, _clock = service_fixture(
        config, loaded=[config.identifier, "user-model"],
    )

    result = service.stop()

    assert result.status == "pass"
    assert backend.unload_calls == [UNLOAD_COMMAND]
    assert backend.loaded == ["user-model"]
    assert_no_forbidden_commands(backend.calls)


def test_stop_is_idempotent_when_exact_identifier_is_absent(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(config, loaded=["user-model"])

    result = service.stop()

    assert result.status == "pass"
    assert not backend.mutating_calls
    assert backend.loaded == ["user-model"]


def test_stop_fails_if_unrelated_loaded_identifiers_change(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(
        config,
        loaded=[config.identifier, "user-model"],
        remove_unrelated_on_stop=True,
    )

    result = service.stop()

    assert result.status == "not_ready"
    assert "unrelated" in (result.error or "")
    assert backend.unload_calls == [UNLOAD_COMMAND]
    assert_no_forbidden_commands(backend.calls)


def test_doctor_and_status_are_read_only(config: InferenceConfig) -> None:
    service, backend, _clock = service_fixture(
        config, loaded=[config.identifier, "user-model"],
    )

    doctor = service.doctor()
    status = service.status()

    assert doctor.status == "pass"
    assert status.status == "pass"
    assert not backend.mutating_calls
    assert backend.loaded == [config.identifier, "user-model"]


def test_capability_inputs_cache_configured_model_beneath_data_root(
    config: InferenceConfig, tmp_path: Path,
) -> None:
    model = Path(config.models_directory) / "publisher" / "model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model-bytes")
    service, backend, _clock = service_fixture(
        config, loaded=[config.identifier, "user-model"],
    )

    inputs = service.capability_inputs(tmp_path / "data")

    assert inputs["model"]["identifier"] == config.identifier
    assert inputs["corpus_version"] == "lm-studio-capability-v4"
    assert list((tmp_path / "data" / "system" / "model-digests").glob("*.json"))
    assert fingerprint_inputs(inputs) == fingerprint_inputs(service.capability_inputs(tmp_path / "data"))
    assert not backend.mutating_calls


def test_current_capability_inputs_recomputes_unloaded_provenance_without_cache_writes(
    config: InferenceConfig, tmp_path: Path,
) -> None:
    model = Path(config.models_directory) / "publisher" / "model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model-bytes")
    service, backend, _clock = service_fixture(config, loaded=[])
    data_root = tmp_path / "data"

    inputs = service.current_capability_inputs(data_root)

    assert inputs["model"]["identifier"] == config.identifier
    assert not (data_root / "system" / "model-digests").exists()
    assert not backend.mutating_calls


def test_chat_completion_uses_injected_loopback_transport_without_credentials(
    config: InferenceConfig,
) -> None:
    service, backend, _clock = service_fixture(config)
    calls: list[
        tuple[str, str, dict[str, object] | None, Mapping[str, str], float]
    ] = []

    def http(
        method: str,
        url: str,
        body: dict[str, object] | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> dict[str, object]:
        calls.append((method, url, body, headers, timeout))
        return {"choices": []}

    backend.http = http  # type: ignore[attr-defined]
    payload = {"model": config.identifier, "stream": False}

    response = service.chat_completion(payload, timeout=9.0)

    assert response == {"choices": []}
    assert calls == [
        (
            "POST",
            "http://127.0.0.1:1234/v1/chat/completions",
            payload,
            {"Content-Type": "application/json", "Accept": "application/json"},
            9.0,
        )
    ]


class ExplodingBackend:
    def __getattr__(self, _name: str) -> Callable[..., object]:
        def explode(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("token=top-secret " + ("x" * (MAX_DIAGNOSTIC_CHARS * 2)))

        return explode


@pytest.mark.parametrize("method", ["doctor", "estimate", "start", "status", "stop"])
def test_all_service_methods_sanitize_unexpected_failures(
    config: InferenceConfig, method: str,
) -> None:
    service = InferenceService(
        config,
        backend=ExplodingBackend(),  # type: ignore[arg-type]
        memory_reader=lambda: 100.0,
    )

    result = getattr(service, method)()

    assert result.status == "fail"
    assert result.retryable is False
    assert result.error is not None
    assert "top-secret" not in result.error
    assert "Traceback" not in result.error
    assert len(result.error) <= MAX_DIAGNOSTIC_CHARS
