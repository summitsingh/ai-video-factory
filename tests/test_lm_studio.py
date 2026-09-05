from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

import ai_video_factory.lm_studio as lm_studio
from ai_video_factory.inference_config import InferenceConfig
from ai_video_factory.lm_studio import LmStudioError, ProcessResult
from ai_video_factory.sanitization import MAX_DIAGNOSTIC_CHARS


EXPECTED_READ_ONLY_COMMANDS = [
    ("lms", "--help"),
    ("lms", "runtime", "ls"),
    ("lms", "runtime", "survey"),
    ("lms", "server", "status"),
    ("lms", "ls", "--json"),
    ("lms", "ps", "--json"),
]


@pytest.fixture
def config(tmp_path: Path) -> InferenceConfig:
    return InferenceConfig.model_validate(
        {
            "schema_version": 1,
            "backend": "lm_studio",
            "base_url": "http://127.0.0.1:1234/v1",
            "lms_binary": "lms",
            "model_key": "qwen3.6-35b-a3b-udt-mtp",
            "identifier": "avf-qwen36-executor",
            "context_length": 65_536,
            "gpu": "max",
            "parallel": 1,
            "ttl_seconds": 3_600,
            "minimum_available_memory_gib": 40,
            "models_directory": str(tmp_path / "models"),
        }
    )


def ok(stdout: str) -> ProcessResult:
    return ProcessResult(0, stdout, "")


def model_inventory_json(
    config: InferenceConfig, *, path: str = "publisher/model.gguf", key: str | None = None
) -> str:
    return json.dumps(
        [{"key": key or config.model_key, "path": path, "sizeBytes": 18_640_894_912}]
    )


class RecordingRunner:
    def __init__(
        self,
        outputs: dict[tuple[str, ...], ProcessResult | BaseException],
    ) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []

    def __call__(self, argv: Sequence[str], timeout: float) -> ProcessResult:
        command = tuple(argv)
        self.calls.append(command)
        self.timeouts.append(timeout)
        result = self.outputs[command]
        if isinstance(result, BaseException):
            raise result
        return result


class FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.read_limits: list[int] = []

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, limit: int) -> bytes:
        self.read_limits.append(limit)
        return self.body[:limit]


class RecordingOpener:
    def __init__(self, response: FakeResponse | BaseException) -> None:
        self.response = response
        self.calls: list[tuple[object, float]] = []

    def open(self, request: object, *, timeout: float) -> FakeResponse:
        self.calls.append((request, timeout))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def patch_opener(
    monkeypatch: pytest.MonkeyPatch, response: FakeResponse | BaseException
) -> RecordingOpener:
    opener = RecordingOpener(response)
    monkeypatch.setattr(lm_studio, "build_opener", lambda *_handlers: opener)
    return opener


def inventory_runner(
    config: InferenceConfig, *, path: str = "publisher/model.gguf", inventory: str | None = None,
    loaded: str = "[]", server_status: str = "The server is running on port 1234.",
) -> RecordingRunner:
    return RecordingRunner(
        {
            ("lms", "--help"): ok("lms is LM Studio's CLI utility (v0.0.47)"),
            ("lms", "runtime", "ls"): ok("llama.cpp-linux-x86_64-amd-rocm-avx2@2.31.2"),
            ("lms", "runtime", "survey"): ok("GPU: 85.67 GiB\nRAM: 122.69 GiB"),
            ("lms", "server", "status"): ok(server_status),
            ("lms", "ls", "--json"): ok(inventory or model_inventory_json(config, path=path)),
            ("lms", "ps", "--json"): ok(loaded),
        }
    )


def test_snapshot_uses_only_read_only_fixed_commands(config: InferenceConfig) -> None:
    from ai_video_factory.lm_studio import LmStudioBackend

    runner = inventory_runner(config)
    snapshot = LmStudioBackend(config, runner=runner).snapshot()

    assert snapshot.configured_model.model_key == config.model_key
    assert snapshot.configured_model_loaded is False
    assert runner.calls == EXPECTED_READ_ONLY_COMMANDS


def test_snapshot_rejects_model_path_outside_configured_root(config: InferenceConfig) -> None:
    from ai_video_factory.lm_studio import LmStudioBackend, LmStudioError

    with pytest.raises(LmStudioError, match="contained"):
        LmStudioBackend(config, runner=inventory_runner(config, path="../../outside.gguf")).snapshot()


def test_snapshot_reports_a_stopped_server(config: InferenceConfig) -> None:
    from ai_video_factory.lm_studio import LmStudioBackend

    snapshot = LmStudioBackend(
        config, runner=inventory_runner(config, server_status="The server is not running.")
    ).snapshot()

    assert snapshot.server_running is False
    assert snapshot.checks["server"].status == "not_ready"


def test_snapshot_retains_loaded_unrelated_models_without_touching_them(
    config: InferenceConfig,
) -> None:
    from ai_video_factory.lm_studio import LmStudioBackend

    runner = inventory_runner(config, loaded='[{"identifier": "user-model"}]')
    snapshot = LmStudioBackend(config, runner=runner).snapshot()

    assert snapshot.loaded_identifiers == ("user-model",)
    assert snapshot.configured_model_loaded is False
    assert runner.calls[-1] == ("lms", "ps", "--json")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FileNotFoundError("lms"), "executable not found"),
        (subprocess.TimeoutExpired(("lms", "--help"), 15), "timed out"),
    ],
)
def test_snapshot_sanitizes_command_startup_errors(
    config: InferenceConfig, error: BaseException, expected: str
) -> None:
    from ai_video_factory.lm_studio import LmStudioBackend, LmStudioError

    runner = inventory_runner(config)
    runner.outputs[("lms", "--help")] = error

    with pytest.raises(LmStudioError, match=expected):
        LmStudioBackend(config, runner=runner).snapshot()


def test_snapshot_rejects_malformed_inventory_json(config: InferenceConfig) -> None:
    from ai_video_factory.lm_studio import LmStudioBackend, LmStudioError

    with pytest.raises(LmStudioError, match="inventory JSON"):
        LmStudioBackend(config, runner=inventory_runner(config, inventory="not json")).snapshot()


def test_snapshot_rejects_duplicate_configured_model_keys(config: InferenceConfig) -> None:
    from ai_video_factory.lm_studio import LmStudioBackend, LmStudioError

    inventory = json.dumps(
        [
            {"key": config.model_key, "path": "publisher/one.gguf", "sizeBytes": 1},
            {"key": config.model_key, "path": "publisher/two.gguf", "sizeBytes": 2},
        ]
    )
    with pytest.raises(LmStudioError, match="duplicate"):
        LmStudioBackend(config, runner=inventory_runner(config, inventory=inventory)).snapshot()


def test_snapshot_rejects_a_missing_configured_model(config: InferenceConfig) -> None:
    from ai_video_factory.lm_studio import LmStudioBackend, LmStudioError

    inventory = model_inventory_json(config, key="some-other-model")
    with pytest.raises(LmStudioError, match="configured model"):
        LmStudioBackend(config, runner=inventory_runner(config, inventory=inventory)).snapshot()


def test_snapshot_sanitizes_and_caps_failed_command_stderr(config: InferenceConfig) -> None:
    from ai_video_factory.lm_studio import LmStudioBackend, LmStudioError

    runner = inventory_runner(config)
    runner.outputs[("lms", "runtime", "survey")] = ProcessResult(
        1, "", "token=top-secret " + ("x" * (MAX_DIAGNOSTIC_CHARS * 2))
    )

    with pytest.raises(LmStudioError) as raised:
        LmStudioBackend(config, runner=runner).snapshot()

    assert "top-secret" not in str(raised.value)
    assert len(str(raised.value)) <= MAX_DIAGNOSTIC_CHARS


def test_snapshot_rejects_malformed_loaded_model_json(config: InferenceConfig) -> None:
    with pytest.raises(LmStudioError, match="loaded-model JSON"):
        lm_studio.LmStudioBackend(
            config, runner=inventory_runner(config, loaded="not json")
        ).snapshot()


def test_run_process_rejects_a_mutating_command_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lm_studio, "_resolved_lms", lambda: "/safe/lms")
    monkeypatch.setattr(
        lm_studio.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not be invoked"),
    )

    with pytest.raises(LmStudioError, match="read-only"):
        lm_studio.run_process(("lms", "server", "stop"), 15)


def test_request_json_rejects_non_loopback_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    opener = patch_opener(monkeypatch, FakeResponse(b"{}"))

    with pytest.raises(LmStudioError, match="loopback"):
        lm_studio.request_json(
            "GET", "http://example.com/v1/models", None,
            {"Content-Type": "application/json", "Accept": "application/json"}, 1,
        )

    assert opener.calls == []


@pytest.mark.parametrize("headers", [{"Accept": "application/json"}, {
    "Content-Type": "application/json", "Accept": "application/json", "Authorization": "secret"
}])
def test_request_json_rejects_nonexact_headers(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    opener = patch_opener(monkeypatch, FakeResponse(b"{}"))

    with pytest.raises(LmStudioError, match="headers only"):
        lm_studio.request_json("GET", "http://127.0.0.1:1234/v1/models", None, headers, 1)

    assert opener.calls == []


@pytest.mark.parametrize("url", [
    "http://user:password@127.0.0.1:1234/v1/models",
    "http://127.0.0.1:1234/v1/models?token=secret",
    "http://127.0.0.1:1234/v1/models#fragment",
])
def test_request_json_rejects_credentialed_or_decorated_loopback_urls(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    opener = patch_opener(monkeypatch, FakeResponse(b"{}"))

    with pytest.raises(LmStudioError, match="loopback"):
        lm_studio.request_json(
            "GET", url, None,
            {"Content-Type": "application/json", "Accept": "application/json"}, 1,
        )

    assert opener.calls == []


@pytest.mark.parametrize(("body", "message"), [(b"not json", "valid JSON"), (b"[]", "object")])
def test_request_json_rejects_invalid_or_nonobject_json(
    monkeypatch: pytest.MonkeyPatch, body: bytes, message: str
) -> None:
    patch_opener(monkeypatch, FakeResponse(body))

    with pytest.raises(LmStudioError, match=message):
        lm_studio.request_json(
            "GET", "http://127.0.0.1:1234/v1/models", None,
            {"Content-Type": "application/json", "Accept": "application/json"}, 1,
        )


def test_request_json_rejects_redirects_without_following_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redirect = HTTPError(
        "http://127.0.0.1:1234/v1/models", 302, "Found", {}, BytesIO(b"redirect")
    )
    opener = patch_opener(monkeypatch, redirect)

    with pytest.raises(LmStudioError, match="status 302"):
        lm_studio.request_json(
            "GET", "http://127.0.0.1:1234/v1/models", None,
            {"Content-Type": "application/json", "Accept": "application/json"}, 1,
        )

    assert len(opener.calls) == 1


def test_request_json_rejects_responses_over_two_mib(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(b"x" * (2 * 1024 * 1024 + 1))
    patch_opener(monkeypatch, response)

    with pytest.raises(LmStudioError, match="2 MiB"):
        lm_studio.request_json(
            "GET", "http://127.0.0.1:1234/v1/models", None,
            {"Content-Type": "application/json", "Accept": "application/json"}, 1,
        )

    assert response.read_limits == [2 * 1024 * 1024 + 1]
