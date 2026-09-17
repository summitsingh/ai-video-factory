from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler

import pytest

import ai_video_factory.lm_studio as lm_studio
from ai_video_factory.inference_config import InferenceConfig
from ai_video_factory.lm_studio import LmStudioError, ProcessResult
from ai_video_factory.sanitization import MAX_DIAGNOSTIC_CHARS


EXPECTED_READ_ONLY_COMMANDS = [
    ("lms", "--help"),
    ("lms", "--version"),
    ("lms", "runtime", "ls"),
    ("lms", "runtime", "survey"),
    ("lms", "server", "status"),
    ("lms", "ls", "--json"),
    ("lms", "ps", "--json"),
]

EXPECTED_ESTIMATE_COMMAND = (
    "lms", "load", "qwen3.6-35b-a3b-udt-mtp", "--gpu", "max",
    "--context-length", "65536", "--no-speculative-draft-mtp",
    "--estimate-only", "-y",
)
EXPECTED_LOAD_COMMAND = (
    "lms", "load", "qwen3.6-35b-a3b-udt-mtp", "--gpu", "max",
    "--context-length", "65536", "--parallel", "1", "--ttl", "3600",
    "--no-speculative-draft-mtp", "--identifier", "avf-qwen36-executor", "-y",
)
EXPECTED_SERVER_START_COMMAND = (
    "lms", "server", "start", "--port", "1234", "--bind", "127.0.0.1",
)
EXPECTED_UNLOAD_COMMAND = ("lms", "unload", "avf-qwen36-executor")


@pytest.fixture
def config(tmp_path: Path) -> InferenceConfig:
    server_config = tmp_path / "http-server-config.json"
    server_config.write_text(
        json.dumps({"port": 1234, "networkInterface": "127.0.0.1", "cors": False})
    )
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
            "server_config_path": str(server_config),
        }
    )


def ok(stdout: str) -> ProcessResult:
    return ProcessResult(0, stdout, "", "/safe/lms")


def model_inventory_json(
    config: InferenceConfig, *, path: str = "publisher/model.gguf", key: str | None = None
) -> str:
    return json.dumps(
        [{"key": key or config.model_key, "path": path, "sizeBytes": 18_640_894_912}]
    )


def loaded_model_json(
    config: InferenceConfig,
    *,
    identifier: str | None = None,
    model_key: str | None = None,
    path: str = "publisher/model.gguf",
    size_bytes: int = 18_640_894_912,
    context_length: int | None = None,
    parallel: int | None = None,
) -> str:
    return json.dumps(
        [
            {
                "identifier": identifier or config.identifier,
                "modelKey": model_key or config.model_key,
                "path": path,
                "sizeBytes": size_bytes,
                "contextLength": context_length or config.context_length,
                "parallel": parallel or config.parallel,
            }
        ]
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
) -> tuple[RecordingOpener, list[object]]:
    opener = RecordingOpener(response)
    handlers: list[object] = []

    def build(*provided: object) -> RecordingOpener:
        handlers.extend(provided)
        return opener

    monkeypatch.setattr(lm_studio, "build_opener", build)
    return opener, handlers


def inventory_runner(
    config: InferenceConfig, *, path: str = "publisher/model.gguf", inventory: str | None = None,
    loaded: str = "[]", server_status: str = "The server is running on port 1234.",
) -> RecordingRunner:
    return RecordingRunner(
        {
            ("lms", "--help"): ok("lms is LM Studio's CLI utility (v0.0.47)"),
            ("lms", "--version"): ok("CLI commit: 07b7252"),
            ("lms", "runtime", "ls"): ok(
                "LLM ENGINE SELECTED MODEL FORMAT\n"
                "llama.cpp-linux-x86_64-amd-rocm-avx2@2.31.2 ✓ GGUF"
            ),
            ("lms", "runtime", "survey"): ok(
                "Survey by llama.cpp-linux-x86_64-amd-rocm-avx2 (2.31.2)\n"
                "AMD Radeon Graphics 85.67 GiB\nRAM: 122.69 GiB"
            ),
            ("lms", "server", "status"): ok(server_status),
            ("lms", "ls", "--json"): ok(inventory or model_inventory_json(config, path=path)),
            ("lms", "ps", "--json"): ok(loaded),
        }
    )


def backend_with_estimate(
    config: InferenceConfig, output: str,
) -> tuple[lm_studio.LmStudioBackend, RecordingRunner]:
    runner = RecordingRunner({EXPECTED_ESTIMATE_COMMAND: ok(output)})
    return lm_studio.LmStudioBackend(config, runner=runner), runner


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

    runner = inventory_runner(
        config,
        loaded=loaded_model_json(config, identifier="user-model", model_key="user-key"),
    )
    snapshot = LmStudioBackend(config, runner=runner).snapshot()

    assert snapshot.loaded_identifiers == ("user-model",)
    assert snapshot.configured_model_loaded is False
    assert runner.calls[-1] == ("lms", "ps", "--json")


def test_snapshot_accepts_only_an_exact_loaded_model_identity(config: InferenceConfig) -> None:
    snapshot = lm_studio.LmStudioBackend(
        config, runner=inventory_runner(config, loaded=loaded_model_json(config))
    ).snapshot()

    assert snapshot.configured_model_loaded is True
    assert snapshot.loaded_models[0].model_key == config.model_key
    assert snapshot.loaded_models[0].context_length == config.context_length


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_key": "wrong-model"},
        {"path": "publisher/wrong.gguf"},
        {"size_bytes": 1},
        {"context_length": 4096},
        {"parallel": 2},
    ],
)
def test_snapshot_rejects_configured_alias_collision(
    config: InferenceConfig, overrides: dict[str, object]
) -> None:
    with pytest.raises(LmStudioError, match="different model"):
        lm_studio.LmStudioBackend(
            config,
            runner=inventory_runner(
                config, loaded=loaded_model_json(config, **overrides)  # type: ignore[arg-type]
            ),
        ).snapshot()


@pytest.mark.parametrize(
    "server_config",
    [
        {"port": 1235, "networkInterface": "127.0.0.1", "cors": False},
        {"port": 1234, "networkInterface": "0.0.0.0", "cors": False},
        {"port": 1234, "networkInterface": "127.0.0.1", "cors": True},
    ],
)
def test_snapshot_rejects_unsafe_running_server_configuration(
    config: InferenceConfig, server_config: dict[str, object]
) -> None:
    Path(config.server_config_path).write_text(json.dumps(server_config))

    with pytest.raises(LmStudioError, match="server configuration"):
        lm_studio.LmStudioBackend(config, runner=inventory_runner(config)).snapshot()


def test_snapshot_does_not_read_server_config_when_server_is_stopped(
    config: InferenceConfig,
) -> None:
    reads: list[Path] = []

    def reader(path: Path) -> str:
        reads.append(path)
        raise AssertionError("stopped server config must not be read")

    snapshot = lm_studio.LmStudioBackend(
        config,
        runner=inventory_runner(config, server_status="The server is not running."),
        file_reader=reader,
    ).snapshot()

    assert snapshot.server_running is False
    assert reads == []


@pytest.mark.parametrize(
    "runtimes",
    [
        "llama.cpp-linux-x86_64-cpu-avx2@2.31.2 ✓ GGUF",
        "llama.cpp-linux-x86_64-cuda-avx2@2.31.2 ✓ GGUF",
        "llama.cpp-linux-x86_64-vulkan-avx@2.31.2 ✓ GGUF",
        "llama.cpp-linux-x86_64-vulkan-avx2@2.31.2 GGUF",
        (
            "llama.cpp-linux-x86_64-vulkan-avx2@2.31.2 ✓ GGUF\n"
            "llama.cpp-linux-x86_64-amd-rocm-avx2@2.31.2 ✓ GGUF"
        ),
    ],
)
def test_snapshot_rejects_missing_wrong_or_multiple_selected_runtimes(
    config: InferenceConfig, runtimes: str
) -> None:
    runner = inventory_runner(config)
    runner.outputs[("lms", "runtime", "ls")] = ok(runtimes)

    with pytest.raises(LmStudioError, match="selected compatible runtime"):
        lm_studio.LmStudioBackend(config, runner=runner).snapshot()


@pytest.mark.parametrize(
    "survey",
    [
        "Survey by llama.cpp-linux-x86_64-vulkan-avx2 (2.31.2)\nAMD GPU 85.67 GiB",
        "Survey by llama.cpp-linux-x86_64-amd-rocm-avx2 (2.31.1)\nAMD GPU 85.67 GiB",
        "Survey by llama.cpp-linux-x86_64-amd-rocm-avx2 (2.31.2)\nNVIDIA GPU 85.67 GiB",
        "Survey by llama.cpp-linux-x86_64-amd-rocm-avx2 (2.31.2)\nAMD GPU 0 GiB",
    ],
)
def test_snapshot_rejects_mismatched_or_invalid_runtime_survey(
    config: InferenceConfig, survey: str
) -> None:
    runner = inventory_runner(config)
    runner.outputs[("lms", "runtime", "survey")] = ok(survey)

    with pytest.raises(LmStudioError, match="runtime survey"):
        lm_studio.LmStudioBackend(config, runner=runner).snapshot()


@pytest.mark.parametrize(
    "help_text",
    [
        "lms is LM Studio's CLI utility (v0.0.47)",
        "CLI commit: not-a-commit",
        "CLI commit: 07b7252\nCLI commit: 07b7252",
    ],
)
def test_snapshot_requires_one_authoritative_cli_commit(
    config: InferenceConfig, help_text: str
) -> None:
    runner = inventory_runner(config)
    runner.outputs[("lms", "--version")] = ok(help_text)

    with pytest.raises(LmStudioError, match="CLI commit"):
        lm_studio.LmStudioBackend(config, runner=runner).snapshot()


def test_snapshot_rejects_conflicting_commit_lines_across_help_and_version(
    config: InferenceConfig,
) -> None:
    runner = inventory_runner(config)
    runner.outputs[("lms", "--help")] = ok("help\nCLI commit: 89abcde")

    with pytest.raises(LmStudioError, match="exactly one CLI commit"):
        lm_studio.LmStudioBackend(config, runner=runner).snapshot()


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
        1, "", "token=top-secret " + ("x" * (MAX_DIAGNOSTIC_CHARS * 2)), "/safe/lms"
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
    opener, _handlers = patch_opener(monkeypatch, FakeResponse(b"{}"))

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
    opener, _handlers = patch_opener(monkeypatch, FakeResponse(b"{}"))

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
    opener, _handlers = patch_opener(monkeypatch, FakeResponse(b"{}"))

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
    opener, _handlers = patch_opener(monkeypatch, redirect)

    with pytest.raises(LmStudioError, match="status 302"):
        lm_studio.request_json(
            "GET", "http://127.0.0.1:1234/v1/models", None,
            {"Content-Type": "application/json", "Accept": "application/json"}, 1,
        )

    assert len(opener.calls) == 1


def test_request_json_disables_ambient_proxies_for_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8443")
    _opener, handlers = patch_opener(monkeypatch, FakeResponse(b"{}"))

    lm_studio.request_json(
        "GET", "http://127.0.0.1:1234/v1/models", None,
        {"Content-Type": "application/json", "Accept": "application/json"}, 1,
    )

    proxy_handlers = [handler for handler in handlers if isinstance(handler, ProxyHandler)]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}
    assert sum(isinstance(handler, lm_studio._NoRedirect) for handler in handlers) == 1


def test_request_json_never_exposes_http_error_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = b"token=SERVER-BODY-SECRET"
    failure = HTTPError(
        "http://127.0.0.1:1234/v1/models", 500, "failure", {}, BytesIO(secret)
    )
    patch_opener(monkeypatch, failure)

    with pytest.raises(LmStudioError) as raised:
        lm_studio.request_json(
            "GET", "http://127.0.0.1:1234/v1/models", None,
            {"Content-Type": "application/json", "Accept": "application/json"}, 1,
        )

    assert "SERVER-BODY-SECRET" not in str(raised.value)
    assert str(raised.value) == "LM Studio HTTP request failed with status 500"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_request_json_rejects_responses_over_two_mib(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(b"x" * (2 * 1024 * 1024 + 1))
    patch_opener(monkeypatch, response)

    with pytest.raises(LmStudioError, match="2 MiB"):
        lm_studio.request_json(
            "GET", "http://127.0.0.1:1234/v1/models", None,
            {"Content-Type": "application/json", "Accept": "application/json"}, 1,
        )

    assert response.read_limits == [2 * 1024 * 1024 + 1]


def test_estimate_uses_exact_nonloading_vector(config: InferenceConfig) -> None:
    backend, runner = backend_with_estimate(
        config,
        "Estimated GPU Memory: 17.36 GiB\n"
        "Estimated Total Memory: 17.36 GiB\nConfidence: LOW\n",
    )

    estimate = backend.estimate()

    assert estimate.gpu_gib == 17.36
    assert estimate.total_gib == 17.36
    assert estimate.confidence == "LOW"
    assert estimate.allowed is True
    assert runner.calls == [EXPECTED_ESTIMATE_COMMAND]
    assert runner.timeouts == [60.0]


def test_estimate_accepts_valid_labels_on_stderr(config: InferenceConfig) -> None:
    runner = RecordingRunner(
        {
            EXPECTED_ESTIMATE_COMMAND: ProcessResult(
                0,
                "Preparing estimate\n",
                "Estimated GPU Memory: 17.36 GiB\n"
                "Estimated Total Memory: 17.36 GiB\nConfidence: LOW\n",
                "/safe/lms",
            )
        }
    )

    estimate = lm_studio.LmStudioBackend(config, runner=runner).estimate()

    assert estimate.gpu_gib == 17.36
    assert estimate.total_gib == 17.36
    assert estimate.confidence == "LOW"


@pytest.mark.parametrize(
    "stderr",
    [
        "Estimated GPU Memory: 17.36 GiB\n",
        "Estimated GPU Memory: 17.49 GiB\n",
    ],
    ids=["duplicate", "conflicting"],
)
def test_estimate_rejects_duplicate_or_conflicting_labels_split_across_streams(
    config: InferenceConfig, stderr: str,
) -> None:
    runner = RecordingRunner(
        {
            EXPECTED_ESTIMATE_COMMAND: ProcessResult(
                0,
                "Estimated GPU Memory: 17.36 GiB\n"
                "Estimated Total Memory: 17.36 GiB\nConfidence: LOW\n",
                stderr,
                "/safe/lms",
            )
        }
    )

    with pytest.raises(LmStudioError, match="estimate output"):
        lm_studio.LmStudioBackend(config, runner=runner).estimate()


@pytest.mark.parametrize(
    "output",
    [
        "Estimated GPU Memory: 17.36 GiB\nConfidence: LOW\n",
        (
            "Estimated GPU Memory: 17.36 GiB\n"
            "Estimated GPU Memory: 17.36 GiB\n"
            "Estimated Total Memory: 17.36 GiB\nConfidence: LOW\n"
        ),
        (
            "Estimated GPU Memory: 17.36 GiB\n"
            "Estimated GPU Memory: malformed\n"
            "Estimated Total Memory: 17.36 GiB\nConfidence: LOW\n"
        ),
        (
            "Estimated GPU Memory: -1 GiB\n"
            "Estimated Total Memory: 17.36 GiB\nConfidence: LOW\n"
        ),
        (
            "Estimated GPU Memory: NaN GiB\n"
            "Estimated Total Memory: 17.36 GiB\nConfidence: LOW\n"
        ),
        (
            "Estimated GPU Memory: 17.36 GiB\n"
            "Estimated Total Memory: inf GiB\nConfidence: LOW\n"
        ),
    ],
)
def test_estimate_rejects_missing_duplicate_or_nonfinite_values(
    config: InferenceConfig, output: str,
) -> None:
    backend, _runner = backend_with_estimate(config, output)

    with pytest.raises(LmStudioError, match="estimate output"):
        backend.estimate()


def test_estimate_converts_runner_timeout_to_bounded_error(config: InferenceConfig) -> None:
    runner = RecordingRunner(
        {EXPECTED_ESTIMATE_COMMAND: subprocess.TimeoutExpired(EXPECTED_ESTIMATE_COMMAND, 60)}
    )

    with pytest.raises(LmStudioError, match="60 seconds"):
        lm_studio.LmStudioBackend(config, runner=runner).estimate()

    assert runner.timeouts == [60.0]


def test_model_start_uses_exact_vector_and_600_second_timeout(
    config: InferenceConfig,
) -> None:
    runner = RecordingRunner({EXPECTED_LOAD_COMMAND: ok("loaded")})

    lm_studio.LmStudioBackend(config, runner=runner).start()

    assert runner.calls == [EXPECTED_LOAD_COMMAND]
    assert runner.timeouts == [600.0]


def test_server_start_and_stop_use_only_targeted_vectors(config: InferenceConfig) -> None:
    runner = RecordingRunner(
        {
            EXPECTED_SERVER_START_COMMAND: ok("server started"),
            EXPECTED_UNLOAD_COMMAND: ok("model unloaded"),
        }
    )
    backend = lm_studio.LmStudioBackend(config, runner=runner)

    backend.start_server()
    backend.stop()

    assert runner.calls == [EXPECTED_SERVER_START_COMMAND, EXPECTED_UNLOAD_COMMAND]
    assert runner.timeouts == [15.0, 15.0]


def test_api_model_identifiers_requires_exact_models_endpoint(
    config: InferenceConfig,
) -> None:
    calls: list[tuple[str, str, object, object, float]] = []

    def http(method: str, url: str, body: object, headers: object, timeout: float) -> dict[str, object]:
        calls.append((method, url, body, headers, timeout))
        return {
            "object": "list",
            "data": [
                {"id": config.identifier, "object": "model", "owned_by": "organization_owner"},
                {"id": "user-model", "object": "model", "owned_by": "organization_owner"},
            ],
        }

    identifiers = lm_studio.LmStudioBackend(config, http=http).api_model_identifiers()

    assert identifiers == (config.identifier, "user-model")
    assert calls == [(
        "GET",
        "http://127.0.0.1:1234/v1/models",
        None,
        {"Content-Type": "application/json", "Accept": "application/json"},
        15.0,
    )]


def test_default_runner_allows_only_exact_task_3_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lm_studio, "_resolved_lms", lambda: "/safe/lms")
    calls: list[tuple[str, ...]] = []

    def run(argv: Sequence[str], **_kwargs: object) -> object:
        calls.append(tuple(argv))
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(lm_studio.subprocess, "run", run)
    approved = [
        EXPECTED_ESTIMATE_COMMAND,
        EXPECTED_SERVER_START_COMMAND,
        EXPECTED_LOAD_COMMAND,
        EXPECTED_UNLOAD_COMMAND,
    ]

    for command in approved:
        lm_studio.run_process(command, 15)

    assert calls == [("/safe/lms", *command[1:]) for command in approved]


def test_resolved_lms_is_canonical_absolute_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    executable = tmp_path / "real-lms"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    alias = tmp_path / "lms"
    alias.symlink_to(executable)
    monkeypatch.setattr(lm_studio.shutil, "which", lambda _name: str(alias))
    lm_studio._resolved_lms.cache_clear()

    assert lm_studio._resolved_lms() == str(executable.resolve())


@pytest.mark.parametrize(
    "command",
    [
        ("lms", "get", "qwen3.6-35b-a3b-udt-mtp"),
        ("lms", "runtime", "select", "llama.cpp"),
        ("lms", "runtime", "update"),
        ("lms", "unload", "--all"),
        ("lms", "server", "stop"),
        (
            "lms", "load", "qwen3.6-35b-a3b-udt-mtp", "--gpu", "max",
            "--context-length", "65536", "--estimate-only", "-y",
        ),
    ],
)
def test_default_runner_rejects_forbidden_or_weakened_vectors_before_subprocess(
    monkeypatch: pytest.MonkeyPatch, command: tuple[str, ...],
) -> None:
    monkeypatch.setattr(lm_studio, "_resolved_lms", lambda: "/safe/lms")
    monkeypatch.setattr(
        lm_studio.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not be invoked"),
    )

    with pytest.raises(LmStudioError, match="approved"):
        lm_studio.run_process(command, 15)
