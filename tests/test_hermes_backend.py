from __future__ import annotations

from pathlib import Path

import pytest

from ai_video_factory.hermes_backend import (
    ALLOWED_COMMANDS,
    HermesBackend,
    HermesError,
    ProcessResult,
    run_process,
)
from ai_video_factory.hermes_config import HermesConfig


def config() -> HermesConfig:
    return HermesConfig(
        schema_version=1,
        hermes_binary="hermes",
        required_version="0.21.0",
        required_commit="b0ab2e16",
        profile="default",
        parent_provider="nous",
        parent_model="stepfun/step-3.7-flash:free",
        delegation_base_url="http://127.0.0.1:1234/v1",
        delegation_model="avf-qwen36-executor",
        delegation_api_mode="chat_completions",
        local_api_key_placeholder="no-key-required",
        max_concurrent_children=1,
        max_iterations=50,
        vision_fixture="assets/vision/capability-probe.png",
        fallback_provider="nous",
        fallback_model="stepfun/step-3.7-flash:free",
    )


def result(stdout: str) -> ProcessResult:
    return ProcessResult(0, stdout, "", "/safe/hermes")


class FakeRunner:
    def __init__(self, outputs: dict[tuple[str, ...], ProcessResult]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(self, argv: tuple[str, ...], *, timeout: float) -> ProcessResult:
        self.calls.append((argv, timeout))
        return self.outputs[argv]


def snapshot_outputs() -> dict[tuple[str, ...], ProcessResult]:
    values = {
        ("hermes", "--version"): "Hermes Agent v0.21.0 (2026.8.31) · upstream b0ab2e16\n",
        ("hermes", "config", "path"): "/safe/config.yaml\n",
        ("hermes", "config", "check"): "Configuration is valid\n",
        ("hermes", "config", "get", "model.provider"): "nous\n",
        ("hermes", "config", "get", "model.default"): "stepfun/step-3.7-flash:free\n",
        ("hermes", "config", "get", "model.base_url"): "https://nous.example/v1\n",
        ("hermes", "config", "get", "delegation.model"): "avf-qwen36-executor\n",
        ("hermes", "config", "get", "delegation.base_url"): "http://127.0.0.1:1234/v1\n",
        ("hermes", "config", "get", "delegation.api_mode"): "chat_completions\n",
        ("hermes", "config", "get", "delegation.max_iterations"): "50\n",
        ("hermes", "config", "get", "delegation.max_concurrent_children"): "1\n",
        ("hermes", "config", "get", "delegation.max_spawn_depth"): "1\n",
        ("hermes", "config", "get", "delegation.orchestrator_enabled"): "false\n",
        ("hermes", "config", "get", "delegation.subagent_auto_approve"): "false\n",
        ("hermes", "config", "get", "delegation.inherit_mcp_toolsets"): "false\n",
    }
    return {argv: result(value) for argv, value in values.items()}


def test_allowlist_contains_only_the_fixed_read_only_vectors() -> None:
    assert ALLOWED_COMMANDS == frozenset(snapshot_outputs())


def test_default_runner_uses_resolved_executable_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class Completed:
        returncode = 0
        stdout = "Hermes Agent v0.21.0 (2026.8.31) · upstream b0ab2e16"
        stderr = ""

    def fake_run(argv: tuple[str, ...], **kwargs: object) -> Completed:
        calls.append((argv, kwargs))
        return Completed()

    monkeypatch.setattr("ai_video_factory.hermes_backend._resolved_hermes", lambda: "/safe/hermes")
    monkeypatch.setattr("ai_video_factory.hermes_backend.subprocess.run", fake_run)

    observed = run_process(("hermes", "--version"), timeout=15.0)

    assert observed.executable_path == "/safe/hermes"
    assert calls == [
        (("/safe/hermes", "--version"), {
            "shell": False,
            "capture_output": True,
            "text": True,
            "timeout": 15.0,
            "check": False,
        })
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ("sh", "-c", "hermes --version"),
        ("hermes", "--version", "--verbose"),
        ("hermes", "config", "get", "delegation.api_key"),
    ],
)
def test_default_runner_rejects_any_noncanonical_vector(argv: tuple[str, ...]) -> None:
    with pytest.raises(HermesError, match="allowlisted"):
        run_process(argv, timeout=15.0)


def test_default_runner_rejects_timeout_above_fixed_bound() -> None:
    with pytest.raises(HermesError, match="timeout"):
        run_process(("hermes", "--version"), timeout=15.1)


def test_snapshot_uses_individual_nonsecret_reads_and_typed_values() -> None:
    runner = FakeRunner(snapshot_outputs())

    snapshot = HermesBackend(config(), runner=runner).snapshot()

    assert snapshot.model_dump() == {
        "hermes_path": "/safe/hermes",
        "version": "0.21.0",
        "commit": "b0ab2e16",
        "config_path": "/safe/config.yaml",
        "profile": "default",
        "config_valid": True,
        "parent_provider": "nous",
        "parent_model": "stepfun/step-3.7-flash:free",
        "parent_base_url": "https://nous.example/v1",
        "delegation_model": "avf-qwen36-executor",
        "delegation_base_url": "http://127.0.0.1:1234/v1",
        "delegation_api_mode": "chat_completions",
        "delegation_max_iterations": 50,
        "delegation_max_concurrent_children": 1,
        "delegation_max_spawn_depth": 1,
        "delegation_orchestrator_enabled": False,
        "delegation_subagent_auto_approve": False,
        "delegation_inherit_mcp_toolsets": False,
        "auxiliary_routes": {
            name: {"enabled": False, "base_url": None, "model": None}
            for name in ("vision", "web_extract", "compression", "title_generation", "background_review")
        },
    }
    assert {call[0] for call in runner.calls} == ALLOWED_COMMANDS


def test_snapshot_represents_missing_delegation_keys_as_unconfigured() -> None:
    outputs = snapshot_outputs()
    for argv in list(outputs):
        if argv[3:4] == ("delegation.model",) or argv[3:4] == ("delegation.base_url",):
            outputs[argv] = ProcessResult(1, "", "key not found", "/safe/hermes")

    snapshot = HermesBackend(config(), runner=FakeRunner(outputs)).snapshot()

    assert snapshot.delegation_model is None
    assert snapshot.delegation_base_url is None


def test_snapshot_rejects_secret_or_multiline_values_without_exposure() -> None:
    outputs = snapshot_outputs()
    outputs[("hermes", "config", "get", "model.default")] = result("token=not-for-report\nsecond\n")

    with pytest.raises(HermesError, match="invalid") as exc_info:
        HermesBackend(config(), runner=FakeRunner(outputs)).snapshot()

    assert "not-for-report" not in str(exc_info.value)
