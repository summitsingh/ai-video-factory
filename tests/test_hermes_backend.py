from __future__ import annotations

import os
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
        delegation_model="avf-tiel-coder-executor",
        delegation_api_mode="chat_completions",
        local_api_key_placeholder="no-key-required",
        max_concurrent_children=1,
        max_iterations=50,
        vision_fixture="assets/vision/capability-probe.png",
        fallback_provider="nous",
        fallback_model="stepfun/step-3.7-flash:free",
    )


def result(stdout: str) -> ProcessResult:
    return ProcessResult(0, stdout, "", "/home/summit/.local/bin/hermes")


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
        ("hermes", "config", "get", "delegation.model"): "avf-tiel-coder-executor\n",
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


def merged_default_outputs() -> dict[tuple[str, ...], ProcessResult]:
    outputs = snapshot_outputs()
    outputs.update({
        ("hermes", "config", "get", "delegation.model"): result("\n"),
        ("hermes", "config", "get", "delegation.base_url"): result("\n"),
        ("hermes", "config", "get", "delegation.api_mode"): result("\n"),
    })
    return outputs


def test_allowlist_contains_only_the_fixed_read_only_vectors() -> None:
    assert ALLOWED_COMMANDS == frozenset(snapshot_outputs())


def test_default_runner_uses_resolved_executable_without_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class Completed:
        returncode = 0
        stdout = "Hermes Agent v0.21.0 (2026.8.31) · upstream b0ab2e16"
        stderr = ""

    def fake_run(argv: tuple[str, ...], **kwargs: object) -> Completed:
        calls.append((argv, kwargs))
        return Completed()

    executable = tmp_path / "hermes"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setattr("ai_video_factory.hermes_backend._HERMES_PATH", executable)
    monkeypatch.setattr("ai_video_factory.hermes_backend.shutil.which", lambda _: str(executable))
    monkeypatch.setattr("ai_video_factory.hermes_backend.subprocess.run", fake_run)

    observed = run_process(("hermes", "--version"), timeout=15.0)

    assert observed.executable_path == str(executable)
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0].startswith("/proc/self/fd/")
    assert argv[1:] == ("--version",)
    assert kwargs == {
        "shell": False,
        "capture_output": True,
        "text": True,
        "timeout": 15.0,
        "check": False,
        "pass_fds": (int(argv[0].rsplit("/", 1)[1]),),
    }


def test_default_runner_maps_only_documented_missing_config_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    executable = tmp_path / "hermes"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "Config key not set: delegation.model\n"

    monkeypatch.setattr("ai_video_factory.hermes_backend._HERMES_PATH", executable)
    monkeypatch.setattr("ai_video_factory.hermes_backend.shutil.which", lambda _: str(executable))
    monkeypatch.setattr("ai_video_factory.hermes_backend.subprocess.run", lambda *_args, **_kwargs: Completed())

    observed = run_process(("hermes", "config", "get", "delegation.model"), timeout=15.0)

    assert observed.missing is True


def test_default_runner_rejects_nonzero_exit_that_is_not_exact_missing_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    executable = tmp_path / "hermes"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "Config key not set: delegation.model\nextra"

    monkeypatch.setattr("ai_video_factory.hermes_backend._HERMES_PATH", executable)
    monkeypatch.setattr("ai_video_factory.hermes_backend.shutil.which", lambda _: str(executable))
    monkeypatch.setattr("ai_video_factory.hermes_backend.subprocess.run", lambda *_args, **_kwargs: Completed())

    with pytest.raises(HermesError, match="unsuccessfully"):
        run_process(("hermes", "config", "get", "delegation.model"), timeout=15.0)


def test_default_runner_rejects_combined_oversized_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    executable = tmp_path / "hermes"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)

    class Completed:
        returncode = 0
        stdout = "x" * (33 * 1024)
        stderr = "y" * (33 * 1024)

    monkeypatch.setattr("ai_video_factory.hermes_backend._HERMES_PATH", executable)
    monkeypatch.setattr("ai_video_factory.hermes_backend.shutil.which", lambda _: str(executable))
    monkeypatch.setattr("ai_video_factory.hermes_backend.subprocess.run", lambda *_args, **_kwargs: Completed())

    with pytest.raises(HermesError, match="64 KiB"):
        run_process(("hermes", "--version"), timeout=15.0)


def test_resolver_rejects_symlink_binds_swap_restore_to_opened_file_and_detects_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o700)
    executable = tmp_path / "hermes"
    executable.symlink_to(target)
    monkeypatch.setattr("ai_video_factory.hermes_backend._HERMES_PATH", executable)
    monkeypatch.setattr("ai_video_factory.hermes_backend.shutil.which", lambda _: str(executable))

    with pytest.raises(HermesError, match="resolved|canonical"):
        run_process(("hermes", "--version"), timeout=15.0)

    executable.unlink()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)

    class Completed:
        returncode = 0
        stdout = "Hermes Agent v0.21.0 (2026.8.31) · upstream b0ab2e16"
        stderr = ""

    def swap_and_restore(argv: tuple[str, ...], **_kwargs: object) -> Completed:
        assert argv[0].startswith("/proc/self/fd/")
        original = tmp_path / "original"
        os.replace(executable, original)
        replacement = tmp_path / "replacement"
        replacement.write_text("#!/bin/sh\n# replacement\n", encoding="utf-8")
        replacement.chmod(0o700)
        os.replace(replacement, executable)
        os.replace(original, executable)
        return Completed()

    monkeypatch.setattr("ai_video_factory.hermes_backend.subprocess.run", swap_and_restore)
    with pytest.raises(HermesError, match="changed"):
        run_process(("hermes", "--version"), timeout=15.0)

    def mutate_in_place(*_args: object, **_kwargs: object) -> Completed:
        executable.write_text("#!/bin/sh\n# changed\n", encoding="utf-8")
        return Completed()

    monkeypatch.setattr("ai_video_factory.hermes_backend.subprocess.run", mutate_in_place)
    with pytest.raises(HermesError, match="changed"):
        run_process(("hermes", "--version"), timeout=15.0)


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


def test_default_runner_rejects_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_video_factory.hermes_backend.shutil.which", lambda _: None)

    with pytest.raises(HermesError, match="not found"):
        run_process(("hermes", "--version"), timeout=15.0)


def test_resolver_rejects_a_path_result_other_than_the_audited_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    executable = tmp_path / "hermes"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setattr("ai_video_factory.hermes_backend._HERMES_PATH", executable)
    monkeypatch.setattr("ai_video_factory.hermes_backend.shutil.which", lambda _: str(tmp_path / "other"))

    with pytest.raises(HermesError, match="audited"):
        run_process(("hermes", "--version"), timeout=15.0)


def test_snapshot_uses_individual_nonsecret_reads_and_typed_values() -> None:
    runner = FakeRunner(snapshot_outputs())

    snapshot = HermesBackend(config(), runner=runner).snapshot()

    assert snapshot.model_dump() == {
        "hermes_path": "/home/summit/.local/bin/hermes",
        "version": "0.21.0",
        "commit": "b0ab2e16",
        "config_path": "/safe/config.yaml",
        "profile": "default",
        "config_valid": True,
        "parent_provider": "nous",
        "parent_model": "stepfun/step-3.7-flash:free",
        "parent_base_url": "https://nous.example/v1",
        "delegation_model": "avf-tiel-coder-executor",
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
            outputs[argv] = ProcessResult(1, "", "key not found", "/home/summit/.local/bin/hermes", missing=True)

    snapshot = HermesBackend(config(), runner=FakeRunner(outputs)).snapshot()

    assert snapshot.delegation_model is None
    assert snapshot.delegation_base_url is None


def test_snapshot_maps_real_merged_empty_route_defaults_to_typed_unconfigured() -> None:
    snapshot = HermesBackend(config(), runner=FakeRunner(merged_default_outputs())).snapshot()

    assert snapshot.delegation_model is None
    assert snapshot.delegation_base_url is None
    assert snapshot.delegation_api_mode is None
    assert snapshot.delegation_max_iterations == 50


def test_snapshot_rejects_secret_or_multiline_values_without_exposure() -> None:
    outputs = snapshot_outputs()
    outputs[("hermes", "config", "get", "model.default")] = result("token=not-for-report\nsecond\n")

    with pytest.raises(HermesError, match="invalid") as exc_info:
        HermesBackend(config(), runner=FakeRunner(outputs)).snapshot()

    assert "not-for-report" not in str(exc_info.value)


def test_snapshot_rejects_url_credentials_and_ambiguous_typed_values() -> None:
    outputs = snapshot_outputs()
    outputs[("hermes", "config", "get", "model.base_url")] = result("https://token@nous.example/v1\n")
    with pytest.raises(HermesError, match="invalid"):
        HermesBackend(config(), runner=FakeRunner(outputs)).snapshot()

    outputs = snapshot_outputs()
    outputs[("hermes", "config", "get", "delegation.orchestrator_enabled")] = result("TRUE\n")
    with pytest.raises(HermesError, match="invalid"):
        HermesBackend(config(), runner=FakeRunner(outputs)).snapshot()


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "ghp_abcdefghijklmnopqrstuvwxyz1234567890\n",
        "github_pat_abcdefghijklmnopqrstuvwxyz1234567890\n",
        "AKIAABCDEFGHIJKLMNOP\n",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature\n",
        "https://nous.example/v1?token=never\n",
        "nous\u200b\n",
        "nous\x1f\n",
    ],
)
def test_snapshot_rejects_secret_like_or_unicode_format_values(unsafe_value: str) -> None:
    outputs = snapshot_outputs()
    outputs[("hermes", "config", "get", "model.base_url")] = result(unsafe_value)

    with pytest.raises(HermesError, match="invalid") as exc_info:
        HermesBackend(config(), runner=FakeRunner(outputs)).snapshot()

    assert "never" not in str(exc_info.value)

@pytest.mark.parametrize(
    "version_text",
    [
        "Hermes Agent v0.21.0 (2026.8.31) · upstream B0AB2E16\n",
        "Hermes Agent v0.21.0 (2026.8.31) · upstream b0ab2e16\nHermes Agent v0.21.0 (2026.8.31) · upstream b0ab2e16\n",
        "Hermes Agent v0.21 (2026.8.31) · upstream b0ab2e16\n",
    ],
)
def test_snapshot_rejects_malformed_or_duplicate_version_identity(version_text: str) -> None:
    outputs = snapshot_outputs()
    outputs[("hermes", "--version")] = result(version_text)

    with pytest.raises(HermesError, match="invalid"):
        HermesBackend(config(), runner=FakeRunner(outputs)).snapshot()


def test_snapshot_rejects_nonzero_injected_config_get_even_when_optional() -> None:
    outputs = snapshot_outputs()
    outputs[("hermes", "config", "get", "delegation.model")] = ProcessResult(
        1, "", "arbitrary failure", "/home/summit/.local/bin/hermes",
    )

    with pytest.raises(HermesError, match="unsuccessfully"):
        HermesBackend(config(), runner=FakeRunner(outputs)).snapshot()

    outputs = snapshot_outputs()
    outputs[("hermes", "config", "get", "delegation.max_iterations")] = result("+50\n")
    with pytest.raises(HermesError, match="invalid"):
        HermesBackend(config(), runner=FakeRunner(outputs)).snapshot()
