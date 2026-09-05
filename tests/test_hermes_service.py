from __future__ import annotations

from ai_video_factory.hermes_backend import HermesError
from ai_video_factory.hermes_config import HermesConfig
from ai_video_factory.hermes_models import HermesSnapshot
from ai_video_factory.hermes_service import HermesService


def config() -> HermesConfig:
    return HermesConfig(
        schema_version=1, hermes_binary="hermes", required_version="0.21.0",
        required_commit="b0ab2e16", profile="default", parent_provider="nous",
        parent_model="stepfun/step-3.7-flash:free",
        delegation_base_url="http://127.0.0.1:1234/v1",
        delegation_model="avf-qwen36-executor", delegation_api_mode="chat_completions",
        local_api_key_placeholder="no-key-required", max_concurrent_children=1,
        max_iterations=50, vision_fixture="assets/vision/capability-probe.png",
        fallback_provider="nous", fallback_model="stepfun/step-3.7-flash:free",
    )


def snapshot(**changes: object) -> HermesSnapshot:
    values: dict[str, object] = {
        "hermes_path": "/home/summit/.local/bin/hermes", "version": "0.21.0", "commit": "b0ab2e16",
        "config_path": "/safe/config.yaml", "profile": "default", "config_valid": True,
        "parent_provider": "nous", "parent_model": "stepfun/step-3.7-flash:free",
        "parent_base_url": "https://nous.example/v1", "delegation_model": "avf-qwen36-executor",
        "delegation_base_url": "http://127.0.0.1:1234/v1", "delegation_api_mode": "chat_completions",
        "delegation_max_iterations": 50, "delegation_max_concurrent_children": 1,
        "delegation_max_spawn_depth": 1, "delegation_orchestrator_enabled": False,
        "delegation_subagent_auto_approve": False, "delegation_inherit_mcp_toolsets": False,
        "auxiliary_routes": {name: {"enabled": False, "base_url": None, "model": None}
            for name in ("vision", "web_extract", "compression", "title_generation", "background_review")},
    }
    values.update(changes)
    return HermesSnapshot.model_validate(values)


class Backend:
    def __init__(self, value: HermesSnapshot | BaseException) -> None:
        self.value = value

    def snapshot(self) -> HermesSnapshot:
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


def test_doctor_passes_only_for_exact_routes_and_current_text_capability() -> None:
    result = HermesService(config(), backend=Backend(snapshot()), text_capability=lambda: True).doctor()

    assert result.status == "pass"
    assert result.retryable is False
    assert set(result.checks) == {
        "installation", "parent_route", "delegation_route", "delegation_safety", "text_capability"
    }
    assert result.attempts == []
    assert result.error is None


def test_unconfigured_delegation_is_not_ready_and_never_leaks_secrets() -> None:
    result = HermesService(
        config(), backend=Backend(snapshot(
            delegation_model=None, delegation_base_url=None, delegation_api_mode=None,
            delegation_max_iterations=None, delegation_max_concurrent_children=None,
            delegation_max_spawn_depth=None, delegation_orchestrator_enabled=None,
            delegation_subagent_auto_approve=None, delegation_inherit_mcp_toolsets=None,
        )),
        text_capability=lambda: True,
    ).doctor()

    assert result.status == "not_ready"
    assert "token=" not in result.model_dump_json()
    assert "api_key" not in result.model_dump_json().casefold()


def test_partial_delegation_is_a_fail_closed_configuration_conflict() -> None:
    result = HermesService(
        config(), backend=Backend(snapshot(delegation_model=None)), text_capability=lambda: True,
    ).doctor()

    assert result.status == "fail"


def test_doctor_fails_closed_for_wrong_provider_paid_model_remote_or_recursion() -> None:
    for changes in (
        {"parent_provider": "other"},
        {"parent_model": "paid-model"},
        {"delegation_base_url": "https://remote.example/v1"},
        {"delegation_api_mode": "responses"},
        {"delegation_max_iterations": 49},
        {"delegation_max_concurrent_children": 2},
        {"delegation_max_spawn_depth": 2},
        {"delegation_orchestrator_enabled": True},
        {"delegation_subagent_auto_approve": True},
        {"delegation_inherit_mcp_toolsets": True},
    ):
        result = HermesService(config(), backend=Backend(snapshot(**changes)), text_capability=lambda: True).doctor()
        assert result.status == "fail"


def test_stale_text_capability_is_not_ready() -> None:
    result = HermesService(config(), backend=Backend(snapshot()), text_capability=lambda: False).doctor()

    assert result.status == "not_ready"
    assert result.retryable is True


def test_doctor_requires_a_boolean_true_capability_and_audited_installation_path() -> None:
    weak_capability = HermesService(
        config(), backend=Backend(snapshot()), text_capability=lambda: 1,  # type: ignore[return-value]
    ).doctor()
    wrong_path = HermesService(
        config(), backend=Backend(snapshot(hermes_path="/tmp/hermes")), text_capability=lambda: True,
    ).doctor()

    assert weak_capability.status == "not_ready"
    assert wrong_path.status == "fail"


def test_unexpected_exception_is_sanitized_failure() -> None:
    result = HermesService(
        config(), backend=Backend(HermesError("token=never-emit")), text_capability=lambda: True
    ).doctor()

    assert result.status == "not_ready"
    assert result.error is not None
    assert "never-emit" not in result.error


def test_truly_unexpected_exception_is_a_nonretryable_failure() -> None:
    result = HermesService(
        config(), backend=Backend(RuntimeError("unanticipated")), text_capability=lambda: True,
    ).doctor()

    assert result.status == "fail"
    assert result.retryable is False
