from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_video_factory.hermes_models import (
    HermesAttempt,
    HermesAuxiliaryRoute,
    HermesCheck,
    HermesResult,
    HermesSnapshot,
)


def auxiliary_routes() -> dict[str, HermesAuxiliaryRoute]:
    return {
        "vision": HermesAuxiliaryRoute(
            enabled=True,
            base_url="http://127.0.0.1:1234/v1",
            model="avf-tiel-coder-executor",
        ),
        "web_extract": HermesAuxiliaryRoute(
            enabled=True,
            base_url="http://127.0.0.1:1234/v1",
            model="avf-tiel-coder-executor",
        ),
        "compression": HermesAuxiliaryRoute(
            enabled=True,
            base_url="http://127.0.0.1:1234/v1",
            model="avf-tiel-coder-executor",
        ),
        "title_generation": HermesAuxiliaryRoute(enabled=False, base_url=None, model=None),
        "background_review": HermesAuxiliaryRoute(enabled=False, base_url=None, model=None),
    }


def snapshot_data() -> dict[str, object]:
    return {
        "hermes_path": "/usr/local/bin/hermes",
        "version": "0.21.0",
        "commit": "b0ab2e16",
        "config_path": "/tmp/hermes/config.yaml",
        "profile": "default",
        "config_valid": True,
        "parent_provider": "nous",
        "parent_model": "stepfun/step-3.7-flash:free",
        "parent_base_url": "https://example.invalid/v1",
        "delegation_model": "avf-tiel-coder-executor",
        "delegation_base_url": "http://127.0.0.1:1234/v1",
        "delegation_api_mode": "chat_completions",
        "delegation_max_iterations": 50,
        "delegation_max_concurrent_children": 1,
        "delegation_max_spawn_depth": 1,
        "delegation_orchestrator_enabled": False,
        "delegation_subagent_auto_approve": False,
        "delegation_inherit_mcp_toolsets": False,
        "auxiliary_routes": auxiliary_routes(),
    }


def test_result_models_preserve_the_public_orchestration_contract() -> None:
    result = HermesResult(
        command="doctor",
        status="pass",
        retryable=False,
        parent_provider="nous",
        parent_model="stepfun/step-3.7-flash:free",
        attempts=[
            HermesAttempt(
                backend="local_lmstudio",
                outcome="pass",
                retryable=False,
                reason_code=None,
            )
        ],
        checks={"routes": HermesCheck(status="ready", detail=True)},
        metrics={"latency_seconds": 1.25},
        artifacts={"report": "data/report.json"},
        error=None,
    )

    assert result.model_dump()["schema_version"] == 1
    assert result.model_dump()["attempts"][0]["backend"] == "local_lmstudio"


def test_snapshot_exposes_only_the_five_known_auxiliary_routes() -> None:
    snapshot = HermesSnapshot.model_validate(snapshot_data())

    assert set(snapshot.auxiliary_routes) == {
        "vision",
        "web_extract",
        "compression",
        "title_generation",
        "background_review",
    }
    assert set(HermesSnapshot.model_fields) == {
        "hermes_path",
        "version",
        "commit",
        "config_path",
        "profile",
        "config_valid",
        "parent_provider",
        "parent_model",
        "parent_base_url",
        "delegation_model",
        "delegation_base_url",
        "delegation_api_mode",
        "delegation_max_iterations",
        "delegation_max_concurrent_children",
        "delegation_max_spawn_depth",
        "delegation_orchestrator_enabled",
        "delegation_subagent_auto_approve",
        "delegation_inherit_mcp_toolsets",
        "auxiliary_routes",
    }
    assert {
        "api_key",
        "delegation_api_key",
        "auxiliary_api_key",
        "raw_config",
    }.isdisjoint(HermesSnapshot.model_fields)


@pytest.mark.parametrize(
    "routes",
    [
        {
            key: value
            for key, value in auxiliary_routes().items()
            if key != "vision"
        },
        {**auxiliary_routes(), "other": HermesAuxiliaryRoute(enabled=False, base_url=None, model=None)},
    ],
)
def test_snapshot_rejects_missing_or_unrecognized_auxiliary_routes(
    routes: dict[str, HermesAuxiliaryRoute],
) -> None:
    data = snapshot_data()
    data["auxiliary_routes"] = routes

    with pytest.raises(ValidationError):
        HermesSnapshot.model_validate(data)


def result_data(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "command": "doctor",
        "status": "pass",
        "retryable": False,
        "parent_provider": "nous",
        "parent_model": "stepfun/step-3.7-flash:free",
        "attempts": [],
        "checks": {},
        "metrics": {},
        "artifacts": {},
        "error": None,
    }
    result.update(overrides)
    return result


def test_result_rejects_boolean_schema_version() -> None:
    with pytest.raises(ValidationError):
        HermesResult.model_validate(result_data(schema_version=True))


def test_result_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HermesResult.model_validate(result_data(unexpected="value"))


@pytest.mark.parametrize(
    "payload",
    [
        {"status": 1, "detail": None},
        {"status": "ready", "detail": ["unsupported"]},
        {"status": "ready", "detail": b"coercion"},
    ],
)
def test_check_rejects_non_strict_public_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        HermesCheck.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"enabled": 1, "base_url": None, "model": None},
        {"enabled": True, "base_url": 1234, "model": None},
        {"enabled": True, "base_url": None, "model": 1234},
        {
            "enabled": True,
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "avf-tiel-coder-executor",
            "api_key": "not-public",
        },
        {
            "enabled": True,
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "avf-tiel-coder-executor",
            "unexpected": "not-public",
        },
    ],
)
def test_auxiliary_route_rejects_non_strict_or_secret_values(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        HermesAuxiliaryRoute.model_validate(payload)
