from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_video_factory.hermes_config import HermesConfig, load_hermes_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def hermes_config_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "hermes_binary": "hermes",
        "required_version": "0.21.0",
        "required_commit": "b0ab2e16",
        "profile": "default",
        "parent_provider": "nous",
        "parent_model": "stepfun/step-3.7-flash:free",
        "delegation_base_url": "http://127.0.0.1:1234/v1",
        "delegation_model": "avf-qwen36-executor",
        "delegation_api_mode": "chat_completions",
        "local_api_key_placeholder": "no-key-required",
        "max_concurrent_children": 1,
        "max_iterations": 50,
        "vision_fixture": "assets/vision/capability-probe.png",
        "fallback_provider": "nous",
        "fallback_model": "stepfun/step-3.7-flash:free",
    }


def test_checked_in_hermes_config_is_exact() -> None:
    config = load_hermes_config(PROJECT_ROOT / "config" / "hermes.toml")

    assert config.model_dump() == hermes_config_data()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("max_concurrent_children", True),
        ("max_iterations", True),
        ("delegation_base_url", "./hermes"),
        ("delegation_base_url", "http://localhost:1234/v1"),
        ("delegation_base_url", "http://127.0.0.1:1234/v1?token=secret"),
        ("delegation_base_url", "http://user:password@127.0.0.1:1234/v1"),
        ("fallback_model", "stepfun/step-3.7-flash:paid"),
        ("fallback_model", "other/free-model:free"),
        ("max_concurrent_children", 2),
        ("vision_fixture", "assets/vision/../../secret.png"),
    ],
)
def test_rejects_invalid_strict_configuration_values(field: str, value: object) -> None:
    config = hermes_config_data()
    config[field] = value

    with pytest.raises(ValidationError):
        HermesConfig.model_validate(config)


def test_rejects_extra_configuration_values() -> None:
    config = hermes_config_data()
    config["unexpected"] = "value"

    with pytest.raises(ValidationError):
        HermesConfig.model_validate(config)
