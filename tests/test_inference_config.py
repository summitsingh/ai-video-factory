from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_video_factory.inference_config import InferenceConfig, load_inference_config


def valid_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
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
        "models_directory": "/home/summit/.lmstudio/models",
    }
    config.update(overrides)
    return config


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_loads_checked_in_lm_studio_configuration(project_root: Path) -> None:
    config = load_inference_config(project_root / "config" / "inference.toml")
    assert config.schema_version == 1
    assert config.backend == "lm_studio"
    assert config.base_url == "http://127.0.0.1:1234/v1"
    assert config.lms_binary == "lms"
    assert config.model_key == "qwen3.6-35b-a3b-udt-mtp"
    assert config.identifier == "avf-qwen36-executor"
    assert config.context_length == 65_536
    assert config.gpu == "max"
    assert config.parallel == 1
    assert config.ttl_seconds == 3_600
    assert config.minimum_available_memory_gib == 40
    assert config.models_directory == "/home/summit/.lmstudio/models"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:1234/v1",
        "http://localhost:1234/v1",
        "http://0.0.0.0:1234/v1",
        "http://127.0.0.1:1234/v1?token=secret",
        "http://user:secret@127.0.0.1:1234/v1",
    ],
)
def test_rejects_noncanonical_or_credentialed_endpoint(base_url: str) -> None:
    with pytest.raises(ValidationError):
        InferenceConfig.model_validate(valid_config(base_url=base_url))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("identifier", "AB"),
        ("models_directory", "relative/models"),
        ("context_length", 0),
        ("parallel", 2),
        ("ttl_seconds", 0),
        ("minimum_available_memory_gib", 0),
    ],
)
def test_rejects_invalid_strict_configuration_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        InferenceConfig.model_validate(valid_config(**{field: value}))


@pytest.mark.parametrize("field", ["schema_version", "parallel"])
def test_rejects_boolean_numeric_literals(field: str) -> None:
    with pytest.raises(ValidationError):
        InferenceConfig.model_validate(valid_config(**{field: True}))
