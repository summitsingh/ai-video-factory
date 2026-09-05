import pytest
from pydantic import ValidationError

from ai_video_factory.inference_models import (
    InferenceCheck,
    InferenceResult,
    MemoryEstimate,
    ModelIdentity,
)


def test_public_inference_models_accept_contract_values() -> None:
    check = InferenceCheck(status="ready", detail="LM Studio is available")
    estimate = MemoryEstimate(
        gpu_gib=24.0,
        total_gib=42.0,
        confidence="HIGH",
        allowed=True,
    )
    identity = ModelIdentity(
        model_key="qwen3.6-35b-a3b-udt-mtp",
        identifier="avf-qwen36-executor",
        relative_path="qwen/Qwen3.6-35B-A3B-UDT-MTP.gguf",
        size_bytes=42_000_000_000,
    )
    result = InferenceResult(
        command="doctor",
        status="pass",
        retryable=False,
        model_identifier=identity.identifier,
        checks={"endpoint": check},
        metrics={"gpu_gib": estimate.gpu_gib},
        artifacts={"config": "config/inference.toml"},
        error=None,
    )

    assert result.schema_version == 1
    assert result.backend == "lm_studio"
    assert identity.sha256 is None


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (InferenceCheck, {"status": "ready", "detail": "ok", "unexpected": True}),
        (
            MemoryEstimate,
            {
                "gpu_gib": 24.0,
                "total_gib": 42.0,
                "confidence": "HIGH",
                "allowed": True,
                "unexpected": True,
            },
        ),
        (
            ModelIdentity,
            {
                "model_key": "model",
                "identifier": "identifier",
                "relative_path": "model.gguf",
                "size_bytes": 1,
                "unexpected": True,
            },
        ),
    ],
)
def test_public_value_models_forbid_unknown_fields(
    model: type[InferenceCheck] | type[MemoryEstimate] | type[ModelIdentity],
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_result_forbids_unknown_fields_and_coercion() -> None:
    payload = {
        "command": "doctor",
        "status": "pass",
        "retryable": False,
        "model_identifier": "avf-qwen36-executor",
        "checks": {},
        "metrics": {},
        "artifacts": {},
        "error": None,
        "unexpected": True,
    }
    with pytest.raises(ValidationError):
        InferenceResult.model_validate(payload)

    payload.pop("unexpected")
    payload["retryable"] = "false"
    with pytest.raises(ValidationError):
        InferenceResult.model_validate(payload)


def test_result_rejects_boolean_schema_version() -> None:
    with pytest.raises(ValidationError):
        InferenceResult.model_validate({
            "schema_version": True,
            "command": "doctor",
            "status": "pass",
            "retryable": False,
            "model_identifier": "avf-qwen36-executor",
            "checks": {},
            "metrics": {},
            "artifacts": {},
            "error": None,
        })
