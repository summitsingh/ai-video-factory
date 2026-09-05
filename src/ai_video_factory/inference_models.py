from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class InferenceCheck(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    status: Literal["ready", "not_ready"]
    detail: str | int | float | bool | None


class MemoryEstimate(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    gpu_gib: float
    total_gib: float
    confidence: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    allowed: bool


class ModelIdentity(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    model_key: str
    identifier: str
    relative_path: str
    size_bytes: int
    sha256: str | None = None


class InferenceResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    command: Literal["doctor", "estimate", "start", "status", "benchmark", "stop"]
    status: Literal["pass", "fail", "not_ready"]
    retryable: bool
    backend: Literal["lm_studio"] = "lm_studio"
    model_identifier: str
    checks: dict[str, InferenceCheck]
    metrics: dict[str, int | float | str | bool | None]
    artifacts: dict[str, str]
    error: str | None

    @field_validator("schema_version", mode="before")
    @classmethod
    def reject_boolean_schema_version(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("schema_version must be the number 1, not a boolean")
        return value
