from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


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
