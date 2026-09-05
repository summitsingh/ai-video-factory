from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class HermesCheck(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    status: Literal["ready", "not_ready"]
    detail: str | int | float | bool | None


class HermesAttempt(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    backend: Literal["local_lmstudio", "nous_free_fallback"]
    outcome: Literal["pass", "fail", "not_attempted"]
    retryable: bool
    reason_code: str | None


class HermesAuxiliaryRoute(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    enabled: bool
    base_url: str | None
    model: str | None


class HermesResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    command: Literal["doctor", "smoke-text", "smoke-vision", "smoke-fallback"]
    status: Literal["pass", "fail", "not_ready"]
    retryable: bool
    parent_provider: Literal["nous"]
    parent_model: str
    attempts: list[HermesAttempt]
    checks: dict[str, HermesCheck]
    metrics: dict[str, int | float | str | bool | None]
    artifacts: dict[str, str]
    error: str | None

    @field_validator("schema_version", mode="before")
    @classmethod
    def reject_boolean_schema_version(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("schema_version must be the number 1, not a boolean")
        return value


_AUXILIARY_ROUTE_NAMES = {
    "vision",
    "web_extract",
    "compression",
    "title_generation",
    "background_review",
}


class HermesSnapshot(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    hermes_path: str
    version: str
    commit: str
    config_path: str
    profile: Literal["default"]
    config_valid: bool
    parent_provider: str
    parent_model: str
    parent_base_url: str
    delegation_model: str | None
    delegation_base_url: str | None
    delegation_api_mode: str | None
    delegation_max_iterations: int | None
    delegation_max_concurrent_children: int | None
    delegation_max_spawn_depth: int | None
    delegation_orchestrator_enabled: bool | None
    delegation_subagent_auto_approve: bool | None
    delegation_inherit_mcp_toolsets: bool | None
    auxiliary_routes: dict[
        Literal[
            "vision",
            "web_extract",
            "compression",
            "title_generation",
            "background_review",
        ],
        HermesAuxiliaryRoute,
    ]

    @model_validator(mode="after")
    def validate_auxiliary_routes(self) -> Self:
        if set(self.auxiliary_routes) != _AUXILIARY_ROUTE_NAMES:
            raise ValueError("auxiliary routes must contain exactly the five supported routes")
        return self
