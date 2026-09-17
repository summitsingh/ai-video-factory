from __future__ import annotations

import tomllib
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HermesConfig(BaseModel):
    """Strict, non-secret expected Hermes routing configuration."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1]
    hermes_binary: str = Field(min_length=1)
    required_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    required_commit: str = Field(pattern=r"^[0-9a-f]{8}$")
    profile: Literal["default"]
    parent_provider: Literal["nous"]
    parent_model: Literal["stepfun/step-3.7-flash:free"]
    delegation_base_url: str
    delegation_model: Literal["avf-qwen36-executor"]
    delegation_api_mode: Literal["chat_completions"]
    local_api_key_placeholder: Literal["no-key-required"]
    max_concurrent_children: Literal[1]
    max_iterations: int = Field(ge=1, le=100)
    vision_fixture: str
    fallback_provider: Literal["nous"]
    fallback_model: Literal["stepfun/step-3.7-flash:free"]

    @field_validator(
        "schema_version", "max_concurrent_children", "max_iterations", mode="before"
    )
    @classmethod
    def reject_boolean_numeric_values(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("numeric configuration values must not be boolean")
        return value

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if self.delegation_base_url != "http://127.0.0.1:1234/v1":
            raise ValueError("delegation endpoint must be the exact LM Studio loopback URL")
        fixture = PurePosixPath(self.vision_fixture)
        if fixture.is_absolute() or ".." in fixture.parts:
            raise ValueError("vision fixture must be a contained project-relative path")
        return self


def load_hermes_config(path: Path) -> HermesConfig:
    """Load and validate a checked-in Hermes configuration contract."""
    with path.open("rb") as config_file:
        return HermesConfig.model_validate(tomllib.load(config_file))
