from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class InferenceConfig(BaseModel):
    """Strict configuration for the local LM Studio inference backend."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1]
    backend: Literal["lm_studio"]
    base_url: str
    lms_binary: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    identifier: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    context_length: int = Field(gt=0)
    gpu: Literal["max"]
    parallel: Literal[1]
    ttl_seconds: int = Field(gt=0)
    minimum_available_memory_gib: int = Field(gt=0)
    models_directory: str = Field(min_length=1)
    server_config_path: str = Field(min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def reject_boolean_schema_version(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("schema_version must be the number 1, not a boolean")
        return value

    @field_validator("parallel", mode="before")
    @classmethod
    def reject_boolean_parallel(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("parallel must be the number 1, not a boolean")
        return value

    @model_validator(mode="after")
    def validate_loopback_endpoint_and_model_directory(self) -> Self:
        endpoint = urlsplit(self.base_url)
        try:
            port = endpoint.port
        except ValueError as exc:
            raise ValueError("base_url must use port 1234") from exc

        if (
            endpoint.scheme != "http"
            or endpoint.hostname != "127.0.0.1"
            or port != 1234
            or endpoint.path != "/v1"
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
            or "?" in self.base_url
            or "#" in self.base_url
        ):
            raise ValueError("base_url must be http://127.0.0.1:1234/v1 without credentials")
        if not Path(self.models_directory).is_absolute():
            raise ValueError("models_directory must be absolute")
        if not Path(self.server_config_path).is_absolute():
            raise ValueError("server_config_path must be absolute")
        return self


def load_inference_config(path: Path) -> InferenceConfig:
    """Load and validate a TOML inference configuration file."""
    with path.open("rb") as config_file:
        return InferenceConfig.model_validate(tomllib.load(config_file))
