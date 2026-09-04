from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class StageStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass(frozen=True)
class ArtifactIntegrity:
    """A content digest for one artifact contained by the configured artifact root."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    stage: str
    input_fingerprint: str
    status: StageStatus
    resumed: bool
    created_at: str
    updated_at: str
    inputs: dict[str, Any]
    artifacts: dict[str, Any]
    artifact_integrity: dict[str, ArtifactIntegrity]
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        return cls(
            schema_version=data["schema_version"],
            run_id=data["run_id"],
            stage=data["stage"],
            input_fingerprint=data["input_fingerprint"],
            status=StageStatus(data["status"]),
            resumed=data["resumed"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            inputs=data["inputs"],
            artifacts=data["artifacts"],
            artifact_integrity={
                key: ArtifactIntegrity(**record)
                for key, record in data.get("artifact_integrity", {}).items()
            },
            error=data["error"],
        )
