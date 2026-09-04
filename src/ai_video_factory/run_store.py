from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_video_factory.models import RunManifest, StageStatus


class FingerprintMismatch(ValueError):
    """Raised when inputs do not match the run's recorded fingerprint."""


def fingerprint_inputs(inputs: dict[str, Any]) -> str:
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def start(self, stage: str, inputs: dict[str, Any]) -> RunManifest:
        fingerprint = fingerprint_inputs(inputs)
        completed = self._newest_completed(stage, fingerprint)
        if completed is not None:
            resumed = replace(completed, resumed=True, updated_at=_timestamp())
            self._write_manifest(resumed)
            return resumed

        now = _timestamp()
        manifest = RunManifest(
            schema_version=1,
            run_id=uuid4().hex,
            stage=stage,
            input_fingerprint=fingerprint,
            status=StageStatus.running,
            resumed=False,
            created_at=now,
            updated_at=now,
            inputs=inputs,
            artifacts={},
            error=None,
        )
        self._write_manifest(manifest)
        return manifest

    def resume(self, run_id: str, inputs: dict[str, Any]) -> RunManifest:
        manifest = self._load_run(run_id)
        if manifest.input_fingerprint != fingerprint_inputs(inputs):
            raise FingerprintMismatch(f"inputs do not match run {run_id}")

        resumed = replace(manifest, resumed=True, updated_at=_timestamp())
        self._write_manifest(resumed)
        return resumed

    def event(self, run_id: str, event: str, fields: dict[str, Any]) -> None:
        manifest = self._load_run(run_id)
        event_path = self._run_directory(manifest) / "events.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": _timestamp(),
            "run_id": manifest.run_id,
            "stage": manifest.stage,
            "event": event,
            "fields": fields,
        }
        with event_path.open("a", encoding="utf-8") as events_file:
            events_file.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
            events_file.write("\n")

    def complete(self, run_id: str, artifacts: dict[str, Any]) -> RunManifest:
        manifest = self._load_run(run_id)
        completed = replace(
            manifest,
            status=StageStatus.completed,
            updated_at=_timestamp(),
            artifacts=artifacts,
            error=None,
        )
        self._write_manifest(completed)
        return completed

    def _newest_completed(self, stage: str, fingerprint: str) -> RunManifest | None:
        stage_directory = self.root / stage
        if not stage_directory.is_dir():
            return None

        matches = []
        for path in stage_directory.glob("*/manifest.json"):
            manifest = self._read_manifest(path)
            if (
                manifest.status is StageStatus.completed
                and manifest.input_fingerprint == fingerprint
            ):
                matches.append(manifest)
        return max(matches, key=lambda manifest: manifest.updated_at, default=None)

    def _load_run(self, run_id: str) -> RunManifest:
        manifests = list(self.root.glob(f"*/{run_id}/manifest.json"))
        if len(manifests) != 1:
            raise FileNotFoundError(f"run {run_id} was not found")
        return self._read_manifest(manifests[0])

    def _write_manifest(self, manifest: RunManifest) -> None:
        directory = self._run_directory(manifest)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        temporary_path = manifest_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)

    def _run_directory(self, manifest: RunManifest) -> Path:
        return self.root / manifest.stage / manifest.run_id

    @staticmethod
    def _read_manifest(path: Path) -> RunManifest:
        return RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
