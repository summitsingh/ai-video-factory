from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_video_factory.models import ArtifactIntegrity, RunManifest, StageStatus
from ai_video_factory.sanitization import sanitize_diagnostic


class FingerprintMismatch(ValueError):
    """Raised when inputs do not match the run's recorded fingerprint."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when a completed run's declared artifacts cannot be verified."""


_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def fingerprint_inputs(inputs: dict[str, Any]) -> str:
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RunStore:
    def __init__(self, root: Path, *, artifact_root: Path | None = None) -> None:
        self.root = Path(root)
        self.artifact_root = Path(artifact_root or root).resolve()

    def start(self, stage: str, inputs: dict[str, Any]) -> RunManifest:
        self._validate_stage(stage)
        fingerprint = fingerprint_inputs(inputs)
        completed = self._newest_completed(stage, fingerprint)
        if completed is not None:
            resumed = replace(completed, resumed=True, updated_at=_timestamp())
            self._write_manifest(resumed)
            self._append_event(resumed, "run_resumed", {"status": resumed.status})
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
            artifact_integrity={},
            error=None,
        )
        self._write_manifest(manifest)
        self._append_event(manifest, "run_started", {"status": manifest.status})
        return manifest

    def completed_read_only(self, stage: str, inputs: dict[str, Any]) -> RunManifest | None:
        """Find the newest integrity-valid completed run without changing any state."""
        self._validate_stage(stage)
        fingerprint = fingerprint_inputs(inputs)
        stage_directory = self.root / stage
        if not stage_directory.is_dir():
            return None
        matches: list[RunManifest] = []
        for path in self._manifest_paths_for_stage(stage_directory):
            manifest = self._read_manifest(path)
            if (
                manifest.status is StageStatus.completed
                and manifest.input_fingerprint == fingerprint
                and self._artifact_integrity_error(manifest) is None
            ):
                matches.append(manifest)
        return max(matches, key=lambda manifest: manifest.updated_at, default=None)

    def resume(self, run_id: str, inputs: dict[str, Any]) -> RunManifest:
        manifest = self._load_run(run_id)
        if manifest.input_fingerprint != fingerprint_inputs(inputs):
            raise FingerprintMismatch(f"inputs do not match run {run_id}")
        if manifest.status is StageStatus.completed:
            integrity_error = self._artifact_integrity_error(manifest)
            if integrity_error is not None:
                self._invalidate(manifest, integrity_error)
                raise ArtifactIntegrityError(integrity_error)

        resumed = replace(manifest, resumed=True, updated_at=_timestamp())
        self._write_manifest(resumed)
        self._append_event(resumed, "run_resumed", {"status": resumed.status})
        return resumed

    def event(self, run_id: str, event: str, fields: dict[str, Any]) -> None:
        manifest = self._load_run(run_id)
        self._append_event(manifest, event, fields)

    def _append_event(
        self, manifest: RunManifest, event: str, fields: dict[str, Any]
    ) -> None:
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

    def complete(
        self,
        run_id: str,
        artifacts: dict[str, Any],
        *,
        expected_artifacts: dict[str, Path],
    ) -> RunManifest:
        manifest = self._load_run(run_id)
        if manifest.status is not StageStatus.running:
            raise ValueError("only running runs can complete")
        integrity = self._record_artifact_integrity(artifacts, expected_artifacts)
        completed = replace(
            manifest,
            status=StageStatus.completed,
            updated_at=_timestamp(),
            artifacts=artifacts,
            artifact_integrity=integrity,
            error=None,
        )
        self._write_manifest(completed)
        self._append_event(completed, "run_completed", {"artifacts": artifacts})
        return completed

    def fail(self, run_id: str, error: object) -> RunManifest:
        """Atomically mark only an active run failed and persist a safe error event."""
        manifest = self._load_run(run_id)
        if manifest.status is not StageStatus.running:
            raise ValueError("only running runs can fail")
        safe_error = sanitize_diagnostic(error)
        failed = replace(
            manifest,
            status=StageStatus.failed,
            updated_at=_timestamp(),
            error=safe_error,
        )
        self._write_manifest(failed)
        self._append_event(failed, "run_failed", {"error": safe_error})
        return failed

    def invalidate_completed(
        self, run_id: str, inputs: dict[str, Any], error: object
    ) -> RunManifest:
        """Invalidate one exact completed run after stage-specific verification fails."""
        manifest = self._load_run(run_id)
        if manifest.input_fingerprint != fingerprint_inputs(inputs):
            raise FingerprintMismatch(f"inputs do not match run {run_id}")
        if manifest.status is not StageStatus.completed:
            raise ValueError("only completed runs can be invalidated")
        return self._invalidate(manifest, sanitize_diagnostic(error))

    def _newest_completed(self, stage: str, fingerprint: str) -> RunManifest | None:
        self._validate_stage(stage)
        stage_directory = self.root / stage
        if not stage_directory.is_dir():
            return None

        matches = []
        for path in self._manifest_paths_for_stage(stage_directory):
            manifest = self._read_manifest(path)
            if (
                manifest.status is StageStatus.completed
                and manifest.input_fingerprint == fingerprint
            ):
                integrity_error = self._artifact_integrity_error(manifest)
                if integrity_error is None:
                    matches.append(manifest)
                else:
                    self._invalidate(manifest, integrity_error)
        return max(matches, key=lambda manifest: manifest.updated_at, default=None)

    def _load_run(self, run_id: str) -> RunManifest:
        self._validate_run_id(run_id)
        manifests: list[Path] = []
        if self.root.is_dir():
            for stage_directory in self.root.iterdir():
                if not stage_directory.is_dir():
                    continue
                try:
                    self._validate_stage(stage_directory.name)
                except ValueError:
                    continue
                manifest_path = stage_directory / run_id / "manifest.json"
                if manifest_path.is_file():
                    manifests.append(manifest_path)
        if len(manifests) != 1:
            raise FileNotFoundError(f"run {run_id} was not found")
        return self._read_manifest(manifests[0])

    @staticmethod
    def _manifest_paths_for_stage(stage_directory: Path) -> list[Path]:
        paths: list[Path] = []
        for run_directory in stage_directory.iterdir():
            if not run_directory.is_dir() or _RUN_ID.fullmatch(run_directory.name) is None:
                continue
            manifest_path = run_directory / "manifest.json"
            if manifest_path.is_file():
                paths.append(manifest_path)
        return paths

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
        self._validate_stage(manifest.stage)
        self._validate_run_id(manifest.run_id)
        return self.root / manifest.stage / manifest.run_id

    def _record_artifact_integrity(
        self,
        artifacts: dict[str, Any],
        expected_artifacts: dict[str, Path],
    ) -> dict[str, ArtifactIntegrity]:
        if not expected_artifacts:
            raise ValueError("a completed run must declare at least one expected artifact")
        records: dict[str, ArtifactIntegrity] = {}
        for key, supplied_path in expected_artifacts.items():
            if key not in artifacts:
                raise ValueError(f"expected artifact {key!r} is absent from artifacts")
            artifact_path, relative_path = self._contained_artifact_path(supplied_path)
            try:
                recorded_path, _ = self._contained_artifact_path(Path(str(artifacts[key])))
            except (TypeError, ValueError) as error:
                raise ValueError(f"artifact {key!r} path is invalid") from error
            if recorded_path != artifact_path:
                raise ValueError(f"artifact {key!r} does not match its expected path")
            if not artifact_path.is_file():
                raise FileNotFoundError(f"expected artifact {key!r} does not exist")
            records[key] = ArtifactIntegrity(
                path=relative_path.as_posix(),
                sha256=_sha256(artifact_path),
                size_bytes=artifact_path.stat().st_size,
            )
        return records

    def _artifact_integrity_error(self, manifest: RunManifest) -> str | None:
        if not manifest.artifact_integrity:
            return "artifact verification failed: manifest has no artifact digests"
        for key, record in manifest.artifact_integrity.items():
            try:
                artifact_path, relative_path = self._contained_artifact_path(
                    self.artifact_root / record.path
                )
                recorded_path, _ = self._contained_artifact_path(
                    Path(str(manifest.artifacts[key]))
                )
            except (KeyError, TypeError, ValueError):
                return f"artifact verification failed: {key!r} has an invalid contained path"
            if relative_path.as_posix() != record.path or recorded_path != artifact_path:
                return f"artifact verification failed: {key!r} path changed"
            if not artifact_path.is_file():
                return f"artifact verification failed: {key!r} is missing"
            if artifact_path.stat().st_size != record.size_bytes:
                return f"artifact verification failed: {key!r} size changed"
            if _sha256(artifact_path) != record.sha256:
                return f"artifact verification failed: {key!r} digest changed"
        return None

    def _contained_artifact_path(self, path: Path) -> tuple[Path, Path]:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.artifact_root / candidate
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(self.artifact_root)
        except ValueError as error:
            raise ValueError("expected artifact must be contained by artifact root") from error
        if relative == Path("."):
            raise ValueError("expected artifact must be contained by artifact root")
        return resolved, relative

    def _invalidate(self, manifest: RunManifest, reason: str) -> RunManifest:
        safe_reason = sanitize_diagnostic(reason)
        invalidated = replace(
            manifest,
            status=StageStatus.failed,
            updated_at=_timestamp(),
            error=safe_reason,
        )
        self._write_manifest(invalidated)
        self._append_event(invalidated, "run_invalidated", {"error": safe_reason})
        return invalidated

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if (
            not stage
            or stage in {".", ".."}
            or Path(stage).is_absolute()
            or "/" in stage
            or "\\" in stage
        ):
            raise ValueError("stage must be a single safe path component")

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run_id must be exactly 32 lowercase hexadecimal characters")

    @staticmethod
    def _read_manifest(path: Path) -> RunManifest:
        manifest = RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        RunStore._validate_stage(manifest.stage)
        RunStore._validate_run_id(manifest.run_id)
        return manifest


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
