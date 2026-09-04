import json
from pathlib import Path

import pytest

from ai_video_factory.models import StageStatus
from ai_video_factory.run_store import FingerprintMismatch, RunStore, fingerprint_inputs


def test_fingerprint_is_order_independent() -> None:
    assert fingerprint_inputs({"b": 2, "a": 1}) == fingerprint_inputs({"a": 1, "b": 2})


def test_completed_run_resumes_with_identical_inputs(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    artifact_root = tmp_path / "artifacts"
    store = RunStore(state_root, artifact_root=artifact_root)
    first = store.start("render", {"fixture": "v1"})
    video = artifact_root / first.run_id / "master.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"rendered-video")
    store.complete(
        first.run_id,
        {"video": str(video)},
        expected_artifacts={"video": video},
    )

    resumed = store.start("render", {"fixture": "v1"})

    assert resumed.run_id == first.run_id
    assert resumed.resumed is True
    assert _event_names(state_root, first.run_id) == [
        "run_started",
        "run_completed",
        "run_resumed",
    ]


def test_explicit_resume_rejects_changed_inputs(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    first = store.start("render", {"fixture": "v1"})

    with pytest.raises(FingerprintMismatch):
        store.resume(first.run_id, {"fixture": "v2"})


def test_explicit_resume_records_lifecycle_event(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    first = store.start("render", {"fixture": "v1"})

    store.resume(first.run_id, {"fixture": "v1"})

    assert _event_names(tmp_path, first.run_id) == ["run_started", "run_resumed"]


def test_event_is_appended_with_required_structured_fields(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run = store.start("render", {"fixture": "v1"})

    store.event(run.run_id, "render_started", {"frame_count": 24})
    store.event(run.run_id, "render_finished", {"frame_count": 24})

    event_path = tmp_path / "render" / run.run_id / "events.jsonl"
    lines = event_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    event = json.loads(lines[1])
    assert event["run_id"] == run.run_id
    assert event["stage"] == "render"
    assert event["event"] == "render_started"
    assert event["fields"] == {"frame_count": 24}
    assert event["timestamp"]


def test_complete_persists_completed_manifest(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    store = RunStore(tmp_path / "state", artifact_root=artifact_root)
    run = store.start("render", {"fixture": "v1"})
    video = artifact_root / run.run_id / "master.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"rendered-video")

    completed = store.complete(
        run.run_id,
        {"video": str(video)},
        expected_artifacts={"video": video},
    )

    assert completed.status is StageStatus.completed
    assert completed.artifacts == {"video": str(video)}
    assert completed.artifact_integrity["video"].path == f"{run.run_id}/master.mp4"
    assert len(completed.artifact_integrity["video"].sha256) == 64


@pytest.mark.parametrize("mutation", ["delete", "tamper"])
def test_start_invalidates_completed_run_when_artifact_is_not_intact(
    tmp_path: Path, mutation: str
) -> None:
    """A missing or changed artifact must force a fresh stage execution."""
    state_root = tmp_path / "state"
    artifact_root = tmp_path / "artifacts"
    store = RunStore(state_root, artifact_root=artifact_root)
    first = store.start("render", {"fixture": "v1"})
    video = artifact_root / first.run_id / "master.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original")
    store.complete(
        first.run_id,
        {"video": str(video)},
        expected_artifacts={"video": video},
    )
    if mutation == "delete":
        video.unlink()
    else:
        video.write_bytes(b"tampered")

    replacement = store.start("render", {"fixture": "v1"})

    assert replacement.run_id != first.run_id
    assert replacement.status is StageStatus.running
    old_manifest = json.loads(
        (state_root / "render" / first.run_id / "manifest.json").read_text()
    )
    assert old_manifest["status"] == "failed"
    assert "artifact verification failed" in old_manifest["error"]
    assert _event_names(state_root, first.run_id)[-1] == "run_invalidated"


def test_complete_rejects_expected_artifact_outside_declared_root(tmp_path: Path) -> None:
    """A manifest must not bless an artifact path outside its contained root."""
    store = RunStore(tmp_path / "state", artifact_root=tmp_path / "artifacts")
    run = store.start("render", {"fixture": "v1"})
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")

    with pytest.raises(ValueError, match="contained by artifact root"):
        store.complete(
            run.run_id,
            {"video": str(outside)},
            expected_artifacts={"video": outside},
        )


def test_fail_atomically_transitions_running_manifest_and_records_sanitized_error(
    tmp_path: Path,
) -> None:
    """An exception must leave the active run failed, not indefinitely running."""
    store = RunStore(tmp_path)
    run = store.start("render", {"fixture": "v1"})

    failed = store.fail(run.run_id, "Authorization: Bearer manifest-secret")

    assert failed.status is StageStatus.failed
    assert failed.error == "Authorization: [REDACTED]"
    manifest_path = tmp_path / "render" / run.run_id / "manifest.json"
    persisted = json.loads(manifest_path.read_text())
    assert persisted["status"] == "failed"
    assert "manifest-secret" not in manifest_path.read_text()
    event_path = tmp_path / "render" / run.run_id / "events.jsonl"
    assert "manifest-secret" not in event_path.read_text()
    assert _event_names(tmp_path, run.run_id)[-1] == "run_failed"
    assert not manifest_path.with_suffix(".tmp").exists()


def test_fail_refuses_to_overwrite_completed_upstream_stage(tmp_path: Path) -> None:
    """Only the currently running stage may transition to failed."""
    artifact = tmp_path / "render" / "placeholder.mp4"
    store = RunStore(tmp_path, artifact_root=tmp_path / "render")
    run = store.start("render", {"fixture": "v1"})
    artifact.write_bytes(b"video")
    store.complete(
        run.run_id,
        {"video": str(artifact)},
        expected_artifacts={"video": artifact},
    )

    with pytest.raises(ValueError, match="only running runs can fail"):
        store.fail(run.run_id, "later stage failed")


@pytest.mark.parametrize(
    "run_id",
    [
        "a" * 31,
        "a" * 33,
        "A" * 32,
        "../" + ("a" * 29),
        "*" * 32,
    ],
)
def test_public_run_operations_reject_noncanonical_run_ids(
    tmp_path: Path, run_id: str
) -> None:
    """Run IDs must be validated before any path is constructed or searched."""
    store = RunStore(tmp_path)

    with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
        store.event(run_id, "event", {})


@pytest.mark.parametrize("stage", ["/tmp/escaped", "../escaped", "render/nested"])
def test_start_rejects_unsafe_stage_paths(tmp_path: Path, stage: str) -> None:
    with pytest.raises(ValueError, match="safe path component"):
        RunStore(tmp_path).start(stage, {"fixture": "v1"})


def _event_names(root: Path, run_id: str) -> list[str]:
    event_path = root / "render" / run_id / "events.jsonl"
    return [json.loads(line)["event"] for line in event_path.read_text().splitlines()]
