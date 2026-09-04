import json
from pathlib import Path

import pytest

from ai_video_factory.models import StageStatus
from ai_video_factory.run_store import FingerprintMismatch, RunStore, fingerprint_inputs


def test_fingerprint_is_order_independent() -> None:
    assert fingerprint_inputs({"b": 2, "a": 1}) == fingerprint_inputs({"a": 1, "b": 2})


def test_completed_run_resumes_with_identical_inputs(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    first = store.start("render", {"fixture": "v1"})
    store.complete(first.run_id, {"video": "output/master.mp4"})

    resumed = store.start("render", {"fixture": "v1"})

    assert resumed.run_id == first.run_id
    assert resumed.resumed is True
    assert _event_names(tmp_path, first.run_id) == [
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
    store = RunStore(tmp_path)
    run = store.start("render", {"fixture": "v1"})

    completed = store.complete(run.run_id, {"video": "output/master.mp4"})

    assert completed.status is StageStatus.completed
    assert completed.artifacts == {"video": "output/master.mp4"}


@pytest.mark.parametrize("stage", ["/tmp/escaped", "../escaped", "render/nested"])
def test_start_rejects_unsafe_stage_paths(tmp_path: Path, stage: str) -> None:
    with pytest.raises(ValueError, match="safe path component"):
        RunStore(tmp_path).start(stage, {"fixture": "v1"})


def _event_names(root: Path, run_id: str) -> list[str]:
    event_path = root / "render" / run_id / "events.jsonl"
    return [json.loads(line)["event"] for line in event_path.read_text().splitlines()]
