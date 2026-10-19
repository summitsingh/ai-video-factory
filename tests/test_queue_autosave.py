"""Tests for the L4 crash-safe queue autosave (nightly_batch + daily_job).

All tests are fast and offline: they cover the snapshot/queue manifest
helpers and the run_nightly / run_daily_job wiring, never rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_video_factory.daily_job import (
    JobConfig,
    daily_queue_path,
    enqueue_daily_job,
    install_daily_autosave,
    load_daily_queue,
    mark_interrupted_stale_entries,
    run_daily_job,
    save_daily_queue,
)
from ai_video_factory.nightly_batch import (
    NightlyConfig,
    _autosave_registered,
    clear_queue_snapshot,
    install_queue_autosave,
    load_queue_items,
    load_queue_snapshot,
    queue_dir,
    run_nightly,
    save_queue_snapshot,
    snapshot_path,
)


def _items(n: int = 2) -> list[dict]:
    return [{"id": f"2026-10-18-{i:02d}", "topic": f"Topic {i}", "status": "pending"} for i in range(n)]


# ========== nightly_batch snapshot ==========


def test_snapshot_round_trip(tmp_path: Path):
    qdir = queue_dir(tmp_path, "2026-10-18")
    save_queue_snapshot(qdir, _items())
    loaded = load_queue_snapshot(qdir)
    assert loaded == _items()
    raw = json.loads(snapshot_path(qdir).read_text())
    assert raw["kind"] == "nightly-queue-snapshot"
    assert raw["schema_version"] == 1


def test_snapshot_missing_or_corrupt_returns_empty(tmp_path: Path):
    qdir = queue_dir(tmp_path, "2026-10-18")
    assert load_queue_snapshot(qdir) == []
    snapshot_path(qdir).parent.mkdir(parents=True, exist_ok=True)
    snapshot_path(qdir).write_text("not json{{")
    assert load_queue_snapshot(qdir) == []
    snapshot_path(qdir).write_text(json.dumps({"kind": "nope", "items": [1]}))
    assert load_queue_snapshot(qdir) == []


def test_clear_queue_snapshot(tmp_path: Path):
    qdir = queue_dir(tmp_path, "2026-10-18")
    save_queue_snapshot(qdir, _items())
    assert snapshot_path(qdir).is_file()
    clear_queue_snapshot(qdir)
    assert not snapshot_path(qdir).is_file()
    clear_queue_snapshot(qdir)  # idempotent, never raises


def test_install_queue_autosave_idempotent(tmp_path: Path):
    qdir = queue_dir(tmp_path, "2026-10-18")
    install_queue_autosave(qdir, lambda: [])
    install_queue_autosave(qdir, lambda: [])
    assert str(qdir) in _autosave_registered


def test_run_nightly_restores_from_fresher_snapshot(tmp_path: Path):
    config = NightlyConfig(topics_per_night=1, date="2026-10-18")
    qdir = queue_dir(tmp_path, "2026-10-18")
    # Simulate a crashed run: snapshot newer than (missing) queue.json.
    save_queue_snapshot(qdir, _items(3))

    def research_fn(n: int):
        return []

    def script_fn(candidate: dict):
        raise AssertionError("must not run: no fresh candidates")

    def pipeline_fn(**kwargs):
        raise AssertionError("must not run: no fresh candidates")

    report = run_nightly(
        tmp_path, config,
        research_fn=research_fn, script_fn=script_fn, pipeline_fn=pipeline_fn,
    )
    assert report["restored_from_snapshot"] is True
    assert report["queue_size"] == 3
    assert report["drafted"] == 0
    # Clean finish clears the snapshot; queue.json holds the restored items.
    assert not snapshot_path(qdir).is_file()
    assert load_queue_items(qdir / "queue.json") == _items(3)


def test_run_nightly_clears_snapshot_on_clean_finish(tmp_path: Path):
    config = NightlyConfig(topics_per_night=1, date="2026-10-18")
    qdir = queue_dir(tmp_path, "2026-10-18")

    def research_fn(n: int):
        return [{"title": "Fresh topic", "description": "d", "url": "https://x.example"}]

    def script_fn(candidate: dict):
        return {"title": "Video: Fresh topic", "summary": "s"}

    def pipeline_fn(**kwargs):
        return {"status": "pass", "run_id": "r1"}

    report = run_nightly(
        tmp_path, config,
        research_fn=research_fn, script_fn=script_fn, pipeline_fn=pipeline_fn,
    )
    assert report["restored_from_snapshot"] is False
    assert report["drafted"] == 1
    assert not snapshot_path(qdir).is_file()


# ========== daily_job queue manifest ==========


def test_daily_queue_empty_and_corrupt(tmp_path: Path):
    assert load_daily_queue(tmp_path) == []
    path = daily_queue_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json{{")
    assert load_daily_queue(tmp_path) == []
    path.write_text(json.dumps({"kind": "nope", "entries": [1]}))
    assert load_daily_queue(tmp_path) == []


def test_enqueue_and_load_round_trip(tmp_path: Path):
    entry = enqueue_daily_job(tmp_path, {"id": "job-1", "topic": "smoke"})
    assert entry["status"] == "queued"
    assert entry["id"] == "job-1"
    assert len(load_daily_queue(tmp_path)) == 1
    raw = json.loads(daily_queue_path(tmp_path).read_text())
    assert raw["kind"] == "daily-job-queue"
    assert raw["schema_version"] == 1


def test_mark_interrupted_stale_entries(tmp_path: Path):
    save_daily_queue(
        tmp_path,
        [
            {"id": "a", "status": "in_progress", "stage": "qc"},
            {"id": "b", "status": "completed"},
        ],
    )
    assert mark_interrupted_stale_entries(tmp_path) == 1
    entries = load_daily_queue(tmp_path)
    assert entries[0]["status"] == "interrupted"
    assert "interrupted_at" in entries[0]
    assert entries[1]["status"] == "completed"
    assert mark_interrupted_stale_entries(tmp_path) == 0  # idempotent


def test_install_daily_autosave_idempotent(tmp_path: Path):
    install_daily_autosave(tmp_path, lambda: [])
    install_daily_autosave(tmp_path, lambda: [])


def test_run_daily_job_failure_marks_queue_entry(tmp_path: Path):
    # No providers -> no candidates -> fast failure through _record_failure.
    config = JobConfig(providers=[])
    result = run_daily_job(tmp_path, config=config)
    assert result.status == "failed"
    entries = load_daily_queue(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "failed"
    assert entry["stage"] == "done"
    assert "started_at" in entry and "finished_at" in entry


def test_run_daily_job_flips_previous_interrupted(tmp_path: Path):
    save_daily_queue(tmp_path, [{"id": "stale", "status": "in_progress", "stage": "qc"}])
    result = run_daily_job(tmp_path, config=JobConfig(providers=[]))
    assert result.status == "failed"
    entries = load_daily_queue(tmp_path)
    assert entries[0]["status"] == "interrupted"
    assert entries[1]["status"] == "failed"
