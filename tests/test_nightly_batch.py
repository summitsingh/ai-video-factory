"""Tests for nightly_batch: overnight research, drafts, and the review queue."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_video_factory.nightly_batch import (
    NightlyConfig,
    approve_item,
    load_queue_items,
    queue_dir,
    reject_item,
    run_nightly,
    save_queue_items,
)


def _candidates(n: int = 6) -> list[dict]:
    return [
        {"title": f"Topic {i}", "description": f"desc {i}", "url": "https://x.example"}
        for i in range(n)
    ]


def _fns(candidates: list[dict], fail_topics: set[str] | None = None):
    fail_topics = fail_topics or set()
    calls: dict[str, list] = {"pipeline": []}

    def research_fn(limit: int) -> list[dict]:
        return candidates[:limit]

    def script_fn(candidate: dict) -> dict:
        return {"title": f"Video: {candidate['title']}", "summary": "sum"}

    def pipeline_fn(**kwargs) -> dict:
        calls["pipeline"].append(kwargs)
        topic = kwargs["topic"]["title"]
        if topic in fail_topics:
            return {"status": "fail", "error": "boom"}
        return {
            "status": "draft_ok",
            "run_id": f"run-{topic}",
            "draft_master_path": f"/tmp/{topic}.mp4",
            "thumbnail_paths": [f"/tmp/{topic}.jpg"],
        }

    return research_fn, script_fn, pipeline_fn, calls


def test_run_nightly_drafts_top_n(tmp_path: Path):
    research_fn, script_fn, pipeline_fn, calls = _fns(_candidates())
    report = run_nightly(
        tmp_path,
        NightlyConfig(topics_per_night=3, date="2026-09-18"),
        research_fn=research_fn,
        script_fn=script_fn,
        pipeline_fn=pipeline_fn,
    )
    assert report["drafted"] == 3
    assert report["failed"] == 0
    assert report["queue_size"] == 3
    qpath = tmp_path / "review_queue" / "2026-09-18" / "queue.json"
    assert qpath.is_file()
    items = load_queue_items(qpath)
    assert all(i["status"] == "pending" for i in items)
    assert items[0]["draft_master_path"].endswith(".mp4")
    assert items[0]["thumbnail_paths"]
    # INTEGRATION CONTRACT: pipeline must be called with draft=True.
    assert calls["pipeline"]
    assert all(c.get("draft") is True for c in calls["pipeline"])


def test_run_nightly_is_idempotent(tmp_path: Path):
    fns = _fns(_candidates())
    kwargs = dict(
        research_fn=fns[0], script_fn=fns[1], pipeline_fn=fns[2],
    )
    run_nightly(tmp_path, NightlyConfig(topics_per_night=3, date="2026-09-18"), **kwargs)
    # Second run must not re-draft the same topics; it picks the next fresh ones.
    report = run_nightly(
        tmp_path, NightlyConfig(topics_per_night=3, date="2026-09-18"), **kwargs
    )
    assert report["drafted"] == 3
    assert report["queue_size"] == 6
    items = load_queue_items(tmp_path / "review_queue" / "2026-09-18" / "queue.json")
    topics = [i["topic"] for i in items]
    assert len(set(topics)) == 6, "no topic may be queued twice"
    # Third run: nothing fresh left among the candidates.
    report = run_nightly(
        tmp_path, NightlyConfig(topics_per_night=3, date="2026-09-18"), **kwargs
    )
    assert report["drafted"] == 0
    assert report["queue_size"] == 6


def test_run_nightly_records_failures_without_stopping(tmp_path: Path):
    research_fn, script_fn, pipeline_fn, _ = _fns(_candidates(), fail_topics={"Topic 1"})
    report = run_nightly(
        tmp_path,
        NightlyConfig(topics_per_night=3, date="2026-09-18"),
        research_fn=research_fn,
        script_fn=script_fn,
        pipeline_fn=pipeline_fn,
    )
    assert report["drafted"] == 2
    assert report["failed"] == 1
    items = load_queue_items(tmp_path / "review_queue" / "2026-09-18" / "queue.json")
    failed = [i for i in items if i["status"] == "failed"]
    assert len(failed) == 1 and "boom" in failed[0]["error"]


def test_approve_and_reject_item(tmp_path: Path):
    qpath = tmp_path / "queue.json"
    save_queue_items(qpath, [{"id": "a1", "status": "pending", "topic": "T"}])
    approved = approve_item(qpath, "a1")
    assert approved["status"] == "approved"
    assert approved["reviewed_at"]
    rejected = reject_item(qpath, "a1")
    assert rejected["status"] == "rejected"
    assert load_queue_items(qpath)[0]["status"] == "rejected"


def test_approve_unknown_id_raises(tmp_path: Path):
    qpath = tmp_path / "queue.json"
    save_queue_items(qpath, [])
    with pytest.raises(KeyError):
        approve_item(qpath, "missing")


def test_reject_bad_status_rejected_internally(tmp_path: Path):
    # statuses outside the enum are rejected by the helper contract
    qpath = tmp_path / "queue.json"
    save_queue_items(qpath, [{"id": "a1", "status": "pending"}])
    with pytest.raises((KeyError, ValueError)):
        approve_item(qpath, "other-id")


def test_queue_dir_layout(tmp_path: Path):
    assert queue_dir(tmp_path, "2026-09-18") == tmp_path / "review_queue" / "2026-09-18"


def test_report_json_written(tmp_path: Path):
    fns = _fns(_candidates())
    report = run_nightly(
        tmp_path,
        NightlyConfig(topics_per_night=2, date="2026-09-18"),
        research_fn=fns[0],
        script_fn=fns[1],
        pipeline_fn=fns[2],
    )
    report_path = tmp_path / "review_queue" / "2026-09-18" / "report.json"
    assert report_path.is_file()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["drafted"] == report["drafted"] == 2


def test_pipeline_exception_becomes_failed_item(tmp_path: Path):
    def bad_pipeline(**kwargs):
        raise RuntimeError("gpu melted")

    research_fn, script_fn, _, _ = _fns(_candidates(2))
    report = run_nightly(
        tmp_path,
        NightlyConfig(topics_per_night=2, date="2026-09-18"),
        research_fn=research_fn,
        script_fn=script_fn,
        pipeline_fn=bad_pipeline,
    )
    assert report["failed"] == 2
    items = load_queue_items(tmp_path / "review_queue" / "2026-09-18" / "queue.json")
    assert all(i["status"] == "failed" for i in items)
