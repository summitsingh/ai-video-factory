"""Tests for the channel scheduler: configs, dedup, locks, due checks, queue."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ai_video_factory.scheduler import (
    ChannelConfig,
    ReviewEntry,
    ReviewQueue,
    SchedulerBusy,
    SchedulerError,
    TopicLedger,
    channel_status,
    is_channel_due,
    load_channels_config,
    normalize_topic,
    pick_topic,
    read_channel_state,
    run_channel,
    scheduler_lock,
    write_channel_state,
)


def _config(**overrides):
    base = {
        "name": "Test Channel",
        "slug": "test-channel",
        "topic_focus": "space stuff",
    }
    base.update(overrides)
    return ChannelConfig(**base)


def test_load_channels_config_example():
    path = Path(__file__).resolve().parents[1] / "config" / "channels.toml"
    configs = load_channels_config(path)
    assert len(configs) == 2
    assert configs[0].slug == "deep-space-diaries"
    assert configs[0].target_duration_minutes == 25.0
    assert configs[1].schedule_time == "09:30"


def test_load_channels_config_missing(tmp_path):
    with pytest.raises(SchedulerError, match="not found"):
        load_channels_config(tmp_path / "nope.toml")


def test_channel_config_validation():
    with pytest.raises(SchedulerError, match="privacy"):
        _config(privacy="everyone")
    with pytest.raises(SchedulerError, match="20-30"):
        _config(target_duration_minutes=10.0)
    with pytest.raises(SchedulerError, match="schedule_time"):
        _config(schedule_time="25:99")
    with pytest.raises(SchedulerError, match="timezone"):
        _config(timezone="Mars/Olympus")


def test_normalize_topic():
    assert normalize_topic("  Water on MARS!! ") == "water on mars"
    assert normalize_topic("NASA's Artemis-II") == "nasas artemisii"


def test_topic_ledger_exact_and_near_duplicate(tmp_path):
    ledger = TopicLedger(tmp_path / "topics.jsonl")
    ledger.add("chan", "Water on Mars")
    assert ledger.is_duplicate("chan", "water on mars")
    assert ledger.is_duplicate("chan", "Water on Mars!!")
    assert not ledger.is_duplicate("chan", "Jupiter's Great Red Spot")
    # Other channels are independent.
    assert not ledger.is_duplicate("other", "Water on Mars")


def test_topic_ledger_persists(tmp_path):
    path = tmp_path / "topics.jsonl"
    TopicLedger(path).add("chan", "Water on Mars", run_id="r1")
    reloaded = TopicLedger(path)
    assert reloaded.is_duplicate("chan", "water on mars")
    assert reloaded.topics_for("chan")[0]["run_id"] == "r1"


def test_scheduler_lock_exclusive(tmp_path):
    with scheduler_lock(tmp_path):
        with pytest.raises(SchedulerBusy):
            with scheduler_lock(tmp_path):
                pass
    # Released after the block.
    with scheduler_lock(tmp_path):
        pass


def test_is_channel_due(tmp_path):
    config = _config(schedule_time="09:00", timezone="UTC")
    state_dir = tmp_path / "state"
    morning = datetime(2026, 9, 18, 10, 0, tzinfo=ZoneInfo("UTC"))
    assert is_channel_due(config, state_dir, morning) is True
    write_channel_state(state_dir, config.slug, {"last_run_date": "2026-09-18"})
    assert is_channel_due(config, state_dir, morning) is False
    # Before the scheduled time it is not due even with no prior run.
    early = datetime(2026, 9, 19, 8, 0, tzinfo=ZoneInfo("UTC"))
    assert is_channel_due(config, state_dir, early) is False


def test_review_queue_round_trip(tmp_path):
    queue = ReviewQueue(tmp_path / "review.jsonl")
    queue.add(
        ReviewEntry(
            video_id="vid1", run_id="r1", channel="chan",
            title="T", url="https://www.youtube.com/watch?v=vid1",
            uploaded_at="2026-09-18T00:00:00",
        )
    )
    assert len(queue.list()) == 1
    assert len(queue.list(status="pending_review")) == 1
    updated = queue.set_status("vid1", "approved")
    assert updated["status"] == "approved"
    assert queue.list(status="pending_review") == []
    with pytest.raises(SchedulerError, match="no review entry"):
        queue.set_status("missing", "approved")


def test_pick_topic_skips_duplicates(tmp_path):
    config = _config()
    ledger = TopicLedger(tmp_path / "topics.jsonl")
    ledger.add(config.slug, "Water on Mars")

    def research(cfg):
        assert cfg is config
        return [
            {"title": "Water on Mars", "description": "d1"},
            {"title": "Europa's Ocean", "description": "d2"},
        ]

    picked = pick_topic(config, ledger, research)
    assert picked is not None and picked["title"] == "Europa's Ocean"

    ledger.add(config.slug, "Europa's Ocean")
    assert pick_topic(config, ledger, research) is None


def test_run_channel_happy_path(tmp_path):
    config = _config()
    ledger = TopicLedger(tmp_path / "topics.jsonl")
    queue = ReviewQueue(tmp_path / "review.jsonl")
    now = datetime(2026, 9, 18, 12, 0)

    def research(cfg):
        return [{"title": "Europa's Ocean", "description": "icy moon"}]

    def pipeline(cfg, title, description):
        assert cfg.target_duration_minutes == 25.0
        return {"status": "pass", "run_id": "run-1", "artifacts": {}}

    def upload(cfg, pipeline_result):
        assert pipeline_result["run_id"] == "run-1"
        return {"video_id": "vid-9", "title": "Europa's Ocean"}

    result = run_channel(
        config, data_dir=tmp_path, ledger=ledger, review_queue=queue,
        research_fn=research, pipeline_fn=pipeline, upload_fn=upload, now=now,
    )
    assert result["status"] == "uploaded_unlisted"
    assert result["video_id"] == "vid-9"
    assert ledger.is_duplicate(config.slug, "Europa's Ocean")
    assert len(queue.list(status="pending_review")) == 1
    state = read_channel_state(tmp_path / "state", config.slug)
    assert state["last_status"] == "uploaded_unlisted"
    assert state["last_run_date"] == "2026-09-18"


def test_run_channel_pipeline_failure_does_not_burn_topic(tmp_path):
    config = _config()
    ledger = TopicLedger(tmp_path / "topics.jsonl")
    queue = ReviewQueue(tmp_path / "review.jsonl")

    def research(cfg):
        return [{"title": "Europa's Ocean", "description": "icy moon"}]

    def pipeline(cfg, title, description):
        return {"status": "fail", "run_id": "run-2", "error": "render blew up"}

    def upload(cfg, pipeline_result):  # pragma: no cover - must not run
        raise AssertionError("upload must not run after pipeline failure")

    result = run_channel(
        config, data_dir=tmp_path, ledger=ledger, review_queue=queue,
        research_fn=research, pipeline_fn=pipeline, upload_fn=upload,
    )
    assert result["status"] == "failed"
    assert not ledger.is_duplicate(config.slug, "Europa's Ocean")
    assert queue.list() == []


def test_channel_status_snapshot(tmp_path):
    configs = [_config(slug="a"), _config(slug="b", name="B")]
    ledger = TopicLedger(tmp_path / "topics.jsonl")
    ledger.add("a", "Some Topic")
    queue = ReviewQueue(tmp_path / "review.jsonl")
    queue.add(
        ReviewEntry(video_id="v", run_id="r", channel="a", title="T",
                    url="u", uploaded_at="x")
    )
    write_channel_state(tmp_path / "state", "a", {"last_run_date": "2026-09-18", "last_status": "uploaded_unlisted"})
    now = datetime(2026, 9, 18, 12, 0, tzinfo=ZoneInfo("UTC"))
    rows = channel_status(configs, tmp_path, now)
    row_a = next(r for r in rows if r["channel"] == "a")
    assert row_a["due"] is False
    assert row_a["topics_used"] == 1
    assert row_a["pending_review"] is True
    row_b = next(r for r in rows if r["channel"] == "b")
    assert row_b["due"] is True
    assert row_b["pending_review"] is False


def test_review_entry_serialization(tmp_path):
    path = tmp_path / "review.jsonl"
    queue = ReviewQueue(path)
    queue.add(ReviewEntry(video_id="v", run_id="r", channel="c", title="T", url="u", uploaded_at="x"))
    raw = json.loads(path.read_text(encoding="utf-8").strip())
    assert raw["status"] == "pending_review"
