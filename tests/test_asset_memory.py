"""Tests for asset_memory: persistent blocklist and relevance scorer."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ai_video_factory.asset_memory import AssetMemory, score_relevance


def test_not_blocked_initially(tmp_path: Path):
    mem = AssetMemory(tmp_path / "memory.json")
    assert mem.is_blocked("nasa-123") is False
    assert "nasa-123" not in mem


def test_block_after_rejection(tmp_path: Path):
    mem = AssetMemory(tmp_path / "memory.json")
    mem.record_rejection("nasa-123", "black_frame", source="nasa")
    assert mem.is_blocked("nasa-123") is True
    assert mem.reason_for("nasa-123") == "black_frame"
    assert len(mem) == 1


def test_rejection_persists_across_instances(tmp_path: Path):
    path = tmp_path / "memory.json"
    AssetMemory(path).record_rejection("nasa-456", "heavy_text_overlay")
    fresh = AssetMemory(path)
    assert fresh.is_blocked("nasa-456") is True
    assert fresh.reason_for("nasa-456") == "heavy_text_overlay"


def test_reject_count_increments(tmp_path: Path):
    path = tmp_path / "memory.json"
    mem = AssetMemory(path)
    mem.record_rejection("nasa-789", "black_frame")
    mem.record_rejection("nasa-789", "black_frame")
    raw = json.loads(path.read_text())
    assert raw["rejections"]["nasa-789"]["count"] == 2


def test_prune_removes_old_entries(tmp_path: Path):
    path = tmp_path / "memory.json"
    mem = AssetMemory(path)
    mem.record_rejection("old-asset", "black_frame")
    raw = json.loads(path.read_text())
    raw["rejections"]["old-asset"]["rejected_at"] = time.time() - 100 * 86400
    path.write_text(json.dumps(raw))
    mem2 = AssetMemory(path)
    removed = mem2.prune(days=90)
    assert removed == 1
    assert mem2.is_blocked("old-asset") is False


def test_prune_keeps_recent_entries(tmp_path: Path):
    path = tmp_path / "memory.json"
    mem = AssetMemory(path)
    mem.record_rejection("new-asset", "static_slate")
    assert mem.prune(days=90) == 0
    assert mem.is_blocked("new-asset") is True


def test_stats_summary(tmp_path: Path):
    mem = AssetMemory(tmp_path / "memory.json")
    mem.record_rejection("a1", "black_frame")
    mem.record_rejection("a2", "black_frame")
    mem.record_rejection("a3", "heavy_text_overlay")
    stats = mem.stats()
    assert stats["blocked"] == 3
    assert stats["reasons"]["black_frame"] == 2
    assert stats["reasons"]["heavy_text_overlay"] == 1


def test_corrupt_file_starts_empty(tmp_path: Path):
    path = tmp_path / "memory.json"
    path.write_text("not json {{{")
    mem = AssetMemory(path)
    assert len(mem) == 0
    assert mem.is_blocked("anything") is False


def test_score_relevance_overlap():
    score = score_relevance(
        "Carina Nebula glowing gas clouds in infrared",
        "slow push through a glowing nebula cloud",
    )
    assert score >= 0.5, score


def test_score_relevance_no_overlap():
    score = score_relevance(
        "Press conference at Johnson Space Center",
        "slow push through a glowing nebula",
    )
    assert score == 0.0, score


def test_score_relevance_empty_visual():
    assert score_relevance("Some title", "") == 0.0
    assert score_relevance("", "some visual") == 0.0
