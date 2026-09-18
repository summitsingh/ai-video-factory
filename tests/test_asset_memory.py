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


def test_event_blocklist_covers_talking_heads():
    from ai_video_factory.nasa_media import _record_is_event_photo

    for title in (
        "NASA astronaut interview in studio",
        "News anchor desk segment about Artemis",
        "Media day Q&A with engineers",
        "Roundtable discussion on Mars",
    ):
        assert _record_is_event_photo(title) is True, title
    assert _record_is_event_photo("Earth from the ISS cupola") is False


def test_fetch_nasa_skips_low_relevance(monkeypatch, tmp_path):
    from ai_video_factory import nasa_media

    records = [
        {
            "title": "News anchor desk segment about space",
            "nasa_id": "junk-1",
            "source_url": "https://example.com/1",
        },
        {
            "title": "The Andromeda galaxy in ultraviolet",
            "nasa_id": "good-1",
            "source_url": "https://example.com/2",
        },
    ]
    monkeypatch.setattr(
        nasa_media, "search_nasa", lambda q, mediatype: records
    )
    monkeypatch.setattr(
        nasa_media, "_nasa_image_url",
        lambda nasa_id: f"https://example.com/{nasa_id}.jpg",
    )
    # Fake download: write a tiny valid PNG so _image_is_usable passes.
    from PIL import Image

    def fake_download(url, dest):
        Image.new("RGB", (800, 450), (10, 10, 40)).save(dest)

    monkeypatch.setattr(nasa_media, "download_asset", fake_download)

    assets = nasa_media.fetch_nasa_for_scene(
        ["galaxy"],
        tmp_path / "scene-00",
        max_images=2,
        relevance_text="a spiral galaxy with bright stars",
    )
    titles = [a.title for a in assets]
    assert titles == ["The Andromeda galaxy in ultraviolet"], titles
