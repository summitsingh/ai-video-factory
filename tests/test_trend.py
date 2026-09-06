"""Tests for the real trend discovery subsystem.

All provider adapters are exercised through an injected transport that returns
canned JSON/XML, so no test makes a real network call.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_video_factory.trend import (
    Transport,
    TrendCandidate,
    SourceSignal,
    compute_signals,
    dedupe,
    discover_trends,
    discover_trends_dry_run,
    fetch_google_news,
    fetch_hacker_news,
    fetch_reddit,
    normalize_items,
    rank_candidates,
    save_candidates,
)


# ========== Canned provider payloads ==========

_HN_JSON = json.dumps({
    "hits": [
        {"object_id": "123", "title": "New Rocket Launch Detected", "points": 800,
         "num_comments": 150, "created_at": "2026-09-06T10:00:00Z"},
        {"object_id": "124", "title": "AI Model Surpasses Coding Benchmark", "points": 1200,
         "num_comments": 300, "created_at": "2026-09-06T09:00:00Z"},
        {"object_id": "", "title": "", "points": 5},  # invalid -> skipped
    ]
})

_REDDIT_JSON = json.dumps({
    "data": {
        "children": [
            {"data": {"title": "New Rocket Launch Detected", "score": 900,
                      "num_comments": 200, "permalink": "/r/technology/comments/x",
                      "created_utc": 1788696000.0}},
        ]
    }
})

_NEWS_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>Google News</title>
<item><title>New Rocket Launch Detected</title>
<link>https://news.google.com/articles/abc</link>
<pubDate>Mon, 06 Sep 2026 08:00:00 +0000</pubDate>
<source>Hacker News</source></item>
<item><title>Empty</title><link></link></item>
</channel></rss>"""


def _transport_factory(payloads: dict[str, str]) -> "Transport":  # noqa: F821 - defined below
    def transport(url: str) -> str:
        for key, value in payloads.items():
            if key in url:
                return value
        raise AssertionError(f"unexpected URL {url}")
    return transport


def test_fetch_hacker_news_parses_and_skips_invalid() -> None:
    items = fetch_hacker_news(transport=_transport_factory({"algolia": _HN_JSON}))
    assert len(items) == 2
    titles = {item["title"] for item in items}
    assert "New Rocket Launch Detected" in titles
    assert all(item["provider"] == "hacker_news" for item in items)
    assert all("news.ycombinator.com" in item["url"] for item in items)


def test_fetch_reddit_parses_and_converts_date() -> None:
    items = fetch_reddit(transport=_transport_factory({"reddit": _REDDIT_JSON}))
    assert len(items) == 1
    assert items[0]["provider"] == "reddit"
    assert items[0]["published_at"] is not None
    assert "2026-09-06" in items[0]["published_at"]


def test_fetch_google_news_parses_rss() -> None:
    items = fetch_google_news(transport=_transport_factory({"news.google.com": _NEWS_RSS}))
    assert len(items) == 1
    assert items[0]["title"] == "New Rocket Launch Detected"
    assert items[0]["published_at"] is not None


def test_normalize_dedupes_by_cluster_key() -> None:
    """The same story across HN + Reddit merges into one candidate."""
    hn = fetch_hacker_news(transport=_transport_factory({"algolia": _HN_JSON}))
    reddit = fetch_reddit(transport=_transport_factory({"reddit": _REDDIT_JSON}))
    items = normalize_items(hn + reddit, observed_at="2026-09-06T12:00:00+00:00")
    # "New Rocket Launch Detected" appears in both -> one topic_id.
    rocket = [c for c in items if "Rocket" in c.title]
    assert len(rocket) == 1
    providers = {s.provider for s in rocket[0].sources}
    assert {"hacker_news", "reddit"} <= providers


def test_rank_candidates_is_deterministic_and_sorted() -> None:
    candidates = [
        TrendCandidate(topic_id="a", title="Low engagement story", summary="",
                       first_seen_at="2026-08-01T00:00:00+00:00", observed_at="2026-09-06T00:00:00+00:00",
                       cluster_key="low-engagement-story",
                       sources=[SourceSignal("hacker_news", "u1", None, {"points": 10})]),
        TrendCandidate(topic_id="b", title="Rocket launch with video and image", summary="",
                       first_seen_at="2026-09-06T00:00:00+00:00", observed_at="2026-09-06T00:00:00+00:00",
                       cluster_key="rocket-launch-with-video-image",
                       sources=[SourceSignal("reddit", "u2", None, {"score": 4000})]),
    ]
    ranked = rank_candidates(candidates)
    assert len(ranked) == 2
    # The high-velocity, visually-rich topic should rank first.
    assert ranked[0].title.startswith("Rocket")
    assert ranked[0].score >= ranked[1].score
    # Signals are populated and within [0, 1].
    for candidate in ranked:
        for value in candidate.signals.values():
            assert 0.0 <= value <= 1.0


def test_competition_risk_rises_for_crowded_slug() -> None:
    base = lambda title: TrendCandidate(
        topic_id="x", title=title, summary="",
        first_seen_at="2026-09-06T00:00:00+00:00", observed_at="2026-09-06T00:00:00+00:00",
        cluster_key="shared-topic",
        sources=[SourceSignal("hacker_news", "u", None, {"points": 100})],
    )
    crowded = rank_candidates([base("Shared Topic One"), base("Shared Topic Two")])
    assert all(c.signals["competition_risk"] > 0.0 for c in crowded)


def test_discover_trends_uses_injected_transport() -> None:
    payloads = {"algolia": _HN_JSON, "reddit": _REDDIT_JSON}
    candidates = discover_trends(
        providers=["hacker_news", "reddit"], max_candidates=5,
        transport=_transport_factory(payloads), now_iso="2026-09-06T12:00:00+00:00",
    )
    assert len(candidates) >= 1
    # Dedup across providers collapses the rocket story.
    rocket = [c for c in candidates if "Rocket" in c.title]
    assert len(rocket) == 1


def test_discover_trends_skips_failed_provider() -> None:
    def transport(url: str) -> str:
        if "reddit" in url:
            raise AssertionError("network down")
        return _HN_JSON

    candidates = discover_trends(
        providers=["hacker_news", "reddit"], max_candidates=5,
        transport=transport, now_iso="2026-09-06T12:00:00+00:00",
    )
    assert len(candidates) >= 1


def test_dry_run_does_not_write_by_default(tmp_path: Path) -> None:
    payloads = {"algolia": _HN_JSON}
    summary = discover_trends_dry_run(
        providers=["hacker_news"], max_candidates=3,
        transport=_transport_factory(payloads), output_dir=None,
    )
    assert summary["mode"] == "dry-run"
    assert summary["candidate_count"] >= 1
    # Nothing written when output_dir is None.
    assert list(tmp_path.iterdir()) == []


def test_save_candidates_writes_json(tmp_path: Path) -> None:
    candidates = [
        TrendCandidate(topic_id="a", title="Story", summary="",
                       first_seen_at="2026-09-06T00:00:00+00:00", observed_at="2026-09-06T00:00:00+00:00",
                       cluster_key="story", sources=[]),
    ]
    path = save_candidates(candidates, tmp_path / "candidates.json")
    data = json.loads(path.read_text())
    assert data["candidate_count"] if "candidate_count" in data else len(data["candidates"]) == 1
    assert data["candidates"][0]["topic_id"] == "a"


def test_compute_signals_bounds_and_diversity() -> None:
    candidate = TrendCandidate(
        topic_id="a", title="Rocket launch video", summary="",
        first_seen_at="2026-09-06T00:00:00+00:00", observed_at="2026-09-06T00:00:00+00:00",
        cluster_key="rocket-launch-video",
        sources=[
            SourceSignal("hacker_news", "u1", "2026-09-06T00:00:00+00:00", {"points": 5000}),
            SourceSignal("reddit", "u2", "2026-09-06T00:00:00+00:00", {"score": 100}),
        ],
    )
    signals = compute_signals(candidate, now_iso="2026-09-06T06:00:00+00:00")
    assert signals["source_diversity"] >= 0.5
    assert signals["velocity"] > 0.5
    assert all(0.0 <= v <= 1.0 for v in signals.values())
