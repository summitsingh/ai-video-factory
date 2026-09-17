"""Research module for finding trending topics from real sources.

Two live providers back trend discovery:

* ``RedditProvider`` reads ``/hot.json`` for a configurable list of
  subreddits (public JSON, no auth).
* ``GoogleNewsProvider`` reads the Google News top-stories RSS feed.

Results are merged, deduplicated by normalized title, and ranked by a
simple recency + source-weight + engagement score. When every provider
fails, a ``ResearchError`` is raised (fail loud) instead of returning
fabricated data. A synthetic fallback exists only for offline development
and requires the ``AVF_ALLOW_SYNTHETIC_RESEARCH=1`` environment variable;
its output is tagged ``synthetic=True`` so it can never be mistaken for
real research.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Protocol


class ResearchError(RuntimeError):
    """Raised when no trend provider returns usable topics."""


_USER_AGENT = "AI-Video-Factory/0.1 (trend research; local only)"
_HTTP_TIMEOUT = 15
_SYNTHETIC_ENV_VAR = "AVF_ALLOW_SYNTHETIC_RESEARCH"

_REDDIT_BASE = "https://www.reddit.com"
_GNEWS_RSS = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"

# Relative trust weighting used when ranking merged topics.
_SOURCE_WEIGHTS = {
    "reddit": 1.0,
    "google_news": 1.2,
}


class TrendProvider(Protocol):
    """A source of trending topics.

    Returns a list of dicts with keys: ``title`` (str), ``source`` (str),
    ``url`` (str), ``published`` (aware datetime or None), ``score``
    (int engagement signal or None), and optionally ``description``.
    """

    def fetch(self, limit: int) -> list[dict]:
        ...


def _http_get(url: str, timeout: int = _HTTP_TIMEOUT) -> bytes:
    """GET a URL with a proper User-Agent and one retry on failure."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - retry once, then surface
            last_error = error
            if attempt == 0:
                time.sleep(1.0)
    raise ResearchError(f"GET {url} failed: {last_error}") from last_error


class RedditProvider:
    """Trending posts from Reddit's public ``/hot.json`` endpoints."""

    def __init__(
        self,
        subreddits: tuple[str, ...] = ("Documentaries", "science", "technology", "news"),
        timeout: int = _HTTP_TIMEOUT,
    ) -> None:
        self.subreddits = subreddits
        self.timeout = timeout

    def fetch(self, limit: int) -> list[dict]:
        topics: list[dict] = []
        per_sub = max(1, (limit + len(self.subreddits) - 1) // len(self.subreddits))
        for subreddit in self.subreddits:
            query = urllib.parse.urlencode({"limit": per_sub})
            url = f"{_REDDIT_BASE}/r/{subreddit}/hot.json?{query}"
            payload = json.loads(_http_get(url, timeout=self.timeout).decode("utf-8"))
            for child in payload.get("data", {}).get("children", []):
                data = child.get("data", {})
                title = (data.get("title") or "").strip()
                if not title or data.get("stickied"):
                    continue
                permalink = data.get("permalink") or ""
                created = data.get("created_utc")
                topics.append(
                    {
                        "title": title,
                        "description": f"Trending on r/{subreddit}",
                        "source": "reddit",
                        "url": f"{_REDDIT_BASE}{permalink}" if permalink else _REDDIT_BASE,
                        "published": (
                            datetime.fromtimestamp(created, tz=timezone.utc)
                            if isinstance(created, (int, float))
                            else None
                        ),
                        "score": data.get("score") if isinstance(data.get("score"), int) else None,
                    }
                )
                if len(topics) >= limit:
                    return topics
        return topics


class GoogleNewsProvider:
    """Top stories from the Google News RSS feed."""

    def __init__(self, feed_url: str = _GNEWS_RSS, timeout: int = _HTTP_TIMEOUT) -> None:
        self.feed_url = feed_url
        self.timeout = timeout

    def fetch(self, limit: int) -> list[dict]:
        raw = _http_get(self.feed_url, timeout=self.timeout)
        root = ET.fromstring(raw)
        topics: list[dict] = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            link = (item.findtext("link") or "").strip()
            published = None
            pub_date = item.findtext("pubDate")
            if pub_date:
                try:
                    published = parsedate_to_datetime(pub_date)
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    published = None
            source_el = item.find("source")
            source_name = (
                (source_el.text or "").strip() if source_el is not None else "Google News"
            )
            topics.append(
                {
                    "title": title,
                    "description": f"Top story via {source_name}",
                    "source": "google_news",
                    "url": link,
                    "published": published,
                    "score": None,
                }
            )
            if len(topics) >= limit:
                break
        return topics


@dataclass
class TrendingSource:
    name: str
    url: str
    priority: int


@dataclass
class TrendingTopic:
    title: str
    description: str
    source: str
    url: str
    timestamp: datetime
    volume: int | None = None


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for dedup."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", title.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _recency_hours(published: datetime | None) -> float:
    """Hours since publication, capped at 48. Unknown counts as stale."""
    if published is None:
        return 48.0
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - published
    return max(0.0, min(48.0, age.total_seconds() / 3600.0))


def _rank_key(topic: dict) -> float:
    """Higher is better: fresh topics first, then engagement, then source."""
    recency_score = (48.0 - _recency_hours(topic.get("published"))) / 48.0
    engagement = topic.get("score")
    engagement_score = min(1.0, (engagement or 0) / 5000.0)
    weight = _SOURCE_WEIGHTS.get(topic.get("source", ""), 1.0)
    return (recency_score * 0.6 + engagement_score * 0.4) * weight


def merge_topics(provider_results: list[list[dict]], limit: int) -> list[dict]:
    """Dedupe provider results by normalized title and rank them."""
    seen: dict[str, dict] = {}
    for topics in provider_results:
        for topic in topics:
            title = (topic.get("title") or "").strip()
            if not title:
                continue
            key = _normalize_title(title)
            if not key or key in seen:
                continue
            seen[key] = {**topic, "title": title}
    ranked = sorted(seen.values(), key=_rank_key, reverse=True)
    return ranked[:limit]


def _select_providers(trend_source: str) -> list[TrendProvider]:
    if trend_source == "reddit":
        return [RedditProvider()]
    if trend_source == "gnews":
        return [GoogleNewsProvider()]
    if trend_source == "all":
        return [RedditProvider(), GoogleNewsProvider()]
    raise ValueError(f"unknown trend_source {trend_source!r}; expected 'reddit', 'gnews', or 'all'")


def _synthetic_topics(max_topics: int, min_engagement: int) -> list[TrendingTopic]:
    """Offline development fallback. Never used unless explicitly enabled."""
    simulated = [
        ("AI Code Generation", "LLMs surpassing human coding speed", 15000),
        ("Climate Tech", "Breakthrough battery technology", 12500),
        ("Quantum Computing", "New quantum advantage demonstrated", 9800),
        ("Space Tourism", "Private spaceflight milestones", 8500),
        ("Neuroscience", "Brain-computer interfaces advance", 7200),
    ]
    topics = []
    for title, desc, volume in simulated:
        if volume >= min_engagement:
            topics.append(
                TrendingTopic(
                    title=title,
                    description=desc,
                    source="synthetic-fallback",
                    url=f"https://example.com/trend/{title.replace(' ', '-')}",
                    timestamp=datetime.now(timezone.utc),
                    volume=volume,
                )
            )
    return topics[:max_topics]


def get_trending_topics(
    max_topics: int = 5,
    min_engagement: int = 1000,
    trend_source: str = "all",
) -> list[TrendingTopic]:
    """Get trending topics from real providers, ranked by recency + engagement.

    Raises ``ResearchError`` when every provider fails. Topics without an
    engagement signal (e.g. news items) are kept regardless of
    ``min_engagement``; scored topics must meet the threshold.
    """
    providers = _select_providers(trend_source)
    results: list[list[dict]] = []
    errors: list[str] = []
    for provider in providers:
        try:
            results.append(provider.fetch(limit=max(10, max_topics * 3)))
        except Exception as error:  # noqa: BLE001 - record and try next provider
            errors.append(f"{type(provider).__name__}: {error}")

    if not any(results):
        if os.environ.get(_SYNTHETIC_ENV_VAR) == "1":
            return _synthetic_topics(max_topics, min_engagement)
        detail = "; ".join(errors) if errors else "no providers configured"
        raise ResearchError(f"all trend providers failed: {detail}")

    topics: list[TrendingTopic] = []
    for item in merge_topics(results, max_topics * 2):
        volume = item.get("score")
        if volume is not None and volume < min_engagement:
            continue
        published = item.get("published")
        topics.append(
            TrendingTopic(
                title=item["title"],
                description=item.get("description") or item["title"],
                source=item.get("source", "unknown"),
                url=item.get("url") or "",
                timestamp=published if isinstance(published, datetime) else datetime.now(timezone.utc),
                volume=volume,
            )
        )
    return topics[:max_topics]


def pick_trending_topic(
    max_topics: int = 5,
    min_engagement: int = 1000,
    trend_source: str = "all",
) -> TrendingTopic:
    """Return the single best trending topic, or raise ``ResearchError``."""
    topics = get_trending_topics(
        max_topics=max_topics, min_engagement=min_engagement, trend_source=trend_source
    )
    if not topics:
        raise ResearchError("trend providers returned no usable topics")
    return topics[0]


@dataclass
class ResearchResult:
    topics: list[TrendingTopic]
    timestamp: datetime
    methodology: str
    synthetic: bool = False


def research_trending_topics(
    max_topics: int = 5,
    min_engagement: int = 1000,
    trend_source: str = "all",
) -> ResearchResult:
    """Research and rank trending topics for video content."""
    topics = get_trending_topics(
        max_topics=max_topics, min_engagement=min_engagement, trend_source=trend_source
    )
    synthetic = os.environ.get(_SYNTHETIC_ENV_VAR) == "1" and all(
        t.source == "synthetic-fallback" for t in topics
    )
    methodology = (
        "synthetic-fallback (AVF_ALLOW_SYNTHETIC_RESEARCH=1); not real research"
        if synthetic
        else f"live providers via trend_source={trend_source} (reddit hot.json, google news rss)"
    )
    return ResearchResult(
        topics=topics,
        timestamp=datetime.now(timezone.utc),
        methodology=methodology,
        synthetic=synthetic,
    )


def save_research_result(result: ResearchResult, path: Path) -> None:
    """Save research result as JSON."""
    data = {
        "topics": [
            {
                "title": t.title,
                "description": t.description,
                "source": t.source,
                "url": t.url,
                "timestamp": t.timestamp.isoformat(),
                "volume": t.volume,
            }
            for t in result.topics
        ],
        "timestamp": result.timestamp.isoformat(),
        "methodology": result.methodology,
        "synthetic": result.synthetic,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
