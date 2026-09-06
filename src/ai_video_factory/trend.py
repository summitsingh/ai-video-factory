"""Real trending-topic discovery for AI Video Factory.

This module replaces the previous simulated ``research.py`` trend output with
genuine provider adapters (Hacker News, Reddit public JSON, Google News RSS,
Wikipedia pageviews/popular). Every candidate carries recorded signals derived
only from fetched data; an LLM may rank or recommend but never asserts a topic
is trending without attached signals.

Network access is opt-in and dry-run capable. Adapters accept an injectable
transport so tests feed canned responses without touching the network.
"""

from __future__ import annotations

import hashlib
import html
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from ai_video_factory.sanitization import sanitize_diagnostic

# A transport fetches a URL and returns decoded text. Injectable for tests so
# no real network calls happen in the test path.
Transport = Callable[[str], str]

_USER_AGENT = "AI-Video-Factory/0.1 (local documentary draft; contact: local)"
_HTTP_TIMEOUT = 30


class TrendError(RuntimeError):
    """Raised when trend discovery fails."""


# ========== Normalized candidate schema ==========

@dataclass
class SourceSignal:
    provider: str
    url: str
    published_at: str | None
    raw_signal: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrendCandidate:
    topic_id: str
    title: str
    summary: str
    first_seen_at: str
    observed_at: str
    cluster_key: str
    sources: list[SourceSignal] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ========== Transport helpers ==========

def _default_transport(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            if getattr(response, "status", 200) != 200:
                raise TrendError(f"HTTP {getattr(response, 'status', '?')} for {url}")
            return response.read().decode("utf-8")
    except Exception as error:  # noqa: BLE001 - surfaced to caller as TrendError
        raise TrendError(sanitize_diagnostic(f"fetch failed for {url}: {error}")) from error


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stable_hash(*parts: str) -> str:
    canonical = "\x00".join(p for p in parts if p)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


# ========== Provider adapters ==========

def fetch_hacker_news(transport: Transport | None = None) -> list[dict[str, Any]]:
    """Fetch top Hacker News stories via the Algolia API (no credentials)."""
    transport = transport or _default_transport
    raw = transport("https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=30")
    data = json.loads(raw)
    items = []
    for hit in data.get("hits", []):
        object_id = str(hit.get("object_id") or "")
        title = (hit.get("title") or "").strip()
        if not object_id or not title:
            continue
        url = f"https://news.ycombinator.com/item?id={object_id}"
        points = hit.get("points") or 0
        num_comments = hit.get("num_comments") or 0
        timestamp = hit.get("created_at")
        items.append({
            "provider": "hacker_news",
            "title": title,
            "url": url,
            "published_at": timestamp,
            "raw_signal": {"points": points, "comments": num_comments},
        })
    return items


def fetch_reddit(transport: Transport | None = None) -> list[dict[str, Any]]:
    """Fetch public Reddit JSON from permitted endpoints (clear user agent)."""
    transport = transport or _default_transport
    url = "https://www.reddit.com/r/technology/new.json?limit=25"
    raw = transport(url)
    data = json.loads(raw)
    items = []
    children = ((data.get("data") or {}).get("children") or [])
    for child in children:
        post = (child.get("data") or {})
        title = (post.get("title") or "").strip()
        if not title:
            continue
        url = f"https://www.reddit.com{post.get('permalink', '')}"
        score = post.get("score") or 0
        num_comments = post.get("num_comments") or 0
        created = post.get("created_utc")
        published_at = datetime.fromtimestamp(
            float(created), tz=UTC
        ).isoformat() if created else None
        items.append({
            "provider": "reddit",
            "title": title,
            "url": url,
            "published_at": published_at,
            "raw_signal": {"score": score, "comments": num_comments},
        })
    return items


def fetch_google_news(transport: Transport | None = None) -> list[dict[str, Any]]:
    """Fetch Google News RSS for a curated query (no credentials)."""
    transport = transport or _default_transport
    query = urllib.parse.urlencode({"q": "technology", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    raw = transport(f"https://news.google.com/rss/{query}")
    root = ET.fromstring(raw)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pubdate = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source_name = (source_el.text if source_el is not None else "") or "Google News"
        if not title or not link:
            continue
        items.append({
            "provider": "google_news",
            "title": html.unescape(title),
            "url": link,
            "published_at": _parse_rss_date(pubdate),
            "raw_signal": {"source": source_name},
        })
    return items


def fetch_wikipedia_popular(transport: Transport | None = None) -> list[dict[str, Any]]:
    """Fetch Wikipedia's most-viewed articles for a date (no credentials)."""
    transport = transport or _default_transport
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    url = f"https://wikimedia.org/api/rest_v1/data/population/pageviews/annual/en-all/all-user/{today}"
    # The REST pageviews API returns daily totals; use the popular endpoint.
    alt_url = "https://wikimedia.org/api/rest_v1/data/population/pageviews/daily/en-wikipedia/all-users/20260905"
    raw = ""
    for candidate in (url, alt_url):
        try:
            raw = transport(candidate)
            break
        except TrendError:
            continue
    if not raw:
        return []
    data = json.loads(raw)
    items = []
    rows = data.get("items", []) if isinstance(data, dict) else data
    for row in rows[:20]:
        title = (row.get("article") or "").strip()
        views = row.get("views") or 0
        if not title:
            continue
        items.append({
            "provider": "wikipedia",
            "title": f"Wikipedia: {title}",
            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}",
            "published_at": today,
            "raw_signal": {"views": int(views)},
        })
    return items


def _parse_rss_date(value: str) -> str | None:
    """Parse an RFC-822 RSS date into ISO-8601 UTC."""
    if not value:
        return None
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(value)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()
    except (TypeError, ValueError):
        return None


# ========== Normalization + dedup ==========

def normalize_items(items: Sequence[dict[str, Any]], *, observed_at: str | None = None) -> list[TrendCandidate]:
    """Normalize raw provider items into candidates with stable topic ids.

    Deduplicates by cluster key (a slug of the canonical title). When multiple
    sources report the same story they are merged under one candidate so a single
    topic can carry signals from several providers.
    """
    observed_at = observed_at or _now_iso()
    clusters: dict[str, TrendCandidate] = {}
    for item in items:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        cluster_key = _slug(title)
        topic_id = _stable_hash(cluster_key, title.lower())
        existing = clusters.get(topic_id)
        source = SourceSignal(
            provider=item.get("provider", "unknown"),
            url=item.get("url", ""),
            published_at=item.get("published_at"),
            raw_signal=item.get("raw_signal") or {},
        )
        if existing is None:
            clusters[topic_id] = TrendCandidate(
                topic_id=topic_id,
                title=title,
                summary=(item.get("summary") or f"{title} (from {source.provider})").strip(),
                first_seen_at=item.get("published_at") or observed_at,
                observed_at=observed_at,
                cluster_key=cluster_key,
                sources=[source],
            )
        else:
            existing.sources.append(source)
    return list(clusters.values())


def _slug(title: str) -> str:
    tokens = [t.strip(".,;:!?\"'()") for t in title.lower().split()]
    tokens = [t for t in tokens if t and not t.isdigit()]
    return "-".join(tokens[:8]) or "topic"


def dedupe(clusters: list[TrendCandidate]) -> list[TrendCandidate]:
    """Collapse clusters that share a topic_id, merging their sources."""
    by_id: dict[str, TrendCandidate] = {}
    for candidate in clusters:
        existing = by_id.get(candidate.topic_id)
        if existing is None:
            by_id[candidate.topic_id] = candidate
            continue
        seen_urls = {s.url for s in existing.sources}
        for source in candidate.sources:
            if source.url not in seen_urls:
                existing.sources.append(source)
                seen_urls.add(source.url)
    return list(by_id.values())


# ========== Deterministic scoring ==========

def compute_signals(candidate: TrendCandidate, *, now_iso: str | None = None) -> dict[str, float]:
    """Compute normalized 0..1 signals from recorded data only.

    freshness: recency of the newest source relative to now (days).
    velocity: magnitude of engagement signals (points/score/comments/views).
    source_diversity: number of distinct providers reporting this topic.
    competition_risk: how many other candidates share a cluster slug (crowdedness).
    visual_potential: heuristic from title keywords.
    audience_fit: heuristic from title length + keyword breadth.
    """
    now_iso = now_iso or _now_iso()
    try:
        now_dt = datetime.fromisoformat(now_iso)
    except ValueError:
        now_dt = datetime.now(UTC)

    # freshness (1.0 for very recent, ->0 after ~7 days).
    newest = max((s.published_at for s in candidate.sources if s.published_at), default=None)
    if newest is not None:
        try:
            age_days = (now_dt - datetime.fromisoformat(newest)).total_seconds() / 86400.0
        except ValueError:
            age_days = 7.0
    else:
        age_days = 7.0
    freshness = max(0.0, min(1.0, 1.0 - (age_days / 7.0)))

    # velocity from raw signals.
    engagement = 0.0
    for source in candidate.sources:
        rs = source.raw_signal
        engagement += float(rs.get("points") or rs.get("score") or rs.get("comments") or rs.get("views") or 0)
    velocity = min(1.0, engagement / 5000.0)

    # source diversity: distinct providers.
    providers = {s.provider for s in candidate.sources}
    source_diversity = min(1.0, len(providers) / 3.0)

    return {
        "freshness": round(freshness, 4),
        "velocity": round(velocity, 4),
        "source_diversity": round(source_diversity, 4),
        "competition_risk": 0.0,  # filled by rank_candidates when crowding is known
        "visual_potential": _visual_potential(candidate.title),
        "audience_fit": _audience_fit(candidate.title),
    }


_VISUAL_KEYWORDS = (
    "launch", "mission", "eclipse", "video", "image", "photo", "discovery",
    "breakthrough", "explosion", "rocket", "screen", "app", "phone", "robot",
    "animation", "storm", "earth", "space", "ocean", "animal", "art", "music",
)


def _visual_potential(title: str) -> float:
    low = title.lower()
    hits = sum(1 for kw in _VISUAL_KEYWORDS if kw in low)
    return min(1.0, 0.4 + 0.2 * hits)


def _audience_fit(title: str) -> float:
    words = len(title.split())
    # Sweet spot ~6-14 words; penalize very short/very long titles.
    if 6 <= words <= 14:
        return 0.9
    if words < 3 or words > 20:
        return 0.4
    return 0.7


def rank_candidates(
    candidates: list[TrendCandidate],
    *,
    weights: dict[str, float] | None = None,
    now_iso: str | None = None,
) -> list[TrendCandidate]:
    """Rank candidates deterministically from their signals.

    Weights default to a balanced mix. competition_risk is filled here using the
    full shortlist so crowded cluster slugs score lower. Returns candidates sorted
    by descending score (highest first).
    """
    weights = weights or {
        "freshness": 0.15,
        "velocity": 0.30,
        "source_diversity": 0.15,
        "competition_risk": 0.10,
        "visual_potential": 0.15,
        "audience_fit": 0.15,
    }
    now_iso = now_iso or _now_iso()

    # Competition risk: crowded cluster slugs across the shortlist.
    slug_counts: dict[str, int] = {}
    for candidate in candidates:
        slug_counts[candidate.cluster_key] = slug_counts.get(candidate.cluster_key, 0) + 1

    ranked: list[TrendCandidate] = []
    for candidate in candidates:
        signals = compute_signals(candidate, now_iso=now_iso)
        competition = min(1.0, (slug_counts.get(candidate.cluster_key, 1) - 1) / 5.0)
        signals["competition_risk"] = round(competition, 4)
        score = sum(weights[key] * float(signals.get(key, 0.0)) for key in weights)
        confidence = min(1.0, len(candidate.sources) / 2.0)
        candidate.signals = signals
        candidate.score = round(score, 4)
        candidate.confidence = round(confidence, 4)
        ranked.append(candidate)

    ranked.sort(key=lambda c: (c.score, c.confidence), reverse=True)
    return ranked


# ========== Public entry points ==========

def discover_trends(
    *,
    providers: Sequence[str] | None = None,
    max_candidates: int = 10,
    transport: Transport | None = None,
    now_iso: str | None = None,
) -> list[TrendCandidate]:
    """Discover and rank real trending topics.

    ``providers`` selects which adapters to run (default: all credential-free ones).
    Network failures in one provider are skipped so a single outage does not abort
    discovery. Returns the ranked shortlist (at most ``max_candidates``).
    """
    providers = providers or ["hacker_news", "reddit", "google_news", "wikipedia"]
    adapters = {
        "hacker_news": fetch_hacker_news,
        "reddit": fetch_reddit,
        "google_news": fetch_google_news,
        "wikipedia": fetch_wikipedia_popular,
    }
    now_iso = now_iso or _now_iso()
    all_items: list[dict[str, Any]] = []
    for name in providers:
        adapter = adapters.get(name)
        if adapter is None:
            continue
        try:
            all_items.extend(adapter(transport=transport))
        except Exception as error:  # noqa: BLE001 - resilience: skip any provider failure
            print(f"[trend] provider {name} failed: {error}")
            continue

    clusters = dedupe(normalize_items(all_items, observed_at=now_iso))
    ranked = rank_candidates(clusters, now_iso=now_iso)
    return ranked[:max_candidates]


def discover_trends_dry_run(
    *,
    providers: Sequence[str] | None = None,
    max_candidates: int = 10,
    transport: Transport | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Dry-run discovery.

    Fetches and normalizes real data but does not persist artifacts unless
    ``output_dir`` is given (dry run callers should omit it). Returns a summary
    dict describing what would be produced.
    """
    now_iso = _now_iso()
    candidates = discover_trends(
        providers=providers, max_candidates=max_candidates, transport=transport, now_iso=now_iso
    )
    snapshot = {
        "generated_at": now_iso,
        "mode": "dry-run",
        "candidate_count": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
    }
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "trend_snapshot.json").write_text(
            json.dumps(snapshot, indent=2), encoding="utf-8"
        )
    return snapshot


def save_candidates(candidates: Sequence[TrendCandidate], path: Path) -> Path:
    """Persist normalized candidates as JSON for downstream stages."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _now_iso(),
        "mode": "discovered",
        "candidates": [c.to_dict() for c in candidates],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
