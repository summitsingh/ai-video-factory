"""Unified stock-footage providers: Pexels and Pixabay.

Free video sources so the pipeline can put real camera footage behind every
scene instead of relying only on procedural visuals. No paid APIs, no Google
Flow / Veo credits are spent here.

Providers
---------
* :class:`PexelsClient` - Pexels video search API. Needs a free API key from
  https://www.pexels.com/api/ in the ``PEXELS_API_KEY`` env var.
  Pexels License: free for commercial use, no attribution required.
* :class:`PixabayClient` - Pixabay video search API. Needs a free API key
  from https://pixabay.com/api/docs/ (free signup) in the
  ``PIXABAY_API_KEY`` env var. Pixabay Content License: free for commercial
  use, no attribution required.

:class:`StockFootageProvider` is the single entry point. ``fetch_clip`` takes
a scene description, derives 2-3 keyword queries from it, tries providers in
order (Pexels -> Pixabay), downloads the best landscape-HD match, and returns a
:class:`StockAsset` from :mod:`ai_video_factory.stock_media` so attribution
flows into the pipeline's existing attribution log. Downloads are cached under
``data/cache/stock/`` keyed by URL hash, so repeat runs never re-download.

Returns ``None`` when every provider misses, so the caller falls back to AI
stills / Ken Burns / procedural visuals unchanged.

INTEGRATION
-----------
The pipeline's visual stage (see ``video_pipeline.py``, step 4.4) currently:

1. ``_populate_stock_clips(...)`` fills per-scene dirs from Pexels/Pixabay stock,
2. ``attach_scene_assets(...)`` binds clips/images onto ``EditScene.clip``
   / ``.image``,
3. ``asset_qc`` verifies each asset; rejections go to the ``AssetMemory``
   blocklist,
4. ``ai_visuals.generate_scene_visual(...)`` paints a cinematic still for
   scenes left with nothing usable.

To plug this module in, add a step between (1) and (2) that, for each scene, runs::

    from ai_video_factory.stock_providers import StockFootageProvider

    provider = StockFootageProvider()  # reads PEXELS_API_KEY / PIXABAY_API_KEY
    fps = edit.fps if edit.fps else 24
    asset = provider.fetch_clip(
        getattr(scene, "visual_direction", None) or scene.visual or scene.title,
        min_duration_sec=max(1.0, scene.duration_frames / fps),
    )
    if asset is not None:
        # Copy asset.path into the scene's assets dir, set scene.clip to the
        # public-relative name, and append asset.to_dict() to the attribution
        # log via stock_media.write_attribution_log.

The fallback chain stays: stock footage -> AI stills / Ken Burns ->
procedural ``ai_visuals``. ``fetch_clip`` returns ``None`` on a total miss,
so existing fallbacks trigger unchanged. API keys are optional: with neither
``PEXELS_API_KEY`` nor ``PIXABAY_API_KEY`` set, every provider is skipped
and ``fetch_clip`` returns ``None``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ai_video_factory.sanitization import sanitize_diagnostic
from ai_video_factory.stock_media import StockAsset, StockMediaError, download_asset

log = logging.getLogger(__name__)


class StockFootageError(RuntimeError):
    """Raised when a stock-footage provider request fails."""


_USER_AGENT = "AI-Video-Factory/0.1 (local documentary draft; contact: local)"
_HTTP_TIMEOUT = 60

# Cache root: <repo>/data/cache/stock (overridable via STOCK_CACHE_DIR).
_CACHE_DIR = Path(
    os.environ.get(
        "STOCK_CACHE_DIR",
        Path(__file__).resolve().parents[2] / "data" / "cache" / "stock",
    )
)

# Landscape HD widths we accept, best first.
_PREFERRED_WIDTHS = (1920, 1280)

# Minimum file size we trust as a real download (bytes).
_MIN_DOWNLOAD_BYTES = 50 * 1024


def _repo_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_path(provider: str, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return _repo_cache_dir() / f"{provider}-{digest}.mp4"


def _http_json(url: str, headers: dict[str, str] | None = None) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, **(headers or {})}
    )
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            if response.status != 200:
                raise StockFootageError(f"provider HTTP {response.status}")
            return json.loads(response.read().decode("utf-8"))
    except StockFootageError:
        raise
    except Exception as error:
        raise StockFootageError(
            sanitize_diagnostic(f"provider request failed: {error}")
        ) from error


@dataclass
class ClipCandidate:
    """One downloadable clip offered by a provider."""

    provider: str
    download_url: str
    title: str
    artist: str
    license: str
    license_url: str
    source_url: str
    width: int
    height: int
    duration_sec: float | None  # None when the provider does not report it.


# ---------------------------------------------------------------------------
# Keyword derivation: scene description -> 2-3 concrete visual queries.
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "with", "into",
    "from", "by", "at", "is", "are", "was", "were", "how", "that", "this",
    "our", "their", "its", "your", "as", "to", "be", "see", "slow", "slowly",
    "camera", "shot", "view", "scene", "showing", "shows", "while", "over",
    "under", "through", "across", "between", "very", "much", "more",
})

# Multi-word anchors first; matched with word boundaries against the
# description so "lab" never fires on "elaborate".
_VISUAL_ANCHORS = (
    "black hole", "space station", "earth from space", "solar system",
    "ocean waves", "city night", "aerial view", "time lapse",
    "nebula", "galaxy", "galaxies", "planet", "planets", "star", "stars",
    "earth", "moon", "mars", "jupiter", "saturn", "sun", "comet", "asteroid",
    "telescope", "astronaut", "spacewalk", "rocket", "launch", "satellite",
    "shuttle", "capsule", "lander", "rover", "aurora", "eclipse", "crater",
    "volcano", "ocean", "waves", "desert", "glacier", "storm", "clouds",
    "forest", "mountain", "mountains", "river", "waterfall", "city",
    "street", "crowd", "highway", "sunset", "sunrise", "night", "rain",
    "snow", "fire", "laboratory", "scientist", "dinosaur", "fossil",
)

def _derive_queries(description: str, limit: int = 3) -> list[str]:
    """Turn a scene description into concrete visual keyword queries.

    Prefers known visual anchors ("nebula", "ocean waves"); falls back to the
    description's most meaningful tokens. Always returns at least one query.
    """
    text = (description or "").lower()
    anchors: list[str] = []
    for anchor in _VISUAL_ANCHORS:
        if re.search(rf"(?<![a-z]){re.escape(anchor)}(?![a-z])", text):
            anchors.append(anchor)
            if len(anchors) >= 3:
                break
    queries: list[str] = []
    if anchors:
        queries.append(" ".join(anchors[:2]))
        queries.extend(a for a in anchors if a not in queries)
    tokens = [
        tok.strip(".,;:!?\"'()")
        for tok in re.split(r"\s+", text)
        if tok.strip(".,;:!?\"'()") and tok.strip(".,;:!?\"'()") not in _STOP_WORDS
    ]
    if tokens:
        fallback = " ".join(tokens[:3])
        if fallback not in queries:
            queries.append(fallback)
    if not queries:
        queries.append((description or "nature").strip()[:80] or "nature")
    return queries[:limit]


# ---------------------------------------------------------------------------
# Pexels
# ---------------------------------------------------------------------------

_PEXELS_SEARCH = "https://api.pexels.com/videos/search"
_PEXELS_LICENSE_URL = "https://www.pexels.com/license/"


class PexelsClient:
    """Pexels video search. Free API key via PEXELS_API_KEY."""

    name = "pexels"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("PEXELS_API_KEY", "")
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _pick_file(self, video: dict) -> dict | None:
        files = video.get("video_files") or []
        mp4s = [
            f for f in files
            if f.get("file_type") == "video/mp4"
            and f.get("width") in _PREFERRED_WIDTHS
            and (f.get("width") or 0) > (f.get("height") or 0)
        ]
        mp4s.sort(key=lambda f: -f["width"])
        return mp4s[0] if mp4s else None

    def search_clips(self, query: str, per_page: int = 8) -> list[ClipCandidate]:
        if not self.available:
            return []
        params = urllib.parse.urlencode({
            "query": query,
            "per_page": str(min(per_page, 80)),
            "orientation": "landscape",
        })
        data = _http_json(
            f"{_PEXELS_SEARCH}?{params}",
            headers={"Authorization": self.api_key},
        )
        candidates: list[ClipCandidate] = []
        for video in data.get("videos") or []:
            picked = self._pick_file(video)
            if not picked or not picked.get("link"):
                continue
            user = video.get("user") or {}
            candidates.append(ClipCandidate(
                provider=self.name,
                download_url=str(picked["link"]),
                title=str(video.get("url") or f"Pexels video {video.get('id')}"),
                artist=str(user.get("name") or "Pexels contributor"),
                license="Pexels License",
                license_url=_PEXELS_LICENSE_URL,
                source_url=str(video.get("url") or ""),
                width=int(picked["width"]),
                height=int(picked["height"]),
                duration_sec=float(video["duration"]) if video.get("duration") else None,
            ))
        return candidates


# ---------------------------------------------------------------------------
# Pixabay
# ---------------------------------------------------------------------------

_PIXABAY_SEARCH = "https://pixabay.com/api/videos/"
_PIXABAY_LICENSE_URL = "https://pixabay.com/service/license/"
# Size renditions, largest first.
_PIXABAY_SIZES = ("large", "medium")


class PixabayClient:
    """Pixabay video search. Free API key via PIXABAY_API_KEY."""

    name = "pixabay"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("PIXABAY_API_KEY", "")
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _pick_file(self, hit: dict) -> dict | None:
        videos = hit.get("videos") or {}
        options = []
        for size in _PIXABAY_SIZES:
            entry = videos.get(size) or {}
            width, height = entry.get("width") or 0, entry.get("height") or 0
            if entry.get("url") and width in _PREFERRED_WIDTHS and width > height:
                options.append(entry)
        options.sort(key=lambda e: -e["width"])
        return options[0] if options else None

    def search_clips(self, query: str, per_page: int = 8) -> list[ClipCandidate]:
        if not self.available:
            return []
        params = urllib.parse.urlencode({
            "key": self.api_key,
            "q": query,
            "per_page": str(min(per_page, 50)),
        })
        data = _http_json(f"{_PIXABAY_SEARCH}?{params}")
        candidates: list[ClipCandidate] = []
        for hit in data.get("hits") or []:
            picked = self._pick_file(hit)
            if not picked or not picked.get("url"):
                continue
            candidates.append(ClipCandidate(
                provider=self.name,
                download_url=str(picked["url"]),
                title=str(hit.get("pageURL") or f"Pixabay video {hit.get('id')}"),
                artist=str(hit.get("user") or "Pixabay contributor"),
                license="Pixabay Content License",
                license_url=_PIXABAY_LICENSE_URL,
                source_url=str(hit.get("pageURL") or ""),
                width=int(picked["width"]),
                height=int(picked["height"]),
                duration_sec=float(hit["duration"]) if hit.get("duration") else None,
            ))
        return candidates


# ---------------------------------------------------------------------------
# Unified provider
# ---------------------------------------------------------------------------

class StockFootageProvider:
    """Fetch one real stock clip per scene, trying free providers in order.

    Order is Pexels -> Pixabay. Providers without API keys are skipped
    silently. Returns ``None`` when every provider misses so the caller
    falls back to AI stills / Ken Burns / procedural visuals.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        pexels_key: str | None = None,
        pixabay_key: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.pexels = PexelsClient(api_key=pexels_key)
        self.pixabay = PixabayClient(api_key=pixabay_key)

    def _ordered_clients(self, description: str) -> list:
        return [c for c in (self.pexels, self.pixabay) if c.available]

    def _probe_ok(self, path: Path, min_duration_sec: float) -> bool:
        """Best-effort ffprobe check: valid video with enough duration."""
        try:
            from ai_video_factory.media_probe import probe_media

            info = probe_media(path)
        except Exception as error:  # noqa: BLE001 - probe optional
            log.debug("probe unavailable for %s: %s", path.name, sanitize_diagnostic(error))
            return True  # Pipeline QC verifies later; don't block on probe.
        duration = info.duration_seconds or 0
        if duration <= 0:
            return False
        if duration < min_duration_sec:
            log.debug("clip too short: %.1fs < %.1fs", duration, min_duration_sec)
            return False
        if info.width and info.width < 640:
            return False
        return True

    def _download_candidate(
        self, candidate: ClipCandidate, min_duration_sec: float, query: str
    ) -> StockAsset | None:
        if (
            candidate.duration_sec is not None
            and candidate.duration_sec < min_duration_sec
        ):
            return None
        destination = _cache_path(candidate.provider, candidate.download_url)
        if not (destination.is_file() and destination.stat().st_size > _MIN_DOWNLOAD_BYTES):
            try:
                download_asset(candidate.download_url, destination)
            except (StockMediaError, StockFootageError, OSError) as error:
                log.debug("download failed: %s", sanitize_diagnostic(error))
                return None
        if destination.stat().st_size <= _MIN_DOWNLOAD_BYTES:
            destination.unlink(missing_ok=True)
            return None
        if not self._probe_ok(destination, min_duration_sec):
            destination.unlink(missing_ok=True)
            return None
        return StockAsset(
            kind="clip",
            path=str(destination),
            title=candidate.title,
            artist=candidate.artist,
            license=candidate.license,
            license_url=candidate.license_url,
            source_url=candidate.source_url,
            query=query,
        )

    def fetch_clip(
        self, scene_description: str, min_duration_sec: float = 5.0
    ) -> StockAsset | None:
        """Download the best stock clip for a scene description.

        Returns a :class:`StockAsset` whose ``path`` is the cached local MP4,
        or ``None`` when no provider has a usable match.
        """
        queries = _derive_queries(scene_description)
        log.info("stock fetch queries for scene: %s", queries)
        for client in self._ordered_clients(scene_description):
            for query in queries:
                try:
                    candidates = client.search_clips(query)
                except StockFootageError as error:
                    log.debug("%s search failed: %s", client.name, sanitize_diagnostic(error))
                    continue
                for candidate in candidates:
                    asset = self._download_candidate(candidate, min_duration_sec, query)
                    if asset is not None:
                        log.info(
                            "stock clip from %s for query %r -> %s",
                            client.name, query, asset.path,
                        )
                        return asset
        log.info("no stock clip found for scene")
        return None
