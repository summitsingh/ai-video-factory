"""Stock media from NASA Image and Video Library (no API key required).

Most NASA imagery is public domain. Attribution records credit NASA with
the source page URL.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_video_factory.stock_media import StockAsset, StockMediaError, download_asset

log = logging.getLogger(__name__)

_NASA_SEARCH = "https://images-api.nasa.gov/search"
_NASA_ASSETS = "https://images-assets.nasa.gov"
_USER_AGENT = "AI-Video-Factory/0.1 (local documentary draft; contact: local)"
_HTTP_TIMEOUT = 60

# Words that carry no search value - stripped from titles before querying NASA.
_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "with", "into",
    "from", "by", "at", "is", "are", "was", "were", "how", "that", "this",
    "our", "their", "its", "your", "as", "to", "be", "see", "everything",
    "rewrote", "cosmic", "history", "james", "webb", "telescope", "telescopes",
})

# Common possessive/contraction suffixes to strip before keyword matching.
_SUFFIXES = ("'s", "'re", "'ve", "'ll", "'d")

# Concrete visual nouns the NASA library is likely to have. Matched against
# a scene's visual direction (not its poetic title) so the query describes
# what should be ON SCREEN. Multi-word anchors come first in matching.
_VISUAL_ANCHORS = (
    "black hole", "space station", "mission control", "launch pad",
    "solar eclipse", "lunar eclipse",
    "nebula", "galaxy", "galaxies", "star", "stars", "planet", "planets",
    "earth", "moon", "mars", "jupiter", "saturn", "sun", "comet", "asteroid",
    "telescope", "observatory", "astronaut", "spacewalk", "spacesuit",
    "rocket", "launch", "satellite", "shuttle", "capsule", "lander", "rover",
    "laboratory", "lab", "scientist", "engineer",
    "aurora", "eclipse", "crater", "volcano", "ocean", "desert", "glacier",
    "wildfire", "hurricane", "storm", "clouds",
    "dinosaur", "fossil",
)

# NASA record titles that signal event/promo imagery (press conferences,
# briefings, award ceremonies). These read as off-topic B-roll for science
# storytelling, so they are skipped unless nothing else matches.
_EVENT_BLOCKLIST = (
    "press conference", "press briefing", "news briefing", "briefing",
    "panel discussion", "town hall", "award", "ceremony", "gala",
    "ribbon cutting", "signing ceremony",
    # Talking-head / studio formats that read as off-topic B-roll.
    "interview", "news desk", "news anchor", "anchor desk",
    "in the studio", "studio interview", "talk show",
    "media day", "q&a", "q & a", "roundtable",
)

# Minimum usable still-image width in pixels. Anything smaller looks soft
# when stretched fullscreen; the scene falls back to the title-card look.
_MIN_IMAGE_WIDTH = 640


def _extract_query(title: str, stop_words: frozenset[str] | None = None) -> str:
    """Turn a verbose scene title into a NASA-searchable query.

    Strips stop words and possessive suffixes, then joins the remaining
    meaningful tokens. Falls back to the raw title if nothing survives so we
    never send an empty query. ``stop_words`` defaults to the module-level
    list; callers pass a theme's list for per-topic theming.
    """
    stop = _STOP_WORDS if stop_words is None else frozenset(stop_words)
    tokens = []
    for token in title.replace(",", " ").split():
        # Drop trailing apostrophe contractions: "Nebula's" -> "Nebula".
        for suffix in _SUFFIXES:
            if token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        token = token.strip(".,;:!?'\"")
        if not token:
            continue
        low = token.lower()
        if low in stop:
            continue
        tokens.append(token)
    return " ".join(tokens) or title.strip()


def _extract_visual_query(
    visual: str | None,
    title: str,
    stop_words: frozenset[str] | None = None,
) -> str:
    """Build a NASA query from the scene's visual direction.

    The visual direction describes what should be on screen ("a planet
    engulfed in fire", "camera pulls back from Earth"), so concrete nouns
    pulled from it match the NASA catalog far better than the scene's poetic
    title ("Wild Explanations"). Falls back to the title-based extraction
    when no known visual anchor appears.
    """
    text = (visual or "").lower()
    hits: list[str] = []
    for anchor in _VISUAL_ANCHORS:
        # Word-boundary match so "lab" doesn't fire on "elaborate".
        if re.search(rf"(?<![a-z]){re.escape(anchor)}(?![a-z])", text):
            # Normalize plurals to the singular anchor for cleaner queries.
            hits.append(anchor)
            if len(hits) >= 3:
                break
    if hits:
        # De-dupe singular/plural pairs ("star"/"stars" -> "stars").
        seen: set[str] = set()
        unique: list[str] = []
        for hit in hits:
            key = hit.rstrip("s")
            if key not in seen:
                seen.add(key)
                unique.append(hit)
        return " ".join(unique[:3])
    return _extract_query(title, stop_words)


def _record_is_event_photo(record_title: str) -> bool:
    low = (record_title or "").lower()
    # Word-boundary match: "gala" must not fire on "galaxy".
    return any(
        re.search(r"\b" + re.escape(term) + r"\b", low)
        for term in _EVENT_BLOCKLIST
    )


def _query_variants(title: str, stop_words: frozenset[str] | None = None) -> list[str]:
    """Return candidate NASA queries for a scene, best first.

    The primary query is the extracted keyword phrase. If that returns no
    media we progressively simplify - dropping leading adjectives down to the
    last two meaningful tokens - so poetic titles like "Horsehead's Hidden
    Structure" still surface something (e.g. "Horsehead"). Single generic words
    are never tried on their own: they pull unrelated hits ("Core" -> rocket
    stages). If nothing multi-word matches, the scene keeps its title-card look
    rather than showing off-topic media.
    """
    stop = _STOP_WORDS if stop_words is None else frozenset(stop_words)
    primary = _extract_query(title, stop)
    variants: list[str] = [primary]
    words = [w for w in primary.split() if w.lower() not in stop]
    # Build a ladder of trailing multi-word candidates, longest first. For a
    # 4-word title this yields the full phrase, then the last 3, 2 tokens -
    # always at least two words so we never fall back to a lone generic noun.
    for tail in range(len(words), 1, -1):
        candidate = " ".join(words[-tail:])
        if candidate not in variants:
            variants.append(candidate)
    return variants


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise StockMediaError(f"NASA API request failed: {error}") from error


def search_nasa(
    query: str, *, mediatype: str, limit: int = 10
) -> list[dict]:
    """Search NASA library; returns records with download candidates."""
    params = urllib.parse.urlencode({
        "q": query, "media_type": mediatype, "page_size": str(limit),
    })
    data = _get_json(f"{_NASA_SEARCH}?{params}")
    items = ((data.get("collection") or {}).get("items") or [])
    records = []
    for item in items:
        nasa_id = ((item.get("data") or [{}])[0].get("nasa_id") or "")
        title = ((item.get("data") or [{}])[0].get("title") or "")
        if not nasa_id:
            continue
        records.append({
            "nasa_id": nasa_id,
            "title": title,
            "source_url": f"https://images.nasa.gov/details/{nasa_id}",
        })
    return records


def _nasa_image_url(nasa_id: str) -> str:
    return f"{_NASA_ASSETS}/image/{nasa_id}/{nasa_id}~medium.jpg"


def _nasa_clip_url(nasa_id: str) -> str | None:
    try:
        files = _get_json(f"{_NASA_ASSETS}/video/{nasa_id}/collection.json")
    except StockMediaError:
        return None
    mp4s = [f for f in files if isinstance(f, str) and f.endswith(".mp4")]
    # Prefer mid-size renditions over huge originals.
    mp4s.sort(key=lambda f: ("orig" in f, "~" in f and "small" not in f))
    return mp4s[0] if mp4s else None


def _image_is_usable(path: Path) -> bool:
    """Reject tiny or unreadable stills; they look soft stretched fullscreen."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, _ = image.size
        return width >= _MIN_IMAGE_WIDTH
    except Exception:
        # If PIL can't read it, don't trust it as B-roll.
        return False


def fetch_nasa_for_scene(
    queries: list[str],
    scene_dir: Path,
    *,
    max_images: int = 2,
    max_clips: int = 1,
    relevance_text: str = "",
    min_relevance: float = 0.08,
) -> list[StockAsset]:
    """Download NASA public-domain assets for one scene's queries.

    When ``relevance_text`` (usually the scene's visual direction) is given,
    each candidate record is scored with
    :func:`ai_video_factory.asset_memory.score_relevance` and records below
    ``min_relevance`` are skipped, so a space documentary does not end up
    with news-desk B-roll.
    """
    from ai_video_factory.asset_memory import score_relevance

    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    assets: list[StockAsset] = []
    images = clips = 0
    for query in queries:
        for mediatype, taken, cap in (("image", images, max_images), ("video", clips, max_clips)):
            if taken >= cap:
                continue
            try:
                records = search_nasa(query, mediatype=mediatype)
            except StockMediaError:
                continue
            for record in records:
                if (mediatype == "image" and images >= max_images) or (
                    mediatype == "video" and clips >= max_clips
                ):
                    break
                # Skip press-conference / ceremony / talking-head imagery: it
                # reads as off-topic B-roll for science storytelling.
                if _record_is_event_photo(record["title"]):
                    continue
                if relevance_text:
                    relevance = score_relevance(
                        str(record["title"]), relevance_text
                    )
                    if relevance < min_relevance:
                        continue
                try:
                    if mediatype == "image":
                        url = _nasa_image_url(record["nasa_id"])
                        destination = scene_dir / f"nasa-image-{len(assets)}.jpg"
                    else:
                        url = _nasa_clip_url(record["nasa_id"])
                        if url is None:
                            continue
                        destination = scene_dir / f"nasa-clip-{len(assets)}.mp4"
                    download_asset(url, destination)
                except StockMediaError:
                    continue
                if mediatype == "image" and not _image_is_usable(destination):
                    destination.unlink(missing_ok=True)
                    continue
                assets.append(StockAsset(
                    kind="image" if mediatype == "image" else "clip",
                    path=str(destination),
                    title=str(record["title"]),
                    artist="NASA",
                    license="Public Domain",
                    license_url="https://www.nasa.gov/nasa-brand-center/images-and-media/",
                    source_url=str(record["source_url"]),
                    query=query,
                ))
                if mediatype == "image":
                    images += 1
                else:
                    clips += 1
    return assets


def populate_assets_from_nasa(
    edit_doc,
    assets_dir: Path,
    *,
    max_images: int = 2,
    max_clips: int = 1,
    stop_words: frozenset[str] | None = None,
) -> dict[str, int]:
    """Download NASA public-domain assets into per-scene folders.

    Each normal content scene is matched to a ``scene-NN/`` directory beneath
    ``assets_dir`` (positional slot from ``scene_asset_slots``) and populated by searching
    NASA for that scene's visual direction first (concrete on-screen nouns),
    falling back to the scene title. Intro/outro scenes are skipped. Records
    that look like press conferences or ceremonies are skipped, and stills
    narrower than 640px are rejected. Returns a small summary of how many
    images/clips were downloaded per scene so the caller can report it in
    pipeline metadata.

    This is best-effort: network failures, empty queries, or missing media are
    logged and skipped so a single bad scene never aborts the whole render.
    """
    from ai_video_factory.edit_schema import EditScene, scene_asset_slots  # local import to avoid cycles

    assets_dir = Path(assets_dir)
    summary: dict[str, int] = {"scenes": 0, "images": 0, "clips": 0}
    lock = threading.Lock()
    slots = scene_asset_slots(edit_doc.scenes)

    def _populate_scene(scene) -> tuple[int, int, int]:
        if not isinstance(scene, EditScene):
            return (0, 0, 0)
        idx = slots.get(str(getattr(scene, "id", "")))
        if idx is None:
            # Intro/outro or non-content scene: no per-scene assets.
            return (0, 0, 0)
        title = str(getattr(scene, "title", "") or "").strip()
        if not title:
            return (0, 0, 0)
        visual = str(getattr(scene, "visual", "") or "").strip()
        scene_dir = assets_dir / f"scene-{idx:02d}"
        try:
            # Query from the visual direction first: it describes what should
            # be on screen ("a planet engulfed in fire"), while the title is
            # often poetic ("Wild Explanations") and pulls junk. Fall back to
            # the title ladder when the visual yields no usable media.
            query = _extract_visual_query(visual, title, stop_words)
            assets: list[StockAsset] = []
            candidates = [query]
            candidates.extend(
                c for c in _query_variants(title, stop_words) if c != query
            )
            for candidate in candidates:
                assets = fetch_nasa_for_scene(
                    [candidate], scene_dir,
                    max_images=max_images, max_clips=max_clips,
                    relevance_text=visual or title,
                )
                if assets:
                    query = candidate
                    break
        except Exception as error:  # noqa: BLE001 - best-effort per scene
            log.warning(
                "nasa populate failed for scene %s: %s",
                getattr(scene, "id", "?"), error,
            )
            return (0, 0, 0)
        if assets:
            return (
                1,
                sum(1 for a in assets if a.kind == "image"),
                sum(1 for a in assets if a.kind == "clip"),
            )
        return (0, 0, 0)

    # Scenes download in parallel (network-bound; each scene writes to its own
    # scene-NN directory so there is no contention).
    scenes = list(edit_doc.scenes)
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(scenes)))) as pool:
        for s_count, i_count, c_count in pool.map(_populate_scene, scenes):
            with lock:
                summary["scenes"] += s_count
                summary["images"] += i_count
                summary["clips"] += c_count
    if summary["scenes"] == 0 and slots:
        log.warning(
            "populate_assets_from_nasa: 0 of %d normal scenes got any NASA "
            "assets; visuals will fall back to later stages",
            len(slots),
        )
    return summary
