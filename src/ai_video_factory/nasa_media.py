"""Stock media from NASA Image and Video Library (no API key required).

Most NASA imagery is public domain. Attribution records credit NASA with
the source page URL.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from ai_video_factory.stock_media import StockAsset, StockMediaError, download_asset

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


def fetch_nasa_for_scene(
    queries: list[str],
    scene_dir: Path,
    *,
    max_images: int = 2,
    max_clips: int = 1,
) -> list[StockAsset]:
    """Download NASA public-domain assets for one scene's queries."""
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

    Each normal content scene (``scene-0``, ``scene-1``, ...) is matched to a
    ``scene-NN/`` directory beneath ``assets_dir`` and populated by searching
    NASA for that scene's title. Intro/outro scenes are skipped. Returns a
    small summary of how many images/clips were downloaded per scene so the
    caller can report it in pipeline metadata.

    This is best-effort: network failures, empty queries, or missing media are
    logged and skipped so a single bad scene never aborts the whole render.
    """
    from ai_video_factory.edit_schema import EditScene  # local import to avoid cycles

    assets_dir = Path(assets_dir)
    summary: dict[str, int] = {"scenes": 0, "images": 0, "clips": 0}
    for scene in edit_doc.scenes:
        if not isinstance(scene, EditScene):
            continue
        if not str(getattr(scene, "id", "")).startswith("scene-"):
            continue
        try:
            idx = int(str(getattr(scene, "id", "")).split("-", 1)[1])
        except (ValueError, IndexError):
            continue
        title = str(getattr(scene, "title", "") or "").strip()
        if not title:
            continue
        scene_dir = assets_dir / f"scene-{idx:02d}"
        try:
            # Try progressively simpler queries until a variant returns media,
            # so poetic titles still surface something from the NASA catalog.
            query = _extract_query(title, stop_words)
            assets: list[StockAsset] = []
            for candidate in _query_variants(title, stop_words):
                assets = fetch_nasa_for_scene(
                    [candidate], scene_dir,
                    max_images=max_images, max_clips=max_clips,
                )
                if assets:
                    query = candidate
                    break
        except Exception as error:  # noqa: BLE001 - best-effort per scene
            print(f"[nasa] scene-{idx} failed: {error}")
            continue
        if assets:
            summary["scenes"] += 1
            summary["images"] += sum(1 for a in assets if a.kind == "image")
            summary["clips"] += sum(1 for a in assets if a.kind == "clip")
    return summary
