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
