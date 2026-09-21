"""Stock media acquisition from Wikimedia Commons (no API key required).

Every downloaded file records full attribution (title, artist, license,
source URL) to an attribution log that the pipeline archives alongside the
video. Only freely licensed files (public domain, CC0, CC-BY, CC-BY-SA) are
accepted; NC/ND and fair-use files are rejected.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ai_video_factory.sanitization import sanitize_diagnostic


class StockMediaError(RuntimeError):
    """Raised when stock search or download fails."""


_COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_USER_AGENT = "AI-Video-Factory/0.1 (local documentary draft; contact: local)"
_HTTP_TIMEOUT = 60

# License URL fragments we accept.
_ACCEPTED_LICENSES = (
    "creativecommons.org/publicdomain/zero",
    "creativecommons.org/publicdomain/mark",
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
    "creativecommons.org/licenses/sa/",
)
# Reject non-commercial / no-derivatives variants.
_REJECTED_LICENSES = ("-nc", "-nd")


@dataclass
class StockAsset:
    kind: Literal["image", "clip"]
    path: str
    title: str
    artist: str
    license: str
    license_url: str
    source_url: str
    query: str
    provider: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _api(params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode({"format": "json", **params})
    request = urllib.request.Request(
        f"{_COMMONS_API}?{query}", headers={"User-Agent": _USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise StockMediaError(
            sanitize_diagnostic(f"Commons API request failed: {error}")
        ) from error


def _license_ok(license_url: str) -> bool:
    url = (license_url or "").lower()
    if not url or any(tag in url for tag in _REJECTED_LICENSES):
        return False
    return any(fragment in url for fragment in _ACCEPTED_LICENSES)


def _pick_video_variant(videoinfo: list[dict[str, Any]]) -> str | None:
    """Choose a directly downloadable MP4 derivative, smallest first."""
    derivatives = videoinfo[0].get("derivatives", []) if videoinfo else []
    mp4s = [
        d for d in derivatives
        if d.get("src", "").endswith(".mp4") and d.get("transcoded", True)
    ]
    mp4s.sort(key=lambda d: d.get("size", 0))
    if mp4s:
        return str(mp4s[0]["src"])
    # Fall back to the original file when it is already an MP4.
    original = (videoinfo[0].get("url", "") if videoinfo else "")
    return original if original.endswith(".mp4") else None


def search_commons(
    query: str, *, mediatype: Literal["image", "clip"], limit: int = 6
) -> list[dict[str, Any]]:
    """Search Commons and return licensed file records (no downloads)."""
    filetype = "bitmap|drawing" if mediatype == "image" else "video"
    data = _api({
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{query} filetype:{filetype}",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|user|extmetadata|size",
        "iiextmetadatafilter": "Artist|LicenseShortName|UsageTerms|LicenseUrl",
        "iiurlwidth": "1920" if mediatype == "image" else "1280",
    })
    pages = (data.get("query") or {}).get("pages") or {}
    records: list[dict[str, Any]] = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        license_url = ((meta.get("LicenseUrl") or {}).get("value") or "")
        if not _license_ok(license_url):
            continue
        if mediatype == "image":
            download_url = info.get("thumburl") or info.get("url")
        else:
            download_url = _pick_video_variant([info])
        if not download_url:
            continue
        artist = ((meta.get("Artist") or {}).get("value") or "unknown")
        records.append({
            "title": page.get("title", ""),
            "artist": artist,
            "license": ((meta.get("LicenseShortName") or {}).get("value") or ""),
            "license_url": license_url,
            "source_url": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(str(page.get('title', '')))}",
            "download_url": download_url,
        })
    return records


def download_asset(url: str, destination: Path) -> Path:
    """Download one file; raise on failure. No redirects off https."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            if response.status != 200:
                raise StockMediaError(f"download HTTP {response.status}: {url}")
            destination.write_bytes(response.read())
    except StockMediaError:
        raise
    except Exception as error:
        raise StockMediaError(
            sanitize_diagnostic(f"download failed for {url}: {error}")
        ) from error
    if destination.stat().st_size == 0:
        raise StockMediaError(f"downloaded empty file: {url}")
    return destination


def fetch_for_scene(
    queries: list[str],
    scene_dir: Path,
    *,
    max_images: int = 2,
    max_clips: int = 1,
    max_bytes_per_file: int = 300 * 1024 * 1024,
) -> list[StockAsset]:
    """Download a small set of licensed assets for one scene's queries."""
    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    assets: list[StockAsset] = []
    images = clips = 0
    for query in queries:
        for mediatype, taken, cap in (("image", images, max_images), ("clip", clips, max_clips)):
            if taken >= cap:
                continue
            try:
                records = search_commons(query, mediatype=mediatype)  # type: ignore[arg-type]
            except StockMediaError:
                continue
            for record in records:
                if (mediatype == "image" and images >= max_images) or (
                    mediatype == "clip" and clips >= max_clips
                ):
                    break
                suffix = ".mp4" if mediatype == "clip" else ".jpg"
                destination = scene_dir / f"{mediatype}-{len(assets)}{suffix}"
                try:
                    download_asset(record["download_url"], destination)
                except StockMediaError:
                    continue
                if destination.stat().st_size > max_bytes_per_file:
                    destination.unlink(missing_ok=True)
                    continue
                assets.append(StockAsset(
                    kind=mediatype,  # type: ignore[arg-type]
                    path=str(destination),
                    title=str(record["title"]),
                    artist=str(record["artist"]),
                    license=str(record["license"]),
                    license_url=str(record["license_url"]),
                    source_url=str(record["source_url"]),
                    query=query,
                ))
                if mediatype == "image":
                    images += 1
                else:
                    clips += 1
    return assets


def write_attribution_log(assets: list[StockAsset], path: Path) -> Path:
    """Write the CC attribution log archived with the video."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "generated_at": datetime.now(UTC).isoformat(),
        "assets": [asset.to_dict() for asset in assets],
    }
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return path
