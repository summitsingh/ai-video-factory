"""Network music providers for the AI Video Factory music bed.

Pluggable royalty-free sources, used by ``music_bed.resolve_music_track`` when
the local mood library (``assets/music/<mood>-bed.mp3``) and the
``MUSIC_BED_PATH`` override both miss. Provider order:

1. Internet Archive advancedsearch API (no key required)
2. Openverse audio API (keyless, ``license_type=commercial``)
3. Freesound API (only when ``FREESOUND_API_KEY`` is set; skipped silently
   otherwise)

License policy: only tracks whose license allows commercial YouTube use are
accepted - CC0 / Public Domain / CC-BY. Anything NonCommercial (NC) or
NoDerivatives (ND) is rejected. Every accepted track still needs the artist
credit in the video description (CC-BY); see ``assets/music/ATTRIBUTION.md``.

All HTTP goes through :func:`_get_json` / :func:`_download`, built on
``urllib.request`` with a shared timeout, so tests can mock the network
without touching provider logic.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass

HTTP_TIMEOUT = 30
USER_AGENT = "ai-video-factory/1.0 (music-bed fetcher; contact: repo owner)"

# Minimum track length so a bed can cover a longform video without sounding
# like a jingle on loop.
MIN_TRACK_SECONDS = 180

# Freesound: https://freesound.org/apiv2/apply/ - create an API key, then
# ``export FREESOUND_API_KEY=<key>``. Without it the provider is skipped.
FREESOUND_API_KEY_ENV = "FREESOUND_API_KEY"


@dataclass
class MusicTrack:
    provider: str  # "internet_archive" | "openverse" | "freesound"
    identifier: str
    title: str
    artist: str
    license: str  # normalized: "cc0" | "pdm" | "by"
    download_url: str
    duration_sec: float | None
    source_url: str  # landing page, for attribution


# ---------------------------------------------------------------------------
# HTTP helpers (mock seam for tests)
# ---------------------------------------------------------------------------

def _get_json(url: str, params: dict | None = None) -> dict | list:
    query = urllib.parse.urlencode(params or {}, doseq=True)
    full = f"{url}?{query}" if query else url
    request = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return json.load(response)


def _download(url: str, dest) -> None:
    """Download *url* to *dest* (a Path). Raises on failure."""
    from pathlib import Path

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response, \
                open(tmp, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    if tmp.stat().st_size < 50_000:
        tmp.unlink()
        raise RuntimeError(f"download suspiciously small: {url}")
    tmp.replace(dest)


# ---------------------------------------------------------------------------
# License policy
# ---------------------------------------------------------------------------

def normalize_commercial_license(url: str | None) -> str | None:
    """Map a license URL to 'cc0' | 'pdm' | 'by' if commercial use is allowed.

    Returns None for missing licenses and for any NonCommercial (NC) or
    NoDerivatives (ND) variant.
    """
    u = (url or "").strip().lower()
    if not u:
        return None
    for bad in ("-nc", "/nc/", "noncommercial", "-nd", "/nd/", "noderivatives",
                "sampling"):
        if bad in u:
            return None
    if "publicdomain/zero" in u or "creativecommons.org/publicdomain/zero" in u \
            or u.endswith("/cc0"):
        return "cc0"
    if "publicdomain/mark" in u:
        return "pdm"
    if "creativecommons.org/licenses/by/" in u:
        return "by"
    return None


# ---------------------------------------------------------------------------
# Provider 1: Internet Archive
# ---------------------------------------------------------------------------

IA_SEARCH_URL = "https://archive.org/advancedsearch.php"
IA_METADATA_URL = "https://archive.org/metadata/{}"
IA_DOWNLOAD_URL = "https://archive.org/download/{}/{}"

# Per-mood title term groups for the IA advancedsearch query.
IA_MOOD_TERMS: dict[str, tuple[str, ...]] = {
    "cosmic": ("space ambient", "dark ambient", "cosmic drone", "deep space"),
    "mystery": ("suspense ambient", "mystery underscore", "dark tension",
                "investigative ambient"),
    "epic": ("epic orchestral", "cinematic orchestral", "heroic orchestral"),
    "calm": ("calm ambient", "peaceful meditation", "nature ambient",
             "serene piano"),
    "tech": ("futuristic ambient", "sci-fi electronic", "synth ambient",
             "cyber ambient"),
}


def _ia_search_identifiers(mood: str, rows: int = 25) -> list[dict]:
    terms = IA_MOOD_TERMS.get(mood, IA_MOOD_TERMS["cosmic"])
    title_clause = " OR ".join(f"title:({t})" for t in terms)
    query = f"mediatype:audio AND ({title_clause})"
    data = _get_json(IA_SEARCH_URL, {
        "q": query,
        "fl[]": ["identifier", "title", "creator", "licenseurl"],
        "rows": rows,
        "sort[]": "downloads desc",
        "output": "json",
    })
    return data.get("response", {}).get("docs", [])


def _ia_pick_file(identifier: str, min_duration_sec: float) -> dict | None:
    """Pick the best MP3 file for an IA item. Returns file dict or None."""
    meta = _get_json(IA_METADATA_URL.format(
        urllib.parse.quote(identifier, safe="")))
    files = meta.get("files", []) if isinstance(meta, dict) else []
    candidates = []
    for entry in files:
        name = entry.get("name", "")
        if not name.lower().endswith(".mp3"):
            continue
        if name.endswith("_meta.sqlite") or "/_/" in name:
            continue
        try:
            bitrate = int(str(entry.get("bitrate", "0")).split()[0])
        except (ValueError, IndexError):
            bitrate = 0
        try:
            length = float(entry.get("length", 0) or 0)
        except (ValueError, TypeError):
            length = 0.0
        candidates.append((bitrate, length, name))
    long_enough = [c for c in candidates if c[1] >= min_duration_sec]
    pool = long_enough or [c for c in candidates if c[1] > 0]
    if not pool:
        return None
    pool.sort(key=lambda c: (c[0], c[1]), reverse=True)
    bitrate, length, name = pool[0]
    return {"name": name, "bitrate": bitrate, "length": length}


# Title markers that almost always mean spoken word, not a music bed.
_IA_SPOKEN_MARKERS = (
    "dramatic reading", "librivox", "audiobook", "audio book", "podcast",
    "interview", "sermon", "lecture", "audiobook",
)


def _ia_is_music(title: str) -> bool:
    lowered = (title or "").lower()
    return not any(marker in lowered for marker in _IA_SPOKEN_MARKERS)


def search_internet_archive(mood: str,
                            min_duration_sec: float = MIN_TRACK_SECONDS,
                            rows: int = 25) -> list[MusicTrack]:
    """Search the Internet Archive for a commercial-use bed for *mood*."""
    tracks: list[MusicTrack] = []
    for doc in _ia_search_identifiers(mood, rows=rows):
        license_kind = normalize_commercial_license(doc.get("licenseurl"))
        if license_kind is None:
            continue
        title = str(doc.get("title", "") or "")
        if not _ia_is_music(title):
            continue
        identifier = doc.get("identifier", "")
        if not identifier:
            continue
        try:
            picked = _ia_pick_file(identifier, min_duration_sec)
        except Exception as exc:
            print(f"[music_providers] IA metadata failed for {identifier}: "
                  f"{exc}")
            continue
        if picked is None:
            continue
        tracks.append(MusicTrack(
            provider="internet_archive",
            identifier=identifier,
            title=title or identifier,
            artist=str(doc.get("creator", "unknown") or "unknown"),
            license=license_kind,
            download_url=IA_DOWNLOAD_URL.format(
                urllib.parse.quote(identifier, safe=""),
                urllib.parse.quote(picked["name"], safe="/")),
            duration_sec=picked["length"] or None,
            source_url=f"https://archive.org/details/{identifier}",
        ))
    return tracks


# ---------------------------------------------------------------------------
# Provider 2: Openverse
# ---------------------------------------------------------------------------

OV_SEARCH_URL = "https://api.openverse.org/v1/audio/"

# Openverse license slugs we accept for monetized YouTube beds.
OV_OK_LICENSES = frozenset({"by", "cc0", "pdm"})

OV_MOOD_QUERIES: dict[str, str] = {
    "cosmic": "dark space ambient drone",
    "mystery": "suspense thriller ambient",
    "epic": "epic cinematic orchestral",
    "calm": "calm peaceful nature ambient",
    "tech": "futuristic sci-fi electronic ambient",
}


def search_openverse(mood: str,
                     min_duration_sec: float = MIN_TRACK_SECONDS,
                     page_size: int = 20) -> list[MusicTrack]:
    """Search Openverse (Freesound/Jamendo/Wikimedia aggregate), keyless."""
    data = _get_json(OV_SEARCH_URL, {
        "q": OV_MOOD_QUERIES.get(mood, OV_MOOD_QUERIES["cosmic"]),
        "license_type": "commercial",
        "page_size": page_size,
    })
    tracks: list[MusicTrack] = []
    for item in data.get("results", []):
        if str(item.get("license", "")).lower() not in OV_OK_LICENSES:
            continue
        duration_ms = item.get("duration") or 0
        try:
            duration_sec = float(duration_ms) / 1000.0
        except (TypeError, ValueError):
            duration_sec = 0.0
        if duration_sec < min_duration_sec:
            continue
        url = item.get("url", "")
        if not url:
            continue
        tracks.append(MusicTrack(
            provider="openverse",
            identifier=str(item.get("id", url)),
            title=str(item.get("title", "untitled")),
            artist=str((item.get("creator") or "unknown")),
            license=str(item.get("license", "")).lower(),
            download_url=url,
            duration_sec=duration_sec,
            source_url=str(item.get("foreign_landing_url", url)),
        ))
    return tracks


# ---------------------------------------------------------------------------
# Provider 3: Freesound (keyed; skipped silently without FREESOUND_API_KEY)
# ---------------------------------------------------------------------------

FS_SEARCH_URL = "https://freesound.org/apiv2/search/text/"

FS_MOOD_QUERIES: dict[str, str] = {
    "cosmic": "space drone ambient",
    "mystery": "suspense tension ambient",
    "epic": "epic orchestral cinematic",
    "calm": "calm nature ambience",
    "tech": "sci-fi ambience electronic",
}


def search_freesound(mood: str,
                     min_duration_sec: float = MIN_TRACK_SECONDS) -> list[MusicTrack]:
    """Search Freesound. Returns [] when no API key is configured.

    Get a free key at https://freesound.org/apiv2/apply/ and export
    FREESOUND_API_KEY. Downloads use the high-quality MP3 preview.
    """
    api_key = os.environ.get(FREESOUND_API_KEY_ENV, "").strip()
    if not api_key:
        return []
    data = _get_json(FS_SEARCH_URL, {
        "query": FS_MOOD_QUERIES.get(mood, FS_MOOD_QUERIES["cosmic"]),
        "filter": f'license:"Creative Commons 0" duration:[{int(min_duration_sec)} TO *]',
        "fields": "id,name,username,license,previews,duration,url",
        "page_size": 20,
        "token": api_key,
    })
    tracks: list[MusicTrack] = []
    for item in data.get("results", []):
        previews = item.get("previews") or {}
        url = previews.get("preview-hq-mp3", "")
        if not url:
            continue
        try:
            duration_sec = float(item.get("duration") or 0)
        except (TypeError, ValueError):
            duration_sec = 0.0
        if duration_sec < min_duration_sec:
            continue
        tracks.append(MusicTrack(
            provider="freesound",
            identifier=str(item.get("id", "")),
            title=str(item.get("name", "untitled")),
            artist=str(item.get("username", "unknown")),
            license="cc0",
            download_url=url,
            duration_sec=duration_sec,
            source_url=str(item.get("url", url)),
        ))
    return tracks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# Provider names in fallback order. The search functions are resolved via
# getattr at call time (not bound here) so tests can patch them.
PROVIDER_NAMES: tuple[str, ...] = (
    "internet_archive",
    "openverse",
    "freesound",
)


def _safe_id(identifier: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", identifier)[:60].strip("_") or "track"


def cache_path_for(mood: str, track: MusicTrack, cache_dir) -> "object":
    from pathlib import Path

    filename = f"{mood}_{track.provider}_{_safe_id(track.identifier)}.mp3"
    return Path(cache_dir) / filename


def fetch_from_providers(mood: str, *, cache_dir,
                         min_duration_sec: float = MIN_TRACK_SECONDS,
                         ) -> tuple | tuple[None, None, None]:
    """Walk the network providers for *mood*; cache hits win.

    Returns ``(path, provider_name, track)`` on success, or
    ``(None, None, None)`` when every provider fails. A provider that raises
    is logged and skipped so the next one is tried.
    """
    from pathlib import Path

    cache_dir = Path(cache_dir)
    for provider_name in PROVIDER_NAMES:
        # Resolved dynamically so tests can patch the search functions.
        search_fn = globals()[f"search_{provider_name}"]
        try:
            tracks = search_fn(mood, min_duration_sec=min_duration_sec)
        except Exception as exc:
            print(f"[music_providers] {provider_name} search failed: {exc}")
            continue
        if not tracks:
            print(f"[music_providers] {provider_name}: no usable tracks "
                  f"for mood '{mood}'")
            continue
        for track in tracks:
            dest = cache_path_for(mood, track, cache_dir)
            if dest.is_file():
                print(f"[music_providers] cache hit: {dest.name}")
                return dest, provider_name, track
            try:
                print(f"[music_providers] downloading {provider_name}: "
                      f"{track.title} ({track.duration_sec:.0f}s)")
                _download(track.download_url, dest)
            except Exception as exc:
                print(f"[music_providers] download failed ({track.title}): "
                      f"{exc}")
                continue
            print(f"[music_providers] cached {dest.name}")
            return dest, provider_name, track
    return None, None, None
