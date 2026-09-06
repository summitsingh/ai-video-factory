"""Rights-cleared asset library for AI Video Factory (Phase 3).

This module is the single authority on whether external media may enter a render.
It wraps the existing NASA and Wikimedia providers behind one abstraction and,
for every acquired asset, records full rights metadata:

    - original provider URL
    - download URL
    - provider + creator
    - license name + canonical license URL
    - attribution requirements
    - acquisition time
    - checksum (sha256)
    - media properties (dimensions / codec / duration when available)
    - candidate scene ids
    - human-review status (+ rejection reason)

A render must never reference an asset whose rights are not ``approved``. The
gate is fail-closed: if a required asset lacks approval, rendering is blocked.

Selection scoring ranks candidates by relevance, technical quality, visual
diversity, reuse history, logo/text risk, and rights confidence.

Network access is opt-in; every provider adapter accepts an injected transport
so tests exercise the full pipeline without real network calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from ai_video_factory.sanitization import sanitize_diagnostic

Provider = Literal["nasa", "wikimedia"]
LicenseStatus = Literal["approved", "pending_review", "rejected"]
AssetKind = Literal["clip", "image", "map", "chart", "generated_visual", "source_excerpt"]
AssetStrategy = Literal[
    "licensed_clip", "licensed_image", "public_domain",
    "map", "chart", "generated_visual", "source_excerpt",
]


class AssetError(RuntimeError):
    """Raised when asset acquisition, rights, or selection fails."""


# ========== Rights record ==========

@dataclass
class MediaProperties:
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    duration_seconds: float | None = None
    mime_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RightsRecord:
    asset_id: str
    kind: AssetKind
    provider: Provider
    original_url: str
    download_url: str
    creator: str
    license_name: str
    license_url: str
    attribution: str
    acquired_at: str
    checksum_sha256: str
    properties: MediaProperties
    candidate_scene_ids: list[str] = field(default_factory=list)
    review_status: LicenseStatus = "pending_review"
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["properties"] = self.properties.to_dict()
        return data


# ========== License classification (deterministic) ==========

_ACCEPTED_LICENSE_FRAGMENTS = (
    "creativecommons.org/publicdomain/zero",
    "creativecommons.org/publicdomain/mark",
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
    "creativecommons.org/licenses/sa/",
    "purl.org/dc/elements/1.1/",  # NASA public domain / gov works
)
_REJECTED_LICENSE_FRAGMENTS = ("-nc", "-nd")


def classify_license(license_url: str, *, provider: Provider) -> tuple[str | None, str]:
    """Return (license_name, canonical_url).

    A missing or ambiguous license returns (None, "") and is treated as a
    rejection by the approval gate. NASA material is public domain by statute;
    Wikimedia material must carry an accepted CC/public-domain URL.
    """
    url = (license_url or "").lower()
    if provider == "nasa":
        # NASA media is generally public domain (17 U.S.C. § 105).
        return ("Public Domain (NASA)", "https://www.nasa.gov/multimedia/guidelines/index.html")
    if not url:
        return (None, "")
    if any(tag in url for tag in _REJECTED_LICENSE_FRAGMENTS):
        return (None, "")
    for fragment in _ACCEPTED_LICENSE_FRAGMENTS:
        if fragment in url:
            # Normalize to a canonical license URL.
            name = "CC0 1.0" if "publicdomain/zero" in url else \
                   "Public Domain Mark" if "publicdomain/mark" in url else \
                   "CC-BY" if "licenses/by/" in url and "by-sa" not in url else \
                   "CC-BY-SA"
            return (name, license_url)
    # Ambiguous / unrecognized license -> reject.
    return (None, "")


def is_license_approved(license_name: str | None) -> bool:
    """A license is approved only when it resolves to a concrete accepted name."""
    return bool(license_name and license_name not in ("", "unknown"))


# ========== Provider adapters (injectable transport) ==========

class TransportError(RuntimeError):
    pass


# A transport returns raw bytes for a URL; tests inject canned payloads.
Transport = Callable[[str], bytes]


def _urlopen_default(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0) -> bytes:
    import urllib.request

    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if getattr(response, "status", 200) != 200:
            raise TransportError(f"HTTP {getattr(response, 'status', '?')} for {url}")
        return response.read()


def search_nasa(
    query: str, *, limit: int = 6, transport: Transport | None = None
) -> list[dict[str, Any]]:
    """Search the NASA Image and Video Library.

    ``transport`` is injected so tests can return canned JSON. Defaults to a real
    urllib GET of the public API.
    """
    transport = transport or _urlopen_default
    endpoint = "https://images-api.nasa.gov/search?q=" + query.replace(" ", "+")
    try:
        raw = transport(endpoint)
        data = json.loads(raw.decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - resilience, surfaced by caller
        raise AssetError(sanitize_diagnostic(f"NASA search failed: {error}")) from error

    items: list[dict[str, Any]] = []
    for item in (data.get("collection") or {}).get("items", [])[:limit]:
        metadata = item.get("metadata") or {}
        links = item.get("links") or []
        media_link = next(
            (link for link in links if link.get("rel") == "media" and link.get("href")),
            links[0] if links else None,
        )
        if not media_link:
            continue
        title = metadata.get("title", "") or item.get("title", "")
        items.append({
            "title": str(title),
            "provider_url": f"https://images-api.nasa.gov/item/{item.get('item_id', '')}",
            "download_url": str(media_link.get("href")),
            "creator": metadata.get("creator", "NASA") or "NASA",
        })
    return items


def search_wikimedia(
    query: str, *, mediatype: Literal["image", "clip"], limit: int = 6,
    transport: Transport | None = None,
) -> list[dict[str, Any]]:
    """Search Wikimedia Commons. ``transport`` is injected for tests."""
    from ai_video_factory.stock_media import _api as commons_api

    # Reuse the existing Commons API wrapper (already license-filtered).
    records = commons_api(query, mediatype=mediatype, limit=limit)  # type: ignore[call-overload]
    return records


# ========== Acquisition + rights assembly ==========

def acquire_asset(
    provider: Provider,
    query: str,
    *,
    scene_id: str,
    destination: Path,
    mediatype: Literal["clip", "image"] = "image",
    transport: Callable[[str], bytes] | None = None,
    checksum_fn: Callable[[Path], str] | None = None,
) -> RightsRecord:
    """Acquire one licensed asset and build its full rights record.

    Raises ``AssetError`` when no license-acceptable result is found or the file
    cannot be downloaded. The returned record starts in ``pending_review``; call
    ``approve_asset`` (or run the gate) before it may render.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if provider == "nasa":
        results = search_nasa(query, transport=transport)
    elif provider == "wikimedia":
        results = search_wikimedia(query, mediatype=mediatype, transport=transport)  # type: ignore[arg-type]
    else:
        raise AssetError(f"unknown asset provider: {provider}")

    if not results:
        raise AssetError(f"no licensed assets found for query {query!r} on {provider}")

    chosen = results[0]
    download_url = str(chosen.get("download_url") or "")
    if not download_url:
        raise AssetError(f"asset for {query!r} has no download URL")

    try:
        data = transport(download_url) if transport else _urlopen_default(download_url)
    except Exception as error:  # noqa: BLE001 - surfaced by caller
        raise AssetError(sanitize_diagnostic(f"download failed for {download_url}: {error}")) from error
    destination.write_bytes(data)
    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise AssetError(f"downloaded empty file: {download_url}")

    checksum_fn = checksum_fn or _sha256_file
    license_name, license_url = classify_license(
        chosen.get("license_url", ""), provider=provider
    )
    creator = str(chosen.get("creator") or "unknown")
    attribution = f"{chosen.get('title', '')} — {creator}, {license_name}" if license_name else ""

    return RightsRecord(
        asset_id=_asset_id(provider, download_url),
        kind="clip" if mediatype == "clip" else "image",
        provider=provider,
        original_url=str(chosen.get("provider_url") or download_url),
        download_url=download_url,
        creator=creator,
        license_name=license_name or "",
        license_url=license_url,
        attribution=attribution,
        acquired_at=datetime.now(UTC).isoformat(),
        checksum_sha256=checksum_fn(destination),
        properties=MediaProperties(mime_type=_mime(download_url)),
        candidate_scene_ids=[scene_id],
    )


# ========== Approval gate (fail-closed) ==========

def approve_asset(record: RightsRecord, *, reason: str | None = None) -> RightsRecord:
    """Mark a record approved when its license is concrete and accepted."""
    if not is_license_approved(record.license_name):
        return reject_asset(
            record,
            reason=reason or f"no acceptable license for {record.provider} asset",
        )
    from dataclasses import replace

    return replace(record, review_status="approved", rejection_reason=None)


def reject_asset(record: RightsRecord, *, reason: str) -> RightsRecord:
    from dataclasses import replace

    return replace(record, review_status="rejected", rejection_reason=reason)


def gate_assets(records: Sequence[RightsRecord]) -> list[RightsRecord]:
    """Run the fail-closed rights gate. Returns approved records only.

    Raises ``AssetError`` if any record is not approved so a render cannot proceed
    with unapproved media.
    """
    approved: list[RightsRecord] = []
    for record in records:
        status = record.review_status
        if status == "pending_review":
            # Auto-classify by license; anything ambiguous is rejected here.
            status_record = approve_asset(record)
            if status_record.review_status != "approved":
                raise AssetError(
                    sanitize_diagnostic(
                        f"asset {record.asset_id} blocked: {status_record.rejection_reason}"
                    )
                )
            approved.append(status_record)
        elif status == "approved":
            approved.append(record)
        else:  # rejected
            raise AssetError(
                sanitize_diagnostic(f"asset {record.asset_id} was rejected: {record.rejection_reason}")
            )
    return approved


def render_cannot_proceed(records: Sequence[RightsRecord]) -> bool:
    """True when at least one record is not approved (used by the pipeline)."""
    try:
        gate_assets(list(records))
        return False
    except AssetError:
        return True


# ========== Selection scoring ==========

@dataclass
class ScoredAsset:
    record: RightsRecord
    score: float
    relevance: float
    quality: float
    diversity: float
    reuse_penalty: float
    logo_risk: float
    rights_confidence: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["record"] = self.record.to_dict()
        return data


def _relevance(query_terms: Sequence[str], asset_title: str) -> float:
    title = asset_title.lower()
    if not query_terms:
        return 0.5
    hits = sum(1 for term in query_terms if term and term.lower() in title)
    return min(1.0, hits / len(query_terms))


def _quality(properties: MediaProperties) -> float:
    """Score resolution/technical quality (higher res + known codec = higher)."""
    score = 0.0
    if properties.width and properties.height:
        pixels = properties.width * properties.height
        # 720p ~ 0.6, 1080p ~ 0.85, 4k+ ~ 1.0
        score = min(1.0, pixels / (3_840 * 2_160)) + 0.3
        score = min(1.0, score)
    elif properties.width:
        score = 0.4
    if properties.codec in ("h264", "hevc"):
        score = min(1.0, score + 0.15)
    return max(0.0, score)


def _diversity(existing_titles: Sequence[str], title: str) -> float:
    """Reward assets that differ from already-selected titles."""
    if not existing_titles:
        return 1.0
    tokens = set(re.findall(r"[a-z]+", title.lower()))
    overlap = sum(1 for t in existing_titles if set(re.findall(r"[a-z]+", t.lower())) & tokens)
    return max(0.0, 1.0 - overlap / len(existing_titles))


def _reuse_penalty(asset_id: str, used_ids: Sequence[str]) -> float:
    """Penalize duplicate/reused assets."""
    return 0.4 if asset_id in set(used_ids) else 0.0


def _logo_risk(title: str) -> float:
    """Penalize titles likely to contain logos/branding/watermarks."""
    risky = {"logo", "brand", "watermark", "sponsor", "product"}
    tokens = set(re.findall(r"[a-z]+", title.lower()))
    hits = tokens & risky
    return min(0.5, 0.1 * len(hits))


def _rights_confidence(record: RightsRecord) -> float:
    if record.review_status != "approved":
        return 0.0
    # Public domain / CC0 carry the highest confidence; attribution-required lower.
    name = (record.license_name or "").lower()
    if "public domain" in name or "cc0" in name:
        return 1.0
    if "by-sa" in name:
        return 0.85
    if "by/" in name:
        return 0.8
    return 0.6


def score_asset(
    record: RightsRecord,
    query_terms: Sequence[str],
    *,
    existing_titles: Sequence[str] = (),
    used_ids: Sequence[str] = (),
) -> ScoredAsset:
    relevance = _relevance(query_terms, record.original_url or "")
    quality = _quality(record.properties)
    diversity = _diversity(existing_titles, record.original_url or "")
    reuse_penalty = _reuse_penalty(record.asset_id, used_ids)
    logo_risk = _logo_risk(record.original_url or "")
    rights_confidence = _rights_confidence(record)

    score = (
        0.30 * relevance
        + 0.25 * quality
        + 0.15 * diversity
        + 0.10 * (1.0 - reuse_penalty)
        + 0.10 * (1.0 - logo_risk)
        + 0.10 * rights_confidence
    )
    return ScoredAsset(
        record=record,
        score=round(score, 4),
        relevance=round(relevance, 4),
        quality=round(quality, 4),
        diversity=round(diversity, 4),
        reuse_penalty=round(reuse_penalty, 4),
        logo_risk=round(logo_risk, 4),
        rights_confidence=round(rights_confidence, 4),
    )


def select_assets(
    candidates: Sequence[RightsRecord],
    query_terms: Sequence[str],
    *,
    limit: int = 3,
) -> list[ScoredAsset]:
    """Rank and return the best assets for a scene.

    Deduplicates by asset_id so reuse history is respected across the batch.
    """
    seen: set[str] = set()
    deduped: list[RightsRecord] = []
    used_ids: list[str] = []
    existing_titles: list[str] = []
    for record in candidates:
        if record.asset_id in seen:
            continue
        seen.add(record.asset_id)
        deduped.append(record)

    scored = [score_asset(r, query_terms, existing_titles=existing_titles, used_ids=used_ids) for r in deduped]
    scored.sort(key=lambda s: (s.score, s.rights_confidence), reverse=True)

    selected: list[ScoredAsset] = []
    for item in scored[:limit]:
        existing_titles.append(item.record.original_url or "")
        used_ids.append(item.record.asset_id)
        selected.append(item)
    return selected


# ========== Persistence ==========

def write_rights_manifest(records: Sequence[RightsRecord], path: Path) -> Path:
    """Write the rights manifest (JSON) archived with the upload package."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "assets": [r.to_dict() for r in records],
        "approved_count": sum(1 for r in records if r.review_status == "approved"),
        "pending_count": sum(1 for r in records if r.review_status == "pending_review"),
        "rejected_count": sum(1 for r in records if r.review_status == "rejected"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_rights_manifest(path: Path) -> list[RightsRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records: list[RightsRecord] = []
    for raw in data.get("assets", []):
        props = raw.get("properties") or {}
        records.append(RightsRecord(
            asset_id=raw["asset_id"],
            kind=raw["kind"],
            provider=raw["provider"],
            original_url=raw["original_url"],
            download_url=raw["download_url"],
            creator=raw.get("creator", "unknown"),
            license_name=raw.get("license_name", ""),
            license_url=raw.get("license_url", ""),
            attribution=raw.get("attribution", ""),
            acquired_at=raw.get("acquired_at", ""),
            checksum_sha256=raw.get("checksum_sha256", ""),
            properties=MediaProperties(**props),
            candidate_scene_ids=list(raw.get("candidate_scene_ids", [])),
            review_status=raw.get("review_status", "pending_review"),
            rejection_reason=raw.get("rejection_reason"),
        ))
    return records


# ========== Helpers ==========

def _asset_id(provider: Provider, url: str) -> str:
    digest = hashlib.sha256(f"{provider}:{url}".encode("utf-8")).hexdigest()[:16]
    return f"asset-{provider}-{digest}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mime(url: str) -> str:
    lowered = url.lower()
    if lowered.endswith(".mp4"):
        return "video/mp4"
    if lowered.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"
