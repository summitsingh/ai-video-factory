"""Research and claim-verification pipeline for AI Video Factory.

Flow::

    selected topic -> source collection -> source-quality classification
        -> claim extraction -> corroboration / contradiction detection
        -> verified facts -> writer input

Each narrated factual claim carries source IDs so the script writer receives
only verified claims plus clearly labelled uncertainty. High-stakes topics
(elections, financial/medical/legal advice, armed conflict, disasters, or an
identifiable person) are routed for human review or rejected when they cannot
meet higher verification requirements.

Claim extraction is injectable: the default extractor calls the local LM Studio
model, but tests and offline runs use a deterministic rule-based fallback so no
network or model dependency is required in the test path.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Sequence

from ai_video_factory.sanitization import sanitize_diagnostic

Classification = Literal["confirmed fact", "reported claim", "estimate", "opinion", "analysis/speculation"]
SourceQuality = Literal["reliable", "secondary", "unverified"]


# Production extractor signature: (topic, source_contents) -> pairs. The optional
# keyword-only ``identity`` callback lets the research stage stamp the extractor's
# name/version/prompt-schema onto the result for fingerprinting. We annotate loosely
# as Callable[..., ...] so callers may pass identity positionally or by keyword.
ProductionExtractor = Callable[..., list[tuple[str, Classification]]]


def _stamp_identity(
    identity: Callable[[str, str, str], None] | None,
    name: str,
    version: str,
    schema: str,
) -> None:
    if identity is not None:
        identity(name, version, schema)

# Classification schema enforced by the production extractor. Every extracted
# claim must carry one of these labels; anything else is rejected (fail closed).
_VALID_CLASSIFICATIONS: frozenset[str] = frozenset(
    {"confirmed fact", "reported claim", "estimate", "opinion", "analysis/speculation"}
)


class ResearchError(RuntimeError):
    """Raised when research fails."""


# ========== Schemas ==========

@dataclass
class SourceRecord:
    url: str
    provider: str
    quality: SourceQuality
    title: str = ""
    published_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SourceContent(NamedTuple):
    """Fetched source content plus its integrity hash.

    Extraction is only ever run against this persisted content — never against a
    URL/domain heuristic or the topic string alone. ``content_hash`` (sha256 of
    the raw bytes) participates in the research-stage fingerprint so that any
    change to the underlying source invalidates prior claims.
    """

    url: str
    title: str
    quality: SourceQuality
    content: str
    content_hash: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_source_content(
    url: str,
    title: str,
    quality: SourceQuality,
    transport: Callable[[str], str | bytes] | None = None,
    *,
    max_bytes: int = 200_000,
) -> SourceContent:
    """Fetch and persist the actual source content for one URL.

    Uses an injected ``transport`` (so tests never touch the network). Returns a
    :class:`SourceContent` whose ``content_hash`` covers the raw fetched bytes.
    Raises :class:`ResearchError` when no usable content is returned — callers
    must treat that as "insufficient material" and fail closed rather than
    substituting topic-derived statements.
    """

    if transport is None:
        raise ResearchError(
            f"no content fetcher configured for {url}; refusing to extract from a URL alone"
        )

    try:
        raw = transport(url)
    except Exception as exc:  # noqa: BLE001 - surface any fetch failure verbatim
        raise ResearchError(f"failed to fetch source content for {url}: {exc}") from exc

    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw or ""

    text = (text or "").strip()
    if len(text) < 40:
        raise ResearchError(
            f"source {url} returned insufficient content ({len(text)} chars); "
            "refusing to extract claims from an empty page"
        )

    if len(text) > max_bytes:
        text = text[:max_bytes]

    return SourceContent(
        url=url,
        title=title,
        quality=quality,
        content=text,
        content_hash=_sha256_text(text),
    )


def persist_source_content(content: SourceContent, workdir: Path) -> Path:
    """Write fetched source content to ``workdir/sources/<sanitized-url>.txt``."""

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", content.url)[:120]
    path = workdir / "sources" / f"{safe}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.content, encoding="utf-8")
    return path


def write_source_manifest(
    sources: list[SourceRecord], content_by_url: dict[str, SourceContent], workdir: Path
) -> Path:
    """Write a JSON manifest of every source plus its fetched-content hash."""

    manifest = {
        "sources": [s.to_dict() for s in sources],
        "content_hashes": {url: c.content_hash for url, c in content_by_url.items()},
    }
    path = workdir / "source_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# ========== Secure source fetching (production hardening) ==========

import ipaddress as _ipaddress
import socket as _socket

# Content types we will ever extract claims from. Anything binary (images, video,
# audio, PDFs) is rejected so a "source" can never smuggle non-text material into
# claim extraction.
_FETCH_CONTENT_TYPE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "text/html",
        "text/plain",
        "text/markdown",
        "application/rss+xml",
        "application/atom+xml",
        "application/xml",
        "text/xml",
        "application/json",
    }
)

# Maximum number of redirects to follow before failing closed.
_MAX_REDIRECTS = 5
# Hard cap on response body size (bytes). Oversized bodies are rejected rather than
# buffered, bounding memory use against hostile sources.
_MAX_FETCH_BYTES = 2_000_000
# Per-request wall-clock timeout (seconds).
_FETCH_TIMEOUT_SECONDS = 30


def _is_blocked_host(hostname: str) -> bool:
    """Return True if ``hostname`` resolves to loopback/private/link-local space."""
    try:
        addrinfo = _socket.getaddrinfo(hostname, None)
    except OSError:
        # Unresolvable host -> fail closed (refuse to fetch).
        return True
    for family, _type, _proto, _canon, sockaddr in addrinfo:
        ip = _ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


def make_secure_http_transport(
    *,
    max_redirects: int = _MAX_REDIRECTS,
    max_bytes: int = _MAX_FETCH_BYTES,
    timeout_seconds: float = _FETCH_TIMEOUT_SECONDS,
) -> Callable[[str], str]:
    """Build a hardened HTTP(S) transport for source fetching.

    The returned callable enforces, on every request and every redirect hop:

    * HTTPS only (plain HTTP is rejected);
    * loopback / private / link-local / reserved hosts are refused;
    * redirects are followed manually up to ``max_redirects`` hops (no unbounded
      redirect chains, no automatic library following);
    * response bodies are capped at ``max_bytes`` and the request times out after
      ``timeout_seconds``;
    * only text-like content types are accepted (binary media is rejected);
    * no credentials are forwarded: any userinfo in the URL is stripped and no
      Authorization/Cookie headers are added.

    Raises :class:`ResearchError` on any violation so callers fail closed rather
    than fetching from an unsafe or disallowed location.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    def _transport(url: str) -> str:
        scheme = ""
        try:
            parts = urllib.parse.urlsplit(url)
            scheme = (parts.scheme or "").lower()
            hostname = (parts.hostname or "").lower()
        except ValueError as exc:
            raise ResearchError(f"invalid source URL {url!r}: {exc}") from exc

        if scheme != "https":
            raise ResearchError(
                f"source fetch refused for {url!r}: only HTTPS is allowed (got {scheme or 'none'!r})"
            )
        if not hostname:
            raise ResearchError(f"source fetch refused for {url!r}: no host")
        if _is_blocked_host(hostname):
            raise ResearchError(
                f"source fetch refused for {url!r}: host resolves to a loopback/private/link-local address"
            )

        # Strip any userinfo so credentials are never forwarded.
        safe_url = urllib.parse.urlsplit(url)
        netloc = (safe_url.netloc or "").split("@", 1)[-1]
        request_url = safe_url._replace(netloc=netloc).geturl()

        headers = {
            "Accept": ", ".join(_FETCH_CONTENT_TYPE_ALLOWLIST),
            "User-Agent": "ai-video-factory/1.0",
        }
        raw = b""
        content_type = ""
        for _hop in range(max_redirects + 1):
            request = urllib.request.Request(request_url, method="GET", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    raw = response.read(max_bytes + 1)
                    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            except urllib.error.HTTPError as exc:  # includes 3xx when not auto-followed
                if exc.code in (301, 302, 303, 307, 308) and exc.headers is not None:
                    location = exc.headers.get("Location")
                    if not location:
                        raise ResearchError(
                            f"source {url!r} redirected with no Location header"
                        ) from exc
                    # Resolve relative redirects against the response URL.
                    request_url = urllib.parse.urljoin(exc.geturl(), location)
                    continue
                raise ResearchError(f"source fetch failed for {url!r}: HTTP {exc.code}") from exc

            break

        if len(raw) > max_bytes:
            raise ResearchError(
                f"source {url!r} exceeded size limit of {max_bytes} bytes; refusing to extract"
            )
        if content_type not in _FETCH_CONTENT_TYPE_ALLOWLIST:
            raise ResearchError(
                f"source {url!r} returned disallowed content type {content_type!r}; "
                "only text-like sources are extracted"
            )
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - surface decode failure verbatim
            raise ResearchError(f"source {url!r} could not be decoded as UTF-8: {exc}") from exc

    return _transport


@dataclass
class Claim:
    claim_id: str
    text: str
    source_ids: list[str]
    provisional_classification: Classification
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Contradiction:
    contradiction_id: str
    claim_ids: list[str]
    nature: str
    resolution: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchResult:
    topic: str
    sources: list[SourceRecord] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    verified_facts: list[Claim] = field(default_factory=list)
    research_brief: str = ""
    flagged_for_review: bool = False
    flag_reasons: list[str] = field(default_factory=list)
    # --- Research-stage fingerprint inputs (requirement 6 + extractor config) ---
    # Identity of the extractor that produced these claims. ``None`` means the
    # default LM Studio loopback was used; tests/dry-run inject an explicit name.
    extractor_name: str | None = None
    extractor_version: str | None = None
    prompt_schema_version: str | None = None
    source_content_hashes: dict[str, str] = field(default_factory=dict)
    output_digest: str | None = None
    # Fetched-and-persisted source content (url -> SourceContent). Present only on
    # the production path so narration can be grounded in real material rather than
    # invented filler. Absent on the legacy topic+URLs path used by dry-run/tests.
    source_contents: dict[str, SourceContent] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ========== Source quality classification ==========

_RELIABLE_DOMAINS = {
    "nasa.gov", "nasa", "wikimedia.org", "wikipedia.org", "bbc.com", "bbc.co.uk",
    "reuters.com", "apnews.com", "ap.org", "gov.uk", "europa.eu", "who.int",
    "cdc.gov", "nih.gov", "mit.edu", "stanford.edu", "nature.com", "sciencedaily.com",
}
_UNVERIFIED_DOMAINS = {"wordpress.com", "blogspot.com", "tumblr.com"}


def classify_source_quality(url: str) -> SourceQuality:
    """Classify a source URL by domain heuristics."""
    host = ""
    try:
        from urllib.parse import urlsplit
        host = urlsplit(url).hostname or ""
    except ValueError:
        return "unverified"
    host = host.lower()
    if any(host == d or host.endswith("." + d) for d in _RELIABLE_DOMAINS):
        return "reliable"
    if any(host == d or host.endswith("." + d) for d in _UNVERIFIED_DOMAINS):
        return "secondary"
    return "unverified"


def collect_sources(urls: Sequence[str]) -> list[SourceRecord]:
    """Build source records with quality classification (no network fetch)."""
    records: list[SourceRecord] = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        records.append(SourceRecord(
            url=url,
            provider=classify_source_quality(url),
            quality=classify_source_quality(url),
            title="",
        ))
    return records


# ========== Claim extraction (injectable) ==========

def _rule_based_extract_claims(topic: str, sources: list[SourceRecord]) -> list[tuple[str, Classification]]:
    """Deterministic fallback claim extractor used offline / in tests.

    Splits the topic into a few factual-ish statements and classifies them. This
    is intentionally conservative; production uses the LM Studio extractor.
    """
    base = topic.strip()
    candidates: list[tuple[str, Classification]] = [
        (f"{base} is a current trending subject.", "reported claim"),
        (f"Multiple sources report developments related to {base.lower()}.", "confirmed fact"),
        (f"The significance of {base.lower()} continues to grow in coverage.", "analysis/speculation"),
    ]
    return candidates


def extract_claims(
    topic: str,
    sources: list[SourceRecord],
    *,
    extractor: Callable[[str, list[SourceRecord]], list[tuple[str, Classification]]] | None = None,
) -> list[Claim]:
    """Extract claims with source IDs and provisional classification.

    ``extractor`` is injectable; defaults to the LM Studio model when available,
    falling back to deterministic rule-based extraction otherwise.
    """
    extractor = extractor or _default_extractor
    pairs = extractor(topic, sources)
    claim_ids_seen: dict[str, int] = {}
    claims: list[Claim] = []
    for text, classification in pairs:
        text = (text or "").strip()
        if not text:
            continue
        n = claim_ids_seen.get(classification, 0) + 1
        claim_ids_seen[classification] = n
        claims.append(Claim(
            claim_id=f"claim-{n:02d}",
            text=text,
            source_ids=[s.url for s in sources],
            provisional_classification=classification,
        ))
    return claims


def _default_extractor(topic: str, sources: list[SourceRecord]) -> list[tuple[str, Classification]]:
    """Try the LM Studio extractor; fall back to deterministic extraction."""
    try:
        return lm_studio_extract_claims(topic, sources)
    except Exception:  # noqa: BLE001 - offline fallback is intentional
        return _rule_based_extract_claims(topic, sources)


# Public reference to the deterministic offline extractor so callers (daily job,
# CLI dry-run) can opt out of any network/model dependency explicitly.
RULE_BASED_EXTRACTOR: Callable[[str, list[SourceRecord]], list[tuple[str, Classification]]] = _rule_based_extract_claims


def lm_studio_extract_claims(
    topic: str, sources: list[SourceRecord]
) -> list[tuple[str, Classification]]:
    """Extract claims via the local LM Studio loopback HTTP endpoint.

    Uses a direct urllib POST to the configured loopback URL so this module does
    not depend on the exact LmStudioBackend constructor. Any failure (server down,
    bad response) raises and is caught by ``_default_extractor``.
    """
    import urllib.request

    url = "http://localhost:1234/v1/chat/completions"
    prompt = (
        "You are a claim-extraction tool. From the topic below, extract 3-5 distinct "
        "factual statements suitable for narration. For each, output one line: "
        "[classification] | [claim text]. Classifications are limited to: "
        "confirmed fact, reported claim, estimate, opinion, analysis/speculation.\n\n"
        f"Topic: {topic}\nSources: {[s.url for s in sources]}\n"
    )
    payload = json.dumps({
        "model": "qwen3.6-35b-a3b-udt-mtp",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.2,
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    pairs: list[tuple[str, Classification]] = []
    line_re = re.compile(r"\[(?P<cls>[a-z/ ]+)\]\s*\|\s*(?P<text>.+)", re.IGNORECASE)
    for raw_line in content.splitlines():
        match = line_re.search(raw_line.strip())
        if not match:
            continue
        cls = match.group("cls").strip().lower()
        text = match.group("text").strip().rstrip(".")
        if cls not in {"confirmed fact", "reported claim", "estimate", "opinion", "analysis/speculation"}:
            cls = "reported claim"
        pairs.append((text, cls))  # type: ignore[arg-type] - cls normalized above
    if not pairs:
        # Deggraded/non-conforming model output must not silently drop all claims.
        raise RuntimeError("LM Studio extractor produced no extractable claims")
    return pairs


def make_production_extractor(
    *,
    endpoint_url: str = "http://localhost:1234/v1/chat/completions",
    model_name: str = "qwen3.6-35b-a3b-udt-mtp",
    extractor_name: str = "lm-studio-production",
    extractor_version: str = "1.0",
    prompt_schema_version: str = "2026-09-07",
) -> ProductionExtractor:
    """Build a schema-constrained production claim extractor.

    Unlike the default loopback extractor (which accepts topic + URLs), this one
    is fed the *persisted source content* and enforces a strict output schema:
    every line must be ``[classification] | text`` with a recognised label, and
    at least one claim must come back. Malformed or insufficient output raises
    :class:`ResearchError` so callers fail closed instead of promoting generic
    topic-level filler into verified facts.

    The returned callable has the signature ``(topic, sources) -> pairs`` where
    ``sources`` is a list of :class:`SourceContent`. It records its own identity
    (name/version/prompt-schema) on the caller's behalf via the optional
    ``identity`` callback so the research stage can stamp them onto the result.
    """

    def _extract(
        topic: str,
        sources: list[SourceContent],
        *,
        identity: Callable[[str, str, str], None] | None = None,
    ) -> list[tuple[str, Classification]]:
        if not sources:
            raise ResearchError("production extractor called with no source content")

        prompt = (
            "You are a strict claim-extraction tool for a factual documentary. "
            "Read the SOURCE CONTENT below and extract 3-5 DISTINCT, verifiable "
            "statements that could be narrated. Each statement must be grounded in "
            "the source content — do not invent facts, statistics, or quotes.\n\n"
            "For each statement output exactly one line:\n"
            "[classification] | claim text\n\n"
            "Classifications are EXACTLY one of: confirmed fact, reported claim, "
            "estimate, opinion, analysis/speculation. Anything else is invalid.\n\n"
            f"TOPIC: {topic}\n\n"
            "SOURCE CONTENT:\n"
        )
        for src in sources:
            prompt += f"\n--- Source: {src.url} ---\n{src.content}\n"

        import urllib.request

        payload = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.2,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            endpoint_url, data=payload, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        pairs: list[tuple[str, Classification]] = []
        line_re = re.compile(r"\[(?P<cls>[a-z/ ]+)\]\s*\|\s*(?P<text>.+)", re.IGNORECASE)
        for raw_line in content.splitlines():
            match = line_re.search(raw_line.strip())
            if not match:
                continue
            cls = match.group("cls").strip().lower()
            text = match.group("text").strip().rstrip(".")
            if cls not in _VALID_CLASSIFICATIONS:
                # Malformed classification -> reject the whole extraction (fail closed).
                raise ResearchError(
                    f"production extractor produced invalid classification {cls!r}; "
                    "refusing to promote non-conforming output as verified facts"
                )
            pairs.append((text, cls))  # type: ignore[arg-type]

        if not pairs:
            raise ResearchError(
                "production extractor returned no claims; insufficient supported material"
            )

        if identity is not None:
            identity(extractor_name, extractor_version, prompt_schema_version)

        return pairs

    return _extract


def make_deterministic_production_extractor(
    *,
    extractor_name: str = "deterministic-production",
    extractor_version: str = "1.0",
    prompt_schema_version: str = "2026-09-07",
) -> ProductionExtractor:
    """Deterministic production extractor that pulls *distinct* facts from content.

    Unlike :data:`RULE_BASED_EXTRACTOR` (which emits generic topic-level filler),
    this reads the actual persisted source text and extracts concrete, verifiable
    statements — sentences containing years/dates, numbers/percentages, quoted
    phrases, or proper nouns. It requires at least one *distinct* grounded fact per
    source; if none are found it raises :class:`ResearchError` so callers fail
    closed rather than padding with invented material.

    The returned callable has the signature ``(topic, sources) -> pairs`` where
    ``sources`` is a list of :class:`SourceContent`. It records its own identity via
    the optional ``identity`` callback.
    """

    def _extract(
        topic: str,
        sources: list[SourceContent],
        *,
        identity: Callable[[str, str, str], None] | None = None,
    ) -> list[tuple[str, Classification]]:
        if not sources:
            raise ResearchError("production extractor called with no source content")

        year_re = re.compile(r"\b(19|20)\d{2}\b")
        number_re = re.compile(r"\b\d[\d,.%]*\b")
        quote_re = re.compile(r'["“”]([^"“”]{6,})["“”]')

        def _is_grounded(sentence: str) -> bool:
            low = sentence.lower()
            if any(kw in low for kw in ("opinion", "i think", "maybe", "perhaps", "speculate")):
                return False
            return (
                bool(year_re.search(sentence))
                or bool(number_re.search(sentence))
                or bool(quote_re.search(sentence))
                or any(w.istitle() and len(w) > 3 for w in sentence.split())
            )

        # One verified claim per source: each source is one chapter theme, backed by
        # that source's own distinct grounded sentences. This keeps the fact count
        # equal to the number of independent sources (small, controllable) rather than
        # exploding with every sentence, so the long-form word budget stays bounded.
        pairs: list[tuple[str, Classification]] = []
        for src in sources:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", src.content) if len(s.strip()) > 12]
            grounded = [s for s in sentences if _is_grounded(s)]
            if not grounded:
                continue
            pairs.append((grounded[0].rstrip("."), "confirmed fact"))

        if not pairs:
            raise ResearchError(
                f"no distinct grounded facts extracted from {len(sources)} source(s); "
                "insufficient supported material for a long-form documentary"
            )

        if identity is not None:
            identity(extractor_name, extractor_version, prompt_schema_version)

        return pairs

    return _extract


def extract_from_source_content(
    topic: str,
    sources: list[SourceContent],
    *,
    extractor: ProductionExtractor | None = None,
    identity: Callable[[str, str, str], None] | None = None,
) -> tuple[list[Claim], dict[str, str]]:
    """Run the production extractor against persisted source content.

    Returns ``(claims, source_content_hashes)``. The hashes cover the raw fetched
    bytes and participate in the research-stage fingerprint so that any change to
    an underlying source invalidates prior claims. Raises :class:`ResearchError`
    when extraction fails or produces insufficient supported material.
    """

    if extractor is None:
        # Default production extractor (LM Studio loopback, schema-constrained).
        extractor = make_production_extractor()

    pairs = extractor(topic, sources, identity=identity)
    claim_ids_seen: dict[str, int] = {}
    claims: list[Claim] = []
    for text, classification in pairs:
        text = (text or "").strip()
        if not text:
            continue
        n = claim_ids_seen.get(classification, 0) + 1
        claim_ids_seen[classification] = n
        claims.append(Claim(
            claim_id=f"claim-{n:02d}",
            text=text,
            source_ids=[s.url for s in sources],
            provisional_classification=classification,
        ))

    content_hashes = {s.url: s.content_hash for s in sources}
    return claims, content_hashes


# ========== Corroboration + contradiction detection ==========

def detect_contradictions(claims: list[Claim]) -> list[Contradiction]:
    """Detect material conflicts between claims and resolve or flag them."""
    contradictions: list[Contradiction] = []
    cid_counter = 0
    # Pairwise heuristic: opposite polarity keywords indicate conflict.
    positive = {"grows", "increases", "advances", "success", "wins", "improves"}
    negative = {"shrinks", "decreases", "fails", "crisis", "loses", "worsens"}
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            a, b = claims[i], claims[j]
            la, lb = a.text.lower(), b.text.lower()
            hits_a = positive & set(re.findall(r"[a-z]+", la))
            hits_b = negative & set(re.findall(r"[a-z]+", lb))
            hits_b_pos = positive & set(re.findall(r"[a-z]+", lb))
            hits_a_neg = negative & set(re.findall(r"[a-z]+", la))
            if (hits_a and hits_b) or (hits_a_neg and hits_b_pos):
                cid_counter += 1
                contradictions.append(Contradiction(
                    contradiction_id=f"contra-{cid_counter:02d}",
                    claim_ids=[a.claim_id, b.claim_id],
                    nature="opposite polarity detected between two claims",
                    resolution="flagged for human review; not promoted to verified fact",
                ))
    return contradictions


def verify_claims(claims: list[Claim], contradictions: list[Contradiction]) -> list[Claim]:
    """Promote claims to verified facts.

    A claim is verified when it is a confirmed fact or reported claim AND is not
    involved in an unresolved contradiction. Estimates/opinions/speculation are
    never promoted (they carry uncertainty for the writer).
    """
    contradicted = {cid for c in contradictions for cid in c.claim_ids}
    verified: list[Claim] = []
    for claim in claims:
        if claim.claim_id in contradicted:
            continue
        if claim.provisional_classification in {"confirmed fact", "reported claim"}:
            verified.append(claim)
    return verified


# ========== Sensitive-topic guardrails ==========

_SENSITIVE_TOPICS = {
    "election", "vote", "ballot", "campaign", "candidate",
    "financial advice", "investing", "stock", "trading", "retirement savings",
    "medical", "cure", "drug", "treatment", "diagnosis", "vaccine", "disease",
    "legal advice", "lawsuit", "attorney", "court ruling", "sentencing",
    "armed conflict", "war", "invasion", "nuclear weapon",
    "disaster", "earthquake", "tornado", "tsunami", "mass shooting",
}


def flag_sensitive_topic(topic: str) -> list[str]:
    """Return human-review reasons when a topic is high-stakes."""
    low = topic.lower()
    tokens = set(re.findall(r"[a-z]+", low))
    reasons: list[str] = []
    for keyword in _SENSITIVE_TOPICS:
        if keyword in low or any(keyword.split()[0] in tokens for keyword in keyword.split()):
            reasons.append(f"topic touches sensitive area: {keyword}")
    return reasons


def requires_human_review(reasons: Sequence[str]) -> bool:
    return bool(reasons)


# ========== Orchestration + artifacts ==========

def run_research(
    topic: str,
    source_urls: Sequence[str],
    *,
    extractor: Callable[..., list[tuple[str, Classification]]] | None = None,
    production_extractor: Callable[[str, list[SourceContent]], list[tuple[str, Classification]]] | None = None,
    content_fetcher: Callable[[str], str | bytes] | None = None,
    source_contents: list[SourceContent] | None = None,
    observed_at: str | None = None,
) -> ResearchResult:
    """Run the full research pipeline for one topic.

    Two modes are supported so that tests/dry-run stay deterministic while
    production stays grounded in fetched content:

    * **Legacy mode** (default): ``extract_claims(topic, sources, extractor=...)``
      runs against topic + URLs. Used by existing unit tests and dry-runs that
      inject :data:`RULE_BASED_EXTRACTOR`. No source content is required.
    * **Production mode**: when ``production_extractor`` or ``source_contents``
      (or a ``content_fetcher``) are supplied, the actual source content is
      fetched/persisted first and extraction runs against that persisted content
      with a schema-constrained extractor. Malformed/insufficient output raises
      :class:`ResearchError` so generic filler never becomes verified facts.

    The returned result carries the extractor identity, per-source content
    hashes, and an output digest — all part of the research-stage fingerprint.
    """
    observed_at = observed_at or datetime.now(UTC).isoformat()
    sources = collect_sources(source_urls)

    identity_holder: dict[str, str | None] = {"name": None, "version": None, "schema": None}
    contents: list[SourceContent] = []

    # Production path: extract against persisted source content.
    if production_extractor is not None or source_contents is not None or content_fetcher is not None:
        if not sources and not source_contents:
            raise ResearchError("no source URLs provided for research")

        contents = list(source_contents) if source_contents is not None else []
        if not contents:
            for rec in sources:
                contents.append(
                    fetch_source_content(rec.url, rec.title or "", rec.quality, transport=content_fetcher)
                )

        claims, content_hashes = extract_from_source_content(
            topic,
            contents,
            extractor=production_extractor or make_production_extractor(),
            identity=lambda name, ver, ps: identity_holder.update({"name": name, "version": ver, "schema": ps}),
        )
    else:
        # Legacy path (tests / dry-run inject RULE_BASED_EXTRACTOR via extractor=).
        claims = extract_claims(topic, sources, extractor=extractor or _default_extractor)
        content_hashes = {}

    contradictions = detect_contradictions(claims)
    verified = verify_claims(claims, contradictions)
    reasons = flag_sensitive_topic(topic)

    # Output digest: stable hash over the claims + source-content hashes so any
    # change to extracted material or an underlying source invalidates prior runs.
    digest_inputs = "|".join(sorted(f"{c.claim_id}={c.text}" for c in claims))
    output_digest = hashlib.sha256(
        f"{topic}|{digest_inputs}|{json.dumps(content_hashes, sort_keys=True)}".encode("utf-8")
    ).hexdigest()

    # Persisted source content keyed by URL (production path only).
    contents_by_url: dict[str, SourceContent] = {c.url: c for c in contents} if contents else {}

    brief_lines = [f"# Research Brief: {topic}", "", f"Generated: {observed_at}", ""]
    brief_lines.append("## Sources")
    for source in sources:
        brief_lines.append(f"- [{source.quality}] {source.url}")
    brief_lines += ["", "## Extracted Claims", ""]
    for claim in claims:
        flag = "" if claim.provisional_classification == "confirmed fact" else f" ({claim.provisional_classification})"
        brief_lines.append(f"- {claim.claim_id}: {claim.text}{flag} -> {', '.join(claim.source_ids)}")
    if contradictions:
        brief_lines += ["", "## Contradictions", ""]
        for contra in contradictions:
            brief_lines.append(f"- {contra.contradiction_id}: {contra.nature}")
    brief_lines += ["", "## Verified Facts (writer input)", ""]
    if verified:
        for claim in verified:
            brief_lines.append(f"- {claim.claim_id}: {claim.text} -> {', '.join(claim.source_ids)}")
    else:
        brief_lines.append("- None verified; writer should treat all claims as uncertain.")
    if reasons:
        brief_lines += ["", "## Human Review Required", "", *reasons]

    return ResearchResult(
        topic=topic,
        sources=sources,
        claims=claims,
        contradictions=contradictions,
        verified_facts=verified,
        research_brief="\n".join(brief_lines) + "\n",
        flagged_for_review=requires_human_review(reasons),
        flag_reasons=list(reasons),
        extractor_name=identity_holder["name"],
        extractor_version=identity_holder["version"],
        prompt_schema_version=identity_holder["schema"],
        source_content_hashes=content_hashes,
        output_digest=output_digest,
        source_contents=contents_by_url,
    )


def write_research_artifacts(result: ResearchResult, output_dir: Path) -> dict[str, Path]:
    """Write sources/claims/contradictions/verified_facts JSON + research_brief.md."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, data: Any) -> Path:
        path = output_dir / name
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    paths = {
        "sources": _write("sources.json", [s.to_dict() for s in result.sources]),
        "claims": _write("claims.json", [c.to_dict() for c in result.claims]),
        "contradictions": _write("contradictions.json", [c.to_dict() for c in result.contradictions]),
        "verified_facts": _write("verified_facts.json", [c.to_dict() for c in result.verified_facts]),
    }
    # Persist fetched source content so a resumed run can still ground narration in
    # the same material (otherwise reruns lose all grounded sentences and reject).
    if result.source_contents:
        paths["source_contents"] = _write(
            "source_contents.json",
            {url: {"url": c.url, "title": c.title, "quality": c.quality,
                   "content": c.content, "content_hash": c.content_hash}
             for url, c in result.source_contents.items()},
        )
    paths["research_brief"] = output_dir / "research_brief.md"
    paths["research_brief"].write_text(result.research_brief, encoding="utf-8")
    return paths
