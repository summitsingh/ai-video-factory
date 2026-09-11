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
import signal
import threading
import time
from urllib.error import HTTPError as _HTTPError
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
ProductionExtractor = Callable[..., "list[Extraction]"]


def _stamp_identity(
    identity: Callable[[str, str, str, str | None, str | None, str | None], None] | None,
    name: str,
    version: str,
    schema: str,
    prompt_version: str | None = None,
    model_id: str | None = None,
    endpoint_url: str | None = None,
) -> None:
    if identity is not None:
        identity(
            name,
            version,
            schema,
            prompt_version=prompt_version,
            model_id=model_id,
            endpoint_url=endpoint_url,
        )

# Classification schema enforced by the production extractor. Every extracted
# claim must carry one of these labels; anything else is rejected (fail closed).
_VALID_CLASSIFICATIONS: frozenset[str] = frozenset(
    {"confirmed fact", "reported claim", "estimate", "opinion", "analysis/speculation"}
)
# Version of the inline production prompt template fed to the local model. Bump this
# when the prompt wording/instructions change so provenance pins which prompt produced
# a given result. Deliberately distinct from EXTRACTION_SCHEMA_VERSION below.
PROMPT_VERSION = "2026-09-07"
# Version of the output-extraction schema: the evidence-line grammar and the set of
# accepted classification labels enforced on every claim line (see _EVIDENCE_LINE_RE
# and _VALID_CLASSIFICATIONS). Bump when that schema changes. Distinct from PROMPT_VERSION.
EXTRACTION_SCHEMA_VERSION = "2026-09-07"
# Bounded retry attempts for the production extractor's model POST. The local
# general-purpose model emits non-deterministic output at temperature=0 for the same
# prompt+content (verified empirically): a rich source that conforms on one call may
# emit preamble/empty/junk on another. A bounded retry lets transient variance recover
# while still failing closed after every attempt is non-conforming. Transport errors
# (endpoint unavailable) are deliberately NOT retried -- they must surface immediately.
_EXTRACTION_RETRIES = 5

# Matches one extraction output line: `[classification] | claim || "verbatim quote"`.
_EVIDENCE_LINE_RE = re.compile(
    r'\[(?P<cls>[a-z/ ]+)\]\s*\|\s*(?P<claim>.+?)\s*\|\|\s*"(?P<quote>.*?)"\s*$',
    re.IGNORECASE,
)
# Tolerant mirror of _EVIDENCE_LINE_RE for the common model slips of omitting the
# square brackets around the label and/or using a dash instead of the double-pipe
# before the quote (e.g. `confirmed fact | claim || "q"`, `[reported claim] | c - "q"`).
# Restricted to the exact valid-classification labels so only a line already carrying
# full claim structure can be normalised; anything else still falls through to the
# strict bracketed grammar first (which rejects unknown labels as invalid) and fails
# closed. Never widens acceptance -- it only rewrites slips into the canonical form.
_TOLERANT_RE = re.compile(
    r'^\[?(?P<cls>confirmed fact|reported claim|estimate|opinion|analysis/speculation)\]?'
    r'\s*[|]\s*(?P<claim>.+?)\s*[|\-]{1,2}\s*"(?P<quote>.*?)"\s*$',
    re.IGNORECASE,
)


@dataclass
class Extraction:
    """One claim pulled from a single source, with its verbatim evidence quote."""

    claim: str
    quote: str
    classification: Classification


@dataclass
class EvidenceRecord:
    """A per-source claim carried through corroboration; every field is auditable.

    ``id`` ties the record to the model's grouping pass (which only *suggests* which
    records share a fact); correctness is enforced later by resolving IDs back to
    records and requiring >=2 distinct NASA URLs with supported labels.
    """

    id: str
    url: str
    label: Classification
    claim: str
    evidence_quote: str
    chunk_id: str = ""
    chunk_ranges: list[tuple[int, int]] = field(default_factory=list)


_SUPPORTED_LABELS = frozenset({"confirmed fact", "reported claim"})


def _normalize_ws(text: str) -> str:
    """Collapse whitespace + lowercase for stable equality comparisons."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()

def _collapse_ws(text: str) -> str:
    """Collapse whitespace only (preserve case/punctuation) for verbatim evidence checks."""
    return re.sub(r"\s+", " ", (text or "")).strip()


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
    except _SourceDeadlineExhausted:
        # Budget exhaustion is a skip signal for the bounded per-source loop; let the
        # caller decide whether to drop or fail closed. Do NOT convert to ResearchError.
        raise
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


class _DeadlineHit(Exception):
    """Internal signal raised by SIGALRM when a single attempt exceeds its wall-clock
    budget. Deliberately NOT a :class:`ResearchError` so the bounded-retry helper can
    treat it as a transient (retryable) failure rather than a fail-closed schema error."""
    pass


class _SourceDeadlineExhausted(ResearchError):
    """Raised when one source exhausts its bounded per-attempt deadline budget on a
    transient failure (timeout/connection/5xx) during fetch or model POST.

    Distinct from other :class:`ResearchError` values so the native per-source loop can
    tell "this source is over budget, skip it" apart from content/schema failures that
    must fail closed. Subclasses :class:`ResearchError` so broad ``except ResearchError``
    handlers keep working while callers may catch this specifically to drop only flagged
    sources and continue the same run."""
    pass


def _enforce_deadline(fn: Callable[[], Any], *, deadline_seconds: float) -> Any:
    """Run ``fn()`` under a hard wall-clock cap using SIGALRM in the main thread.

    ``urlopen(timeout=...)`` is a per-operation socket timeout, not a wall-clock cap:
    connect + read operations can cumulatively exceed it (observed in diagnostics), so
    it cannot alone guarantee a fixed per-attempt budget. SIGALRM interrupts the blocking
    syscall -- PEP 475 lets the handler's exception escape rather than retrying EINTR --
    so total wall time stays bounded with no worker threads and no overlap between
    attempts. The prior signal handler and any pre-existing one-shot timer are restored
    on exit so an enclosing deadline is never clobbered. Fails closed (never runs
    unbounded) where SIGALRM is unavailable or off the main thread."""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        raise ResearchError(
            f"cannot enforce {deadline_seconds}s deadline without SIGALRM on the main "
            "thread; refusing to run unbounded"
        )

    def _on_deadline(_signum: int, _frame: Any) -> None:
        raise _DeadlineHit(f"operation exceeded {deadline_seconds}s wall-clock deadline")

    if deadline_seconds <= 0:
        raise ResearchError(
            f"deadline must be positive, got {deadline_seconds}s; refusing to run unbounded"
        )
    old_handler = signal.signal(signal.SIGALRM, _on_deadline)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.setitimer(signal.ITIMER_REAL, deadline_seconds)
        return fn()
    finally:
        # Cancel our own timer, restore the prior handler, then restore any enclosing
        # one-shot timer we superseded so its remaining interval/period is preserved.
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)


def _run_transient_bounded(
    fn: Callable[[], Any],
    *,
    deadline_seconds: float,
    max_attempts: int,
) -> Any:
    """Run ``fn()`` up to ``max_attempts`` times, each capped by a SIGALRM wall-clock
    deadline. Retry ONLY transient failures (timeout/connection/5xx); fail immediately
    on schema/grounding :class:`ResearchError` or HTTP 4xx so they never consume the
    transient retry budget. After exhausting attempts, raise
    :class:`_SourceDeadlineExhausted` to mark the source as over-budget."""
    max_attempts = max(1, int(max_attempts))
    last_transient: Exception | None = None
    start = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        try:
            return _enforce_deadline(fn, deadline_seconds=deadline_seconds)
        except ResearchError:
            # Non-transient (4xx, redirect error, insufficient content, malformed
            # schema/grounding output): fail closed immediately without consuming retries.
            raise
        except (TimeoutError, OSError, _DeadlineHit) as exc:
            last_transient = exc
            if attempt >= max_attempts:
                err = _SourceDeadlineExhausted(
                    f"source exceeded {deadline_seconds}s per-attempt budget after "
                    f"{max_attempts} attempt(s): {type(exc).__name__}: {exc}"
                )
                # Rich diagnostics so the native per-source loop can record a structured
                # SourceDrop (stage/limit/attempts/elapsed/detail) instead of a bare URL.
                err.limit_seconds = deadline_seconds
                err.attempts = max_attempts
                err.elapsed_seconds = round(time.monotonic() - start, 3)
                err.detail = f"{type(exc).__name__}: {exc}"
                raise err from exc
    assert last_transient is not None  # pragma: no cover - loop always returns/raises
def make_secure_http_transport(
    *,
    max_redirects: int = _MAX_REDIRECTS,
    max_bytes: int = _MAX_FETCH_BYTES,
    timeout_seconds: float = _FETCH_TIMEOUT_SECONDS,
    max_attempts: int = 1,
) -> Callable[[str], str]:
    """Build a hardened HTTP(S) transport for source fetching.

    The returned callable enforces, on every request and every redirect hop:

    * HTTPS only (plain HTTP is rejected);
    * loopback / private / link-local / reserved hosts are refused;
    * redirects are followed manually up to ``max_redirects`` hops (no unbounded
      redirect chains, no automatic library following);
    * response bodies are capped at ``max_bytes`` and each attempt is bounded by a hard
      SIGALRM wall-clock deadline (not merely the per-operation socket timeout, which can
      cumulatively exceed it) of ``timeout_seconds``;
    * only text-like content types are accepted (binary media is rejected);
    * no credentials are forwarded: any userinfo in the URL is stripped and no
      Authorization/Cookie headers are added.

    Each attempt is retried up to ``max_attempts`` times for transient failures only
    (timeout/connection/HTTP 5xx). HTTP 4xx, redirect errors, oversized bodies, disallowed
    content types, and insufficient content fail closed immediately without consuming the
    retry budget. After exhausting attempts on a transient failure, raises
    :class:`_SourceDeadlineExhausted` so the native per-source loop can skip only flagged
    sources and continue the same run rather than aborting all research.

    Raises :class:`ResearchError` on any non-transient violation so callers fail closed
    rather than fetching from an unsafe or disallowed location.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    headers = {
        "Accept": ", ".join(_FETCH_CONTENT_TYPE_ALLOWLIST),
        # Browser-compatible UA so authoritative sites that require one (e.g. JPL) accept the request; the tool identity is retained for transparency. This does not forward credentials or weaken any invariant below.
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36 ai-video-factory/1.0",
    }

    def _do_request(request_url: str) -> tuple[bytes, str]:
        """Perform one GET; return (body, ctype). ``urlopen(timeout=...)`` bounds each
        individual socket operation; the total per-attempt wall-clock cap is enforced once
        by :func:`_run_transient_bounded` around the whole redirect chain + body read."""
        response = urllib.request.urlopen(
            urllib.request.Request(request_url, method="GET", headers=headers),
            timeout=timeout_seconds,
        )
        try:
            raw = response.read(max_bytes + 1)
            ctype = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        finally:
            response.close()
        return raw, ctype

    def _fetch_once(request_url: str) -> str:
        """One attempt across all redirect hops. Returns the decoded body. Raises
        :class:`ResearchError` on 4xx / redirect error / oversized / disallowed-type (fail
        closed); re-raises raw ``HTTPError`` for HTTP 5xx so the bounded caller treats it
        as a transient and retries once."""
        raw = b""
        content_type = ""
        for _hop in range(max_redirects + 1):
            try:
                raw, content_type = _do_request(request_url)
            except urllib.error.HTTPError as exc:  # includes 3xx when not auto-followed
                if exc.code in (301, 302, 303, 307, 308) and exc.headers is not None:
                    location = exc.headers.get("Location")
                    if not location:
                        raise ResearchError(
                            "source redirected with no Location header"
                        ) from exc
                    request_url = urllib.parse.urljoin(exc.geturl(), location)
                    continue
                if 500 <= exc.code < 600:
                    raise  # transient server error -> bounded caller retries once (HTTPError is OSError)
                raise ResearchError(f"source fetch failed: HTTP {exc.code}") from exc
            break
        if len(raw) > max_bytes:
            raise ResearchError(
                f"source exceeded size limit of {max_bytes} bytes; refusing to extract"
            )
        if content_type not in _FETCH_CONTENT_TYPE_ALLOWLIST:
            raise ResearchError(
                f"source returned disallowed content type {content_type!r}; "
                "only text-like sources are extracted"
            )
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - surface decode failure verbatim
            raise ResearchError(f"source could not be decoded as UTF-8: {exc}") from exc

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

        # Bounded transient retry: each attempt is capped by a hard SIGALRM wall-clock
        # deadline. Transient failures (timeout/connection/5xx) retry up to max_attempts;
        # 4xx / redirect errors / oversized / disallowed-type fail closed immediately. After
        # exhausting attempts, raise _SourceDeadlineExhausted so the native per-source loop
        # can skip only flagged sources and continue the same run.
        return _run_transient_bounded(
            lambda: _fetch_once(request_url),
            deadline_seconds=timeout_seconds,
            max_attempts=max_attempts,
        )

    return _transport


@dataclass
class Claim:
    claim_id: str
    text: str
    source_ids: list[str]
    provisional_classification: Classification
    confidence: float = 0.0
    # Auditable per-source evidence backing this claim: each entry records the source
    # URL, the label that source assigned, and the verbatim evidence quote (validated
    # as a substring of one selected chunk slice during extraction). Entries are
    # persisted with the claim so the research-stage fingerprint covers evidence, not
    # just claim text. Each entry is a dict carrying the source url, label, quote,
    # chunk id, and raw ``chunk_ranges`` (a list of [start, end] char spans into that
    # source); ranges are nested lists, so the value type is ``dict[str, Any]``.
    evidence: list[dict[str, Any]] = field(default_factory=list)

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
class SourceDrop:
    """Structured diagnostic for a source skipped because it exhausted its bounded
    per-attempt wall-clock deadline budget during fetch or extraction.

    Retained on :attr:`ResearchResult.dropped_sources` so provenance never falsely credits
    the dropped source with supporting the film and the operator keeps why/where it was
    dropped (stage, attempt limit, elapsed time, final failure detail)."""
    url: str
    stage: Literal["fetch", "extraction"]
    limit_seconds: float
    attempts: int
    elapsed_seconds: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise for provenance artifacts/state (consistent with other records)."""
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
    prompt_version: str | None = None
    extraction_schema_version: str | None = None
    # Resolved active model + endpoint for the production extractor. Recorded so the
    # pilot provenance pins exactly which local model produced these claims; ``None``
    # means no production extraction ran (legacy/dry-run path).
    extractor_model_id: str | None = None
    extractor_endpoint: str | None = None
    source_content_hashes: dict[str, str] = field(default_factory=dict)
    output_digest: str | None = None
    # Fetched-and-persisted source content (url -> SourceContent). Present only on
    # the production path so narration can be grounded in real material rather than
    # invented filler. Absent on the legacy topic+URLs path used by dry-run/tests.
    source_contents: dict[str, SourceContent] = field(default_factory=dict)
    # Structured diagnostics for sources skipped because they exhausted their bounded
    # per-attempt deadline budget during fetch or extraction. Recorded so provenance never
    # falsely credits a dropped source (e.g. Mars) with supporting the film, and so the
    # operator retains why/where each drop happened; downstream asset planning uses the
    # retained sources only. Empty when nothing was dropped.
    dropped_sources: list[SourceDrop] = field(default_factory=list)

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



class ModelResolutionError(RuntimeError):
    """Raised when an active local model cannot be resolved from LM Studio at runtime."""


def _models_list_url(endpoint_url: str) -> str:
    """Derive the ``/v1/models`` listing URL from a chat-completions endpoint URL."""
    base = endpoint_url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return f"{base}/models"


def _lmstudio_models_bytes(url: str) -> bytes:
    """Fetch raw bytes from a local LM Studio URL (injectable for tests)."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=15) as response:
        return response.read()


def resolve_active_model(
    *,
    endpoint_url: str = "http://127.0.0.1:1234/v1/chat/completions",
    preferred: str | None = None,
    transport: Callable[[str], bytes] | None = None,
) -> str:
    """Resolve an active local model ID from LM Studio's ``/v1/models`` at runtime.

    Never assumes a hard-coded default. Queries the running server for available
    models. When ``preferred`` is configured it MUST appear in that listing or the
    build fails closed; only when no preference is set does it return the first id in
    the server's returned order. Fails closed with :class:`ModelResolutionError` if the
    server is unreachable, exposes no models, or a configured preferred model is absent,
    so production never silently extracts against an unverified or unavailable model.
    """
    models_url = _models_list_url(endpoint_url)
    fetch = transport or _lmstudio_models_bytes
    try:
        raw = fetch(models_url)
        data = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise ModelResolutionError(
            f"could not resolve active model from {models_url}: {sanitize_diagnostic(str(error))}"
        ) from error
    records = data.get("data") if isinstance(data, dict) else None
    identifiers = [r["id"] for r in records if isinstance(r, dict) and r.get("id")] if records else []
    if not identifiers:
        raise ModelResolutionError(f"LM Studio {models_url} exposed no models")
    if preferred is not None:
        if preferred not in identifiers:
            raise ModelResolutionError(
                f"configured model {preferred!r} not available at {models_url}; "
                f"available ids: {', '.join(identifiers)}"
            )
        return preferred
    return identifiers[0]


def make_production_extractor(
    *,
    endpoint_url: str = "http://localhost:1234/v1/chat/completions",
    model_name: str | None = None,
    extractor_name: str = "lm-studio-production",
    extractor_version: str = "1.0",
    prompt_version: str = PROMPT_VERSION,
    extraction_schema_version: str = EXTRACTION_SCHEMA_VERSION,
    timeout_seconds: float = 180,
    max_attempts: int = _EXTRACTION_RETRIES,
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
    (name/version/prompt-schema/resolved model/endpoint) on the caller's behalf via
    the optional ``identity`` callback so the research stage can stamp them onto the
    result. When ``model_name`` is omitted the active local model is resolved at
    runtime from LM Studio; a missing or unreachable server fails the build closed.
    """

    # Always verify the configured model actually exists at runtime rather than
    # assuming a hard-coded default; fail closed if LM Studio is unreachable, exposes
    # no models, or a configured ``model_name`` is absent. The resolved id + endpoint
    # are reported through ``identity`` for provenance.
    resolved_model_name = resolve_active_model(
        endpoint_url=endpoint_url, preferred=model_name
    )


    def _extract(
        topic: str,
        sources: list[SourceContent],
        *,
        identity: Callable[[str, str, str, str | None, str | None, str | None], None] | None = None,
    ) -> list[Extraction]:
        if not sources:
            raise ResearchError("production extractor called with no source content")

        prompt = (
            "You are a strict claim-extraction tool for a factual documentary. "
            "Read the SOURCE CONTENT below and extract 3-5 DISTINCT, verifiable "
            "statements that could be narrated. Each statement must be grounded in "
            "the source content — do not invent facts, statistics, or quotes.\n\n"
            "For each statement output exactly one line:\n"
            "[classification] | claim text || \"verbatim quote from the source\"\n\n"
            "The verbatim quote MUST appear word-for-word in the source content and "
            "is the evidence that supports the claim. Classifications are EXACTLY one "
            "of: confirmed fact, reported claim, estimate, opinion, analysis/speculation. "
            "Anything else is invalid.\n\n"
            f"TOPIC: {topic}\n\n"
            "SOURCE CONTENT:\n"
        )
        contents: dict[str, str] = {}
        for src in sources:
            contents[src.url] = (src.content or "")
            prompt += f"\n--- Source: {src.url} ---\n{src.content}\n"

        import urllib.request

        payload = json.dumps({
            "model": resolved_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.0,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            endpoint_url, data=payload, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        def _parse(raw_content: str) -> list[Extraction]:
            """Parse model output into validated extractions; fail closed on any slip."""
            # Each claim must carry a verbatim evidence quote that actually appears in the
            # source it was drawn from. This makes every retained claim auditable: without
            # its exact supporting text we refuse to promote it as a verified fact.
            extractions: list[Extraction] = []
            for raw_line in raw_content.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue  # tolerate blank lines / trailing newline only
                # Tolerate common model slips of omitting brackets around the label and/or
                # using a dash instead of the double-pipe before the quote (e.g. `confirmed
                # fact | claim || "q"`, `[reported claim] | c - "q"`). Only lines already
                # carrying full claim structure are rewritten to canonical form; the strict
                # grammar, valid-classification set, and verbatim-grounding checks below still
                # apply unchanged, so nothing that would fail-closed is now accepted.
                if not _EVIDENCE_LINE_RE.match(stripped):
                    tolerant = _TOLERANT_RE.match(stripped)
                    if tolerant:
                        stripped = (
                            f"[{tolerant.group('cls').strip().lower()}] | "
                            f"{tolerant.group('claim').strip()} || "
                            f'"{tolerant.group('quote').strip()}"'
                        )
                match = _EVIDENCE_LINE_RE.match(stripped)
                if not match:
                    # Any nonblank line that is not a well-formed claim line fails closed.
                    raise ResearchError(
                        "production extractor produced a malformed output line; "
                        "refusing to promote non-conforming output as verified facts"
                    )
                cls = match.group("cls").strip().lower()
                if cls not in _VALID_CLASSIFICATIONS:
                    # Malformed classification -> reject the whole extraction (fail closed).
                    raise ResearchError(
                        f"production extractor produced invalid classification {cls!r}; "
                        "refusing to promote non-conforming output as verified facts"
                    )
                claim = match.group("claim").strip().rstrip(".")
                quote = match.group("quote").strip()
                if not claim or not quote:
                    raise ResearchError(
                        "production extractor produced a claim without claim text or "
                        "evidence quote; refusing to promote non-conforming output"
                    )
                # Require the verbatim quote to be a substring of one of the source's
                # clean excerpts (whitespace-normalised). A quote absent from the sourced
                # material means the claim is not actually grounded -> fail closed.
                norm_quote = _normalize_ws(quote)
                if not any(norm_quote in _normalize_ws(body) for body in contents.values()):
                    raise ResearchError(
                        f"production extractor produced a quote not present in any source "
                        f"({quote[:60]!r}); claim is not grounded and is rejected"
                    )
                extractions.append(Extraction(claim=claim, quote=quote, classification=cls))

            if not extractions:
                raise ResearchError(
                    "production extractor returned no claims with verbatim evidence; "
                    "insufficient supported material"
                )
            return extractions

        accepted: list[Extraction] | None = None
        last_error: Exception | None = None
        for _attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except _HTTPError as exc:
                # 5xx is transient (server overload) -> retry; 4xx fails closed immediately.
                if 500 <= exc.code < 600:
                    last_error = exc
                    continue
                raise ResearchError(f"model POST failed: HTTP {exc.code}") from exc
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            try:
                accepted = _parse(content)
                break
            except ResearchError as exc:
                # Non-conforming model output is transient (temperature=0 variance); retry.
                last_error = exc

        if accepted is None:
            raise last_error if last_error is not None else ResearchError(
                "production extractor returned no claims with verbatim evidence"
            )

        if identity is not None:
            identity(
                extractor_name,
                extractor_version,
                extraction_schema_version,
                prompt_version=prompt_version,
                model_id=resolved_model_name,
                endpoint_url=endpoint_url,
            )

        # Expose the resolved model/endpoint so the corroboration pass fingerprints
        # the same active local model (never a hard-coded default).
        _extract.resolved_model_name = resolved_model_name  # type: ignore[attr-defined]
        _extract.endpoint_url = endpoint_url  # type: ignore[attr-defined]

        return accepted

    return _extract


def make_deterministic_production_extractor(
    *,
    extractor_name: str = "deterministic-production",
    extractor_version: str = "1.0",
    prompt_version: str = PROMPT_VERSION,
    extraction_schema_version: str = EXTRACTION_SCHEMA_VERSION,
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
        identity: Callable[[str, str, str, str | None, str | None, str | None], None] | None = None,
    ) -> list[Extraction]:
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

        # Emit a small bounded set of *distinct* grounded sentences per source so the
        # daily-job gate (>= MIN_VERIFIED_FACTS verified facts) is reachable even when
        # corroboration merges identical claims across sources. Dedup by normalized text
        # *per source*: the same sentence appearing in multiple sources is exactly the
        # independent corroboration we need, so a global set must not suppress it.
        MAX_EXTRACTIONS_PER_SOURCE = 5
        extractions: list[Extraction] = []
        for src in sources:
            seen_norm: set[str] = set()
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", src.content) if len(s.strip()) > 12]
            emitted = 0
            for sentence in sentences:
                if emitted >= MAX_EXTRACTIONS_PER_SOURCE:
                    break
                if not _is_grounded(sentence):
                    continue
                norm = _normalize_ws(sentence)
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                # The sentence is verbatim from the source, so it doubles as its own
                # auditable evidence quote.
                extractions.append(Extraction(
                    claim=sentence.rstrip("."),
                    quote=sentence,
                    classification="confirmed fact",
                ))
                emitted += 1

        if not extractions:
            raise ResearchError(
                f"no distinct grounded facts extracted from {len(sources)} source(s); "
                "insufficient supported material for a long-form documentary"
            )

        if identity is not None:
            identity(
                extractor_name,
                extractor_version,
                extraction_schema_version,
                prompt_version=prompt_version,
                model_id=None,
                endpoint_url=None,
            )

        return extractions

    return _extract


# Maximum characters of clean, topic-relevant excerpt fed to the local model per
# source. The local LM Studio model has a tight context window and degrades (empty
# output / timeout) on large input, so each source is reduced to a bounded excerpt
# rather than concatenated with every other source into one oversized prompt.
_EXCERPT_MAX_CHARS = 2_500

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {"what", "which", "when", "where", "why", "how", "this", "that", "these",
     "those", "with", "without", "into", "upon", "their", "there", "here",
     "about", "after", "before", "more", "most", "some", "such", "than", "then"}
)
# Domain lexicon for the pilot topic (water beyond Earth / solar-system science).
_WATER_SPACE_WORDS = frozenset(
    {"water", "ice", "ocean", "subsurface", "liquid", "hydrogen", "oxidizer",
     "rover", "mission", "nasa", "mars", "europa", "enceladus", "lunar", "moon",
     "jupiter", "saturn", "cassini", "juno", "clipper", "crater", "polar",
     "glacier", "river", "lake", "vapor", "humidity", "propellant", "isru",
     "solvent", "habitable", "habitability", "life"}
)


def _topic_keywords(topic: str) -> set[str]:
    """Significant words from the topic (>=4 chars, no stopwords)."""
    return {w for w in _WORD_RE.findall(topic.lower()) if len(w) >= 4 and w not in _STOPWORDS}


@dataclass(frozen=True)
class SelectedChunk:
    """A bounded, topic-relevant slice of one source with auditable provenance.

    ``text`` is the selected excerpt (topic-bearing sentences joined by single spaces).
    ``source_ranges`` is the ordered list of half-open ``(start, end)`` character spans
    into the *original fetched* :class:`SourceContent.content` that produced each
    selected sentence in order -- so an auditor can slice the raw source and see exactly
    what grounded a claim. ``id`` is a full SHA-256 over those ranges plus the excerpt
    text: stable for identical content, invalidating when it changes.
    """

    text: str
    id: str
    source_ranges: tuple[tuple[int, int], ...]


_SCRIPT_STYLE_RE = re.compile(r"(?s)<(script|style)[^>]*>.*?</\1>")
_TAG_RE = re.compile(r"<[^>]+>")


def _normalize_with_offsets(content: str) -> tuple[str, list[int], list[int]]:
    """Normalize fetched text -- drop script/style blocks and tags, collapse whitespace
    -- while recording, for each output character, the raw ``(start, end_inclusive)`` span
    in ``content`` it came from. Consecutive whitespace/tag/script regions collapse to a
    single space whose raw span covers all of them, so any selected normalized span maps
    back to the exact original bytes rather than to derived/normalized positions.
    """
    src = content or ""
    n = len(src)
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []

    def emit_space(raw_start: int, raw_end_inclusive: int) -> None:
        # Merge with a previously emitted space so whitespace/tag runs collapse to one.
        if out and out[-1] == " ":
            ends[-1] = raw_end_inclusive
        else:
            out.append(" ")
            starts.append(raw_start)
            ends.append(raw_end_inclusive)

    i = 0
    while i < n:
        m_script = _SCRIPT_STYLE_RE.match(src, i)
        if m_script:
            emit_space(m_script.start(), m_script.end() - 1)
            i = m_script.end()
            continue
        m_tag = _TAG_RE.match(src, i)
        if m_tag:
            emit_space(m_tag.start(), m_tag.end() - 1)
            i = m_tag.end()
            continue
        if src[i].isspace():
            j = i
            while j < n and src[j].isspace():
                j += 1
            emit_space(i, j - 1)
            i = j
            continue
        out.append(src[i])
        starts.append(i)
        ends.append(i)
        i += 1

    # Trim leading/trailing collapsed spaces to match str.strip() of the original path.
    while out and out[-1] == " ":
        out.pop()
        starts.pop()
        ends.pop()
    while out and out[0] == " ":
        out.pop(0)
        starts.pop(0)
        ends.pop(0)
    return "".join(out), starts, ends


def _clean_excerpt(
    content: str, topic: str, max_chars: int = _EXCERPT_MAX_CHARS
) -> SelectedChunk:
    """Reduce raw fetched page text to a bounded, water/space-relevant excerpt.

    Strips scripts/styles/HTML tags, splits into sentences, then keeps the densest
    topic-bearing sentences up to ``max_chars`` and fills any remaining budget with
    other substantive (non-navigation) sentences. This keeps each source within the
    local model's tight context window while focusing on substantive content rather
    than page chrome -- without discarding every non-keyworded sentence when a single
    high-scoring one would otherwise consume the whole excerpt budget.

    Returns a :class:`SelectedChunk` whose ``source_ranges`` are exact half-open character
    spans into the *original fetched* ``content`` for each selected sentence in order, and
    whose ``id`` is a stable full SHA-256 over those spans plus the excerpt text. Together
    they pin exactly which slice of the source produced each claim, so provenance stays
    auditable (and invalidates when underlying content changes).
    """
    normalized, raw_starts, raw_ends = _normalize_with_offsets(content)
    if not normalized:
        return SelectedChunk(text="", id="", source_ranges=())

    # Record every candidate sentence's character span in the normalized source up
    # front (cursor advances for all candidates so offsets never regress), then
    # score/dedup/fit against the bounded budget below.
    candidates: list[tuple[str, int, int]] = []
    cursor = 0
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        start = normalized.find(sentence, cursor)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        cursor = max(cursor, end)
        candidates.append((sentence, start, end))

    keywords = _WATER_SPACE_WORDS | _topic_keywords(topic)
    scored: list[tuple[int, str, int, int]] = []  # (hits, sentence, start, end)
    filler: list[tuple[str, int, int]] = []        # (sentence, start, end)
    seen: set[str] = set()
    for sentence, start, end in candidates:
        if len(sentence.split()) < 6:
            continue
        key = re.sub(r"\s+", " ", sentence.strip()).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        hits = sum(1 for w in _WORD_RE.findall(sentence.lower()) if w in keywords)
        if hits > 0:
            scored.append((hits, sentence, start, end))
        else:
            filler.append((sentence, start, end))

    parts: list[str] = []
    total = 0
    ranges: list[tuple[int, int]] = []

    def _fit(sentence: str, start: int, end: int) -> bool:
        nonlocal total
        if total + len(sentence) + 1 > max_chars:
            return False
        parts.append(sentence)
        total += len(sentence) + 1
        # Map the normalized sentence span to a half-open raw span into the original
        # fetched content so an auditor can slice ``content[start:end]`` directly.
        ranges.append((raw_starts[start], raw_ends[end - 1] + 1))
        return True

    # Continue (not break): a single oversized sentence must not suppress later,
    # smaller evidence that still fits the bounded extraction budget.
    for _, sentence, start, end in sorted(scored, key=lambda item: item[0], reverse=True):
        if not _fit(sentence, start, end):
            continue
    for sentence, start, end in filler:
        if not _fit(sentence, start, end):
            continue

    chunk_payload = (
        f"{len(ranges)}\x00"
        + "|".join(f"{s}-{e}" for s, e in ranges)
        + "\x00"
        + " ".join(parts)
        + "\x00"
    )
    chunk_id = hashlib.sha256(chunk_payload.encode("utf-8")).hexdigest()
    return SelectedChunk(text=" ".join(parts), id=chunk_id, source_ranges=tuple(ranges))


_CLASS_ORDER: dict[str, int] = {
    "confirmed fact": 0, "reported claim": 1, "estimate": 2,
    "opinion": 3, "analysis/speculation": 4,
}


def _build_claims_from_records(records: list[EvidenceRecord]) -> list[Claim]:
    """Merge per-source evidence records into claims via auditable corroboration.

    Records are grouped by normalized claim text (whitespace-collapsed, lowercased,
    trailing period stripped). A group is promoted only when it is backed by >=2
    distinct NASA URLs with supported labels -- genuine cross-source corroboration,
    not a model heuristic. Unmatched records become single-source claims so no sourced
    material is dropped. Every retained claim carries an auditable verbatim evidence
    quote (validated as a substring of its source during extraction).
    """
    groups: list[dict[str, Any]] = []
    for r in records:
        norm = _normalize_ws(r.claim)
        if not norm:
            continue
        group = next((g for g in groups if g["norm"] == norm), None)
        if group is None:
            group = {"norm": norm, "members": []}
            groups.append(group)
        group["members"].append(r)

    claims: list[Claim] = []
    counter = 0
    for g in groups:
        members = g["members"]
        supported = [m for m in members if m.label in _SUPPORTED_LABELS]
        urls = sorted({m.url for m in members})
        supported_urls = sorted({m.url for m in supported})
        # Corroboration (and promotion) requires >=2 distinct independent sources.
        if len(supported_urls) >= 2:
            cls = min(
                (m.label for m in supported),
                key=lambda c: _CLASS_ORDER.get(c, 9),
            )
            source_ids = supported_urls
        else:
            # Not corroborated; keep with its strongest available label. verify_claims
            # will never promote these to verified facts.
            cls = min(
                (m.label for m in members),
                key=lambda c: _CLASS_ORDER.get(c, 9),
            )
            source_ids = urls
        counter += 1
        claims.append(Claim(
            claim_id=f"claim-{counter:02d}",
            text=members[0].claim,
            source_ids=source_ids,
            provisional_classification=cls,
            evidence=[
                {
                    "url": m.url,
                    "label": m.label,
                    "quote": m.evidence_quote,
                    "chunk_id": m.chunk_id,
                    "chunk_ranges": [list(r) for r in m.chunk_ranges],
                }
                for m in members
            ],
        ))
    return claims


def extract_from_source_content(
    topic: str,
    sources: list[SourceContent],
    *,
    extractor: ProductionExtractor | None = None,
    identity: Callable[[str, str, str, str | None, str | None], None] | None = None,
    droppable_urls: frozenset[str] = frozenset(),
    drops: list[SourceDrop] | None = None,
) -> tuple[list[Claim], dict[str, str]]:
    """Extract claims grounded in persisted source content.

    Extraction runs **per source** against a bounded, topic-relevant excerpt (the
    local model has a tight context window and degrades on large input), so each
    claim is tagged only with the URL that actually produced it -- genuine per-source
    provenance rather than tagging every claim with every input URL. Claims from
    independent sources that are semantically similar are then merged into single
    corroborated claims carrying all their supporting URLs.

    Returns ``(claims, source_content_hashes)``. The hashes cover the raw fetched
    bytes and participate in the research-stage fingerprint so any change to an
    underlying source invalidates prior claims. Raises :class:`ResearchError` when
    extraction fails or produces no supported material (fail closed).
    """

    if extractor is None:
        # Default production extractor (LM Studio loopback, schema-constrained).
        extractor = make_production_extractor()

    # Build one auditable record per extracted claim, tagged with the source URL that
    # produced it. Extraction runs per source against a bounded excerpt (the local
    # model has a tight context window), so each record's provenance is genuine and
    # its verbatim evidence quote was already validated as a substring of that source.
    records: list[EvidenceRecord] = []
    rec_counter = 0
    # Sources that survived BOTH fetch and extraction; provenance, hashes, and the brief
    # are built from these only, so a dropped source never falsely appears to support the
    # film.
    retained: list[SourceContent] = []
    dropped_seen: set[str] = set()
    for sc in sources:
        try:
            chunk = _clean_excerpt(sc.content, topic)
            single = SourceContent(
                url=sc.url, title=sc.title, quality=sc.quality,
                content=chunk.text, content_hash=sc.content_hash,
            )
            # Central grounding gate: applies to EVERY extractor (not just the LM Studio
            # one), so an injected/custom extractor cannot bypass the verbatim-evidence
            # requirement. Each claim must carry a valid label and a quote that actually
            # appears in this source's selected chunk; anything else fails closed.
            for ext in extractor(topic, [single], identity=identity):
                if ext.classification not in _VALID_CLASSIFICATIONS:
                    raise ResearchError(
                        f"extractor produced invalid classification {ext.classification!r}; "
                        "refusing to promote non-conforming output as verified facts"
                    )
                norm_claim = _collapse_ws(ext.claim)
                norm_quote = _collapse_ws(ext.quote)
                if not norm_claim or not norm_quote:
                    raise ResearchError(
                        "extractor produced a claim without claim text or evidence quote; "
                        "refusing to promote non-conforming output"
                    )
                # Strengthened central grounding gate (applies to EVERY extractor, not
                # just the LM Studio one): the quote must occur wholly within at least ONE
                # selected raw slice, so a fabricated quote spanning the join of two
                # reordered/non-contiguous sentences cannot pass. Each raw slice is normalized
                # with _normalize_with_offsets (removing script/style bodies + tags, collapsing
                # whitespace) exactly as the excerpt was built, so markup never causes a false
                # rejection and script junk can't supply a matching quote. The empty-quote check
                # above still rejects whitespace-only quotes (which would match every slice).
                slices = [
                    _normalize_with_offsets(sc.content[s:e])[0]
                    for s, e in chunk.source_ranges
                ]
                if not any(norm_quote in sl for sl in slices):
                    raise ResearchError(
                        f"extractor produced a quote not present wholly within source "
                        f"{sc.url!r} chunk slice; claim is not grounded and is rejected "
                        f"({norm_quote[:60]!r})"
                    )
                rec_counter += 1
                records.append(EvidenceRecord(
                    id=f"rec-{rec_counter:03d}",
                    url=sc.url,
                    label=ext.classification,
                    claim=ext.claim,
                    evidence_quote=ext.quote,
                    chunk_id=chunk.id,
                    chunk_ranges=list(chunk.source_ranges),
                ))
            retained.append(sc)
        except _SourceDeadlineExhausted as exc:
            # Extraction-stage deadline exhaustion: skip ONLY flagged sources and continue
            # the same run. Other ResearchError (malformed schema / ungrounded quote) still
            # fails closed immediately -- it is not a transient budget failure.
            if sc.url not in droppable_urls or sc.url in dropped_seen:
                raise
            dropped_seen.add(sc.url)
            if drops is not None:
                drops.append(SourceDrop(
                    url=sc.url,
                    stage="extraction",
                    limit_seconds=getattr(exc, "limit_seconds", 0.0),
                    attempts=getattr(exc, "attempts", 1),
                    elapsed_seconds=getattr(exc, "elapsed_seconds", 0.0),
                    detail=getattr(exc, "detail", str(exc)),
                ))
            continue

    if not records:
        raise ResearchError(
            "production extractor returned no claims across all sources; "
            "insufficient supported material"
        )

    claims = _build_claims_from_records(records)
    content_hashes = {s.url: s.content_hash for s in retained}
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

    A claim is verified when it (1) is a confirmed fact or reported claim, (2) is
    not involved in an unresolved contradiction, and (3) is independently
    corroborated by at least two distinct sources. Estimates/opinions/speculation
    are never promoted (they carry uncertainty for the writer). Corroboration via
    source count prevents reporting a single-source statement as a verified fact.
    """
    contradicted = {cid for c in contradictions for cid in c.claim_ids}
    verified: list[Claim] = []
    for claim in claims:
        if claim.claim_id in contradicted:
            continue
        if claim.provisional_classification not in {"confirmed fact", "reported claim"}:
            continue
        if len(set(claim.source_ids)) < 2:
            continue
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
    production_extractor: Callable[[str, list[SourceContent]], "list[Extraction]"] | None = None,
    content_fetcher: Callable[[str], str | bytes] | None = None,
    source_contents: list[SourceContent] | None = None,
    observed_at: str | None = None,
    droppable_urls: frozenset[str] = frozenset(),
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

    ``droppable_urls`` lists sources that may be skipped (rather than fail the run)
    when they exhaust their bounded per-attempt deadline budget during fetch or
    extraction; such sources are recorded in ``result.dropped_sources`` and excluded
    from provenance, hashes, content, and the brief. Every other source still fails
    closed on deadline exhaustion.
    """
    observed_at = observed_at or datetime.now(UTC).isoformat()
    sources = collect_sources(source_urls)

    identity_holder: dict[str, str | None] = {"name": None, "version": None, "schema": None, "prompt_version": None, "model_id": None, "endpoint_url": None}
    contents: list[SourceContent] = []
    # Structured diagnostics for sources skipped due to exhausting their bounded
    # per-attempt deadline budget during fetch or extraction. Retained on the result so
    # provenance never falsely credits a dropped source and the operator keeps why/where.
    drops: list[SourceDrop] = []

    # Production path: extract against persisted source content.
    if production_extractor is not None or source_contents is not None or content_fetcher is not None:
        if not sources and not source_contents:
            raise ResearchError("no source URLs provided for research")

        contents = list(source_contents) if source_contents is not None else []
        if not contents:
            for rec in sources:
                try:
                    contents.append(
                        fetch_source_content(rec.url, rec.title or "", rec.quality, transport=content_fetcher)
                    )
                except _SourceDeadlineExhausted as exc:
                    # Fetch-stage deadline exhaustion: skip ONLY flagged sources (e.g. Mars)
                    # and continue the same run; other sources fail closed.
                    if rec.url not in droppable_urls:
                        raise
                    drops.append(SourceDrop(
                        url=rec.url,
                        stage="fetch",
                        limit_seconds=getattr(exc, "limit_seconds", 0.0),
                        attempts=getattr(exc, "attempts", 1),
                        elapsed_seconds=getattr(exc, "elapsed_seconds", 0.0),
                        detail=getattr(exc, "detail", str(exc)),
                    ))
                    continue

        claims, content_hashes = extract_from_source_content(
            topic,
            contents,
            extractor=production_extractor or make_production_extractor(),
            identity=lambda name, ver, ps, prompt_version=None, model_id=None, endpoint_url=None: identity_holder.update({"name": name, "version": ver, "schema": ps, "prompt_version": prompt_version, "model_id": model_id, "endpoint_url": endpoint_url}),
            droppable_urls=droppable_urls,
            drops=drops,
        )
    else:
        # Legacy path (tests / dry-run inject RULE_BASED_EXTRACTOR via extractor=).
        claims = extract_claims(topic, sources, extractor=extractor or _default_extractor)
        content_hashes = {}

    contradictions = detect_contradictions(claims)
    verified = verify_claims(claims, contradictions)
    reasons = flag_sensitive_topic(topic)

    # Output digest: stable hash over the COMPLETE extraction structure -- every
    # claim's text, classification, supporting URLs, and auditable evidence quotes --
    # plus the per-source content hashes. Any change to extracted material, its
    # evidence, or an underlying source invalidates prior runs.
    claims_data = sorted((c.to_dict() for c in claims), key=lambda c: c["claim_id"])
    digest_inputs = json.dumps(claims_data, sort_keys=True)
    # Dropped sources are part of the provenance fingerprint too: two runs with identical
    # retained claims/hashes but materially different source failures must not collide.
    drops_digest = json.dumps(
        sorted(
            (
                {
                    "url": d.url,
                    "stage": d.stage,
                    "attempts": d.attempts,
                    "limit_seconds": d.limit_seconds,
                    "elapsed_seconds": d.elapsed_seconds,
                    "detail": d.detail,
                }
                for d in drops
            ),
            key=lambda x: (x["url"], x["stage"]),
        ),
        sort_keys=True,
    )
    output_digest = hashlib.sha256(
        f"{topic}|{digest_inputs}|{drops_digest}|{json.dumps(content_hashes, sort_keys=True)}".encode("utf-8")
    ).hexdigest()

    # Sources actually retained after bounded per-source deadline handling (fetch +
    # extraction). Provenance, hashes, the brief, and downstream asset planning use these
    # only -- a dropped source must never appear to support the film.
    dropped_url_set = {d.url for d in drops}
    retained_sources = [s for s in sources if s.url not in dropped_url_set]

    # Persisted source content keyed by URL (production path only), excluding any dropped
    # source so the fingerprint and narration material cover retained sources only.
    contents_by_url: dict[str, SourceContent] = {
        c.url: c for c in contents if c.url not in dropped_url_set
    } if contents else {}

    brief_lines = [f"# Research Brief: {topic}", "", f"Generated: {observed_at}", ""]
    brief_lines.append("## Sources")
    for source in retained_sources:
        brief_lines.append(f"- [{source.quality}] {source.url}")
    if dropped_url_set:
        brief_lines += ["", "## Skipped Sources (over per-attempt deadline)", ""]
        for d in drops:
            brief_lines.append(
                f"- [{d.stage}] {d.url}: exceeded {d.limit_seconds}s after {d.attempts} attempt(s) "
                f"({d.elapsed_seconds}s); detail: {d.detail}"
            )
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
        sources=retained_sources,
        claims=claims,
        contradictions=contradictions,
        verified_facts=verified,
        research_brief="\n".join(brief_lines) + "\n",
        flagged_for_review=requires_human_review(reasons) or bool(drops),
        flag_reasons=list(reasons),
        extractor_name=identity_holder["name"],
        extractor_version=identity_holder["version"],
        extraction_schema_version=identity_holder["schema"],
        prompt_version=identity_holder["prompt_version"],
        extractor_model_id=identity_holder["model_id"],
        extractor_endpoint=identity_holder["endpoint_url"],
        source_content_hashes=content_hashes,
        output_digest=output_digest,
        source_contents=contents_by_url,
        dropped_sources=drops,
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
