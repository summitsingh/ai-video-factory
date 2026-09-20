"""Source URL verification and credible source seeding.

Every citation URL that reaches the final render must be a live page:
each URL gets an HTTP HEAD request (falling back to GET when the server
rejects HEAD), must return 200 after redirects, and dead URLs are dropped
and logged. Placeholder URLs (example.com, lorem, ...) can never reach a
render: ``guard_no_placeholders`` raises ``SourceVerificationError`` the
moment one is detected, failing the run loud instead of publishing a
video that cites a fake source.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class SourceVerificationError(RuntimeError):
    """Raised when a placeholder source URL is detected or verification fails.

    Fail-loud by design: a placeholder citation must abort the run, never
    silently reach the final render.
    """


# Tokens that mark a URL as a placeholder rather than a real citation.
# Matched against the lowercased host and path of each URL.
_PLACEHOLDER_TOKENS = (
    "example.com",
    "example.org",
    "example.net",
    "example.edu",
    "example.io",
    "placeholder",
    "lorem",
    "your-url",
    "your_url",
    "insert-url",
    "insert_url",
    "changeme",
    "change-me",
    "todo-url",
    "fake-url",
    "foo.bar",
    "test.invalid",
)

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _looks_like_placeholder(url: str) -> str | None:
    """Return the matched placeholder token, or None when the URL looks real."""
    lowered = (url or "").strip().lower()
    for token in _PLACEHOLDER_TOKENS:
        if token in lowered:
            return token
    return None


def guard_no_placeholders(urls: list[str] | None) -> None:
    """Fail loud if any URL is a placeholder.

    Raises ``SourceVerificationError`` naming the offending URL and the
    matched token. This is the last line of defense before the render:
    call it on the final source list in the pipeline's script step.
    """
    for url in urls or []:
        token = _looks_like_placeholder(url)
        if token is not None:
            raise SourceVerificationError(
                f"placeholder source URL detected ({url!r} matches {token!r}); "
                "refusing to render a video that cites a fake source"
            )


def _request_status(url: str, method: str, timeout: float) -> int | None:
    """Return the HTTP status for one request, or None on transport failure."""
    request = urllib.request.Request(
        url, method=method, headers={"User-Agent": _BROWSER_UA}
    )
    try:
        # HTTPRedirectHandler follows 301/302/303/307 automatically, so the
        # status we see is the final one after redirects.
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as error:
        # HTTPError still carries the real status code (404, 403, 405...).
        return error.code
    except Exception:
        return None


def verify_url(url: str, timeout: float = 15.0) -> bool:
    """Check that a citation URL is live.

    Sends HEAD first (cheap), falling back to GET when the server rejects
    HEAD (405/501) or otherwise refuses it. Returns True only when the
    final status after redirects is 200.
    """
    url = (url or "").strip()
    if not url or not re.match(r"^https?://", url, re.IGNORECASE):
        return False
    status = _request_status(url, "HEAD", timeout)
    if status in (405, 501, 403, 400):
        # Server rejects HEAD: retry with GET before giving up.
        status = _request_status(url, "GET", timeout)
    return status == 200


@dataclass
class SourceVerificationReport:
    """Outcome of verifying a script's citation URLs."""

    live: list[str] = field(default_factory=list)
    dead: list[str] = field(default_factory=list)
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def dropped(self) -> list[str]:
        return self.dead


def verify_sources(
    urls: list[str] | None, timeout: float = 15.0
) -> SourceVerificationReport:
    """Verify every citation URL; return the live ones and log the dead.

    Dead URLs are dropped from the returned ``live`` list and logged with
    their status so the run record shows exactly what was removed.
    """
    report = SourceVerificationReport()
    seen: set[str] = set()
    for url in urls or []:
        url = (url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        if verify_url(url, timeout=timeout):
            report.live.append(url)
            log.info("source OK: %s", url)
        else:
            report.dead.append(url)
            log.warning("source DEAD, dropping citation: %s", url)
    log.info(
        "source verification: %d live, %d dropped",
        len(report.live),
        len(report.dead),
    )
    return report


# Credible, verified-live Fermi Paradox sources. Every URL below returned
# HTTP 200 (after redirects) when seeded on 2026-09-20; the verification
# step re-checks them at generation time and drops any that went dead.
FERMI_PARADOX_SOURCES: tuple[str, ...] = (
    # Overview + the canonical "where is everybody?" framing.
    "https://en.wikipedia.org/wiki/Fermi_paradox",
    # The institute actually running the search; plain-language explainer.
    "https://www.seti.org/research/seti-101/fermi-paradox/",
    # NASA astrobiology: the agency's search-for-life program home.
    "https://astrobiology.nasa.gov/",
    # Corbet (2025), "A Less Terrifying Universe? Mundanity as an Explanation
    # for the Fermi Paradox" (Open J. Astrophys.): a recent peer-reviewed
    # review of paradox explanations.
    "https://arxiv.org/abs/2509.22878v2",
    # Likavcan (2025), "The Grass of the Universe": the sustainability
    # solution to the Fermi paradox (Haqq-Misra/Baum lineage).
    "https://arxiv.org/abs/2411.08057v2",
)


def seed_fermi_sources() -> list[str]:
    """Return the credible Fermi Paradox seed sources (verified live)."""
    return list(FERMI_PARADOX_SOURCES)


def is_fermi_topic(topic: str | None) -> bool:
    """True when the topic is about the Fermi paradox / SETI silence."""
    lowered = (topic or "").lower()
    return "fermi" in lowered or (
        "where is everybody" in lowered and "paradox" in lowered
    )
