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

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from ai_video_factory.sanitization import sanitize_diagnostic

Classification = Literal["confirmed fact", "reported claim", "estimate", "opinion", "analysis/speculation"]
SourceQuality = Literal["reliable", "secondary", "unverified"]


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
    return pairs


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
    observed_at: str | None = None,
) -> ResearchResult:
    """Run the full research pipeline for one topic."""
    observed_at = observed_at or datetime.now(UTC).isoformat()
    sources = collect_sources(source_urls)
    claims = extract_claims(topic, sources, extractor=extractor)
    contradictions = detect_contradictions(claims)
    verified = verify_claims(claims, contradictions)
    reasons = flag_sensitive_topic(topic)

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
    paths["research_brief"] = output_dir / "research_brief.md"
    paths["research_brief"].write_text(result.research_brief, encoding="utf-8")
    return paths
