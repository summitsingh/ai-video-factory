"""Tests for the research/claim-verification pipeline (offline, fixture-based)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_video_factory.research_pipeline import (
    Claim,
    Contradiction,
    Extraction,
    ResearchResult,
    SourceDrop,
    _SourceDeadlineExhausted,
    classify_source_quality,
    collect_sources,
    detect_contradictions,
    extract_claims,
    flag_sensitive_topic,
    run_research,
    verify_claims,
    write_research_artifacts,
)


def test_classify_source_quality() -> None:
    assert classify_source_quality("https://nasa.gov/image") == "reliable"
    assert classify_source_quality("https://en.wikipedia.org/wiki/X") == "reliable"
    assert classify_source_quality("https://blog.wordpress.com/post") == "secondary"
    assert classify_source_quality("https://example.com/x") == "unverified"


def test_collect_sources_skips_blank() -> None:
    sources = collect_sources(["https://nasa.gov/a", "", "   ", "https://reddit.com/r/tech/comments/x"])
    assert len(sources) == 2
    assert sources[0].quality == "reliable"
    assert sources[1].quality == "unverified"


def test_extract_claims_assigns_ids_and_source_refs() -> None:
    sources = collect_sources(["https://nasa.gov/a"])

    def extractor(topic, src):
        return [("The telescope launched in 2021.", "confirmed fact")]

    claims = extract_claims("JWST", sources, extractor=extractor)
    assert len(claims) == 1
    assert claims[0].claim_id == "claim-01"
    assert claims[0].source_ids == ["https://nasa.gov/a"]
    assert claims[0].provisional_classification == "confirmed fact"


def test_extract_claims_falls_back_deterministically() -> None:
    sources = collect_sources(["https://nasa.gov/a"])
    # No extractor passed -> default tries LM Studio (offline) then rule fallback.
    claims = extract_claims("Quantum Computing", sources)
    assert len(claims) >= 1
    # Every claim references the collected source ids.
    for claim in claims:
        assert claim.source_ids


def test_detect_contradictions_finds_opposite_polarity() -> None:
    claims = [
        Claim("claim-01", "Revenue grows every quarter.", ["s1"], "confirmed fact"),
        Claim("claim-02", "The company faces a financial crisis.", ["s1"], "reported claim"),
    ]
    contradictions = detect_contradictions(claims)
    assert len(contradictions) == 1
    assert set(contradictions[0].claim_ids) == {"claim-01", "claim-02"}


def test_verify_claims_promotes_only_supported_and_excludes_contradicted() -> None:
    claims = [
        Claim("claim-01", "Revenue grows.", ["s1"], "confirmed fact"),
        Claim("claim-02", "The company faces a crisis.", ["s1"], "reported claim"),
        Claim("claim-03", "I think it will be fine.", ["s1"], "opinion"),
    ]
    contradictions = detect_contradictions(claims)
    verified = verify_claims(claims, contradictions)
    ids = {c.claim_id for c in verified}
    # claim-03 is an opinion -> never promoted.
    assert "claim-03" not in ids
    # The two contradicting claims are excluded from verified facts.
    assert "claim-01" not in ids and "claim-02" not in ids


def test_flag_sensitive_topic() -> None:
    assert flag_sensitive_topic("upcoming presidential election results")
    assert flag_sensitive_topic("how to invest your savings")
    assert not flag_sensitive_topic("a documentary about rocket launches")


def test_run_research_end_to_end(tmp_path: Path) -> None:
    sources = ["https://nasa.gov/image", "https://reddit.com/r/tech/comments/x"]

    def extractor(topic: str, src: list) -> list[tuple[str, str]]:
        return [
            ("The mission launched successfully.", "confirmed fact"),
            ("Analysts speculate about future funding.", "analysis/speculation"),
        ]

    result = run_research("Deep Space Mission", sources, extractor=extractor)  # type: ignore[arg-type]
    assert isinstance(result, ResearchResult)
    assert len(result.claims) == 2
    assert len(result.verified_facts) == 1
    assert result.verified_facts[0].provisional_classification == "confirmed fact"
    assert result.research_brief.count("Deep Space Mission") >= 1

    paths = write_research_artifacts(result, tmp_path)
    for key in ("sources", "claims", "contradictions", "verified_facts"):
        data = json.loads(paths[key].read_text())
        assert isinstance(data, list)
    assert (tmp_path / "research_brief.md").is_file()


def test_run_research_flags_sensitive_topic(tmp_path: Path) -> None:
    result = run_research("2026 election fraud allegations", ["https://nasa.gov/a"])
    assert result.flagged_for_review is True
    assert result.flag_reasons
    # Sensitive topics must not silently promote unverified claims.
    assert "Human Review Required" in result.research_brief


def test_run_research_drops_fetch_deadline_source(tmp_path: Path) -> None:
    """A source that exhausts its fetch deadline is skipped (not failed closed),
    recorded in dropped_sources, and excluded from provenance/hashes/brief."""
    mars = "https://images-api.nasa.gov/item/nasa-456"
    europa = "https://images-api.nasa.gov/item/europa-ocean"

    def fetcher(url: str) -> bytes:
        if url == mars:
            raise _SourceDeadlineExhausted("fetch timed out")
        return b"Europa has a subsurface ocean of liquid water."

    def extractor(topic: str, src: list, identity=None):
        for s in src:
            yield Extraction(
                claim="Europa harbors a subsurface ocean.",
                quote="Europa has a subsurface ocean of liquid water.",
                classification="confirmed fact",
            )

    result = run_research(
        "Water in the outer solar system",
        [mars, europa],
        content_fetcher=fetcher,
        production_extractor=extractor,  # type: ignore[arg-type]
        droppable_urls=frozenset({mars}),
    )
    assert len(result.dropped_sources) == 1
    drop = result.dropped_sources[0]
    assert isinstance(drop, SourceDrop)
    assert drop.url == mars and drop.stage == "fetch"
    # Europa retained; Mars absent from provenance.
    assert [s.url for s in result.sources] == [europa]
    assert europa in result.source_content_hashes
    assert mars not in result.source_content_hashes
    # The dropped URL is recorded as a skip but never credited to any claim.
    for line in result.research_brief.splitlines():
        if line.startswith("- claim-"):
            assert mars not in line
    assert "Skipped Sources" in result.research_brief
    # Only the retained source produced a claim.
    assert len(result.claims) == 1


def test_run_research_fails_closed_on_non_droppable_deadline(tmp_path: Path) -> None:
    """A non-droppable source that exhausts its deadline fails the whole run."""
    europa = "https://images-api.nasa.gov/item/europa-ocean"

    def fetcher(url: str) -> bytes:
        raise _SourceDeadlineExhausted("fetch timed out")

    def extractor(topic: str, src: list, identity=None):
        yield Extraction("x", "x", "confirmed fact")

    with pytest.raises(_SourceDeadlineExhausted):
        run_research(
            "Water in the outer solar system",
            [europa],
            content_fetcher=fetcher,
            production_extractor=extractor,  # type: ignore[arg-type]
        )


def test_run_research_drops_extraction_deadline_source(tmp_path: Path) -> None:
    """A source that exhausts its extraction deadline is skipped and recorded,
    while retained sources still yield claims."""
    mars = "https://images-api.nasa.gov/item/nasa-456"
    europa = "https://images-api.nasa.gov/item/europa-ocean"

    def fetcher(url: str) -> bytes:
        return b"Europa has a subsurface ocean of liquid water beneath its icy crust."

    def extractor(topic: str, src: list, identity=None):
        for s in src:
            if s.url == mars:
                raise _SourceDeadlineExhausted("extraction timed out")
            yield Extraction(
                claim="Europa harbors a subsurface ocean.",
                quote="Europa has a subsurface ocean",
                classification="confirmed fact",
            )

    result = run_research(
        "Water in the outer solar system",
        [mars, europa],
        content_fetcher=fetcher,
        production_extractor=extractor,  # type: ignore[arg-type]
        droppable_urls=frozenset({mars}),
    )
    assert len(result.dropped_sources) == 1
    assert result.dropped_sources[0].stage == "extraction"
    assert [s.url for s in result.sources] == [europa]
    assert mars not in result.source_content_hashes
    assert europa in result.source_content_hashes
    assert len(result.claims) == 1


def test_run_research_drop_is_auditable_in_digest_and_flag(tmp_path: Path) -> None:
    """A dropped source must surface in the provenance digest and review flag even
    when the retained claims/hashes are otherwise identical to a run with no drop."""
    mars = "https://images-api.nasa.gov/item/nasa-456"
    europa = "https://images-api.nasa.gov/item/europa-ocean"
    content = b"Europa has a subsurface ocean of liquid water beneath its icy crust."

    def extractor(topic: str, src: list, identity=None):
        for s in src:
            yield Extraction(
                claim="Europa harbors a subsurface ocean.",
                quote="Europa has a subsurface ocean",
                classification="confirmed fact",
            )

    # Baseline: only Europa fetched and retained; nothing dropped.
    def ok_fetcher(url: str) -> bytes:
        return content

    base = run_research(
        "Water in the outer solar system", [europa],
        content_fetcher=ok_fetcher, production_extractor=extractor,  # type: ignore[arg-type]
    )
    assert base.dropped_sources == []
    assert base.flagged_for_review is False

    # Same retained evidence (Europa only), but Mars was dropped at fetch.
    def drop_mars_fetcher(url: str) -> bytes:
        if url == mars:
            raise _SourceDeadlineExhausted("fetch timed out")
        return content

    dropped = run_research(
        "Water in the outer solar system", [mars, europa],
        content_fetcher=drop_mars_fetcher, production_extractor=extractor,  # type: ignore[arg-type]
        droppable_urls=frozenset({mars}),
    )
    assert [s.url for s in dropped.sources] == [europa]
    assert dropped.flagged_for_review is True
    # Identical retained claims/hashes but a drop -> digest must differ (auditable).
    assert base.source_content_hashes == dropped.source_content_hashes
    assert [c.claim_id for c in base.claims] == [c.claim_id for c in dropped.claims]
    assert base.output_digest != dropped.output_digest
