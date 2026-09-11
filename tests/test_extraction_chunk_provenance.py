"""Focused tests for bounded context sizing and per-chunk verbatim grounding.

These prove the two invariants the pilot fix must hold: (1) a source far larger than
the extraction budget is reduced to a bounded, relevance-ranked chunk whose claims still
carry quotes found *in that chunk*, and (2) any extractor -- including an injected one --
is rejected when its quoted evidence is absent from the selected chunk.
"""

from __future__ import annotations

import hashlib

import pytest

from ai_video_factory.research_pipeline import (
    Extraction,
    ResearchError,
    SourceContent,
    _clean_excerpt,
    _normalize_with_offsets,
    _collapse_ws,
    _EXCERPT_MAX_CHARS,
    extract_from_source_content,
    make_deterministic_production_extractor,
)
_TOPIC = "Water Beyond Earth: Where It Is and Why It Means We're Looking"


def _source(text: str, url: str = "https://nasa.gov/water") -> SourceContent:
    return SourceContent(
        url=url,
        title="chunk-provenance fixture",
        quality="reliable",
        content=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def test_large_source_is_bounded_and_still_yields_evidence_backed_claims() -> None:
    """A source far larger than the budget is reduced to a bounded chunk; claims pulled
    from that chunk must still carry quotes found *in the chunk* (grounding survives)."""
    big = "\n".join(
        [
            "NASA confirmed water ice existed on Mars in 2021.",
            "Frozen subsurface water was detected below the icy crust of Europa.",
            "Plumes of water vapor eject from the south pole of Enceladus.",
            "Juno has orbited Jupiter since 2016 and mapped subsurface water ice.",
        ]
        + [f"Site navigation link number {i} about unrelated page chrome here." for i in range(400)]
    )
    content = _source(big)

    chunk = _clean_excerpt(content.content, _TOPIC)
    excerpt, ranges, chunk_id = chunk.text, list(chunk.source_ranges), chunk.id
    # Bounded to the extraction budget regardless of how large the raw source was.
    assert len(excerpt) <= _EXCERPT_MAX_CHARS == 2_500
    # Exact selected spans are recorded and index the ORIGINAL raw content (not a
    # derived/normalized copy), so an auditor can slice ``content[start:end]`` directly.
    assert ranges, "expected at least one selected sentence span"
    for start, end in ranges:
        assert 0 <= start < end <= len(content.content)
    # Reconstruction invariant: normalizing each raw slice and joining the results in range
    # order reproduces the excerpt exactly, proving the offsets map back to real source chars
    # (not a derived/normalized copy). Holds for plain text; an HTML/script fixture below
    # proves it when slices contain inline tags and dropped script bodies.
    reconstructed = " ".join(
        _normalize_with_offsets(content.content[s:e])[0] for s, e in ranges
    )
    assert reconstructed == excerpt
    # Determinism: identical content yields an identical full-SHA-256 chunk id (64 hex).
    assert len(chunk_id) == 64 and all(c in "0123456789abcdef" for c in chunk_id)
    assert _clean_excerpt(content.content, _TOPIC).id == chunk_id

    claims, hashes = extract_from_source_content(
        _TOPIC, [content], extractor=make_deterministic_production_extractor()
    )
    assert hashes == {content.url: content.content_hash}
    # Claims were actually produced from the bounded chunk...
    assert claims
    # ...and every claim's evidence quote is a substring of that same bounded chunk, and
    # carries the chunk identity so provenance stays auditable end to end.
    for claim in claims:
        assert claim.evidence
        for ev in claim.evidence:
            assert _collapse_ws(ev["quote"]) in _collapse_ws(excerpt)
            assert ev["chunk_id"] == chunk_id


def test_claim_with_quote_absent_from_chunk_is_rejected() -> None:
    """A claim whose quoted evidence is not present in its selected chunk must be rejected
    (fail closed). This holds for ANY extractor, including injected ones -- the gate cannot
    be bypassed by swapping in a custom extractor."""

    def bogus_extractor(topic: str, sources: list[SourceContent], *, identity=None) -> list[Extraction]:
        # Valid label + nonempty claim/quote, but the quote is NOT in the source chunk.
        return [
            Extraction(
                claim="Mars currently has abundant flowing liquid water.",
                quote="abundant flowing liquid water covering the plains",
                classification="confirmed fact",
            )
        ]

    content = _source("NASA confirmed water ice on Mars in 2021.")
    with pytest.raises(ResearchError, match="not grounded"):
        extract_from_source_content(_TOPIC, [content], extractor=bogus_extractor)


def test_valid_label_and_grounded_quote_pass_through_same_gate() -> None:
    """Sanity check the gate accepts a properly grounded claim rather than rejecting all
    injected output: the quote is an exact substring of the selected chunk."""

    def good_extractor(topic: str, sources: list[SourceContent], *, identity=None) -> list[Extraction]:
        return [
            Extraction(
                claim="NASA confirmed water ice on Mars.",
                quote="NASA confirmed water ice",
                classification="confirmed fact",
            )
        ]

    content = _source("In 2021 NASA confirmed water ice on Mars.")
    claims, _ = extract_from_source_content(_TOPIC, [content], extractor=good_extractor)
    assert len(claims) == 1
    assert claims[0].evidence[0]["quote"] == "NASA confirmed water ice"

def test_whitespace_only_quote_is_rejected_not_vacuously_accepted() -> None:
    """A whitespace-only quote must fail closed, not pass vacuously: ``"" in excerpt`` is
    always True, so the normalized quote (and claim) must be required nonempty first."""

    def spaces_extractor(topic: str, sources: list[SourceContent], *, identity=None) -> list[Extraction]:
        return [
            Extraction(
                claim="   ",
                quote="     ",
                classification="confirmed fact",
            )
        ]

    content = _source("NASA confirmed water ice on Mars in 2021.")
    with pytest.raises(ResearchError, match="claim text or evidence quote"):
        extract_from_source_content(_TOPIC, [content], extractor=spaces_extractor)


def test_cross_slice_join_quote_is_rejected() -> None:
    """A fabricated quote spanning the JOIN of two reordered sentences must be rejected:
    it is not present wholly within any single selected raw slice, even though its words
    appear across the chunk. The cross-slice join gap defeats a quote that stitches the end
    of one evidence sentence to unrelated following text."""

    def join_extractor(topic, sources, *, identity=None):
        # "water ice on Mars in 2021." + ". Site navigation link number 0 about" straddles
        # the boundary between the selected evidence sentence (range 0) and the very next
        # filler slice (nav item 0); both are selected, so the quote spans two ranges.
        return [
            Extraction(
                claim="fabricated",
                quote="water ice on Mars in 2021. Site navigation link number 0 about",
                classification="confirmed fact",
            )
        ]

    big = "\n".join(
        ["NASA confirmed water ice on Mars in 2021."]
        + [f"Site navigation link number {i} about unrelated page chrome here." for i in range(400)]
    )
    content = _source(big)
    with pytest.raises(ResearchError, match="not present wholly within"):
        extract_from_source_content(_TOPIC, [content], extractor=join_extractor)


def test_html_script_body_cannot_supply_matching_quote() -> None:
    """Raw slices are normalized the same way as the excerpt (script/style bodies + tags
    removed), so a phrase that appears ONLY inside a <script> block cannot satisfy the gate,
    while a grounded quote embedded in tagged sentences still passes."""

    # The <script> block is embedded BETWEEN words of one >=6-word selected sentence, so its
    # raw source_range includes the script bytes. "hidden ocean under ice" appears only inside
    # that script body; no real sentence has it. A normalized slice drops the body (gate rejects),
    # whereas a tag-only stripper would leak it -- so this fails closed only with the fix.
    sc_script = _source(
        'NASA confirmed that <script>var secret="hidden ocean under ice";'
        '</script> water ice exists beneath the surface in 2021.',
        url="https://nasa.gov/script",
    )

    def script_quote(topic, sources, *, identity=None):
        return [Extraction(claim="x", quote="hidden ocean under ice", classification="confirmed fact")]

    with pytest.raises(ResearchError, match="not present wholly within"):
        extract_from_source_content(_TOPIC, [sc_script], extractor=script_quote)

    # A grounded quote embedded in <b> tags must still pass: normalization strips the tags
    # consistently for both the excerpt and the gate slices.
    sc_tagged = _source(
        "<p>NASA confirmed <b>water ice on Mars</b> in 2021.</p>", url="https://nasa.gov/tags"
    )

    def grounded(topic, sources, *, identity=None):
        return [Extraction(claim="NASA confirmed water ice on Mars", quote="water ice on Mars", classification="confirmed fact")]

    claims, _ = extract_from_source_content(_TOPIC, [sc_tagged], extractor=grounded)
    assert len(claims) == 1
    assert claims[0].evidence[0]["quote"] == "water ice on Mars"


def test_normalized_raw_ranges_reconstruct_html_excerpt() -> None:
    """Normalized raw ranges reconstruct an HTML/script-derived excerpt exactly:
    normalizing each content[start:end] and joining in range order yields chunk.text,
    even when a slice carries inline tags and a dropped <script> body."""
    html = (
        'NASA confirmed <b>water ice</b> on Mars '
        '<script>var junk="navigation"; bad();</script>'
        'in 2021.'
        ' Rovers found subsurface water ice beneath Europa crust.'
    )
    content = _source(html)
    chunk = _clean_excerpt(content.content, _TOPIC)
    ranges = list(chunk.source_ranges)
    assert ranges, "expected selected sentence spans from HTML/script source"
    reconstructed = " ".join(
        _normalize_with_offsets(content.content[s:e])[0] for s, e in ranges
    )
    assert reconstructed == chunk.text
    # Both keyworded sentences are selected; each raw range maps back to its slice.
    assert len(ranges) == 2
