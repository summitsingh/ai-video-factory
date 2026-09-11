"""Provenance + model-resolution tests for the production research pipeline.

Covers Phase-4 Step 1: the active local model is resolved from LM Studio at
runtime (never a hard-coded default), provenance (resolved model id/endpoint,
schema version, source-content hashes, output digest) is recorded on the
research result, and production fails closed when the configured model is
unavailable or schema output is invalid.

Both ``_lmstudio_models_bytes`` (the ``/v1/models`` listing) and the extraction
chat-completions POST go through ``urllib.request.urlopen``; a single dispatching
fake installed under ``sys.modules`` stands in for the network for both.
"""

import json
import sys
import types

import pytest

from ai_video_factory.research_pipeline import (
    ModelResolutionError,
    ResearchError,
    ResearchResult,
    SourceContent,
    EvidenceRecord,
    _build_claims_from_records,
    _models_list_url,
    make_production_extractor,
    resolve_active_model,
    PROMPT_VERSION,
    EXTRACTION_SCHEMA_VERSION,
    run_research,
)

_MODELS_BYTES = json.dumps({"data": [{"id": "m-a"}, {"id": "m-b"}]}).encode()


def _source(text: str) -> SourceContent:
    return SourceContent(
        url="https://nasa.gov/example",
        title="Example NASA asset",
        quality="reliable",
        content=text,
        content_hash="",
    )


def _install_fake_urllib(monkeypatch, claims_content: str):
    """Stand in for LM Studio: ``/v1/models`` returns model ids; chat-completions
    returns ``claims_content`` as the assistant message text."""

    class FakeResponse:
        def __init__(self, target: object) -> None:
            # urlopen is called with a URL string (models listing) or a Request
            # object (chat-completions POST); normalize to the full URL either way.
            if isinstance(target, str):
                self._url = target
            else:
                getter = getattr(target, "get_full_url", None)
                self._url = getter() if callable(getter) else ""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            if self._url.endswith("/models"):
                return _MODELS_BYTES
            return json.dumps(
                {"choices": [{"message": {"content": claims_content}}]}
            ).encode()

    request_mod = types.ModuleType("urllib.request")
    request_mod.Request = lambda *a, **k: object()  # type: ignore[attr-defined]
    request_mod.urlopen = lambda url, *a, **k: FakeResponse(url)  # type: ignore[attr-defined]
    urllib_mod = types.ModuleType("urllib")
    urllib_mod.request = request_mod
    monkeypatch.setitem(sys.modules, "urllib", urllib_mod)
    monkeypatch.setitem(sys.modules, "urllib.request", request_mod)


# ---- URL derivation ---------------------------------------------------------


def test_models_list_url_from_chat_completions_endpoint():
    assert (
        _models_list_url("http://localhost:1234/v1/chat/completions")
        == "http://localhost:1234/v1/models"
    )


def test_models_list_url_bare_v1_base():
    # A bare ``/v1`` base must not gain a second ``/v1``.
    assert _models_list_url("https://h-1.inference.ai/v1") == "https://h-1.inference.ai/v1/models"


# ---- resolve_active_model ---------------------------------------------------


def test_resolve_prefers_configured_model():
    transport = lambda url: _MODELS_BYTES  # noqa: E731
    assert (
        resolve_active_model(
            endpoint_url="http://x/v1/chat/completions", preferred="m-b", transport=transport
        )
        == "m-b"
    )


def test_resolve_returns_first_id_when_no_preferred():
    transport = lambda url: _MODELS_BYTES  # noqa: E731
    assert (
        resolve_active_model(endpoint_url="http://x/v1/chat/completions", transport=transport)
        == "m-a"
    )


def test_resolve_fails_closed_when_preferred_absent():
    """A configured preferred model the server does not expose fails closed rather
    than silently extracting against a different model."""
    def transport(url: str) -> bytes:
        return _MODELS_BYTES  # exposes m-a, m-b only

    with pytest.raises(ModelResolutionError):
        resolve_active_model(
            endpoint_url="http://x/v1/chat/completions",
            preferred="m-z",
            transport=transport,
        )


def test_resolve_fails_closed_when_server_unreachable():
    def boom(url):
        raise OSError("connection refused")

    with pytest.raises(ModelResolutionError, match="could not resolve"):
        resolve_active_model(endpoint_url="http://127.0.0.1:1/v1/chat/completions", transport=boom)


def test_resolve_fails_closed_when_no_models_exposed():
    def transport(url):
        return json.dumps({"data": []}).encode()

    with pytest.raises(ModelResolutionError, match="exposed no models"):
        resolve_active_model(endpoint_url="http://x/v1/chat/completions", transport=transport)


def test_resolve_fails_closed_on_malformed_json():
    def transport(url):
        return b"not-json"

    with pytest.raises(ModelResolutionError, match="could not resolve"):
        resolve_active_model(endpoint_url="http://x/v1/chat/completions", transport=transport)


# ---- make_production_extractor: provenance + fail-closed --------------------


def test_production_extractor_resolves_and_captures_provenance(monkeypatch):
    _install_fake_urllib(monkeypatch, '[confirmed fact] | NASA carried water on Mars in 2021. || "NASA carried water on Mars in 2021."')
    captured: dict[str, object] = {}

    extractor = make_production_extractor(endpoint_url="http://127.0.0.1:9/v1/chat/completions")

    pairs = extractor(
        "Water Beyond Earth",
        [_source("NASA carried water on Mars in 2021.")],
        identity=lambda name, ver, schema, prompt_version=None, model_id=None, endpoint_url=None: captured.update(
            {"model_id": model_id, "endpoint": endpoint_url, "prompt_version": prompt_version, "schema_version": schema}
        ),
    )

    assert len(pairs) == 1
    # Provenance stamped through the identity callback for pilot provenance.
    assert captured["model_id"] == "m-a"
    assert captured["endpoint"] == "http://127.0.0.1:9/v1/chat/completions"
    assert captured["prompt_version"] == PROMPT_VERSION
    assert captured["schema_version"] == EXTRACTION_SCHEMA_VERSION


def test_production_extractor_fails_closed_when_server_down(monkeypatch):
    def boom(url):
        raise OSError("no server")

    monkeypatch.setattr(
        "ai_video_factory.research_pipeline._lmstudio_models_bytes", boom
    )
    # Model resolution happens at construction time, so the build itself fails.
    with pytest.raises(ModelResolutionError, match="could not resolve"):
        make_production_extractor(endpoint_url="http://127.0.0.1:1/v1/chat/completions")


def test_production_extractor_rejects_invalid_classification(monkeypatch):
    """Schema output that is ``[label] | text`` with an unknown label fails closed."""
    _install_fake_urllib(monkeypatch, "[bogus label] | a narratable statement || \"verbatim evidence text\"")

    extractor = make_production_extractor(
        endpoint_url="http://127.0.0.1:9/v1/chat/completions", model_name="m-a"
    )
    with pytest.raises(ResearchError, match="invalid classification"):
        extractor("Water Beyond Earth", [_source("ignored")])


# A response mixing one valid claim line with a malformed line must fail closed,
# not silently accept the good line and drop the junk.
def test_production_extractor_fails_closed_on_mixed_output(monkeypatch):
    _install_fake_urllib(
        monkeypatch,
        '[confirmed fact] | NASA carried water on Mars in 2021. || "NASA carried water on Mars in 2021."\n'
        "this is a hallucinated line with no schema",
    )
    extractor = make_production_extractor(
        endpoint_url="http://127.0.0.1:9/v1/chat/completions", model_name="m-a"
    )
    with pytest.raises(ResearchError, match="malformed output line"):
        extractor("Water Beyond Earth", [_source("NASA carried water on Mars in 2021.")])
# The model occasionally omits brackets around the label or uses a dash instead of
# the double-pipe before the quote. Those slips are normalised to canonical form and
# then validated identically (valid-class set + verbatim grounding); they must be
# accepted, while unknown labels and junk lines keep failing closed (see the two tests
# above). This tolerance widens acceptance only for formatting -- never semantics.
@pytest.mark.parametrize(
    "line",
    [
        '[confirmed fact] | NASA carried water on Mars in 2021. || "NASA carried water on Mars in 2021."',
        '[confirmed fact] | NASA carried water on Mars in 2021. - "NASA carried water on Mars in 2021."',
        'confirmed fact | NASA carried water on Mars in 2021. || "NASA carried water on Mars in 2021."',
        'confirmed fact | NASA carried water on Mars in 2021. - "NASA carried water on Mars in 2021."',
    ],
)
def test_production_extractor_accepts_label_format_slips(monkeypatch, line):
    _install_fake_urllib(monkeypatch, line)
    extractor = make_production_extractor(
        endpoint_url="http://127.0.0.1:9/v1/chat/completions", model_name="m-a"
    )
    extractions = extractor("Water Beyond Earth", [_source("NASA carried water on Mars in 2021.")])
    assert len(extractions) == 1
def test_production_extractor_retries_then_succeeds(monkeypatch):
    """A transient non-conforming response is retried; a later conforming one succeeds.

    The local general-purpose model emits non-deterministic output at temperature=0, so a
    rich source that conforms on one call may emit preamble/junk on another. The extractor
    must retry a bounded number of times and recover, while still failing closed if every
    attempt is non-conforming (see test_production_extractor_fails_closed_on_mixed_output).
    """
    chat_calls = {"n": 0}

    def _chat_json(content: str) -> bytes:
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    class FakeResponse:
        def __init__(self, target):
            self._url = target if isinstance(target, str) else getattr(target, "get_full_url", lambda: "")()
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            if self._url.endswith("/models"):
                return _MODELS_BYTES
            chat_calls["n"] += 1
            if chat_calls["n"] == 1:
                # First attempt: preamble junk with no valid claim lines -> non-conforming.
                return _chat_json("Just some preamble notes here, nothing schema-shaped.")
            # Recovered conforming output on the retry.
            return _chat_json('[confirmed fact] | NASA carried water on Mars in 2021. || "NASA carried water on Mars in 2021."\n')

    request_mod = types.ModuleType("urllib.request")
    request_mod.Request = lambda *a, **k: object()
    request_mod.urlopen = lambda url, *a, **k: FakeResponse(url)
    urllib_mod = types.ModuleType("urllib")
    urllib_mod.request = request_mod
    monkeypatch.setitem(sys.modules, "urllib", urllib_mod)
    monkeypatch.setitem(sys.modules, "urllib.request", request_mod)

    extractor = make_production_extractor(
        endpoint_url="http://127.0.0.1:9/v1/chat/completions", model_name="m-a"
    )
    extractions = extractor("Water Beyond Earth", [_source("NASA carried water on Mars in 2021.")])
    assert len(extractions) == 1
    # It had to retry past the first non-conforming response to reach the conforming one.
    assert chat_calls["n"] >= 2


def test_production_extractor_fails_closed_after_exhausting_retries(monkeypatch):
    """If every attempt is non-conforming, the extractor still fails closed."""

    def _chat_json(content: str) -> bytes:
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    class FakeResponse:
        def __init__(self, target):
            self._url = target if isinstance(target, str) else getattr(target, "get_full_url", lambda: "")()
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            if self._url.endswith("/models"):
                return _MODELS_BYTES
            # Always non-conforming (unknown label) -> must fail closed after all retries.
            return _chat_json('[bogus label] | x || "y"')

    request_mod = types.ModuleType("urllib.request")
    request_mod.Request = lambda *a, **k: object()
    request_mod.urlopen = lambda url, *a, **k: FakeResponse(url)
    urllib_mod = types.ModuleType("urllib")
    urllib_mod.request = request_mod
    monkeypatch.setitem(sys.modules, "urllib", urllib_mod)
    monkeypatch.setitem(sys.modules, "urllib.request", request_mod)

    extractor = make_production_extractor(
        endpoint_url="http://127.0.0.1:9/v1/chat/completions", model_name="m-a"
    )
    with pytest.raises(ResearchError, match="invalid classification"):
        extractor("Water Beyond Earth", [_source("NASA carried water on Mars in 2021.")])
def test_build_claims_from_records_requires_two_distinct_sources():
    # Same normalized claim from two distinct NASA URLs + supported labels => one
    # corroborated claim carrying both source URLs.
    records = [
        EvidenceRecord(
            id="rec-001", url="https://nasa.gov/europa-clipper", label="confirmed fact",
            claim="Europa has a subsurface ocean.", evidence_quote="subsurface ocean"),
        EvidenceRecord(
            id="rec-002", url="https://nasa.gov/juno", label="reported claim",
            claim="Europa has a subsurface ocean.", evidence_quote="subsurface ocean"),
    ]
    claims = _build_claims_from_records(records)
    assert len(claims) == 1
    assert claims[0].source_ids == [
        "https://nasa.gov/europa-clipper", "https://nasa.gov/juno"]
    assert claims[0].provisional_classification == "confirmed fact"

    # The same claim from a single URL is NOT corroborated (one source only).
    single = _build_claims_from_records([
        EvidenceRecord(
            id="rec-003", url="https://nasa.gov/mars", label="confirmed fact",
            claim="Mars had ancient rivers.", evidence_quote="ancient rivers"),
    ])
    assert len(single) == 1
    assert single[0].source_ids == ["https://nasa.gov/mars"]

    # Two DIFFERENT claims from two sources => two separate single-source claims.
    mixed = _build_claims_from_records([
        EvidenceRecord(
            id="rec-004", url="https://nasa.gov/europa-clipper", label="confirmed fact",
            claim="Europa has a subsurface ocean.", evidence_quote="subsurface ocean"),
        EvidenceRecord(
            id="rec-005", url="https://nasa.gov/mars", label="confirmed fact",
            claim="Mars had ancient rivers.", evidence_quote="ancient rivers"),
    ])
    assert len(mixed) == 2
    for c in mixed:
        assert len(c.source_ids) == 1


# ---- run_research: provenance lands on the result ---------------------------


def test_run_research_stamps_resolved_model_provenance(monkeypatch):
    _install_fake_urllib(monkeypatch, '[confirmed fact] | Perseverance landed in Jezero in 2021. || "Perseverance rover touched down in Jezero Crater in 2021."')
    extractor = make_production_extractor(endpoint_url="http://127.0.0.1:9/v1/chat/completions")

    contents = [_source("NASA's Perseverance rover touched down in Jezero Crater in 2021.")]
    result = run_research(
        "Water Beyond Earth",
        source_urls=["https://nasa.gov/mars"],
        source_contents=contents,
        production_extractor=extractor,
    )

    assert isinstance(result, ResearchResult)
    assert result.extractor_model_id == "m-a"
    assert result.extractor_endpoint == "http://127.0.0.1:9/v1/chat/completions"
    # Prompt + extraction-schema versions are stamped for pilot provenance.
    assert result.prompt_version == PROMPT_VERSION
    assert result.extraction_schema_version == EXTRACTION_SCHEMA_VERSION
    # Source-content hashes + output digest participate in the fingerprint.
    assert result.source_content_hashes, "per-source content hashes must be recorded"
    assert result.output_digest


def test_output_digest_covers_evidence_and_classification(monkeypatch):
    """The output fingerprint must change when auditable evidence or a claim's
    classification changes -- not only when source bytes do."""
    contents = [_source("NASA's Perseverance rover touched down in Jezero Crater in 2021.")]

    def _run(quote: str, label: str = "confirmed fact") -> ResearchResult:
        body = f'[{label}] | Perseverance landed in Jezero in 2021. || "{quote}"'
        _install_fake_urllib(monkeypatch, body)
        extractor = make_production_extractor(
            endpoint_url="http://127.0.0.1:9/v1/chat/completions"
        )
        return run_research(
            "Water Beyond Earth",
            source_urls=["https://nasa.gov/mars"],
            source_contents=contents,
            production_extractor=extractor,
        )

    base = _run("Perseverance rover touched down in Jezero Crater in 2021.")
    # Same claim text + classification; only the evidence quote differs.
    by_evidence = _run("rover touched down in Jezero")
    assert by_evidence.output_digest != base.output_digest

    # Same evidence; only the classification label differs.
    by_class = _run(
        "Perseverance rover touched down in Jezero Crater in 2021.",
        label="reported claim",
    )
    assert by_class.output_digest != base.output_digest


def test_run_research_legacy_mode_has_no_model_provenance():
    # Legacy/dry-run path never runs production extraction -> model stays None.
    from ai_video_factory.research_pipeline import RULE_BASED_EXTRACTOR

    result = run_research(
        "Off-topic dry run",
        source_urls=["https://example.com/a", "https://example.com/b"],
        extractor=RULE_BASED_EXTRACTOR,
    )
    assert result.extractor_model_id is None
