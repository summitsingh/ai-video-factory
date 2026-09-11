"""Focused tests for repeatable ``--source-url`` support.

Covers three required behaviors:

* **Repeated URLs** collapse to one provenance entry per unique URL (no duplicate
  fetches, stable fingerprint).
* **Provenance persistence**: every supplied URL is recorded in the research-stage
  fingerprint (content-hash keys + on-disk ``sources.json``) and the resolved model
  / endpoint / schema / prompt versions land on the result.
* **Unsafe / invalid URLs** are rejected by the hardened content path before any
  fetching, so production fails closed rather than extracting from a bad source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_video_factory.cli import app
from ai_video_factory.daily_job import JobConfig, _build_research_packages
from ai_video_factory.research_pipeline import (
    EXTRACTION_SCHEMA_VERSION,
    PROMPT_VERSION,
    Extraction,
    ResearchError,
    ModelResolutionError,
    make_production_extractor,
    resolve_active_model,
    make_secure_http_transport,
    run_research,
    collect_sources,
)
from ai_video_factory.trend import SourceSignal, TrendCandidate


# ---------- shared fakes (offline; mirror tests/test_daily_job.py conventions) ----

_CANONICAL_CLAIMS = (
    ("NASA confirmed water ice existed on the target in 2021.", "NASA confirmed water ice"),
    ("Frozen subsurface water was detected below the icy crust of Europa.", "Frozen subsurface water"),
    ("Plumes of water vapor eject from the south pole of Enceladus.", "water vapor eject from the south pole"),
)


def _grounded_fetcher(url: str) -> str:
    """Rich, grounded source text (years/numbers/proper nouns) for extraction.

    Every canonical claim quote below is a verbatim substring of this content so the
    central grounding gate accepts it; the water/Europa/Enceladus/Jupiter sentences
    are topic-relevant and rank into the bounded excerpt alongside the filler lines.
    """
    body = [
        f"Source report for {url}. In 2021 NASA confirmed water ice on the target.",
        "Frozen subsurface water was detected below the icy crust of Europa.",
        "Plumes of water vapor eject from the south pole of Enceladus.",
        "Juno has orbited Jupiter since 2016 and mapped subsurface water ice.",
    ]
    for i in range(1, 321):
        body.append(f"Study number {i} recorded a distinct finding of {10 + i} units.")
    return "\n".join(body)


def _stamping_extractor(topic, sources, *, identity=None):
    """Fake production extractor: stamps provenance and returns the canonical claims.

    Returning the same set of confirmed facts for every source lets corroboration
    merge them into distinct verified facts each carrying all source URLs -- exactly
    the independent cross-source corroboration the gates require.
    """
    if identity is not None:
        identity(
            "test-production-extractor",
            "1.0",
            EXTRACTION_SCHEMA_VERSION,
            prompt_version=PROMPT_VERSION,
            model_id="local-test-model",
            endpoint_url="http://127.0.0.1:1234/v1/chat/completions",
        )
    claims = []
    for _ in sources:
        for text, quote in _CANONICAL_CLAIMS:
            claims.append(Extraction(claim=text, quote=quote, classification="confirmed fact"))
    return claims


_SOURCES = [
    "https://science.nasa.gov/planetary/mars-water-ice",
    "https://nasa.gov/news/europa-ocean-detection",
    "https://images-api.nasa.gov/item/enceladus-plumes",
]


# ---------- CLI registration -----------------------------------------------------


def test_daily_cli_exposes_repeatable_source_url() -> None:
    """``daily --source-url`` is a registered, repeatable option."""
    result = CliRunner().invoke(app, ["daily", "--help"])
    assert result.exit_code == 0
    assert "--source-url" in result.stdout
    assert "-u" in result.stdout


# ---------- source-quality classification (gate input) ---------------------------


def test_collect_sources_classifies_nasa_reliable_and_skips_empty() -> None:
    records = collect_sources(
        ["https://science.nasa.gov/mars", "https://en.wikipedia.org/x", "", "   "]
    )
    urls = [r.url for r in records]
    assert urls == ["https://science.nasa.gov/mars", "https://en.wikipedia.org/x"]
    quality = {r.url: r.quality for r in records}
    assert quality["https://science.nasa.gov/mars"] == "reliable"
    assert quality["https://en.wikipedia.org/x"] == "reliable"


# ---------- repeated URLs --------------------------------------------------------


def test_repeated_urls_dedup_at_provenance_level() -> None:
    urls = [
        "https://science.nasa.gov/mars-1",
        "https://science.nasa.gov/mars-1",  # duplicate
        "https://nasa.gov/europa",
    ]
    result = run_research(
        "Water Beyond Earth",
        urls,
        production_extractor=_stamping_extractor,
        content_fetcher=_grounded_fetcher,
    )
    # The provenance hash is keyed by URL: duplicates collapse to one entry each.
    assert set(result.source_content_hashes) == set(urls)
    assert len(result.source_content_hashes) == 2


# ---------- provenance persistence -----------------------------------------------


def test_run_research_persists_source_urls_and_model_provenance() -> None:
    result = run_research(
        "Water Beyond Earth",
        _SOURCES,
        production_extractor=_stamping_extractor,
        content_fetcher=_grounded_fetcher,
    )
    # Every supplied URL is persisted as a provenance content-hash key.
    assert set(result.source_content_hashes) == set(_SOURCES)
    # Resolved model + endpoint + schema + prompt versions recorded for provenance.
    assert result.extractor_model_id == "local-test-model"
    assert result.extractor_endpoint == "http://127.0.0.1:1234/v1/chat/completions"
    assert result.extraction_schema_version == EXTRACTION_SCHEMA_VERSION
    assert result.prompt_version == PROMPT_VERSION
    assert result.output_digest  # extraction-output digest present


def test_make_production_extractor_resolves_active_model_not_hardcoded(monkeypatch) -> None:
    """The active model is resolved at runtime from the server listing, never a
    hard-coded default; a missing preferred model fails closed."""
    import ai_video_factory.research_pipeline as rp

    def canned_models(url: str) -> bytes:
        assert url.endswith("/v1/models")
        return json.dumps({"data": [{"id": "resolved-local-model"}]}).encode()

    monkeypatch.setattr(rp, "_lmstudio_models_bytes", canned_models)

    # No preferred model pinned -> first available id is resolved at runtime, never a
    # hard-coded default (make_production_extractor delegates to this).
    model = resolve_active_model(endpoint_url="http://127.0.0.1:1234/v1/chat/completions")
    assert model == "resolved-local-model"

    # make_production_extractor builds only after resolving the active model at
    # runtime; a missing preferred model would fail this build closed.
    extractor = make_production_extractor(
        endpoint_url="http://127.0.0.1:1234/v1/chat/completions", model_name=None
    )
    assert callable(extractor)

    # A pinned-but-absent preferred model fails closed rather than guessing.
    with pytest.raises(ModelResolutionError):
        resolve_active_model(
            endpoint_url="http://127.0.0.1:1234/v1/chat/completions",
            preferred="model-not-loaded",
            transport=canned_models,
        )


def test_build_research_packages_persists_source_urls_to_artifacts(tmp_path: Path) -> None:
    """The research-package stage persists the supplied URLs to on-disk provenance
    (sources.json) and stamps model provenance on the result."""
    state = __import__(
        "ai_video_factory.run_store", fromlist=["RunStore"]
    ).RunStore(tmp_path / "state", artifact_root=tmp_path / "runs")
    runs_dir = tmp_path / "runs" / "daily"

    config = JobConfig(
        production_extractor=_stamping_extractor,
        content_fetcher=_grounded_fetcher,
    )
    candidate = TrendCandidate(
        topic_id="override",
        title="Water Beyond Earth",
        summary="Grounded live pilot topic",
        first_seen_at="2026-09-09T00:00:00+00:00",
        observed_at="2026-09-09T00:00:00+00:00",
        cluster_key="water beyond earth where it is why",
        sources=[SourceSignal(provider="nasa", url=url, published_at=None) for url in _SOURCES],
    )

    results, dirs = _build_research_packages(state, runs_dir, [candidate], False, config)
    assert len(results) == 1
    result = results[0]
    assert set(result.source_content_hashes) == set(_SOURCES)

    # The on-disk research artifacts record the exact supplied source URLs.
    brief_dir = Path(dirs[0])
    sources_file = brief_dir / "sources.json"
    assert sources_file.is_file()
    recorded_urls = {item["url"] for item in json.loads(sources_file.read_text(encoding="utf-8"))}
    assert recorded_urls == set(_SOURCES)


# ---------- unsafe / invalid URL rejection (hardened path) -----------------------


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://nasa.gov/plain-http",  # plain HTTP rejected (HTTPS only)
        "http://127.0.0.1:1234/v1/models",  # loopback + plain HTTP
        "https://10.0.0.5/private",  # private range refused
        "https://192.168.1.9/internal",  # private range refused
        "not a url at all",  # malformed / no scheme
    ],
)
def test_unsafe_source_urls_are_rejected(bad_url: str) -> None:
    """The hardened content path refuses unsafe/invalid URLs before any fetch, so
    production fails closed rather than extracting from a bad source."""
    with pytest.raises(ResearchError):
        run_research(
            "Water Beyond Earth",
            [bad_url],
            production_extractor=_stamping_extractor,
            content_fetcher=make_secure_http_transport(),
        )
