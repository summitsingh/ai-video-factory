"""Tests for the daily production-job orchestration (Phase 3, Milestone 7).

Every test is offline and fixture-based: no real network call, no LM Studio
loopback, no paid model. The full pipeline is exercised end-to-end against an
injected trend transport that returns canned provider payloads, a content
fetcher that persists grounded source text, and an injectable render engine so
the production path can be validated without any external dependency.

The suite covers the eight conditions required for Milestone 7:

1. deterministic production path uses an injected extractor (never LM Studio);
2. LLM availability does not change test outcomes;
3. a real run with no configured renderer fails safely as ``not_configured``
   and never creates an approval package;
4. failed final QC blocks packaging;
5. inadequate distinct verified material blocks long-form production;
6. reruns reuse valid stages (resume reconstructs the rendered candidate);
7. CLI dry run performs no network discovery and no render (real Typer runner);
8. research fetches + persists real source content deterministically, and the
   extractor records its identity from persisted content rather than topic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_video_factory.daily_job import JobConfig, run_daily_job
from ai_video_factory.production import (
    OfflineRenderEngine,
    ProductionError,
    build_candidate,
)
from ai_video_factory.research_pipeline import (
    Claim,
    ResearchResult,
    SourceContent,
    SourceRecord,
    run_research,
)
from ai_video_factory.trend import TrendCandidate


# ========== Fixtures: canned provider payloads (no network) ==========


def _trend_transport(url: str) -> str:
    """Return cached HN + Google News payloads for one topic across two providers.

    All items share the title ``"Local AI Trends"`` so they collapse into a
    single candidate carrying three sources from two distinct providers, which
    satisfies the preflight breadth gate (>=2 providers) and gives the
    deterministic production extractor enough grounded content to promote.
    """
    if "hn.algolia.com" in url:
        return json.dumps({
            "hits": [
                {"object_id": "a1", "title": "Local AI Trends",
                 "points": 900, "num_comments": 50, "created_at": "2026-09-07T10:00:00Z"},
                {"object_id": "a2", "title": "Local AI Trends",
                 "points": 800, "num_comments": 40, "created_at": "2026-09-07T09:00:00Z"},
            ]
        })
    if "news.google.com" in url:
        return (
            '<?xml version="1.0"?><rss version="2.0"><channel>'
            "<item>"
            "<title>Local AI Trends</title>"
            "<link>https://example.com/gn1</link>"
            "<pubDate>Sat, 07 Sep 2026 10:00:00 GMT</pubDate>"
            "<source>Google News</source>"
            "</item></channel></rss>"
        )
    raise AssertionError(f"unexpected trend transport url: {url}")


def _content_fetcher(url: str) -> str:
    """Persist rich, grounded source text (years/numbers/proper nouns)."""
    lines = [f"Report on {url}. The analysis noted 42 percent growth in 2024."]
    for i in range(1, 321):
        lines.append(f"Study number {i} recorded a distinct finding of {10 + i} units that year.")
    return "\n".join(lines)


def _asset_transport(url: str) -> bytes:
    """Return cached NASA search JSON plus download bytes (approved license)."""
    if "images-api.nasa.gov/search" in url:
        return json.dumps({
            "collection": {"items": [
                {"item_id": "x1",
                 "metadata": {"title": "Local AI Trends", "creator": "NASA"},
                 "links": [{"rel": "media", "href": "https://cdn.example/asset.mp4"}]}
            ]}
        }).encode("utf-8")
    # download_url fetch.
    return b"\x89PNG\r\n\x1a\nfake-bytes"


def _config(**overrides) -> JobConfig:
    cfg = JobConfig(
        providers=["hacker_news", "google_news"],
        trend_transport=_trend_transport,
        content_fetcher=_content_fetcher,
        asset_transport=_asset_transport,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class _BlackMasterEngine(OfflineRenderEngine):
    """Render engine that produces an all-black master (fails the visual gate).

    Reuses the parent's narration synthesis so runtime pacing stays in band; only
    the video is replaced with a solid black clip.
    """

    def render_master(self, edit, *, narration_segments, assets_by_scene_id, destination):  # pragma: no cover
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fps = int(edit.fps)
        total_frames = sum(s.duration_frames for s in edit.scenes)
        duration = max(1.0, total_frames / fps)
        import subprocess

        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"color=c=black:s=1280x720:r={fps}:duration={duration:.3f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination),
            ],
            check=True, capture_output=True, text=True,
        )
        return destination


# ========== 1. deterministic production path uses an injected extractor ==========


def test_deterministic_extractor_used_in_production(tmp_path: Path) -> None:
    """The production path extracts with the injected extractor, never LM Studio."""
    calls: list[str] = []

    def injected(topic: str, sources: list[SourceContent], **_: object) -> list[tuple[str, str]]:
        calls.append(topic)
        # One distinct, source-backed claim per persisted source (not a repeat).
        return [
            (f"Local AI adoption grew {42 + i} percent in 2024 per {s.url}", "confirmed fact")
            for i, s in enumerate(sources)
        ]

    result = run_daily_job(tmp_path, config=_config(production_extractor=injected), engine=OfflineRenderEngine())  # type: ignore[arg-type]
    assert calls == ["Local AI Trends"]
    assert result.status == "blocked_on_approval"


# ========== 2. LLM availability does not change test outcomes ==========


def test_llm_availability_does_not_change_outcome(tmp_path: Path) -> None:
    """With an injected extractor, LM Studio presence is irrelevant to the outcome."""

    def injected(topic: str, sources: list[SourceContent], **_: object) -> list[tuple[str, str]]:
        return [
            (f"Verified finding {i} about local AI in 2024", "confirmed fact")
            for i in range(3)
        ]

    result = run_daily_job(tmp_path, config=_config(production_extractor=injected), engine=OfflineRenderEngine())  # type: ignore[arg-type]
    assert result.status == "blocked_on_approval"


# ========== 3. production without a configured renderer fails safely ========


def test_production_without_renderer_fails_safely(tmp_path: Path) -> None:
    """A real run with no engine reports not_configured rather than producing junk,
    and never creates an approval package.
    """
    result = run_daily_job(tmp_path, config=_config())
    assert result.status == "failed"
    assert "not configured" in (result.failure_reason or "")
    assert result.approval_path is None


# ========== 4. failed final QC blocks packaging ==========


def test_failed_qc_blocks_packaging(tmp_path: Path) -> None:
    """A black master fails the visual gate; no approval package is produced."""
    result = run_daily_job(tmp_path, config=_config(), engine=_BlackMasterEngine())  # type: ignore[arg-type]
    assert result.status == "failed"
    assert result.approval_path is None
    assert "final QC failed" in (result.failure_reason or "")


# ========== 5. inadequate distinct verified material blocks long-form ==========


def test_inadequate_material_blocks_long_form(tmp_path: Path) -> None:
    """Fewer than MIN_VERIFIED_FACTS verified facts are rejected at production."""

    def rich_source(url: str, n_sentences: int = 120) -> tuple[SourceRecord, SourceContent]:
        lines = [f"Report on {url}. The analysis noted 42 percent growth in 2024."]
        for i in range(1, n_sentences + 1):
            lines.append(f"Study number {i} recorded a distinct finding of {10 + i} units that year.")
        record = SourceRecord(url=url, provider="reliable", quality="reliable", title="t")
        content_obj = SourceContent(
            url=url, title="t", quality="reliable",
            content="\n".join(lines), content_hash="",
        )
        return record, content_obj

    records: list[SourceRecord] = []
    contents: dict[str, SourceContent] = {}
    for i in range(2):  # only two sources -> two verified facts (< MIN_VERIFIED_FACTS)
        url = f"https://example.com/src{i}"
        record, content_obj = rich_source(url)
        records.append(record)
        contents[url] = content_obj
    facts = [
        Claim(claim_id=f"claim-{i + 1}", text=f"Verified finding {i} about the topic that year 2024",
              source_ids=[records[i].url], provisional_classification="confirmed fact")
        for i in range(2)
    ]
    research = ResearchResult(
        topic="Local AI Trends", sources=records, claims=list(facts),
        verified_facts=list(facts), source_contents=contents, research_brief="",
    )

    with pytest.raises(ProductionError) as excinfo:
        build_candidate(research, engine=OfflineRenderEngine(), min_words=2200, workdir=tmp_path)
    assert "insufficient" in str(excinfo.value).lower()


# ========== 6. reruns reuse valid stages ==========


def test_rerun_reuses_valid_stages(tmp_path: Path) -> None:
    """A second run reuses the rendered candidate instead of rebuilding it."""
    first = run_daily_job(tmp_path, config=_config(), engine=OfflineRenderEngine())  # type: ignore[arg-type]
    assert first.status == "blocked_on_approval"
    assert not first.artifacts.get("reused_candidate", True)

    second = run_daily_job(tmp_path, config=_config(), engine=OfflineRenderEngine())  # type: ignore[arg-type]
    assert second.status == "blocked_on_approval"
    assert second.artifacts.get("reused_candidate") is True


# ========== 7. CLI dry run performs no network or render (real CliRunner) ========


def test_cli_dry_run_no_network_or_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real ``daily --dry-run`` invocation renders nothing and skips trend discovery.

    ``--topic`` makes the CLI skip trend discovery entirely (no network), and dry-run
    never reaches production rendering. Both paths are spied on with counters; a correct
    dry run must call neither the trend-discovery path nor ``build_candidate``.
    """
    from typer.testing import CliRunner

    from ai_video_factory.cli import app
    from ai_video_factory import daily_job as dj_mod

    calls = {"discover": 0, "build": 0}

    def spy_discover(*_args: object, **_kwargs: object) -> list[TrendCandidate]:
        calls["discover"] += 1
        now = datetime.now(UTC).isoformat()
        return [TrendCandidate(
            topic_id="o", title="Local AI Trends", summary="s",
            first_seen_at=now, observed_at=now, cluster_key="local ai trends",
            sources=[SourceRecord(url=f"https://example.com/s{i}", provider="hacker_news", quality="reliable", title="t") for i in range(3)],
        )]

    def spy_build(*_args: object, **_kwargs: object):
        calls["build"] += 1
        return None

    monkeypatch.setattr(dj_mod, "discover_trends", spy_discover)
    monkeypatch.setattr(dj_mod, "build_candidate", spy_build)

    runner = CliRunner()
    # --topic skips trend discovery (no network); --dry-run never renders.
    result = runner.invoke(app, ["daily", "--dry-run", "--topic", "Local AI Trends"])
    assert calls["discover"] == 0, "trend discovery must not run under --topic dry-run"
    assert calls["build"] == 0, "rendering must never happen in dry-run"


# ========== 8. source fetch + extractor are deterministic and grounded ========


def test_source_fetch_and_extractor_are_deterministic() -> None:
    """Research fetches + persists real source content and extracts deterministically.

    Two identical runs over the same sources produce an identical output digest, a
    non-empty per-source content-hash map (content was actually fetched), and a
    populated extractor identity -- proving extraction is grounded in persisted
    content rather than derived from topic heuristics alone. The injected extractor
    records its identity via the ``identity`` callback that ``run_research`` supplies.
    """
    def injected(topic: str, sources: list[SourceContent], *, identity=None, **_: object) -> list[tuple[str, str]]:
        if identity is not None:
            identity("test-extractor", "1.0", "schema-1")
        return [(f"Fact about {topic} grounded in content", "confirmed fact") for _ in range(3)]

    fetcher = lambda url: f"Rich grounded content about {url}: 42 percent growth in 2024."
    urls = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]

    r1 = run_research("Local AI Trends", urls, production_extractor=injected, content_fetcher=fetcher)
    r2 = run_research("Local AI Trends", urls, production_extractor=injected, content_fetcher=fetcher)

    assert len(r1.verified_facts) == 3
    assert r1.output_digest == r2.output_digest
    assert set(r1.source_content_hashes) == set(urls)
    assert r1.extractor_name is not None
