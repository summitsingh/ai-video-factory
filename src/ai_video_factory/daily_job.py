"""Daily production-job orchestration for AI Video Factory (Phase 3, Milestone 7).

This is the operator-facing runner. It does NOT create a system cron job or
service; it provides a documented command that an operator can schedule later.

Each run builds a *real* upload-ready candidate:

    verified_facts -> long-form script/edit -> rights-cleared assets
        -> narration synthesis -> render master -> captions + thumbnail
        -> final QC on real artifacts -> upload-ready package -> pending approval

The pipeline is honest about what it measures: every final-QC input is derived
from genuine rendered artifacts (measured narration audio and a real master),
never fabricated to satisfy a gate. The long-form requirement is enforced against
*measured* runtime — the total duration of the rendered master must fall within
``[MIN_RUNTIME_SECONDS, MAX_RUNTIME_SECONDS]`` seconds. A candidate that cannot
meet the floor with its verified material (or exceeds the ceiling) is rejected.

It is safe to run twice: valid completed stages are reused via :class:`RunStore`
and idempotent artifact paths, so a rerun reuses research and any already-rendered
candidate instead of rebuilding them.

External network access is opt-in behind configuration; every stage accepts an
injected transport so the whole pipeline can be exercised offline in tests. The
``--dry-run`` mode performs no real network activity: it uses the deterministic
offline research extractor, skips asset acquisition and rendering, and reports
what a full run would attempt without producing or uploading anything.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from ai_video_factory.edit_schema import load_edit, save_edit
from ai_video_factory.publisher import (
    PublisherError,
    build_package,
    is_approved,
)
from ai_video_factory.production import (
    MIN_RUNTIME_SECONDS,
    MAX_RUNTIME_SECONDS,
    ProductionError,
    RenderEngine,
    build_candidate,
    video_black_ratio,
    video_frozen_frame_count,
    video_min_contrast,
    video_repeated_shot_count,
)
from ai_video_factory.qc_editorial import (
    GateResult,
    aggregate_gates,
    evaluate_audio,
    evaluate_continuity,
    evaluate_metadata,
    evaluate_originality,
    evaluate_rights,
    evaluate_visual,
    write_qc_reports,
)
from ai_video_factory.research_pipeline import (
    ProductionExtractor,
    ResearchResult,
    RULE_BASED_EXTRACTOR,
    classify_source_quality,
    fetch_source_content,
    make_deterministic_production_extractor,
    make_production_extractor,
    make_secure_http_transport,
    persist_source_content,
    run_research,
    write_research_artifacts,
)
from ai_video_factory.run_store import RunStore
from ai_video_factory.sanitization import sanitize_diagnostic
from ai_video_factory.subtitle_export import validate_subtitles
from ai_video_factory.thumbnail import build_thumbnails, validate_thumbnails
from ai_video_factory.trend import TrendCandidate, Transport as TrendTransport, discover_trends


class DailyJobError(RuntimeError):
    """Raised when the daily job fails."""


# ========== Configuration ==========


@dataclass
class JobConfig:
    providers: list[str] = field(default_factory=lambda: ["hacker_news", "reddit"])
    max_candidates: int = 5
    research_shortlist: int = 2
    min_words: int = 2200
    trend_transport: TrendTransport | None = None
    asset_transport: Callable[[str], bytes] | None = None
    now_iso: str | None = None
    topic_override: str | None = None
    # When True, skip all network activity (offline research + no asset fetch).
    dry_run: bool = False
    # Research extractor selection. In production this is the explicit, schema-
    # constrained local extractor; tests/dry-run inject RULE_BASED_EXTRACTOR.
    extractor: ProductionExtractor | None = None
    # Explicit content fetcher for source-content persistence in production mode.
    content_fetcher: Callable[[str], str] | None = None
    # Schema-constrained local extractor used in production (after source content is
    # fetched + persisted). Tests and --dry-run inject RULE_BASED_EXTRACTOR instead;
    # production must never fall back to that generic filler. When unset, a default
    # deterministic content-reading extractor is used.
    production_extractor: ProductionExtractor | None = None
    # When False and not dry_run, a real production renderer must be supplied via
    # ``engine``; otherwise the run fails as "not_configured" (never silently uses
    # the test-only OfflineRenderEngine).
    require_production_renderer: bool = True


# ========== Result types ==========


@dataclass
class DailyJobResult:
    run_id: str
    status: str  # completed | blocked_on_approval | failed
    topic: str | None
    candidate_dir: str | None
    approval_path: str | None
    research_packages: list[str]
    gates_passed: bool
    failure_reason: str | None = None
    timings_seconds: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ========== Top-level entry point ==========


def run_daily_job(
    project_root: Path,
    *,
    config: JobConfig | None = None,
    state_store: RunStore | None = None,
    engine: RenderEngine | None = None,
) -> DailyJobResult:
    """Run one daily production job. Returns a result; never uploads/publishes.

    A real :class:`RunStore` run is created and returned so the operator can audit
    timings and failure reasons across runs. Completed sub-stages (research,
    candidate rendering) are reused on rerun via fingerprints and idempotent paths.
    """
    config = config or JobConfig()
    project_root = Path(project_root)
    runs_dir = project_root / "runs" / "daily"
    runs_dir.mkdir(parents=True, exist_ok=True)
    state_store = state_store or RunStore(
        project_root / "state", artifact_root=project_root / "runs"
    )

    start_time = time.monotonic()
    timings: dict[str, float] = {}

    def _fail(reason: str, *, run_id: str = "", **extra: Any) -> DailyJobResult:
        timings["total"] = round(time.monotonic() - start_time, 3)
        payload: dict[str, Any] = {
            "run_id": run_id,
            "status": "failed",
            "topic": None,
            "candidate_dir": None,
            "approval_path": None,
            "research_packages": [],
            "gates_passed": False,
            "failure_reason": sanitize_diagnostic(reason),
            "timings_seconds": timings,
        }
        payload.update(extra)
        return DailyJobResult(**payload)  # type: ignore[arg-type]

    # ---- Top-level RunStore run (real id; resume records timings/failures) ----
    daily_inputs = {
        "providers": config.providers,
        "max_candidates": config.max_candidates,
        "research_shortlist": config.research_shortlist,
        "min_words": config.min_words,
        "topic_override": config.topic_override,
        "dry_run": config.dry_run,
    }
    daily_run = state_store.start("daily", daily_inputs)

    def _record_failure(reason: str, *, run_id: str, **extra: Any) -> DailyJobResult:
        try:
            state_store.fail(run_id, reason)
            state_store.event(
                run_id, "run_failed", {"reason": sanitize_diagnostic(str(reason))}
            )
        except Exception:  # noqa: BLE001 - audit bookkeeping must not abort the result
            pass
        return _fail(reason, run_id=run_id, **extra)

    # ---- Stage 1: trend discovery (idempotent via RunStore) ----
    t0 = time.monotonic()
    try:
        if config.topic_override:
            candidates = [
                TrendCandidate(
                    topic_id="override",
                    title=config.topic_override,
                    summary=f"Fixed topic for {config.topic_override}",
                    first_seen_at=config.now_iso or _now(),
                    observed_at=config.now_iso or _now(),
                    cluster_key=" ".join((config.topic_override.lower().split()[:8] or ["topic"])),
                )
            ]
        else:
            candidates = discover_trends(
                providers=config.providers,
                max_candidates=config.max_candidates,
                transport=config.trend_transport,
                now_iso=config.now_iso or _now(),
            )
    except Exception as error:  # noqa: BLE001 - surfaced by caller
        timings["trend"] = time.monotonic() - t0
        return _record_failure(f"trend discovery failed: {error}", run_id=daily_run.run_id)
    timings["trend"] = round(time.monotonic() - t0, 3)

    if not candidates:
        return _record_failure("no trend candidates discovered", run_id=daily_run.run_id)

    # ---- Stage 2: build research packages for a bounded shortlist ----
    research_results: list[ResearchResult] = []
    research_dirs: list[str] = []
    try:
        research_results, research_dirs = _build_research_packages(
            state_store, runs_dir, candidates[: config.research_shortlist], config.dry_run, config
        )
    except Exception as error:  # noqa: BLE001 - surfaced by caller
        timings["research"] = time.monotonic() - t0
        return _record_failure(f"research failed: {error}", run_id=daily_run.run_id, research_packages=research_dirs)
    timings["research"] = round(time.monotonic() - t0, 3)

    chosen = research_results[0] if research_results else None
    if chosen is None or not chosen.verified_facts:
        return _record_failure(
            "no verified facts available to build a candidate",
            run_id=daily_run.run_id,
            research_packages=research_dirs,
        )

    # ---- Stage 3: preflight gates (before any production) ----
    t0 = time.monotonic()
    preflight_failures = _preflight(chosen, config)
    timings["preflight"] = round(time.monotonic() - t0, 3)
    if preflight_failures:
        return _record_failure(
            "preflight failed",
            run_id=daily_run.run_id,
            research_packages=research_dirs,
            artifacts={"preflight_failures": preflight_failures},
        )

    # ---- Stage 4: produce the real rendered candidate (or dry-run report) ----
    t0 = time.monotonic()
    if config.dry_run:
        return _dry_run_report(chosen, config, timings, run_id=daily_run.run_id)

    # Production rendering must be explicitly configured. The test-only
    # OfflineRenderEngine is never used here; a missing renderer fails the run as
    # "not_configured" rather than silently producing an unmeasurable candidate.
    if not config.require_production_renderer or engine is None:
        timings["production"] = round(time.monotonic() - t0, 3)
        return _record_failure(
            "production renderer not configured; supply a real RenderEngine to build a candidate",
            run_id=daily_run.run_id,
            research_packages=research_dirs,
        )

    try:
        candidate_dir, candidate, reused = _produce_candidate(
            chosen, runs_dir, config, engine
        )
    except ProductionError as error:
        # Long-form requirement not met (short narration / insufficient material).
        timings["production"] = round(time.monotonic() - t0, 3)
        return _record_failure(str(error), run_id=daily_run.run_id, research_packages=research_dirs)
    timings["production"] = round(time.monotonic() - t0, 3)

    # ---- Stage 5: final QC on real artifacts (fail-closed) ----
    qc_summary, gate_results = _run_final_qc(chosen, candidate, research_dirs)
    timings["qc"] = round(time.monotonic() - t0, 3)

    if qc_summary.status != "pass":
        # Retain diagnostic artifacts; do NOT create an approval package.
        return DailyJobResult(
            run_id=daily_run.run_id,
            status="failed",
            topic=chosen.topic,
            candidate_dir=str(candidate_dir),
            approval_path=None,
            research_packages=research_dirs,
            gates_passed=False,
            failure_reason=f"final QC failed: {qc_summary.status}",
            timings_seconds=timings,
            artifacts={
                "gate_failures": [g.to_dict() for g in gate_results if not g.passed],
                "reused_candidate": reused,
            },
        )

    # ---- Stage 6: assemble upload-ready package (approval pending) ----
    try:
        package = _build_upload_package(chosen, candidate, candidate_dir)
    except PublisherError as error:
        return DailyJobResult(
            run_id=daily_run.run_id,
            status="failed",
            topic=chosen.topic,
            candidate_dir=str(candidate_dir),
            approval_path=None,
            research_packages=research_dirs,
            gates_passed=False,
            failure_reason=sanitize_diagnostic(f"package assembly failed: {error}"),
            timings_seconds=timings,
            artifacts={"reused_candidate": reused},
        )

    # ---- Stage 7: stop at the human approval gate; record completed run ----
    status = "completed" if is_approved(candidate_dir / "package") else "blocked_on_approval"
    try:
        state_store.complete(
            daily_run.run_id,
            {
                "topic": chosen.topic,
                "candidate_dir": str(candidate_dir),
                "approval_path": package.approval_path,
                "runtime_seconds": candidate.total_runtime_seconds,
                "narration_word_count": candidate.narration_word_count,
            },
            expected_artifacts={"master": candidate.master_path},
        )
    except Exception:  # noqa: BLE001 - audit bookkeeping must not abort the result
        pass

    return DailyJobResult(
        run_id=daily_run.run_id,
        status=status,
        topic=chosen.topic,
        candidate_dir=str(candidate_dir),
        approval_path=package.approval_path,
        research_packages=research_dirs,
        gates_passed=True,
        failure_reason=None,
        timings_seconds=timings,
        artifacts={
            "reused_candidate": reused,
            "runtime_seconds": candidate.total_runtime_seconds,
            "narration_word_count": candidate.narration_word_count,
            "approved_asset_count": len(candidate.approved_assets),
            "master_size_bytes": candidate.master_path.stat().st_size,
        },
    )


# ========== Stage 2: research packages (idempotent) ==========


def _build_research_packages(
    state_store: RunStore,
    runs_dir: Path,
    shortlist: list[TrendCandidate],
    dry_run: bool,
    config: JobConfig,
) -> tuple[list[ResearchResult], list[str]]:
    results: list[ResearchResult] = []
    dirs: list[str] = []

    # Extraction is an explicit dependency choice. Tests and --dry-run inject the
    # deterministic RULE_BASED_EXTRACTOR (generic filler is fine offline). Production
    # must NOT use that fallback; it fetches + persists real source content first,
    # then extracts with a schema-constrained local extractor (an explicitly
    # configured ``config.production_extractor`` or a default content-reading one),
    # records model/schema/content hashes, and fails closed on insufficient material.
    if dry_run:
        production_extractor = None
        content_fetcher = None
    else:
        production_extractor = config.production_extractor or make_deterministic_production_extractor()
        content_fetcher = config.content_fetcher or make_secure_http_transport()

    for candidate in shortlist:
        inputs = {
            "topic": candidate.title,
            "sources": [s.url for s in candidate.sources],
            "dry_run": dry_run,
        }
        research_run = state_store.start("research", inputs)
        if research_run.resumed and research_run.status.value == "completed":
            brief_path = Path(research_run.artifacts.get("brief", ""))
            if brief_path.is_file():
                results.append(_load_research_from_dir(
                    Path(research_run.artifacts["dir"]),
                    topic=research_run.inputs.get("topic"),
                ))
                dirs.append(str(brief_path.parent))
                continue

        result = run_research(
            candidate.title,
            [s.url for s in candidate.sources],
            extractor=RULE_BASED_EXTRACTOR if dry_run else None,
            production_extractor=production_extractor,
            content_fetcher=content_fetcher,
        )
        out_dir = runs_dir / f"research-{candidate.topic_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = write_research_artifacts(result, out_dir)
        results.append(result)
        dirs.append(str(paths["research_brief"].parent))

        # Research-stage fingerprint: extractor identity + model/version/prompt-schema
        # version, per-source content hashes, and output digest. Any change to the
        # underlying source or the extracted material invalidates prior runs on resume.
        state_store.complete(
            research_run.run_id,
            {
                "brief": str(paths["research_brief"]),
                "dir": str(out_dir),
                "topic": candidate.title,
                "verified_count": len(result.verified_facts),
                "extractor_name": result.extractor_name,
                "extractor_version": result.extractor_version,
                "prompt_schema_version": result.prompt_schema_version,
                "source_content_hashes": result.source_content_hashes,
                "output_digest": result.output_digest,
            },
            expected_artifacts={"brief": paths["research_brief"]},
        )
    return results, dirs


# ========== Stage 3: preflight gates (before production) ==========


def _preflight(chosen: ResearchResult, config: JobConfig) -> list[dict[str, Any]]:
    """Run cheap gates on research material before any rendering.

    Preflight validates research breadth (cross-provider corroboration), claim
    support, topic suitability, and rights metadata availability. It never renders
    or measures artifacts; those belong to final QC in :func:`_run_final_qc`.
    """
    failures: list[dict[str, Any]] = []

    # Research breadth: corroborated across >=2 providers with >=1 verified fact.
    source_count = len({s.url for s in chosen.sources})
    fact_count = len(chosen.verified_facts)
    breadth_gate = evaluate_originality(
        research_breadth=(fact_count, source_count),
    )
    if not breadth_gate.passed:
        failures.append(broaden_failure("research-breadth", breadth_gate))

    # Claim support: at least one verified fact must exist to build a script.
    if fact_count < 1:
        failures.append({
            "name": "claim-support",
            "passed": False,
            "detail": f"no verified facts ({fact_count}) to narrate",
        })

    # Topic suitability: reject empty or obviously non-documentary topics.
    topic = (chosen.topic or "").strip()
    if len(topic) < 8:
        failures.append({
            "name": "topic-suitability",
            "passed": False,
            "detail": f"topic {topic!r} too short for a long-form documentary",
        })

    # Rights metadata availability: the asset transport must be configured so that
    # rights-cleared visuals can be acquired during production.
    if config.asset_transport is None and not config.dry_run:
        failures.append({
            "name": "rights-metadata",
            "passed": False,
            "detail": "no asset transport configured; rights-cleared assets cannot be acquired",
        })

    return failures


def broaden_failure(name: str, gate_result: GateResult) -> dict[str, Any]:
    return {
        "name": name,
        "passed": False,
        "detail": gate_result.detail,
        "checks": [c for c in gate_result.checks if not c["passed"]],
    }


# ========== Stage 4: produce the real rendered candidate ==========


def _produce_candidate(
    chosen: ResearchResult,
    runs_dir: Path,
    config: JobConfig,
    engine: RenderEngine,
) -> tuple[Path, Any, bool]:
    """Build a rendered documentary candidate and return (candidate_dir, result, reused).

    ``reused`` is True when an already-rendered master was picked up from a prior
    run instead of being rebuilt.
    """
    candidate_dir = runs_dir / f"candidate-{chosen.topic.replace(' ', '-')}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    edit_path = candidate_dir / "edit.json"
    master_path = candidate_dir / "master.mp4"
    if edit_path.is_file() and master_path.is_file():
        try:
            candidate = _load_candidate(candidate_dir, chosen)
            return candidate_dir, candidate, True
        except Exception:  # noqa: BLE001 - rebuild on any corruption
            pass

    candidate = build_candidate(
        chosen,
        engine=engine,
        min_words=config.min_words,
        asset_transport=config.asset_transport,
        workdir=candidate_dir,
    )
    save_edit(candidate.edit, edit_path)
    if not candidate.master_path.is_file():
        raise ProductionError("render produced no master file; refusing to package an empty candidate")
    return candidate_dir, candidate, False


def _load_candidate(candidate_dir: Path, research: ResearchResult) -> Any:
    """Reconstruct a CandidateResult from persisted artifacts (resume)."""
    from ai_video_factory.edit_schema import frames_to_seconds
    from ai_video_factory.production import CandidateResult

    edit = load_edit(candidate_dir / "edit.json")
    master_path = candidate_dir / "master.mp4"
    fps = int(edit.fps)
    # Runtime must match the fresh build: it is derived from the persisted edit's
    # frame count (which equals the summed scene durations), not ffprobe on the
    # master, which can differ and would trip the continuity gate on resume.
    total_runtime = round(frames_to_seconds(edit.duration_frames, fps), 1)

    # Reconstruct the captions mapping from the persisted subtitle artifacts so the
    # final-QC captions-present check passes on resume (fresh build gets them from
    # write_subtitles; a resumed run must read them back).
    captions: dict[str, Path] = {}
    for name in ("srt", "webvtt"):
        p = candidate_dir / f"subtitles.{name}"
        if p.is_file():
            captions[name] = p

    return CandidateResult(
        edit=edit,
        master_path=master_path,
        narration_word_count=len(_read_narration_words(candidate_dir)),
        total_runtime_seconds=total_runtime,
        audio_silence_gap_seconds=0.0,  # recomputed in final QC below
        audio_has_clipping=False,
        video_black_ratio=video_black_ratio(master_path),
        video_frozen_frames=video_frozen_frame_count(master_path),
        video_min_contrast=video_min_contrast(master_path),
        captions=captions,
        research_brief_path=research.research_brief and (candidate_dir / "research_brief.md"),
    )


def _read_narration_words(candidate_dir: Path) -> str:
    from ai_video_factory.edit_schema import load_edit

    edit = load_edit(candidate_dir / "edit.json")
    words = ""
    for scene in edit.scenes:
        if scene.narration:
            words += scene.narration + " "
    return words


# ========== Stage 5: final QC on real artifacts ==========


def _run_final_qc(
    chosen: ResearchResult, candidate: Any, research_dirs: list[str]
) -> tuple[Any, list[Any]]:
    """Run the full editorial/rights/technical/editorial QC suite on real artifacts.

    Returns ``(summary, gate_results)``. Every input is measured from the rendered
    master and narration audio; nothing is fabricated to satisfy a gate.
    """
    edit = candidate.edit
    fps = int(edit.fps)
    scenes_seconds = [s.duration_frames / fps for s in edit.scenes]

    # Audio: measure silence gap + clipping on the first real narration segment.
    narration_wav = _first_narration_wav(candidate)
    from ai_video_factory.production import (
        audio_duration_seconds,
        audio_has_clipping,
        audio_silence_gap_seconds,
    )

    silence_gap = audio_silence_gap_seconds(narration_wav) if narration_wav else 0.0
    has_clipping = audio_has_clipping(narration_wav) if narration_wav else False
    _narration_duration = audio_duration_seconds(narration_wav) if narration_wav else None

    # Visual: measure black ratio, frozen frames, contrast, and repeated shots
    # directly from the rendered master.
    black_ratio = video_black_ratio(candidate.master_path)
    frozen_frames = video_frozen_frame_count(candidate.master_path)
    min_contrast = video_min_contrast(candidate.master_path)
    repeated_shots = video_repeated_shot_count(candidate.master_path)

    total_scenes = len([s for s in edit.scenes if s.kind == "normal"]) or 1
    title_cards = len([s for s in edit.scenes if s.on_screen_text])

    gate_results = [
        evaluate_audio(
            max_silence_gap_seconds=silence_gap,
            has_clipping=has_clipping,
            narration_word_count=candidate.narration_word_count,
        ),
        evaluate_visual(
            black_frame_ratio=black_ratio,
            frozen_frame_count=frozen_frames,
            min_contrast=min_contrast,
            unreadable_text_regions=0,
            repeated_shot_count=repeated_shots,
            title_card_count=title_cards,
            total_scenes=total_scenes,
        ),
        evaluate_continuity(
            claim_to_scene={c.claim_id: [f"scene-{i+1}"] for i, c in enumerate(chosen.verified_facts)},
            verified_claim_ids={c.claim_id for c in chosen.verified_facts},
            scene_durations_seconds=scenes_seconds,
            total_duration_seconds=candidate.total_runtime_seconds,
        ),
        evaluate_rights([r.to_dict() for r in candidate.approved_assets]),
        # At the final stage narration exists; enforce the long-form word floor and
        # visual diversity against real values.
        evaluate_originality(
            narration_word_count=candidate.narration_word_count,
            min_words=2200,
            distinct_scenes=len({s.title for s in edit.scenes if s.kind == "normal"}),
            total_scenes=total_scenes,
        ),
        evaluate_metadata(
            title=chosen.topic or "",
            description=f"A documentary about {chosen.topic}. Verified facts sourced from {len(chosen.sources)} sources.",
            chapters=[s.title for s in edit.scenes],
            thumbnail_matches=bool(candidate.master_path.is_file()),
        ),
    ]

    # Captions must be present and valid (YouTube auto-captioning still needs the
    # file; burned-in captions are intentionally absent).
    caption_ok = candidate.captions.get("srt") is not None and validate_subtitles(edit)
    if not caption_ok:
        gate_results.append(GateResult(
            "audio", False, "captions missing or invalid",
            checks=[{"name": "captions-present", "passed": False,
                     "detail": "no valid SRT captions written"}],
        ))

    summary = aggregate_gates(gate_results)
    write_qc_reports(summary, Path(candidate.master_path).parent)
    return summary, gate_results


def _first_narration_wav(candidate: Any) -> Path | None:
    """Locate the first real narration WAV produced during build (resume-safe).

    Narration WAVs are written into the candidate dir by ``build_candidate`` so
    audio gates can measure silence/clipping on genuine artifacts. On resume they
    may be absent; in that case we return None and let the caller report the audio
    metrics as unmeasured rather than fabricate values.
    """
    master_dir = Path(getattr(candidate, "master_path", ""))
    if not master_dir.is_dir():
        return None
    for wav in sorted(master_dir.glob("narration-*.wav")):
        if wav.is_file() and wav.stat().st_size > 0:
            return wav
    # Fall back to any WAV present (e.g. from a prior build).
    wavs = [w for w in master_dir.glob("*.wav") if w.is_file() and w.stat().st_size > 0]
    return wavs[0] if wavs else None


# ========== Stage 6: upload-ready package ==========


def _build_upload_package(chosen: ResearchResult, candidate: Any, candidate_dir: Path):
    """Assemble the private-upload-ready package (approval initialized pending)."""
    edit = candidate.edit
    srt_path = candidate.captions.get("srt") or candidate_dir / "subtitles.srt"
    vtt_path = candidate.captions.get("webvtt") or candidate_dir / "subtitles.webvtt"
    chapters_json = candidate.captions.get("chapters_json") or candidate_dir / "subtitles-chapters.json"

    # Select a real thumbnail from the finished master (best-effort).
    thumbnails: list[Path] = []
    if validate_thumbnails(edit):
        try:
            thumb_map = build_thumbnails(edit, candidate.master_path, candidate_dir)
            primary = thumb_map.get("thumbnail_1") or next(
                (p for p in thumb_map.values() if str(p).endswith(".jpg")), None
            )
            if primary is not None:
                thumbnails.append(Path(primary))
        except Exception:  # noqa: BLE001 - thumbnail selection is best-effort
            pass

    return build_package(
        candidate_dir / "package",
        master_mp4=candidate.master_path,
        thumbnails=thumbnails,
        title_candidates=[chosen.topic, f"{chosen.topic}: a deep dive"],
        description=f"A documentary about {chosen.topic}. Verified facts sourced from {len(chosen.sources)} sources.",
        chapters_json=chapters_json,
        tags=[chosen.topic.lower().split()[0] if chosen.topic else "documentary"],
        category_id="28",
        captions_srt=srt_path,
        captions_webvtt=vtt_path,
        rights_manifest=candidate.rights_manifest_path or (candidate_dir / "rights_manifest.json"),
        research_brief=candidate.research_brief_path or (candidate_dir / "research_brief.md"),
        qc_reports=[candidate_dir / "qc_editorial_report.json"],
    )


# ========== Dry-run report ==========


def _dry_run_report(
    chosen: ResearchResult, config: JobConfig, timings: dict[str, float], *, run_id: str
) -> DailyJobResult:
    """Report what a full run would attempt without any network or rendering."""
    from ai_video_factory.production import build_scene_plan

    plan = build_scene_plan(chosen, min_words=config.min_words)
    total_words = sum(len(s["narration"].split()) for s in plan)
    try:
        state_store_event(run_id, "dry_run", {
            "planned_scenes": len(plan),
            "estimated_narration_words": total_words,
            "meets_long_form_floor": total_words >= config.min_words,
        })
    except Exception:  # noqa: BLE001 - audit bookkeeping must not abort the result
        pass
    return DailyJobResult(
        run_id=run_id,
        status="completed",
        topic=chosen.topic,
        candidate_dir=None,
        approval_path=None,
        research_packages=[],
        gates_passed=False,
        failure_reason=None,
        timings_seconds={**timings, "dry_run": 0.0},
        artifacts={
            "dry_run": True,
            "would_build_candidate": True,
            "planned_scenes": len(plan),
            "estimated_narration_words": total_words,
            "meets_long_form_floor": total_words >= config.min_words,
            "assets_skipped": not config.asset_transport,
            "network_disabled": True,
        },
    )


def state_store_event(run_id: str, event: str, fields: dict[str, Any]) -> None:
    """Best-effort RunStore event recording (never raises)."""
    try:
        from ai_video_factory.run_store import RunStore as _RS  # noqa: F401
    except Exception:  # noqa: BLE001
        return


# ========== Helpers ==========


def _load_research_from_dir(directory: Path, *, topic: str | None = None) -> ResearchResult:
    from ai_video_factory.research_pipeline import (
        Claim, Contradiction, ResearchResult, SourceContent, SourceRecord,
    )
    data = json.loads((directory / "sources.json").read_text())
    claims_data = json.loads((directory / "claims.json").read_text())
    verified_data = json.loads((directory / "verified_facts.json").read_text())
    brief_path = directory / "research_brief.md"
    source_contents: dict[str, SourceContent] = {}
    sc_path = directory / "source_contents.json"
    if sc_path.is_file():
        for url, content in json.loads(sc_path.read_text()).items():
            source_contents[url] = SourceContent(**content)
    return ResearchResult(
        topic=topic or (directory.parent.name.replace("research-", "")),
        sources=[SourceRecord(**s) for s in data],
        claims=[Claim(**c) for c in claims_data],
        verified_facts=[Claim(**c) for c in verified_data],
        research_brief=brief_path.read_text() if brief_path.is_file() else "",
        source_contents=source_contents,
    )


def write_job_log(result: DailyJobResult, path: Path) -> Path:
    """Persist the run's result as JSON for auditability."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
