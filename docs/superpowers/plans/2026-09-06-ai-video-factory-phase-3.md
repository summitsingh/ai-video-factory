# AI Video Factory Phase 3 Implementation Plan

## Overview

Build a reliable daily production pipeline on top of the existing local renderer. Incremental, testable milestones. Each milestone adds one subsystem with unit tests (fixture-based, no real network) and runs the full suite before proceeding. Preserve existing behavior/tests unless intentionally migrating an interface with documentation.

**Uncommitted files at session start (reported, not modified unless required):** my voice/transition work in `remotion/src/SyntheticVideo.tsx`, `remotion/src/schema.ts`, `src/ai_video_factory/edit_schema.py`, `src/ai_video_factory/video_pipeline.py`. New Phase 3 modules are additive; I avoid editing those four unless a change is unavoidable, and if so I keep it minimal.

## Milestones

### M1 — Trend subsystem (`trend.py`)
- Real adapters: Hacker News (algolia API), Reddit public JSON, Google News RSS, Wikipedia pageviews/popular. Each accepts an injectable transport for tests.
- Normalizer -> normalized candidate schema (exact fields from design). Dedup by `topic_id` (stable hash of cluster_key + canonical title). Store raw snapshots per provider.
- Deterministic scoring from recorded signals; editorial ranking separate and advisory only.
- CLI command `trend-discovery --dry-run` + `config/trend.toml`.
- Tests: adapter parsing with canned JSON, dedup, signal computation, score determinism, dry-run does not write artifacts.

### M2 — Research pipeline (`research_pipeline.py`)
- Source collection from real URLs; source-quality classification (reliable/secondary/unverified) by domain heuristics.
- Claim extraction via injectable extractor (default = LM Studio; tests use deterministic fixture extractor). Each claim carries source IDs + provisional classification.
- Corroboration / contradiction detection -> verified_facts + contradictions.
- Sensitive-topic guardrails (elections/financial/medical/legal/conflict/disaster/person) route to human review or reject.
- Artifacts: sources.json, claims.json, contradictions.json, verified_facts.json, research_brief.md.
- Tests: classification, claim->source traceability, contradiction detection, guardrail routing.

### M3 — Long-form edit schema (`edit_schema.py` extension)
- Add optional long-form fields with defaults (start_seconds, duration_seconds, claim_ids, editorial_purpose, visual_brief, asset_queries, asset_strategy, on_screen_text, transition, music_energy). Backward-compatible: existing short-form docs still validate.
- Tests: new fields parse; old schema still validates; seconds->frames conversion helper.

### M4 — Rights-cleared asset library (`assets.py`)
- Provider abstraction over NASA + Wikimedia (reuse existing download/search functions). Full rights metadata per record + sha256 checksum.
- License gate: missing/ambiguous license = rejection. Selection scoring (relevance, quality, diversity, reuse history, logo/text risk, rights confidence).
- Render requires approved assets only.
- Tests: license allowlist accept/reject, ambiguous rejection, checksum computation, selection scoring ordering, provider search/download with injected transport.

### M5 — QC gates (`qc.py` extension)
- Keep existing technical checks/functions intact. Add `evaluate_editorial_qc` (continuity claim_ids->verified facts, originality scene variety + narration substance, metadata no-overclaim), `evaluate_rights_qc` (every external asset approved provenance). Combine into fail-closed master gate with machine JSON + human Markdown reports.
- Tests: each gate pass/fail, fail-closed blocks downstream, report formats.

### M6 — Upload package (`publisher.py`)
- Assemble master.mp4, thumbnails/alternates, title candidates, description (with sources/attributions), chapters, tags/category/draft metadata, captions, rights manifest, research brief, QC reports, approval.json=pending.
- YouTube publisher interface: class stub, disabled + unconfigured by default; requires approved approval.json, OAuth outside Git, private-first, resumable transfer when later enabled.
- Tests: package completeness, approval.json pending, publish blocked without approval/config.

### M7 — Daily orchestration (`daily_job.py` + CLI)
- Stages via RunStore (trend-discovery, research, candidate-edit, asset-acquisition, render, qc, upload-package). Discover -> rank -> bounded research -> <=1 candidate -> stop at approval gate. Resumable + idempotent (reuse completed stages on fingerprint match). Logs/artifacts/timings/failure reasons. No system cron; documented command.
- CLI `daily-job` with `--dry-run`, `--max-candidates`, config path.
- Tests: idempotency (run twice reuses stages), <=1 candidate, approval gate stops publish, resumability.

### M8 — Integration + final verification
- Wire new commands into `cli.py`. Integrate rights gate into render path so unapproved assets cannot render. Run full suite + lint/typecheck. Update README with local config, dry runs, credentials boundary, safety gates, operator approval.

## Verification gates per milestone
- `uv run pytest -q` green (existing + new tests).
- New module imports cleanly; no real network in test path.
- Deterministic assertions where applicable.

## Definition of done (final)
- Trend discovery uses real adapter interfaces and fixture-tested normalization; no simulated trend output in production path.
- Dry run creates source-backed ranked topic candidates.
- Research provides traceability from narration claims to verified evidence.
- 15-20 min edit timed from actual narration, chaptered + varied visual planning.
- Assets without approved rights metadata cannot render.
- Editorial, rights, technical QC can fail closed.
- Completed job produces private-upload-ready package but cannot publish autonomously.
- Daily job resumable and idempotent.
- All tests pass; existing suite preserved.
- Documentation covers local config, dry runs, credentials boundaries, safety gates, operator approval.

## Risks / notes
- LM Studio script generation is the one non-deterministic external dependency; claim extraction default extractor calls it but is injectable so tests stay offline. If LM Studio is unavailable, research falls back to deterministic rule-based extraction (documented).
- Real adapters must respect rate limits + robots rules; all network opt-in behind config with dry-run capability.
- Keep `EditScene` strict/forbid schema valid: new fields optional with defaults.
