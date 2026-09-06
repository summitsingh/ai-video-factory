# AI Video Factory Phase 3 Design

## Goal

Transform the existing local video renderer into a reliable daily production pipeline that can discover real trending topics, research them with traceable claim verification, produce a 15-20 minute documentary-style video package, run editorial + rights + technical QC gates, and assemble a private-upload-ready package. The system stops at an explicit human approval gate before any upload or scheduling action. No automatic public publishing is implemented in this phase.

## Scope

Phase 3 adds: real trend discovery (no simulated output), a research/claim-verification pipeline, long-form script + storyboard generation, a rights-cleared asset library with license gates, editorial QC gates layered over existing technical checks, an upload-package builder that cannot publish autonomously, and a resumable/idempotent daily job runner.

Phase 3 does NOT add: automatic public YouTube publishing, paid asset providers, system-level cron/services, model downloads, driver/GPU changes, or any change to the local LM Studio inference boundary beyond consuming its script-generation capability through the existing adapter.

## Architecture

The pipeline is a sequence of RunStore-backed stages keyed by content-addressed input fingerprints so each stage is resumable and idempotent:

```text
trend-discovery -> topic-shortlist -> research(<bounded shortlist>) -> candidate-edit -> asset-acquisition(rights gate) -> render -> qc(technical + editorial + rights) -> upload-package(approval=pending) -> [human approval] -> publish(later phase, disabled)
```

Each stage writes a manifest (see `run_store.py`), an append-only JSONL event log, and its artifacts under `data/projects/<project-id>/`. The daily job reuses completed stages when inputs are unchanged and produces at most one final candidate per run.

### Component boundaries

| Component | Module | Responsibility | Inputs | Outputs |
|---|---|---|---|---|
| Trend discovery | `trend.py` | Real provider adapters, normalization, dedup, deterministic scoring, editorial ranking | config (providers/weights/cadence/exclusions), optional network | ranked candidate shortlist + raw snapshots |
| Research pipeline | `research_pipeline.py` | Source collection, source-quality classification, claim extraction, corroboration/contradiction, verified facts | topic + sources | `sources.json`, `claims.json`, `contradictions.json`, `verified_facts.json`, `research_brief.md` |
| Long-form script/storyboard | `script_generator.py` (extended) | 15-20 min narration timed from TTS, 8-12 chapters, varied visual plan, per-scene schema | research brief + topic | long-form edit document |
| Asset library | `assets.py` | Provider abstraction over NASA/Wikimedia, rights metadata, license gate, selection scoring | scene asset_queries | approved/rejected asset records with provenance |
| QC gates | `qc.py` (extended) | Technical checks (existing) + editorial/rights/continuity/originality/metadata gates | master + edit + assets | machine JSON + human Markdown; fail-closed |
| Upload package | `publisher.py` | Assemble upload-ready package, initialize approval.json=pending; YouTube interface stubbed/disabled | all artifacts | private-upload-ready package dir |
| Daily orchestration | `daily_job.py` + CLI | Discover -> rank -> bounded research -> <=1 candidate -> stop at approval gate; resumable/idempotent | config + RunStore | one candidate package, logs, timings |

## Schemas

### Trend candidate (normalized)

```json
{
  "topic_id": "stable hash",
  "title": "string",
  "summary": "string",
  "first_seen_at": "ISO-8601",
  "observed_at": "ISO-8601",
  "cluster_key": "string",
  "sources": [
    {"provider": "string", "url": "https://...", "published_at": "ISO-8601 or null", "raw_signal": {}}
  ],
  "signals": {
    "freshness": 0.0, "velocity": 0.0, "source_diversity": 0.0,
    "competition_risk": 0.0, "visual_potential": 0.0, "audience_fit": 0.0
  },
  "score": 0.0,
  "confidence": 0.0
}
```

- `topic_id` is a stable hash of the cluster key + canonical title so dedup across runs is deterministic.
- Raw provider snapshots are stored separately (per-provider JSON) so scoring is reproducible and auditable.
- Scoring is fully deterministic from recorded signals; an LLM may rank/recommend but must never claim a topic is trending without attached signals.

### Research artifacts

- `sources.json` — collected sources with quality classification (`reliable` / `secondary` / `unverified`) and provider metadata.
- `claims.json` — narrated factual claims, each carrying source IDs and a provisional classification (confirmed fact / reported claim / estimate / opinion / analysis/speculation).
- `contradictions.json` — detected conflicts between sources with resolution notes.
- `verified_facts.json` — claims that survived corroboration/contradiction, tagged confirmed; the writer receives only these plus clearly labelled uncertainty.
- `research_brief.md` — human-readable synthesis for the writer.

### Long-form edit scene (extended EditScene)

```json
{
  "id": "scene-01",
  "start_seconds": 0.0,
  "duration_seconds": 6.8,
  "narration": "string",
  "claim_ids": ["claim-01"],
  "editorial_purpose": "hook | evidence | explanation | transition | payoff",
  "visual_brief": "string",
  "asset_queries": ["string"],
  "asset_strategy": "licensed_clip | licensed_image | public_domain | map | chart | generated_visual | source_excerpt",
  "on_screen_text": "string or null",
  "transition": "cut | dissolve | fade",
  "music_energy": "low | medium | high"
}
```

Backward compatibility: existing `EditScene` fields (`from_frame`, `duration_frames`, `title`, `caption`, etc.) remain valid. New long-form fields default sensibly so short-form documents still validate; the pipeline converts seconds->frames at render time using fps.

### Asset rights record

Every asset records: original provider URL, download URL, provider + creator, license name + canonical license URL, attribution requirements, acquisition time, checksum (sha256), media properties, candidate scene IDs, human-review status (`pending` / `approved` / `rejected`), and rejection reason where applicable. An asset without approved rights metadata cannot enter a render.

Selection scoring weights: relevance to scene, resolution/technical quality, visual diversity, duplicate/reuse history, logo/text risk, rights confidence.

### Approval gate

`approval.json` initialized as `{"status": "pending", ...}`. The upload-package stage completes but cannot publish. A later publisher (Phase 4) requires an approved `approval.json`, OAuth credentials stored outside Git, private-first upload, and resumable transfer.

## Data flow

1. Trend discovery fetches real provider data, normalizes to candidates, dedups by cluster key, computes deterministic signals/scores, returns a ranked shortlist.
2. Research pipeline runs for a bounded number of top candidates (configurable), producing traceable artifacts from narration claims back to verified evidence. Sensitive-topic guardrails route high-stakes topics for human review or rejection.
3. Script generator produces one long-form candidate edit document timed from actual TTS duration, with 8-12 chapters and varied visual planning.
4. Asset library acquires rights-cleared media per scene; license gate rejects assets without approved provenance before render.
5. Render (existing Remotion/FFmpeg path) produces master.mp4.
6. QC runs technical + editorial + rights gates; any failure blocks the upload-package stage.
7. Publisher assembles the private-upload-ready package and initializes approval.json=pending.

## Configuration

`config/trend.toml` (new) holds: enabled providers, provider-specific settings (user agent, rate limits), scoring weights, cadence, topic exclusions, regional settings, research bounds (max candidates, max sources per candidate), asset provider toggles, and QC gate thresholds. Optional credential-backed providers are disabled by default. Secrets live outside Git; configuration is non-secret.

## Safety model

- Missing or ambiguous license = rejection. No paid provider until API access + license policy are represented in config and explicitly enabled.
- Never invent facts, trend scores, engagement metrics, citations, or asset licenses. Scores derive only from recorded signals; sources carry real URLs.
- External network operations are opt-in behind configuration and dry-run capable (dry run fetches/normalizes but does not download media or write artifacts).
- Fail-closed: a failed gate blocks the upload-package stage; no publish without approved approval.json.
- No scraping/republishing of arbitrary YouTube/TikTok/news/social video. Assets only from explicitly allowed providers (NASA public domain, Wikimedia Commons licensed files) with recorded provenance.
- Credentials/OAuth/API keys/cookies/browser profiles never written to Git.

## Test strategy

- Unit tests use recorded fixtures; no real network calls in the production test path. Provider adapters accept an injectable transport so tests feed canned JSON responses.
- Scoring/dedup/clustering logic tested against fixed inputs with deterministic assertions.
- Research classification + contradiction detection tested against fixture source/claim sets.
- Rights gate tested: approved asset passes, missing-license asset rejected, ambiguous license rejected.
- QC gates tested for pass/fail on both machine and human reports; fail-closed behavior asserted.
- Daily job idempotency/resumability tested via RunStore with fixed fingerprints.
- Existing test suite preserved; regression tests added for all new schemas and gates.

## Migration plan

- `research.py` simulated functions are replaced by real adapters in `trend.py`. The old module is retained as a thin deprecated shim only if any external caller depends on it; otherwise callers switch to the new API. No production path uses simulated output after migration.
- `EditScene` gains optional long-form fields with defaults so existing short-form documents and tests continue to validate unchanged.
- `qc.py` keeps its existing technical checks and function signatures (`evaluate_qc`, `write_qc_reports`) intact; editorial/rights gates are added as new functions and integrated into the pipeline without breaking existing callers/tests.

## Out of scope (explicit)

- Automatic public YouTube publishing (Phase 4).
- Paid asset providers.
- System-level cron/systemd services (operator schedules a documented command later).
- Model downloads, driver/GPU changes, system-wide configuration.
- Changes to the LM Studio inference boundary beyond consuming script generation through the existing adapter.

## Verification

- Dry run creates source-backed ranked topic candidates with recorded signals.
- Research output provides traceability from narration claims to verified evidence (claim_ids -> sources.json).
- A 15-20 minute edit is timed from actual narration and has chaptered, varied visual planning.
- Assets without approved rights metadata cannot render.
- Editorial, rights, and technical QC can fail closed.
- A completed job produces a private-upload-ready package but cannot publish autonomously.
- The daily job is resumable and idempotent.
- All tests pass; existing suite preserved.
