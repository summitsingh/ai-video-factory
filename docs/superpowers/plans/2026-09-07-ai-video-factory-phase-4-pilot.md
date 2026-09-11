# Phase 4 — Real Production Pilot (Plan)

**Status:** APPROVED by operator 2026-09-08. Execute end-to-end; stop at `approval.json: pending`.

## Topic

"Water Beyond Earth: Where It Is and Why It Means We're Looking"
- Channel lane: cinematic science / space / future-tech. Audience: curious adults. Tone: warm, authoritative, investigative. Evergreen.
- Audience promise: a journey to every place in the solar system where water exists — Mars's ancient rivers, hidden oceans under Europa/Enceladus, lunar polar ice — and why it is the key to life and humanity's next step into space.

## Architecture review (what exists vs. what I must build)

### Existing (verified present, will reuse)
- `research_pipeline.run_research(topic, source_urls, production_extractor=..., content_fetcher=...)` — production mode fetches + persists real source content, extracts with a schema-constrained extractor, records per-source content hashes + output digest. Verified: returns 4 claims from genuine NASA content when the correct model is used.
- `research_pipeline.make_production_extractor(model_name=...)` — posts to LM Studio loopback, enforces `[classification] | text` schema, fails closed on malformed/insufficient output. Records identity via `identity()` callback.
- `research_pipeline.make_secure_http_transport()` — HTTPS-only, rejects private/loopback/link-local, caps size/redirects/timeouts, allowlists text content types. Verified working.
- `production.build_candidate(research, engine=..., min_words=..., asset_transport=..., workdir=...)` — builds scene plan from verified facts + disjoint grounded source sentences, synthesates narration via the injected engine, enforces runtime band [900,1200]s, acquires+approves rights-cleared assets, renders master via `engine.render_master`, writes captions.
- `production.RenderEngine` abstract interface: `synthesize_narration(text, voice) -> Path` and `render_master(edit, narration_segments=..., assets_by_scene_id=..., destination=...)`. Only `OfflineRenderEngine` (test-only synthetic testsrc/tone) exists today.
- `daily_job.run_daily_job(project_root, config=JobConfig(...), engine=...)` — orchestrates trend→research→preflight→produce→QC→package→approval gate. When `topic_override` is set it builds a single candidate with **no sources** (empty list). Requires an explicit real `engine` or fails as "not_configured".
- `narration.synthesize_to_wav(text, output, engine="kokoro", voice=...)` — real Kokoro TTS. Verified pacing: ~405 ms/word → 2600 words ≈ 1053s narration + 8s bookend ≈ **1061s runtime** (in-band).
- `remotion/src/SyntheticVideo.tsx` + `schema.ts` — Remotion renderer reads an EditDocument JSON, renders scenes with per-scene `image`/`clip`, starfield, data viz, lower thirds, chapter markers. Verified: smoke render produces valid 720p h264 master.
- `assets.py` — NASA + Wikimedia providers behind a rights gate. **BUG FOUND**: current `search_nasa` reads `data["collection"]["items"]` with nested `metadata` and `rel="media"` links, but the live API returns items under `data["data"]["items"]` (list) with per-item nested metadata (`nasa_id`, `title`, `creator`/`secondary_creator`, `keywords`) and canonical/preview links. It also records **no license field** anywhere on NASA items. Must fix to match actual structure + capture provenance fields for the operator's per-asset PD-by-statute policy.

### What I must build (Phase 4 deliverables)
1. **`RemotionKokoroRenderEngine`** — real production `RenderEngine`:
   - `synthesize_narration(text, voice)` → calls `narration.synthesize_to_wav(..., engine="kokoro", voice=voice or "af_heart")`.
   - `render_master(edit, narration_segments, assets_by_scene_id, destination)` → writes edit JSON to Remotion props, runs `npm run render` (Remotion headless via google-chrome), muxes the real Kokoro narration track into the muted Remotion master with FFmpeg. No testsrc/tone fallback.
2. **Extractor model fix via discovery** — resolve the loaded model id from LM Studio `/v1/models` at runtime; pass it to `make_production_extractor(model_name=...)`. Record resolved id in pilot provenance. NOT a new hard-coded default.
3. **NASA asset provider fix + per-asset provenance** — correct `search_nasa` to the live API structure; capture `nasa_id`, creator, canonical URL, retrieved metadata; classify public domain by statute (17 U.S.C. § 105) with explicit recorded basis; reject anything with an explicit restrictive license marker or ambiguous provenance. ESA excluded unless a specific asset carries an accepted CC/PD license confirmed per-item.
4. **Pilot runner** — `pilot.py` that: discovers the model, builds the production extractor + secure transport + NASA/Wikimedia asset transport, supplies real source URLs for the topic-override research path, runs `run_daily_job(engine=RemotionKokoroRenderEngine(...))`, and stops at `approval.json: pending`.

## Safety constraints (verbatim from operator)
- Do not upload, schedule, publish, or authenticate to YouTube.
- Do not install models, drivers, packages, or system services without reporting the exact need and obtaining approval. (Remotion/node/chrome/Kokoro already installed — no installs.)
- Do not create cron jobs.
- Never store API keys, OAuth credentials, cookies, or browser profiles in Git.
- Do not silently fall back from real production dependencies to test/offline components.
- Per-asset rights: record NASA ID, creator, canonical URL, retrieved metadata, exact PD basis; reject ambiguous provenance. ESA excluded unless individual asset has explicitly compatible license.

## Acceptance criteria
- Measured master runtime in [900, 1200]s (target ~1061s).
- ≥3 verified facts, each mapped to a source ID and grounded in persisted content.
- Every narrated claim distinguishes confirmed observation from habitability/resource-extraction hypothesis.
- All approved assets carry full rights records; no ambiguous/ESA-without-license asset reaches render.
- Final QC passes on real artifacts (audio, visual, continuity, originality, rights, metadata).
- `approval.json` written with status `pending`; nothing uploaded/published/authenticated/scheduled.

## Verification plan
- Run pilot end-to-end; capture the DailyJobResult JSON.
- ffprobe master duration + full-decode check.
- Confirm provenance files: extractor model id, source content hashes, rights manifest with per-asset NASA fields.
- Focused test additions for the new engine + asset fix (fast subset).

## Risks / unknowns
- Remotion headless render time (~60–120s) — budget generously; run in foreground with high timeout.
- Kokoro model load is cached per process but first synthesis ~5s.
- NASA search may return few relevant items for some queries — use multiple targeted queries (Mars water, Europa ocean, Enceladus plume, lunar ice).
