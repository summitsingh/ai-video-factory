# Workflow Improvements (living list)

Goal: Hollywood-style 20-30 min videos people watch, produced by a fast,
resumable, self-improving pipeline. Add items here whenever a bottleneck or
quality gap is found. Status: done / in-progress / planned.

## Done (2026-09-18)

1. **Stage resume** - `video-render`, `video-audio`, `video-mux` are separate
   RunStore stages with content fingerprints. A mux failure no longer forces a
   full re-render; re-running the command resumes completed stages.
2. **Scaling timeouts + retries** - mux timeout = max(300s, 4x duration), one
   automatic retry on timeout/failure (`_run_command(..., retries=...)`).
3. **Parallel TTS** - per-scene synthesis runs in a 4-worker thread pool.
4. **Parallel asset downloads** - per-scene NASA fetch runs in a 4-worker pool.
5. **Draft mode** - `--draft` renders at 640x360 for fast review iterations,
   fingerprinted separately from full renders.
6. **Retry/timeout regression tests** - `tests/test_pipeline_reliability.py`.

## In progress (workstreams, 2026-09-18)

7. **Asset pre-check** (A) - sample frames from video assets, reject black
   frames and title slates before the expensive render.
8. **Visual QC** (A) - black-frame, text-overlay, watermark heuristics on
   sampled frames.
9. **Asset memory** (A) - cross-run blocklist of rejected asset IDs + relevance
   scorer, so bad assets are never re-downloaded.
10. **Title/thumbnail pipeline** (B) - 3 title variants + thumbnail briefs +
    YouTube description from winning patterns (awaits YouTube research).
11. **AI visuals for gaps** (C) - generated cinematic visuals when stock fails,
    replacing the phone-mockup fallback.
12. **Multi-output** (D) - one script -> 25-min long-form + 90s short +
    vertical shorts.
13. **Nightly batch** (D) - overnight research -> draft renders -> morning
    review queue on reference workstation.

## Done (2026-09-18, format-driven rebuild)

14. **Format presets** (`formats.py`) - 7 winning YouTube formats from the
    format research (business_autopsy, systems_explainer,
    history_reconstruction, mystery_deep_dive, science_doc,
    armchair_true_crime, horror_anthology), each with title formulas,
    3-beat 60s hook plan, chapter template, narration register, WPM.
    Tier 1 first. `test_formats.py` (11 tests).
15. **Hook-first scripts** - `longform.py` accepts `--format`; beats, bible
    tone, and cold-open retention device come from the preset. Verified:
    3,750 words = 25.0 min at 150 WPM for business_autopsy.
16. **Packaging wired** - `packaging.py` runs after the script stage
    (5 scored title variants + chosen title, 3 thumbnail briefs, full
    description with chapters/sources -> `packaging.json`); brief overlays
    feed the thumbnail renderer.
17. **Shorts cutter** (`shorts.py`) - renders 20-40s vertical 1080x1920
    Shorts from the finished master (hook scene + burnt-in caption);
    wired into the pipeline after thumbnails + standalone `shorts` CLI.
18. **Asset QC + memory + AI visuals wired** - every attached asset verified
    (black frames, slates, dimensions), rejections recorded in the
    persistent blocklist, cinematic generated stills for empty scenes.
19. **Nightly CLI** - `nightly-batch` (research -> format script -> draft
    render -> review queue) and `review list/approve/reject` commands.

## Planned / candidate (add more as discovered)

- **Strict resolution QC** - fail when output is not exactly the target
  (1280x536 must not pass as 1280x720).
- **Clip start-offset** - skip the first 0.5-1s of stock clips (fades from
  black) or sample darkness and offset automatically.
- **Slate/text-frame rejection** - OCR-based filter for title cards and
  broadcast graphics with burned-in text.
- **Faster script drafts** - smaller/faster model for draft scripts, 27B only
  for finals; or parallel beat generation.
- **Thumbnail A/B** - generate 3 thumbnails, score via heuristic, keep best.
- **Learning loop** - track per-topic/per-format retention signals back into
  research scoring.
- **Chapter-aware pacing** - retention dips mapped to chapters; auto-suggest
  re-hooks.
