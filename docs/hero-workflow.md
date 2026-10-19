# Master-Image Hero Workflow

Date: 2026-10-18. Status: workflow discipline (doc change, no code).

Hero clips cover the "impossible" subjects: alien worlds, impossible camera
moves, abstract concepts, exact brand-matched visuals - everything real
stock footage cannot deliver (see `src/ai_video_factory/hero_video.py`).
Because these clips are the most expensive seconds in the pipeline (local
LTX-2 generation, ~7-12 min per 720p 8s clip on summit-amd), they get the
strictest workflow discipline. This document codifies it.

Concept adapted from the WanGP agent's "Long Video Workflows" discipline
(master-image workflow, end-frame anchors); reimplemented here as pipeline
convention, not code.

## The rule

**One master reference image per impossible subject. Every hero clip for
that subject starts from the master (image-to-video). Never chain
clip-to-clip.**

Concretely:

1. **Generate the master once.** One high-quality reference image per
   impossible subject (character, creature, location, impossible object).
   Review it like a final: identity, clothing, environment, style, and
   composition must be exactly right, because every clip inherits from it.
   Save it as `output/hero-clips/<subject>/master.png` (override with
   `HERO_CLIP_DIR`).

2. **Every clip starts from the master.** Each hero clip is an independent
   image-to-video generation conditioned on the same master image. Clip 3
   never takes clip 2's output as its input. The prompt for each shot
   describes only the motion, camera move, or action needed for that shot,
   anchored to what the master already shows.

3. **Fixed shot durations per scene.** Decide the shot plan up front (e.g.
   four 8s shots for a 32s scene) and generate exactly those durations.
   Do not "extend and see" by chaining continuations; LTX-2 snaps
   `duration * fps` to 8n+1 frames and widths/heights to the 32px grid
   (`hero_video._snap32`), so planned durations keep every clip on-grid
   and interchangeable in the edit.

4. **One subject, one master, one folder.** All clips for a subject live
   next to their master: `output/hero-clips/<subject>/hero-<shot>.mp4`.
   A new subject means a new master, never a recycled one.

## Why chaining is banned

Feeding clip N's output into clip N+1 as the start frame looks convenient
and permits improvisation, but it has two compounding costs:

- **Drift accumulation.** Each generation inherits every artifact, identity
  shift, and composition error from the previous one. After three chained
  clips the subject no longer matches the master.
- **Repeated lossy encode/decode.** Every continuation re-decodes and
  re-encodes video, softening detail generation over generation.

Starting every clip from the same master resets the error budget each
shot: the worst any single clip can drift is one generation away from the
reference.

## The one exception

When the next scene genuinely cannot be planned until the latest output is
reviewed, a single continuation from the latest clip is allowed - once, and
only to discover the next shot. The discovered shot is then re-generated
from the master as the canonical clip. Improvisation is a scouting tool,
not the production path.

## Checklist (per impossible subject)

- [ ] Master image generated, reviewed, saved as `<subject>/master.png`
- [ ] Shot plan written: shot count, fixed duration per shot, camera/motion
      per shot (prompts reference the master, not each other)
- [ ] Every clip generated image-to-video from the master (no clip-to-clip
      inputs in the generation log)
- [ ] Clip durations match the plan; dims on the 32px grid, frames 8n+1
- [ ] Per-clip settings recorded next to the output (prompt, seed, model,
      quant, timings) - the L5 settings-sidecar convention, so a bad clip
      can be reproduced or diagnosed without manifest archaeology

## Non-goals

This doc does not change `hero_video.py` or any pipeline code. Enforcement
is by convention plus the settings sidecar paper trail. If a future change
adds master-image tracking to the pipeline, it must follow this doc, not
redefine it.
