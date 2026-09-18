# Packaging module integration guide

`src/ai_video_factory/packaging.py` is the YouTube packaging layer: titles,
thumbnail briefs, and description text. It is deterministic and LLM-free, so
it runs anywhere the pipeline runs. It does not touch rendering, uploading,
or any existing module.

## Signatures

```python
from ai_video_factory.packaging import (
    generate_title_variants,
    generate_thumbnail_briefs,
    generate_description,
    score_title,
)

generate_title_variants(topic, script_title, description="", count=3) -> list[str]
# Best-first YouTube titles built only from words in the topic material,
# so every variant is defensible from the script. Formulas: curiosity gap,
# stakes, number/spec, contrarian, direct question.

score_title(title) -> float
# 0.0 to 1.0 heuristic. Rewards 40-60 char length, power words, concrete
# numbers. Penalizes ALL CAPS words, em dashes, shouty punctuation.

generate_thumbnail_briefs(script, count=3) -> list[dict]
# script: ScriptOutput or dict with title/scenes keys. Each brief has:
# concept, foreground_subject, background, text_overlay (max 4 words,
# uppercase), color_mood, why_it_works. Built for the 1280x720 canvas.

generate_description(topic, script, sources) -> str
# Hook paragraph (seeded from the first scene's narration), CHAPTERS section
# with estimated timestamps from scene duration_frames, SOURCES section with
# full URLs and clean publisher labels. Links belong here, never in the video.
```

## Where to hook into the pipeline

All hooks live in `src/ai_video_factory/video_pipeline.py`. No existing
function needs to change shape; these are new call sites.

1. **After the script stage** (`run_video_pipeline`, right after the script
   artifact is finalized): call `generate_title_variants` and
   `generate_description`, persist both next to the script JSON
   (e.g. `packaging.json`). The chosen title then flows into downstream
   stages instead of the raw script title.
2. **After render, before/inside the thumbnail step** (near the existing
   `build_thumbnails(...)` call around line 1789): call
   `generate_thumbnail_briefs(script, count=3)`. The briefs' `text_overlay`
   strings can feed `build_thumbnails` as the headline copy (it already
   accepts per-topic `power_words`; overlay text is the natural next
   parameter), and `foreground_subject`/`color_mood` can drive an AI image
   generation step for thumbnails that do not reuse video frames.
3. **Upload path** (`youtube.py` / `cli.py youtube-upload`): pass the stored
   description and the winning title variant as the upload metadata instead
   of hand-written strings.

## TODO: fill in when the YouTube research report lands

- [ ] Replace `_TITLE_FORMULAS` weights with the report's ranked title
      patterns. Add any proven formulas missing here (e.g. "I tried X for
      N days", bracket qualifiers like "[Documentary]") as new entries.
- [ ] Retune `score_title`: set the length sweet spot, power-word list, and
      penalties from the report's CTR findings. Consider per-category
      weights (history vs. science vs. finance titles behave differently).
- [ ] Extend `_THUMBNAIL_ARCHETYPES` with the report's winning thumbnail
      patterns (face close-ups, red arrows/circles, before/after splits,
      text placement rules). Keep `text_overlay` at max 4 words.
- [ ] Add a `pick_winner(titles) -> str` policy: today the pipeline should
      take `variants[0]`; the report may justify a different rule (e.g.
      prefer question titles for science, stakes titles for history).
- [ ] Description: adopt the report's proven description structure (link
      placement, hashtag strategy, pinned-comment CTA) in
      `generate_description`.
- [ ] Wire briefs to actual thumbnail image generation: the briefs are
      written for a generator (foreground/background/mood), but no image
      model call exists yet. Add it behind the thumbnail step from hook 2.
- [ ] A/B loop: store published title/thumbnail with the video's later
      view/CTR data so future picks learn from real performance.

---

# Workstream D: multi-output + nightly batch integration guide

Modules added (no existing files modified):

- `src/ai_video_factory/multi_output.py`: one script -> longform + 90s short + vertical shorts.
- `src/ai_video_factory/nightly_batch.py`: overnight research/draft/review-queue orchestration.
- `tests/test_multi_output.py`, `tests/test_nightly_batch.py`: 19 tests.

## 1. The `draft=True` contract (for the pipeline integrator)

`nightly_batch.run_nightly` calls its injected `pipeline_fn` like this:

```python
pipeline_fn(
    topic=candidate,          # dict with title/description/url
    script=script_result,     # dict from script_fn
    draft=True,               # REQUIRED: draft-only render
    theme=config.theme,
    target_duration_minutes=config.target_duration_minutes,
    llm_url=config.llm_url,
)
```

The integrator wiring `run_video_pipeline` (or a wrapper) behind this seam
must:

1. Accept `draft: bool = False` as a keyword argument.
2. When `draft=True`: render at reduced resolution (suggest 854x480),
   skip the YouTube upload step entirely, skip the final loudness-normalized
   master (a plain mux is fine for review), and still write
   `draft_master_path`, `thumbnail_paths`, and `run_id` into the result dict.
3. Return a result dict with at least `status` (`"pass"`/`"draft_ok"` on
   success), `run_id`, `draft_master_path`, and `thumbnail_paths`
   (list, may be empty).

Timeouts must scale with duration (the fixed 180s mux timeout that killed
the 2026-09-18 smoke run is the failure this avoids).

## 2. CLI command sketch (`nightly` in cli.py)

```python
@nightly_app.command("run")
def nightly_run(
    data_root: str = typer.Option("data", "--data-root"),
    topics: int = typer.Option(3, "--topics"),
    theme: str = typer.Option("space", "--theme"),
    minutes: float = typer.Option(25.0, "--minutes"),
    llm_url: str = typer.Option("http://localhost:8080/v1/chat/completions", "--llm-url"),
) -> None:
    from ai_video_factory.nightly_batch import NightlyConfig, run_nightly
    from ai_video_factory.research import research_trending_topics
    from ai_video_factory.scheduler import scheduler_lock  # one run at a time

    config = NightlyConfig(
        topics_per_night=topics, theme=theme,
        target_duration_minutes=minutes, llm_url=llm_url,
    )

    def research_fn(limit: int) -> list[dict]:
        result = research_trending_topics(max_topics=limit, trend_source="all")
        return [
            {"title": t.title, "description": t.description, "url": t.url}
            for t in result.topics
        ]

    def script_fn(candidate: dict) -> dict:
        # call generate_longform_script(...) and return
        # {"title": ..., "summary": ..., "script": ...}
        ...

    def pipeline_fn(**kwargs) -> dict:
        # build a VideoJob from kwargs["topic"]/kwargs["script"],
        # call run_video_pipeline(..., draft=kwargs["draft"], ...)
        # and map the PipelineResult to the contract dict above
        ...

    with scheduler_lock(Path(data_root)):
        report = run_nightly(
            Path(data_root), config,
            research_fn=research_fn, script_fn=script_fn, pipeline_fn=pipeline_fn,
        )
    typer.echo(json.dumps(report, indent=2))
```

Also add `nightly queue --date <YYYY-MM-DD>` (list items) and
`nightly approve <id>` / `nightly reject <id>` thin wrappers over
`approve_item` / `reject_item`.

## 3. Cron spec (matches scheduler.py conventions)

scheduler.py runs channels from TOML with `schedule_time`/`timezone` and
guards overlap with `scheduler_lock` (fcntl, non-blocking). The nightly
batch reuses that lock (see sketch above), so the cron entry is a plain
daily trigger:

```cron
# summit-amd, Asia/Kolkata. Research + drafts overnight, review queue by morning.
0 2 * * *  cd /home/summit/ai-video-factory && .venv/bin/ai-video-factory nightly run --data-root data >> /home/summit/ai-video-factory/data/nightly.log 2>&1
```

02:00 IST is 20:30 UTC previous day; the Vulkan llama-server on :8080 must
be up (the batch does not start it). If a run is still holding the lock at
02:00 the next night, `scheduler_lock` raises `SchedulerBusy` and cron
logs it instead of stacking renders.

## 4. Review queue layout

```
data/review_queue/<YYYY-MM-DD>/queue.json   # {"items": [...]}
data/review_queue/<YYYY-MM-DD>/report.json  # run_nightly report
```

Item fields: `id`, `topic`, `title`, `description`, `status`
(`pending`|`approved`|`rejected`|`failed`), `draft_master_path`,
`thumbnail_paths`, `script_summary`, `run_id`, `generated_at`,
`reviewed_at`, `error` (failed items only).

Morning flow: user opens the day's `queue.json`, watches each
`draft_master_path`, then runs `nightly approve <id>`; approved items are
picked up by the final-render/upload path (out of scope for this
workstream).

## 5. Multi-output usage

```python
from ai_video_factory.multi_output import plan_outputs, derive_short_edit
from ai_video_factory.longform import longform_to_edit_document

plan = plan_outputs(longform_script, edit_doc)
# plan["outputs"]["short90"]["scene_ids"] -> feed to derive_short_edit
short_edit = derive_short_edit(edit_doc, plan["outputs"]["short90"]["scene_ids"])
# plan["outputs"]["shorts"] -> list of {"hook", "scenes", "duration_seconds",
#   "orientation": "vertical", "aspect": "9:16"}; derive one edit per segment
# with derive_short_edit(edit_doc, seg["scenes"])
```

The vertical aspect is a render-time concern (the renderer workstream owns
it); the edit docs carry scene selection and timing only.

---

# ai_visuals: AI-generated visuals for asset gaps

## Capability verdict (summit-amd, checked 2026-09-18)

**Active today: procedural.** No diffusion stack exists on the workstation:

- No `torch`, no `diffusers`, no `transformers` in any Python env
- No Stable Diffusion / SDXL / Flux checkpoints anywhere on disk
  (`find /home/summit -maxdepth 4` for `*stable-diffusion*`, `*sdxl*`,
  `*comfyui*`, `*flux*` found nothing usable; only a ComfyUI skill doc
  under `~/.hermes`, not installed)
- `models/` holds TTS models only
- 836 GB free disk, ROCm 7.2 HIP stack present, 128 GB unified memory

So `generate_scene_visual` currently renders the deterministic
Pillow/numpy cinematic fallback. It is honest about this: the module
docstring and `ACTIVE_BACKEND` both say `procedural` until the unlock
below is completed. The fallback is genuinely better than the
phone-mockup card it replaces: themed 1280x720 canvases (space, earth,
fire, ocean, tech, abstract) with gradient, nebula clouds, starfield,
theme extras (planet limb glow, embers, light rays, grid), vignette,
and film grain. Seeded from the scene text, so reruns are identical.

## Signatures

```python
from ai_video_factory.ai_visuals import (
    ACTIVE_BACKEND,            # "procedural" | "diffusion"
    needs_generated_visual,    # (scene_assets) -> bool
    generate_scene_visual,     # (direction, narration, out_path, style="cinematic documentary") -> Path
    detect_theme,              # (direction, narration=None) -> str
    render_procedural,         # (direction, narration=None, out_path=None, width=1280, height=720) -> Path
)
```

`needs_generated_visual` accepts a dict (`{"image":..., "clip":...,
"qc_failed":...}`), a list/tuple of asset paths, an object with
`image`/`clip` attributes (e.g. `EditScene`), or `None`. It returns
True when there is nothing usable: no candidates, files missing, or
`qc_failed` set. Conservative by design: any doubt means generate.

## Where to hook (video_pipeline.py, asset stage)

After `populate_assets_from_nasa(...)` (around the asset stage), per
normal scene:

```python
from ai_video_factory.ai_visuals import needs_generated_visual, generate_scene_visual

for scene in edit_doc.scenes:
    if scene.kind != "normal":
        continue
    if needs_generated_visual(scene):
        gen_path = assets_dir / scene.id / "generated.jpg"
        generate_scene_visual(scene.visual, scene.narration, gen_path)
        scene.image = str(gen_path)   # renderer picks it up as the Ken Burns still
```

Do not wire this into intro/outro scenes; keep those on title cards.
Runtime cost of the procedural path is under a second per scene.

## Unlocking the diffusion path (not done)

Exact steps on summit-amd:

1. Python deps (workstation venv):
   `pip install torch --index-url https://download.pytorch.org/whl/rocm6.3`
   then `pip install diffusers transformers accelerate safetensors`
   (use the current ROCm manylinux wheel index at download.pytorch.org;
   the wheels are self-contained and work on Ubuntu 26.04).
   Expected download: ~5 GB torch + ~1 GB deps.
2. Model: `stabilityai/sdxl-turbo` (~6.9 GB, fp16 safetensors, 4-step
   inference, the fastest quality option for an iGPU).
   `huggingface-cli download stabilityai/sdxl-turbo --include "*.safetensors" --include "*.json" --include "*.txt"`
   into `/home/summit/ai-video-factory/models/sdxl-turbo/`
   (or set `AI_VISUALS_MODEL_DIR` to another location).
3. No code change needed: `generate_scene_visual` auto-detects torch +
   diffusers + the checkpoint and switches `ACTIVE_BACKEND` to
   `diffusion`. Expected speed on Strix Halo iGPU: roughly 1-3 minutes
   per 1280x720 still at 4 steps.
4. Slower, higher-quality alternative: `stabilityai/stable-diffusion-xl-base-1.0`
   (~6.9 GB, 25-30 steps, several minutes per image).

Runtime deps the parent should add to `pyproject.toml` when wiring the
hook: `numpy`, `pillow` (already present on summit-amd; installed in the
local venv for tests only).

## Tests

`tests/test_ai_visuals.py`: 15 tests, all passing. Covers gap
detection (None/empty/missing files/qc_failed/duck-typed scene),
theme keyword routing, procedural output size/mode/determinism/
variation/parent-dir creation, procedural fallback dispatch, diffusion
dispatch with a mocked generator, and a no-em-dashes lint on the
module. Diffusion is mocked because no GPU model is installed.

---

# Asset QC + Asset Memory integration guide

Two new modules defend the pipeline against bad downloaded assets
(black frames, title slates, broadcast graphics, static slates).
They are standalone: no existing module was modified to add them.

- `src/ai_video_factory/asset_qc.py` — pixel-level checks on downloaded files.
- `src/ai_video_factory/asset_memory.py` — persistent cross-run blocklist
  plus a keyword relevance scorer stub.

## Function signatures

### asset_qc

```python
def verify_image_asset(path: str | Path) -> dict
def verify_video_asset(path: str | Path, work_dir: str | Path | None = None) -> dict
```

Both return a verdict dict:

```python
{
    "ok": bool,            # True when the asset passed every check
    "reasons": [...],      # machine-readable slugs, empty when ok
    "scores": {...},       # numeric diagnostics (brightness, text_edge_density, ...)
}
```

Reason slugs: `unreadable`, `too_small` (images), `black_frame`,
`dark_frame` / `dark_frames`, `heavy_text_overlay`, `static_slate` (video),
`probe_unavailable` (video, ffmpeg missing).

### asset_memory

```python
class AssetMemory:
    def __init__(self, path: str | Path | None = None) -> None
    def is_blocked(self, asset_id: str) -> bool
    def record_rejection(self, asset_id: str, reason: str, source: str = "") -> None
    def reason_for(self, asset_id: str) -> str | None
    def prune(self, days: int = 90) -> int      # returns number removed
    def stats(self) -> dict
    def __len__(self) -> int
    def __contains__(self, asset_id: str) -> bool   # same as is_blocked

def score_relevance(asset_title: str, scene_visual: str) -> float  # 0-1 keyword overlap
```

Default store path: `data/asset_memory.json` (cwd-relative), overridable via
the `AI_VIDEO_FACTORY_ASSET_MEMORY` env var or the constructor `path` arg.
Writes are atomic (temp file + rename).

## Where to call in the pipeline

### 1. Asset download loop (`nasa_media.fetch_nasa_for_scene`)

The primary hook. After `download_asset(url, destination)` succeeds and
after the existing `_image_is_usable` check:

```python
from ai_video_factory.asset_qc import verify_image_asset, verify_video_asset
from ai_video_factory.asset_memory import AssetMemory

memory = AssetMemory()  # create once per pipeline run, reuse across scenes

# inside fetch_nasa_for_scene, per record (record["nasa_id"] is the asset ID):
if memory.is_blocked(record["nasa_id"]):
    continue  # rejected in a previous run, never re-download

download_asset(url, destination)

verdict = (verify_image_asset(destination) if mediatype == "image"
           else verify_video_asset(destination))
if not verdict["ok"]:
    memory.record_rejection(record["nasa_id"], ",".join(verdict["reasons"]), source="nasa")
    destination.unlink(missing_ok=True)
    continue  # try the next record
```

Notes:

- Keep the existing event-photo title filter first: it is cheaper than
  downloading.
- Optionally rank surviving candidates with
  `score_relevance(record["title"], scene_visual)` and keep the top-N per
  scene instead of the first-N.
- `verify_video_asset` samples 3 frames via ffmpeg; on a scene with many
  candidate clips this adds seconds, not minutes.

### 2. Pre-render gate (`video_pipeline`, before the Remotion render step)

A second pass over whatever landed in each `scene-NN/` directory, in case
assets arrived through another path (stock_media, manual drops):

```python
from pathlib import Path
from ai_video_factory.asset_qc import verify_image_asset, verify_video_asset

def _scene_assets_ok(scene_dir: Path) -> list[str]:
    bad = []
    for f in sorted(scene_dir.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            verdict = verify_image_asset(f)
        elif f.suffix.lower() in (".mp4", ".mov", ".webm"):
            verdict = verify_video_asset(f)
        else:
            continue
        if not verdict["ok"]:
            bad.append(f"{f.name}: {','.join(verdict['reasons'])}")
    return bad
```

Call it per scene after `populate_assets_from_nasa` returns and before
render. On failure: drop the bad file and let the scene fall back to a
generated card, or trigger one more NASA query round for that scene.

## Config knobs (`asset_qc` module constants)

| Constant | Default | Effect |
|---|---|---|
| `BLACK_MEAN_THRESHOLD` | 12.0 | Mean brightness below which a frame may be black |
| `BLACK_CONTENT_THRESHOLD` | 15.0 | Brightest-1% mean below which a dark frame is truly empty (keeps starfields) |
| `DARK_MEAN_THRESHOLD` | 28.0 | Below this a frame is suspiciously dark |
| `TEXT_EDGE_THRESHOLD` | 0.04 | Band edge density that means obvious large text |
| `TEXT_CONCENTRATION_THRESHOLD` | 1.4 | Max-band/mean-band ratio marking text-like clustering (dark frames) |
| `TEXT_DARK_EDGE_FLOOR` | 0.016 | Minimum band edge density for the dark-frame text rule |
| `STATIC_SIMILARITY_THRESHOLD` | 0.995 | Frame similarity above which a clip is a static slate |
| `OCR_WORD_THRESHOLD` | 8 | Words that trigger text rejection when tesseract exists |
| `MIN_IMAGE_WIDTH` / `MIN_IMAGE_HEIGHT` | 640 / 360 | Minimum still dimensions |
| `SAMPLE_FRACTIONS` | (0.02, 0.5, 0.95) | Where video frames are sampled |

## Known limits (be honest about these)

- The text heuristic is tuned for clear-cut slates and dark broadcast
  graphics. Subtle cases (dim text on mid-tone footage, UI mockups like
  phone frames) can pass; that is what `asset_memory` plus the review queue
  are for. If tesseract ever becomes available in the environment, OCR
  takes over automatically and is far more precise.
- `static_slate` needs at least 2 sampled frames; clips shorter than ~1s
  may yield only one.
- Heuristics err toward recall on dark space content: starfields and night
  scenes pass the black-frame check via the brightest-1% content test.

## Tests

`tests/test_asset_qc.py` (13 tests) and `tests/test_asset_memory.py`
(11 tests) build synthetic fixtures at test time (PIL images, ffmpeg
lavfi clips in tmp dirs). No binary fixtures are committed.
