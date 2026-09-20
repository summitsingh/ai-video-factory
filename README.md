# AI Video Factory

Turn a topic into a long-form, documentary-style video: researched and
citation-verified script, storyboard, visuals, voiceover, topic-matched music,
karaoke captions, transitions, QC, thumbnail, and YouTube metadata. Built for
20-30 minute videos, running on your own machine.

## What it does

Give it a topic, get back a finished video package:

1. **Research** - gathers sources for the topic and verifies every citation
   URL actually resolves (HTTP 200), failing loudly on placeholders.
2. **Script** - generates a multi-beat documentary script with a cold-open
   hook, then enforces the target duration (extends underweight beats,
   ~150 wpm) so a 25-minute target really is 25 minutes.
3. **Visuals** - per-scene routing: stock footage (Pexels, Pixabay, NASA),
   local LTX-2 AI hero clips for impossible shots, cinematic AI stills,
   procedural fallbacks. Real footage is preferred over synthetic.
4. **Voiceover** - full narration via local TTS.
5. **Music** - a topic-matched ambient bed (cosmic, mystery, epic, calm,
   tech) fetched from royalty-free sources, ducked under narration and
   loudness-normalized.
6. **Captions** - karaoke-style word-highlight captions burned in, plus
   SRT/VTT exports for YouTube.
7. **Assembly** - FFmpeg xfade dissolves between scenes, final mix at
   -16 LUFS.
8. **QC gate** - automated checks fail the run on dark frames, digital
   silence, low audio, wrong resolution, or runtime drift.
9. **Packaging** - AI-picked thumbnail with title text, and YouTube
   metadata (titles, hook-first description, chapters, tags, sources).

## Quickstart

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), FFmpeg with
ffprobe, and an OpenAI-compatible chat completions endpoint for script
writing (LM Studio locally, llama-server, or any hosted API).

```sh
git clone https://github.com/summitsingh/ai-video-factory.git
cd ai-video-factory
uv sync
cp .env.example .env   # add free Pexels/Pixabay API keys if you want stock footage
```

Run a long-form video (25 minutes, science documentary format):

```sh
uv run ai-video-factory video-pipeline "The Fermi Paradox" \
  --longform \
  --duration-minutes 25 \
  --format science_doc \
  --llm-url http://localhost:1234/v1/chat/completions \
  --output data/projects/fermi-paradox
```

Short-form (90 seconds) is the default when `--longform` is omitted.
Format presets: `business_autopsy`, `systems_explainer`,
`history_reconstruction`, `mystery_deep_dive`, `science_doc`,
`armchair_true_crime`, `horror_anthology`.

Other useful commands:

```sh
uv run ai-video-factory test-pipeline --json   # smoke test on the checked-in fixture
uv run ai-video-factory doctor                 # machine report
uv run ai-video-factory nightly-batch          # unattended trending-topic run
```

## Configuration

Everything is configured through `.env` (see `.env.example`) and CLI flags.
No credentials are ever committed; `.env` is gitignored.

| Variable | Purpose |
|---|---|
| `PEXELS_API_KEY` / `PIXABAY_API_KEY` | Stock footage search (free tiers) |
| `MUSIC_BED_PATH` | Your own royalty-free bed track (optional override) |
| `FREESOUND_API_KEY` | Extra music provider (optional) |
| `AVF_KARAOKE_CAPTIONS` | `0` disables burned-in karaoke captions |
| `LTX2_MODEL_DIR` / `LTX2_VENV` | Local LTX-2 hero-clip backend paths |
| `AI_VISUALS_MODEL_DIR` | SDXL-Turbo checkpoint dir for AI stills |

Local LM Studio inference can be managed with
`ai-video-factory inference doctor|start|status|stop`
(see `config/inference.toml`).

## Architecture

```
topic
  -> research (research.py, source_verifier.py)
  -> script (script_generator.py, duration_enforcer.py)
  -> storyboard / edit doc (video_pipeline.py)
  -> visuals per scene (stock_media.py, hero_video.py, ai_visuals.py)
  -> narration (narration.py)
  -> music bed (music_bed.py, music_providers.py)
  -> captions (subtitle_export.py)
  -> assembly + xfade (pipeline.py, production.py)
  -> QC gate (qc_final.py)
  -> thumbnail + metadata (thumbnail.py, yt_metadata.py)
```

State is content-addressed under `state/` and run outputs under
`data/projects/`; both are gitignored and recreated at runtime. Completed
stages are reused only when input and tool fingerprints still match.

## Optional: LTX-2 hero clips

For custom "impossible" shots, a local
[LTX-2](https://huggingface.co/Lightricks/LTX-2) backend generates short
clips on your GPU. It needs a ROCm/CUDA torch build and ~100 GB of model
weights. See `docs/ltx2-setup-notes.md` and `scripts/download_ltx2.py`.
This is optional; the pipeline works fully without it.

## Roadmap

- Whisper forced alignment for exact word-level caption timing
- Crossfaded music loops and more mood profiles
- Cloud clip backends (Veo/Sora) as optional providers
- Multi-language narration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
