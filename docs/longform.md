# Long-form documentaries (20-30 minutes)

The default pipeline makes 90-second shorts. Pass `--longform` to
`video-pipeline` for a full documentary built on a Hollywood three-act
structure.

## Structure

A 25-minute script targets ~3,750 words of narration (150 wpm) across
seven beats:

| Beat | Share | Job |
| --- | --- | --- |
| COLD OPEN | 4% | Hook in 30s, plant one open loop |
| ACT I | 16% | World, players, central question |
| ACT II (3 beats) | 60% | Evidence, twist, deepening |
| ACT III | 12% | Payoff: answer the cold-open question |
| OUTRO | 8% | Recap, sources, CTA |

Each beat is generated in its own LM Studio call against a shared series
bible (topic, angle, research brief, prior scene titles), so the model
sustains coherence across thousands of words. Beats that come back thin,
malformed, or off-brief fail the run loudly instead of producing a thin
video.

## Cinematic render

- Act title cards (`COLD OPEN`, `ACT I` ...) open each beat
- Lower thirds render names, dates, and locations from the script
- Scene timing derives from narration word counts, so the timeline
  matches the spoken track
- Letterboxed 2.39:1, per-scene color grade, film grain, cross-dissolves,
  transient captions, Ken Burns on stills, PiP insets

## Usage

```bash
# 25-minute documentary (default)
ai-video-factory video-pipeline "The search for water on Mars" --longform

# 20 or 30 minutes
ai-video-factory video-pipeline "Europa's hidden ocean" --longform --duration-minutes 30
```

`--duration-minutes` must be between 20 and 30. Content QC gates the
finished master: runtime within 5% of target, at least 10 scenes, 90%+
narration coverage, act structure present, and captions/chapters/thumbnail
artifacts in place. A long-form run that misses its runtime fails QC
instead of shipping short.

## Cost notes

A 25-minute video means ~25 minutes of TTS audio, a 25-minute Remotion
render, and 7 LM Studio generations. Expect the render to take a while on
a workstation; the pipeline checkpoints every stage in the run manifest,
so an interrupted run resumes where it left off.
