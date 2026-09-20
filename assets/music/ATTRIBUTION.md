# Music bed attribution

The curated mood library in this directory is seeded with tracks by
**Scott Buckley**, released under **CC-BY 4.0**
(https://creativecommons.org/licenses/by/4.0/).

CC-BY 4.0 allows commercial use, including monetized YouTube videos, as long
as the artist is credited. Paste the credit block below into every video
description that uses one of these beds (swap in the track you used).

| File | Mood | Track | Artist | License |
|------|------|-------|--------|---------|
| `cosmic-bed.mp3` | cosmic (space, astronomy, universe) | Decoherence | Scott Buckley | CC-BY 4.0 |
| `mystery-bed.mp3` | mystery (suspense, crime, unexplained) | Intervention | Scott Buckley | CC-BY 4.0 |
| `epic-bed.mp3` | epic (history, ancient civilizations, war) | Emergent | Scott Buckley | CC-BY 4.0 |
| `calm-bed.mp3` | calm (nature, wildlife, environment) | Ephemera | Scott Buckley | CC-BY 4.0 |
| `tech-bed.mp3` | tech (AI, robotics, future) | Machina | Scott Buckley | CC-BY 4.0 |

All five are 320 kbps MP3, 44.1 kHz stereo, 3+ minutes, instrumental, and
narration-friendly. Source: https://www.scottbuckley.com.au/library/

## Credit block for video descriptions

```
Music: '<Track>' by Scott Buckley
https://www.scottbuckley.com.au/library/
Licensed under CC-BY 4.0: https://creativecommons.org/licenses/by/4.0/
```

## Notes

- The MP3s themselves are gitignored (large, machine-local). This file is
  tracked so the license record survives.
- If the pipeline ever fetches a replacement bed from a network provider
  (Internet Archive / Openverse / Freesound), only CC0 / Public Domain /
  CC-BY tracks are accepted, and the run log (`[music_bed] source=...`)
  records which track served. Add its credit here if it becomes permanent.
- `MUSIC_BED_PATH` remains an explicit operator override; `FREESOUND_API_KEY`
  (from https://freesound.org/apiv2/apply/) enables the Freesound leg of the
  fallback chain.
