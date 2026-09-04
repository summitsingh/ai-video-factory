# AI Video Factory

Phase 1 is a local-only, deterministic synthetic-video workflow. It does not
download models or media, access credentials, publish content, install system
software, or change device drivers.

## Operator commands

```text
uv run ai-video-factory doctor
uv run ai-video-factory benchmark
uv run ai-video-factory test-pipeline --json
```

The test pipeline renders the checked-in 1280x720, 30 fps fixture with
Remotion, muxes a silent 48 kHz stereo AAC track, runs ffprobe plus a complete
FFmpeg decode, and writes the resulting MP4 and QC reports beneath
`data/projects/synthetic/runs/`. Generated run data is intentionally ignored
by Git. Rendering requires an already-installed local Chrome or Chromium
executable (or `REMOTION_CHROME_EXECUTABLE`); it never downloads a browser.

Hermes uses only these CLI commands and parses the JSON response from
`test-pipeline --json`; see [the Hermes command contract](docs/hermes-command-contract.md).
