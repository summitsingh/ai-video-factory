# AI Video Factory

Phase 1 is a local-only, deterministic synthetic-video workflow. It does not
download models or media, access credentials, publish content, install system
software, or change device drivers.

## Clean-checkout bootstrap

Prerequisites are Python 3.12 through `uv`, Node 22 with npm, FFmpeg/ffprobe,
and an already-installed Chrome or Chromium executable. From the repository
root:

```sh
uv sync --dev --locked
cd remotion
npm ci
cd ..
export REMOTION_CHROME_EXECUTABLE=/absolute/path/to/google-chrome
```

`uv sync` and `npm ci` may contact their package registries on a new machine;
they do not download models, media, or a browser. If the browser is discoverable
as `google-chrome`, `google-chrome-stable`, `chromium`, or `chromium-browser`,
the environment variable may be omitted. Run all commands below from the
repository root and keep any `.env` local and untracked.

## Operator commands

```sh
uv run ai-video-factory doctor
uv run ai-video-factory benchmark
uv run ai-video-factory test-pipeline --json
```

The test pipeline renders the checked-in 1280x720, 30 fps fixture with
Remotion, muxes a silent 48 kHz stereo AAC track, runs ffprobe plus a complete
FFmpeg decode, and writes the resulting MP4 and QC reports beneath
`data/projects/synthetic/runs/`. Generated run data is intentionally ignored
by Git. Completed stages are reused only after their input/tool fingerprints
and recorded artifact digests still match.

The `doctor` command emits the current JSON machine report. The checked-in
`system_report.md` is the separate human-readable bootstrap audit; `doctor`
does not rewrite it.

At render time, an explicit local browser path prevents Remotion from
automatically downloading a browser, and the checked-in composition contains
no remote assets. Chrome is not process-level egress sandboxed, so this is not
an operating-system offline guarantee.

Hermes uses only these CLI commands and parses the JSON response from
`test-pipeline --json`; see [the Hermes command contract](docs/hermes-command-contract.md).
