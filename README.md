# AI Video Factory

Phase 1 is a local-only, deterministic synthetic-video workflow. Phase 2A adds
a controlled LM Studio boundary for one already-installed Qwen model. Neither
workflow downloads models or media, accesses credentials, publishes content,
installs system software, changes runtimes, or changes device drivers.

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

## Local inference lifecycle

Run the following commands separately and in order from the repository root.
Record the loaded identifiers from both `lms ps --json` calls and require the
unrelated before and after sets to match exactly.

```sh
uv run ai-video-factory inference doctor
uv run ai-video-factory inference estimate
lms ps --json
uv run ai-video-factory inference start
uv run ai-video-factory inference status
uv run ai-video-factory inference benchmark
uv run ai-video-factory doctor
uv run ai-video-factory benchmark
uv run ai-video-factory test-pipeline --json
uv run ai-video-factory inference stop
uv run ai-video-factory inference status
lms ps --json
```

Every `inference` command emits exactly one versioned JSON document. A passing
result exits `0`; `fail` and `not_ready` both exit `2`. The final inference
status is therefore expected to exit `2` with a sanitized `not_ready` document
after a successful stop confirms that the configured model is absent.

`inference doctor`, `status`, and both `lms ps` calls are read-only.
`inference estimate` uses LM Studio's estimate-only mode and applies the 40 GiB
available-memory gate without loading a model. `inference start` may start only
the `127.0.0.1:1234` server and loads only the existing Qwen model as
`avf-qwen36-executor`. `inference benchmark` makes three loopback requests and
may read/hash the existing model for provenance; it writes only ignored report
and run-state data beneath `data/`. The generated primary-model digest cache is
kept only beneath ignored `data/system/model-digests/`; model files and
companions are never written. `inference stop` unloads only the stable
identifier and never stops the shared server or unloads another model.

Every discovery snapshot resolves the canonical executable and records its
authoritative CLI commit, requires exactly one selected Vulkan-or-AMD-ROCm AVX2
runtime plus a matching positive AMD accelerator survey, and—whenever the
server is running—checks `server_config_path` for port `1234`, interface
`127.0.0.1`, and disabled CORS. Residency is accepted only when the alias,
inventory model key and contained path, package size, context length, and
parallelism all match. An alias collision fails closed, including for stop, so
the factory cannot unload a different model. Loopback HTTP ignores ambient
proxy settings and never follows redirects or exposes HTTP error bodies.

If an inference step or probe fails, immediately run the targeted
`uv run ai-video-factory inference stop`, verify with `lms ps --json` that
`avf-qwen36-executor` is absent and unrelated identifiers are unchanged, and do
not treat the host as ready for Phase 2B. Phase 2B may consume only a passing,
provenance-verified capability report for the stable loopback identifier.

At render time, an explicit local browser path prevents Remotion from
automatically downloading a browser, and the checked-in composition contains
no remote assets. Chrome is not process-level egress sandboxed, so this is not
an operating-system offline guarantee.

Future Hermes orchestration may use only the documented CLI commands and their
JSON responses; Hermes is not installed or configured by Phase 2A. See
[the Hermes command contract](docs/hermes-command-contract.md).
