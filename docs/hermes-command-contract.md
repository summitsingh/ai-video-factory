# Hermes command contract

Hermes owns orchestration, but it must invoke only the supported command-line
interface. It must not call npm, FFmpeg, ffprobe, or internal Python modules
directly.

Bootstrap a clean checkout once from the repository root with
`uv sync --dev --locked` and `(cd remotion && npm ci)`. Dependency bootstrap may use package
registries, but it must not download models, media, or a browser. At runtime,
Hermes must use the repository root as its working directory and may set
`REMOTION_CHROME_EXECUTABLE` to an absolute path for an already-installed
Chrome or Chromium executable.

```sh
uv run ai-video-factory inference doctor
uv run ai-video-factory inference estimate
uv run ai-video-factory inference start
uv run ai-video-factory inference status
uv run ai-video-factory inference benchmark
uv run ai-video-factory inference stop
uv run ai-video-factory doctor
uv run ai-video-factory benchmark
uv run ai-video-factory test-pipeline --json
```

The six `inference` commands each emit exactly one JSON document with these
stable fields:

- `schema_version`
- `command`
- `status`
- `retryable`
- `backend`
- `model_identifier`
- `checks`
- `metrics`
- `artifacts`
- `error`

They exit `0` only when `status` is `pass`; `fail` and `not_ready` exit `2`.
Unexpected exceptions are converted to the same versioned contract with a
sanitized error and no traceback. All paths resolve from the repository root,
not Hermes's ambient working directory.

Before any lifecycle mutation, the adapter validates the canonical LM Studio
executable and CLI commit, the single selected compatible GPU runtime and AMD
survey, the running server's exact loopback/CORS safety fields, and the complete
resident-model identity. It rechecks server configuration immediately after a
server start. A matching already-resident model is idempotently accepted before
the memory/estimate gates; an alias collision is never loaded over or unloaded.
Loopback API calls disable inherited proxies, reject redirects, and discard HTTP
error bodies at the adapter boundary.

`doctor` and `benchmark` emit JSON diagnostic documents. The checked-in
`system_report.md` is a separate Markdown bootstrap audit and is not rewritten
by `doctor`. `test-pipeline --json`
emits exactly one JSON object on stdout with these stable fields:

- `schema_version`
- `command`
- `status`
- `run_id`
- `resumed`
- `retryable`
- `artifacts`
- `error`

Hermes must parse stdout as JSON, persist `run_id`, and use `retryable` when
deciding whether to retry a command. A `status` of `fail` is a stopping
condition for QC; Hermes must not continue to a later stage or publication.
If `run_id` is `null`, setup failed before an active stage existed, so there is
no run event to retain. The response still contains every stable result field.

The synthetic render requires a verified local browser executable. It uses
`REMOTION_CHROME_EXECUTABLE` when configured, otherwise a locally discoverable
Chrome or Chromium executable. This prevents Remotion's automatic browser
download, and the composition has no remote assets; absence is a retryable
command failure before Remotion starts. Chrome is not process-level egress
sandboxed, so this contract does not claim operating-system network isolation.

Exit status is `0` on QC pass and `2` on a pipeline or QC failure. JSON mode
never writes a traceback to stdout.

Phase 2B may use local inference only after `inference benchmark` has produced
a passing, integrity-checked capability report whose current provenance still
matches `config/inference.toml`. The report must identify
`avf-qwen36-executor`, served only through `http://127.0.0.1:1234/v1`. A stale,
missing, invalid, or non-passing report is a stopping condition. Hermes must not
load a fallback model, select or update an LM Studio runtime, enable LAN/CORS,
stop the shared server, or unload unrelated models. This contract prepares a
future Hermes integration; it does not claim Hermes is installed or configured.
