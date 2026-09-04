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
uv run ai-video-factory doctor
uv run ai-video-factory benchmark
uv run ai-video-factory test-pipeline --json
```

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
