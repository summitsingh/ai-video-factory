# Hermes command contract

Hermes owns orchestration, but it must invoke only the supported command-line
interface. It must not call npm, FFmpeg, ffprobe, or internal Python modules
directly.

```text
ai-video-factory doctor
ai-video-factory benchmark
ai-video-factory test-pipeline --json
```

`doctor` and `benchmark` emit JSON diagnostic documents. `test-pipeline --json`
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
Chrome or Chromium executable. It never downloads a browser; absence is a
retryable command failure before Remotion starts.

Exit status is `0` on QC pass and `2` on a pipeline or QC failure. JSON mode
never writes a traceback to stdout.
