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

Exit status is `0` on QC pass and `2` on a pipeline or QC failure. JSON mode
never writes a traceback to stdout.
