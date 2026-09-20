# Workstation render test, 2026-09-18 (reference workstation)

## Verdict: smoke test FAILED deterministically at script generation. Long-form NOT attempted (gated on smoke pass, correctly).

## Failure
- Command: `python -m ai_video_factory.cli video-pipeline "The Fermi Paradox" --duration 90 --output data/projects/smoke-fermi --json`
- Run dir: `~/ai-video-factory/data/projects/the-fermi-paradox/runs/00ef0bf281314ed98f4741999797b9c5/` (research.json only, no script.json)
- Error: `ScriptGenerationError: model output contained no JSON object` (fail-loud worked, no placeholder video)
- Root cause: loaded LM Studio model `tiel-coder-35b-a3b-mtp` is a reasoning model. With the full script prompt at max_tokens=2048 it burns the budget in `reasoning_content` and hits `finish_reason=length` with empty/brace-free `message.content`. Pipeline reads only `message.content` (`script_generator.py:157`, `longform.py:218`).
- Reproduced 2x independently (full CLI run + direct `generate_script_with_lm_studio` call). With a simplified prompt the model emitted valid JSON start (`{"title": "Where Is Everybody? The Fermi Paradox Explained"...}`) but still truncated at 2048 tokens.
- Fix needs pipeline code change (parse `reasoning_content`, raise max_tokens, add retry, log raw output). NOT done: out of scope.

## Design flaws found (non-crashing)
1. Positional topic ignored for content: trend research overrode "The Fermi Paradox" with a CNBC news story ("UN mission finds evidence signaling U.S. war crimes in Iran..."). Confirms scheduler concern #1.
2. `--output` flag silently ignored: runs always go to `data/projects/<topic-slug>/runs/<uuid>/` (`video_pipeline.py:1238`).

## Verified working
- Research stage: real gnews results written to research.json.
- Remotion render: `npx remotion render SyntheticVideo` with headless-shell Chrome -> /tmp/remotion-test.mp4, 90 frames, 397KB, RC=0.
- Kokoro TTS: works with venv patch (kokoro_onnx float32 speed fix, backup `.bak-int32`).
- Fail-loud contract: run marked failed, no silent fallback.

## Environment fixes (no pipeline code touched)
- `~/run-smoke.sh`: removed `set -u` (silently killed nvm, the real cause of earlier instant exits); added `PYTHONUNBUFFERED=1`.
- Runner: `~/run-smoke.sh`; log: `~/smoke-fermi.log`.

## LM Studio model inventory (remote)
- Loaded: `tiel-coder-35b-a3b-mtp` (reasoning model, the problem).
- Downloaded, not loaded: Qwen3.6-35B-A3B-UDT (matches pipeline default model id), Ornith-1.5-35B-A3B, Qwen3.8-Flash-Next, Ling-3.0-flash-Q3_K_M.

## Recommended next steps
1. Either patch pipeline for reasoning models OR load a non-reasoning model (Ornith-1.5 candidate) in LM Studio, then re-run smoke.
2. Fix `--output` being ignored and topic override before scheduler use.
3. Then run the 25-min longform and inspect the master.
