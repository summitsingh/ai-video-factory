# AI Video Factory Phase 2B Hermes Orchestration Design

## Goal

Configure the already-installed Hermes Agent default profile so its existing
Nous free cloud model remains the parent orchestrator while substantive video
production work runs through the already-installed Qwen 3.6 35B-A3B model in
LM Studio. Add a guarded local vision capability and a deliberate fallback to
the same Nous free cloud model when a local worker fails.

Phase 2B proves routing, lifecycle, vision, fallback, and audit behavior with
synthetic tasks. It does not yet build the research, TTS, Whisper, image
generation, upload, or publishing stages.

## Approved decisions

- Work on `main` with Git identity `Summit Singh Thakur` and
  `400621+summitsingh@users.noreply.github.com`.
- Preserve the Hermes default profile and its parent route:
  `provider: nous`, model `stepfun/step-3.7-flash:free`.
- Use the cloud parent only for orchestration and explicit fallback.
- Route ordinary worker inference to LM Studio at
  `http://127.0.0.1:1234/v1` using model identifier
  `avf-qwen36-executor`.
- Permit local workers to use web search and web extraction for trend and
  source research. This is not an offline-only system; the model inference is
  local, while research sources may be remote.
- Enable local vision only after a deterministic image-input capability probe
  passes on the current model, projection file, runtime, and CLI provenance.
- If a local worker fails, times out, produces an invalid result, or fails a
  stage validator, the parent may perform one fallback attempt using only the
  configured Nous free model. No paid model or other cloud provider is allowed.
- Preserve the approval gate for credentials, external uploads, and YouTube
  publication.
- Do not install or download models, replace GPU drivers, install CUDA, select
  or update LM Studio runtimes, or manage unrelated loaded models.
- The operator explicitly accepts the current LM Studio setting that persists
  sensitive prompts, responses, and reasoning in local LM Studio logs. Phase
  2B documents this exception and does not change or enforce LM Studio logging
  settings.

## Audited starting state

- Hermes Agent is already installed at version `0.21.0 (2026.8.31)`, upstream
  commit `b0ab2e16`, with its own Python 3.11 runtime.
- The Hermes default profile is active. Its current parent provider is `nous`
  and its current default model is `stepfun/step-3.7-flash:free`.
- Phase 2A commit `6781666` provides guarded LM Studio doctor, estimate,
  start, status, benchmark, and targeted stop commands.
- The Phase 2A text capability report passed for Qwen 3.6 on the selected
  LM Studio Vulkan runtime. The model was unloaded after verification.
- Phase 2A did not validate multimodal inference. Vision remains unavailable
  to Hermes until the new probe passes.

The implementation must re-read and validate these facts before editing any
Hermes configuration. A version, provider, model, endpoint, or capability
mismatch is a stopping condition, not an invitation to migrate or upgrade.

## Architecture

### 1. Cloud parent orchestration

The Hermes default model remains the top-level agent. It may decompose a user
request, invoke the factory lifecycle commands, dispatch local workers, inspect
structured results, request clarification, and present the final status.

The parent must not directly produce research synthesis, scripts,
storyboards, scene plans, captions, metadata, thumbnails, or other production
artifacts on the successful path. Those are local-worker responsibilities.
The cloud parent sees worker summaries because Hermes native delegation returns
them to the parent; the design does not claim that production data never leaves
the machine.

### 2. Local worker route

Hermes native `delegate_task` is the worker rail. The default profile receives
an explicit delegation route with:

- model `avf-qwen36-executor`;
- base URL `http://127.0.0.1:1234/v1`;
- OpenAI-compatible chat-completions mode;
- a non-secret local placeholder API key only if Hermes requires a non-empty
  SDK value;
- no provider fallback inherited by the child;
- one concurrent child, matching the LM Studio model's configured
  `parallel = 1`;
- bounded iterations and timeouts suitable for a local reasoning model.

The direct custom endpoint is intentional. Hermes must not use its first-class
LM Studio model-management path because the factory, not Hermes, owns safe
estimate, exact-identity load, and targeted unload operations.

Local workers inherit only the toolsets needed for the verified task. Phase 2B
allows repository file access, terminal access through the documented factory
CLI, web search/extraction, and local image input. It does not grant messaging,
cron creation, credential management, publication, unrestricted browser
automation, or recursive delegation.

### 3. Workspace policy

A repository-scoped Hermes instruction file defines the binding workflow:

1. Run `inference doctor` and require `pass`.
2. Run `inference start`; never call `lms load` directly.
3. Require current text capability and, for image tasks, vision capability.
4. Delegate each substantive production task to the local worker with a
   complete goal, context, source bundle, output schema, and artifact path.
5. Validate the returned output before accepting it.
6. Use the cloud fallback only for a classified local failure.
7. Record route, attempt, validator result, and fallback reason without
   recording raw prompts, responses, or reasoning in factory reports.
8. Run `inference stop` when the workflow is complete; never stop the shared
   LM Studio server or unload unrelated models.
9. Stop before any credential entry, upload, or publication and obtain explicit
   operator approval.

Prompt policy is reinforced by code-level validators and tests. It is not
treated as the sole enforcement boundary.

### 4. Local vision gate

Extend the Phase 2A benchmark with a versioned multimodal corpus. The fixture is
a tiny repository-owned image containing deterministic shapes, colors, and
short text. The request embeds the local fixture without a remote URL and asks
for a strict JSON description.

The probe passes only when:

- the exact configured model is resident and API-visible;
- LM Studio reports the model as vision-capable;
- the contained projection file is a regular non-symlink file;
- primary and projection digests, sizes, mtimes, runtime, CLI path/commit, and
  corpus version match the report provenance;
- the response validates against the exact JSON schema and identifies all
  required visual facts;
- no request, response, image bytes, or reasoning is persisted in the factory
  capability report.

Any provenance change invalidates the prior vision report. Hermes must not send
images to the local worker without a current passing report. Vision failure may
use the approved Nous free fallback, and that fallback necessarily sends the
image/task data to Nous.

### 5. Fallback behavior

Hermes delegation does not provide a child fallback chain. The parent therefore
owns a small, explicit state machine:

```text
local attempt -> validate -> accept
      | failure / timeout / invalid output
      v
record classified reason -> one Nous-free parent attempt -> validate -> accept or stop
```

A fallback is allowed only for:

- local endpoint unavailable after the guarded start attempt;
- bounded timeout or model transport failure;
- local worker tool-call failure;
- malformed or schema-invalid output;
- deterministic stage-quality failure.

The fallback may use only the unchanged parent route
`nous / stepfun/step-3.7-flash:free`. If that route is not active, is no longer
free, or fails validation, the stage stops. Fallback is never allowed to bypass
credential, upload, publication, provenance, or QC gates.

Every attempt record contains the stage, backend class (`local_lmstudio` or
`nous_free_fallback`), outcome, timestamps, retryability, validator summary,
and a sanitized reason. It excludes prompts, responses, reasoning, tool
arguments, image bytes, and credentials.

### 6. Auxiliary model calls

Hermes can make auxiliary inference calls for activities such as extraction,
compression, title generation, and vision. Phase 2B must prevent these from
silently becoming unclassified cloud production work:

- required auxiliary production calls route to the same local endpoint;
- optional automatic title generation is disabled unless it is proven local;
- vision routes locally only after the vision gate passes;
- an auxiliary local failure follows the same explicit Nous-free fallback
  policy, where supported;
- unsupported auxiliary routing is disabled and reported rather than silently
  inheriting the cloud parent.

The implementation must test the resolved configuration, not merely compare
YAML text.

## Configuration mutation and rollback

Hermes configuration is external to the Git repository and may contain
credentials. Before mutation, create a permission-preserving backup in the
Hermes configuration directory. Never copy that file into the repository,
test fixtures, reports, terminal transcripts, or commits.

Change only the minimum default-profile fields required for delegation and
local auxiliary routing. Preserve `model.provider`, `model.default`,
`model.base_url`, authentication material, memories, skills, sessions, gateway
settings, and unrelated tool configuration.

After mutation:

- run Hermes's configuration validator and doctor;
- read back resolved non-secret route fields;
- prove the parent still resolves to the same Nous free model;
- prove delegation resolves to loopback and the stable Qwen identifier;
- provide a documented restore command for the backup.

If validation fails, restore the backup and stop.

## CLI and test surface

Add factory-owned commands for stable machine-readable integration checks:

- `hermes doctor`: validate installed Hermes identity, parent route, delegation
  route, tool boundary, capability reports, and non-secret configuration.
- `hermes smoke-text`: guarded model start, one synthetic delegated text task,
  strict validation, route audit, and targeted stop.
- `hermes smoke-vision`: guarded model start, current vision gate, one synthetic
  local image task, strict validation, route audit, and targeted stop.
- `hermes smoke-fallback`: use an injected or deliberately non-networking local
  failure seam, prove exactly one Nous-free fallback, validate output, and avoid
  changing the live LM Studio endpoint.

All commands return versioned JSON with stable status, retryability, backend,
run ID, attempt summaries, artifacts, checks, metrics, and sanitized errors.
Unexpected errors must not print tracebacks or configuration contents.

Unit and integration tests cover:

- strict Hermes version and route parsing;
- preservation of the parent model and unrelated configuration;
- local delegation resolution without a real credential;
- rejection of non-loopback endpoints and model aliases;
- single-worker concurrency;
- companion-file containment and digest invalidation;
- positive and negative vision fixtures;
- local success without cloud fallback;
- each allowed fallback class;
- no fallback on policy, credential, upload, publication, or QC gates;
- rejection when the parent is not the approved Nous free route;
- redaction and report non-persistence boundaries;
- rollback after failed configuration validation;
- targeted unload and preservation of unrelated loaded models.

The controlled live verification sequence is text success, vision success, an
injected safe fallback proof, targeted unload, and confirmation that the final
loaded-model set matches the initial unrelated-model set.

## Security and operating boundaries

- No CUDA or NVIDIA driver installation.
- No kernel, firmware, ROCm, Vulkan, or LM Studio runtime changes.
- No model downloads or replacement of existing weights.
- No credentials are entered, printed, copied, or committed.
- No LAN bind, CORS enablement, redirects, or inherited HTTP proxy use for LM
  Studio inference.
- No upload, external message, YouTube API call, or publication without a new
  explicit approval.
- Web research is allowed and therefore may disclose search queries and visited
  URLs to external sites.
- Nous fallback is allowed and therefore may disclose the failed task's input
  and required image/source context to Nous.
- LM Studio sensitive local logging remains enabled by explicit operator
  choice. Factory reports continue to exclude sensitive model content, but the
  system does not claim end-to-end non-persistence.

## Success criteria

Phase 2B is complete only when:

1. The default Hermes parent route remains the audited Nous free model.
2. A successful production-style text task is executed by the local Qwen
   delegate and recorded as local without invoking cloud fallback.
3. The versioned local vision probe and delegated vision smoke test pass.
4. An injected local failure produces exactly one validated fallback through
   the approved Nous free parent route.
5. Resolved auxiliary routing cannot silently use an unclassified cloud model.
6. Hermes invokes only the documented factory lifecycle contract for LM Studio.
7. Configuration rollback is proven without exposing secrets.
8. All automated tests, existing synthetic pipeline checks, and live controlled
   checks pass.
9. The configured Qwen model is unloaded at the end and unrelated loaded models
   are unchanged.
10. No external upload or publication occurs.

## Non-goals

- Installing or upgrading Hermes Agent.
- Replacing the default cloud orchestrator.
- Adding paid-model fallback or multiple cloud fallbacks.
- Building full trend discovery, research, scripting, TTS, Whisper, image
  generation, or YouTube publication stages.
- Enabling unattended cron or gateway automation.
- Deleting or disabling LM Studio's sensitive local logs.
- Claiming that a cloud-orchestrated or fallback-enabled workflow is private or
  offline-only.
