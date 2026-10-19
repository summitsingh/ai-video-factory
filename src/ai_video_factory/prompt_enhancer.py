"""Per-model prompt enhancement for hero-clip generation.

Structure follows WanGP's ``resolve_prompt_enhancer_settings`` concept:
base cinematic templates plus per-model override instructions, resolved
per model and then rendered through the existing local LLM endpoint
(:func:`ai_video_factory.longform._lm_studio_chat`). Reimplemented here;
no WanGP code is copied.

Resolution order for a model name:
  1. per-model override instructions (:data:`MODEL_OVERRIDES`), when present;
  2. otherwise the base T2V cinematic template.

The enhancer is best-effort: if the LLM call fails or returns nothing
usable, :func:`enhance_prompt` logs a warning and returns the raw prompt
so generation never dies on an enhancement hiccup.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger(__name__)

MAX_WORDS = 150

_BASE_STRUCTURE = """\
Structure the paragraph like a shot list:
- open with the main action in one sentence,
- add specific movements and gestures,
- describe characters and objects precisely (appearance, clothing, age),
- describe the background and environment,
- specify camera angle and movement (static, pan, tilt, dolly, close-up),
- describe lighting, colors, and mood,
- note any sudden changes or events in chronological order.

Rules: start directly with the action; stay literal and concrete, no
abstract metaphors; describe one continuous shot; do not change the user's
intent; never ask for on-screen text, titles, or logos; do not exceed
150 words. Output the enhanced prompt only, no preamble or commentary."""

BASE_T2V_CINEMATIC = f"""\
You are an expert cinematic director. Expand the user's idea into a single
flowing paragraph of detailed, chronological shot description.

{_BASE_STRUCTURE}"""

MODEL_OVERRIDES: dict[str, str] = {
    "ltx2": f"""\
You are an expert cinematic director writing prompts for the LTX-2
text-to-video model. Expand the user's idea into a single flowing
paragraph of detailed, chronological shot description.

{_BASE_STRUCTURE}

LTX-2 notes: favor short declarative sentences inside the paragraph;
describe motion as continuous physical actions rather than quick cuts;
keep the camera vocabulary simple; avoid dialogue, lyrics, and any
readable text; one subject and one location per prompt.""",
    "ltx2.5": f"""\
You are an expert cinematic director writing prompts for the LTX-2.5
distilled text-to-video model (8-step checkpoint; it also generates
synchronized audio). Expand the user's idea into a single flowing
paragraph of detailed, chronological shot description.

{_BASE_STRUCTURE}

LTX-2.5 notes: favor short declarative sentences; describe motion as
continuous physical actions; keep the camera vocabulary simple; one
subject and one location per prompt. You may include brief diegetic
sound cues (wind, footsteps, room tone) and at most one short spoken
line in quotes when the scene needs it; keep dialogue minimal and never
ask for on-screen text.""",
}

ChatFn = Callable[..., str]


def resolve_instructions(model: str) -> str:
    """Return the enhancer instructions for ``model`` (override or base)."""
    return MODEL_OVERRIDES.get(model, BASE_T2V_CINEMATIC)


def build_messages(user_prompt: str, model: str = "ltx2.5") -> list[dict[str, str]]:
    """Build the system/user message pair for the chat endpoint."""
    return [
        {"role": "system", "content": resolve_instructions(model)},
        {"role": "user", "content": user_prompt},
    ]


def _default_chat(messages: list[dict[str, str]], max_tokens: int, **kwargs: Any) -> str:
    from ai_video_factory.longform import _lm_studio_chat

    return _lm_studio_chat(messages, max_tokens, **kwargs)


def _cap_words(text: str, limit: int = MAX_WORDS) -> str:
    words = text.split()
    if len(words) > limit:
        return " ".join(words[:limit])
    return text


def enhance_prompt(
    user_prompt: str,
    *,
    model: str = "ltx2.5",
    chat_fn: ChatFn | None = None,
    max_tokens: int = 400,
    temperature: float = 0.6,
    api_url: str | None = None,
) -> str:
    """Enhance ``user_prompt`` via the local LLM. Falls back to raw prompt.

    ``chat_fn`` defaults to the repo's LM Studio chat endpoint. Any failure
    (or an empty response) returns ``user_prompt`` unchanged with a warning.
    The result is hard-capped at :data:`MAX_WORDS` words.
    """
    if not user_prompt or not user_prompt.strip():
        return user_prompt
    chat = chat_fn or _default_chat
    messages = build_messages(user_prompt, model)
    kwargs: dict[str, Any] = {"temperature": temperature}
    if api_url is not None:
        kwargs["api_url"] = api_url
    try:
        enhanced = chat(messages, max_tokens, **kwargs)
    except Exception as exc:  # noqa: BLE001 - enhancement must not break generation
        log.warning("prompt enhancement failed (%s); using raw prompt", exc)
        return user_prompt
    enhanced = (enhanced or "").strip()
    if not enhanced:
        log.warning("prompt enhancement returned empty text; using raw prompt")
        return user_prompt
    return _cap_words(enhanced)
