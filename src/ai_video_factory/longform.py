"""Long-form documentary script engine.

Produces 20-30 minute documentary scripts with a Hollywood three-act
structure: a cold-open hook, setup, three development beats with rising
stakes, a climax, and a payoff outro. Each beat is generated separately
against a shared series bible so a local LM Studio model can sustain
coherence across ~4,000 words without losing the thread.

Retention design (baked into every beat prompt):
- cold open plants an open loop the climax pays off
- every beat ends on a forward pull (question, tease, or reversal)
- scenes run 45-75 seconds of narration so the visual never sits still
- each scene carries a visual direction and an optional lower third
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ai_video_factory.edit_schema import EditDocument, EditScene
from ai_video_factory.formats import FormatPreset, get_preset
from ai_video_factory.sanitization import sanitize_diagnostic


WORDS_PER_MINUTE = 150
MIN_LONGFORM_MINUTES = 20
MAX_LONGFORM_MINUTES = 30
# Reasoning models spend a large share of their token budget thinking
# before they write. Beats budget content tokens plus thinking headroom,
# and retry once on empty/truncated output, so a thinking model can never
# silently collapse a beat the way a 2048-token cap did.
BEAT_MIN_TOKENS = 8192
BEAT_TOKEN_HEADROOM = 4
BEAT_ATTEMPTS = 2
DEFAULT_LONGFORM_MINUTES = 25.0


class LongformError(RuntimeError):
    """Raised when a long-form script cannot be produced or validated.

    Fails loud: a beat that comes back short, malformed, or off-brief
    aborts the run instead of producing a thin video.
    """


@dataclass(frozen=True)
class BeatSpec:
    """Structural spec for one story beat."""

    key: str
    label: str
    fraction: float
    purpose: str
    retention: str


LONGFORM_BEATS: tuple[BeatSpec, ...] = (
    BeatSpec(
        key="cold_open",
        label="COLD OPEN",
        fraction=0.04,
        purpose=(
            "Hook the viewer in the first 30 seconds. Open on the most "
            "visually striking or emotionally charged moment of the story, "
            "state the stakes plainly, and promise the payoff without giving "
            "it away."
        ),
        retention="Plant one open loop: a question the video will answer in the climax.",
    ),
    BeatSpec(
        key="act1_setup",
        label="ACT I",
        fraction=0.16,
        purpose=(
            "Set up the world, the key players, and the central question. "
            "Give the viewer just enough context to care, then raise the "
            "stakes."
        ),
        retention="End the beat on a complication: something is not what it seemed.",
    ),
    BeatSpec(
        key="act2a_evidence",
        label="ACT II",
        fraction=0.20,
        purpose=(
            "Lay out the core evidence or story. Build the case scene by "
            "scene with concrete details, numbers, and sourced facts."
        ),
        retention="End on a twist or a deepening mystery that reframes what came before.",
    ),
    BeatSpec(
        key="act2b_twist",
        label="ACT II",
        fraction=0.20,
        purpose=(
            "Follow the twist. Explore the competing explanations, the "
            "skeptics, and the details most coverage misses."
        ),
        retention="End on the biggest unanswered question of the story.",
    ),
    BeatSpec(
        key="act2c_deepening",
        label="ACT II",
        fraction=0.20,
        purpose=(
            "Go deeper: the human story, the history, or the science behind "
            "the question. Make the abstract tangible."
        ),
        retention="Pay off the cold-open loop partially, but hold the final answer for the climax.",
    ),
    BeatSpec(
        key="act3_climax",
        label="ACT III",
        fraction=0.12,
        purpose=(
            "Deliver the payoff. Answer the cold-open question, resolve the "
            "central tension, and land the single takeaway the viewer will "
            "remember."
        ),
        retention="One clear, quotable takeaway. No new threads opened.",
    ),
    BeatSpec(
        key="outro",
        label="OUTRO",
        fraction=0.08,
        purpose=(
            "Recap the journey in two or three sentences, credit the "
            "sources, and close with a call to action (subscribe, next "
            "video)."
        ),
        retention="Tease the next video's topic in one line when it fits naturally.",
    ),
)


def _words(text: str) -> int:
    return len(text.split())


def _beats_for_format(preset: FormatPreset | None) -> tuple[BeatSpec, ...]:
    """Build beat specs from a format preset, or fall back to the default arc."""
    if preset is None:
        return LONGFORM_BEATS
    specs: list[BeatSpec] = []
    for beat in preset.beats:
        retention = "End on a forward pull: a question, tease, or reversal."
        if beat["key"] == "cold_open":
            retention = (
                "Universal 3-beat 60s hook: 0:00-0:10 pattern interrupt, "
                "0:10-0:30 promise + credibility, 0:30-1:00 the map. "
                f"Hook plan: {preset.hook.beat1_0_10} / {preset.hook.beat2_10_30} "
                f"/ {preset.hook.beat3_30_60}"
            )
        specs.append(
            BeatSpec(
                key=beat["key"],
                label=beat.get("label", beat["key"].replace("_", " ").upper()),
                fraction=beat["fraction"],
                purpose=beat["purpose"],
                retention=retention,
            )
        )
    return tuple(specs)


def _format_voice_line(preset: FormatPreset | None) -> str:
    if preset is None:
        return "Tone: cinematic documentary. Confident, curious, precise. No hype, no filler, no invented facts."
    return (
        f"Format: {preset.label}. Narration: {preset.narration_person}, "
        f"{preset.narration_tense}. Tone: {preset.narration_tone}. "
        f"{preset.sentence_guidance} Target pace {preset.wpm} words per minute."
    )


def beat_word_target(spec: BeatSpec, target_minutes: float) -> int:
    return round(target_minutes * WORDS_PER_MINUTE * spec.fraction)


LONGFORM_SYSTEM_PROMPT = """You are an award-winning documentary scriptwriter. You write narration for long-form \
YouTube documentaries in a cinematic, confident voice: short declarative sentences, vivid concrete detail, \
zero filler. Every scene you write must earn its place.

Rules:
- Write narration only, no stage directions outside the visual_direction field.
- Ground claims in the provided research; never invent facts, quotes, or statistics.
- Vary sentence rhythm: punchy fragments for tension, longer sweeps for wonder.
- Each scene's narration should run 45-75 seconds when spoken (110-190 words).
- Output ONLY valid JSON, no markdown fences, no commentary."""


def _beat_user_prompt(
    *,
    spec: BeatSpec,
    bible: str,
    word_target: int,
    scene_target: int,
    previous_beats: list[str],
) -> str:
    context = ""
    if previous_beats:
        context = (
            "What came before (scene titles, in order):\n"
            + "\n".join(f"- {title}" for title in previous_beats)
            + "\nDo not repeat these scenes. Build on them.\n"
        )
    return f"""Series bible:
{bible}

Now write the "{spec.label}" beat ({spec.key}).

Purpose: {spec.purpose}
Retention device: {spec.retention}

Requirements:
- CRITICAL: Write AT LEAST {word_target} words of narration total across the beat. Do not write less.
- Exactly {scene_target} scenes.
- Each scene: a short title, narration (110-190 words), a visual_direction \
describing what the viewer sees (archival footage, NASA imagery, data visualization, \
dramatic reenactment description, etc.), and optionally a lower_third (a name, \
date, or location to display on screen, or null).

Output JSON only, exactly this shape:
{{"scenes": [{{"title": str, "narration": str, "visual_direction": str, "lower_third": str | null}}]}}
{context}"""


def _lm_studio_chat(
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    api_url: str = "http://localhost:1234/v1/chat/completions",
    model: str = "qwen3.6-35b-a3b-udt-mtp",
    temperature: float = 0.7,
    timeout: int | None = None,
    json_mode: bool = False,
) -> str:
    if timeout is None:
        # Thinking models are slow: budget a ~6 tok/s floor so a large
        # beat cannot time out mid-stream on a CPU-only box.
        timeout = max(600, max_tokens // 6)
    # Qwen3 thinking models burn the token budget on chain-of-thought and
    # return an empty message body. /no_think disables thinking for these
    # structured JSON beats, where deliberation adds nothing.
    if "qwen3" in model.lower():
        messages = [
            (
                {**m, "content": "/no_think\n" + m["content"]}
                if m.get("role") == "user"
                else m
            )
            for m in messages
        ]
    payload_dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7 if json_mode else temperature,
        "stream": False,
    }
    # JSON mode constrains the model to output valid JSON only.
    # Critical for reasoning models like Bonsai that otherwise leak
    # chain-of-thought into the response.
    if json_mode:
        payload_dict["response_format"] = {"type": "json_object"}
    payload = json.dumps(payload_dict).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise LongformError(f"LM Studio request failed: {error}") from error
    try:
        message = result["choices"][0]["message"] or {}
    except (KeyError, IndexError, TypeError) as error:
        raise LongformError("LM Studio response had no message content") from error
    content = message.get("content") or ""
    if not content.strip():
        # Thinking models can exhaust the budget inside reasoning_content;
        # the answer is sometimes recoverable from the thinking trace.
        content = message.get("reasoning_content") or ""
    if not content.strip():
        raise LongformError("LM Studio response had no message content")
    return content


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise LongformError("model output contained no JSON object")
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as error:
        raise LongformError(f"model output was not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise LongformError("model output JSON was not an object")
    return parsed


def _generate_beat_text(
    chat: Callable[..., str],
    messages: list[dict[str, str]],
    max_tokens: int,
    beat_key: str,
    validate: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Call the chat backend for one beat, retrying once on empty output.

    Reasoning models can burn their whole token budget thinking and return
    an empty message (or a truncated stream) with ``finish_reason=length``.
    The first attempt uses ``max_tokens``; the retry doubles it. A response
    is only accepted when it actually contains a JSON object. When
    ``validate`` is given it also runs inside the retry loop, and a
    validation failure is fed back to the model so the retry can correct
    the exact problem instead of guessing again. Transport errors are not
    retried here; they fail loud immediately.
    """
    budgets = [max_tokens, max_tokens * 2][:BEAT_ATTEMPTS]
    last_error: LongformError | None = None
    raw_text = ""
    current_messages = list(messages)
    for attempt, budget in enumerate(budgets, start=1):
        try:
            raw_text = chat(current_messages, budget)
        except LongformError as error:
            last_error = error
            raw_text = ""
            continue
        except Exception as error:
            raise LongformError(
                f"beat {beat_key!r} generation failed: {error}"
            ) from error
        if not (raw_text or "").strip():
            last_error = LongformError(
                f"beat {beat_key!r} returned empty content (attempt {attempt}; "
                "the model likely spent its token budget thinking)"
            )
            continue
        try:
            parsed = _extract_json_object(raw_text)
            if validate is not None:
                validate(parsed)
        except LongformError as error:
            last_error = error
            current_messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        "Your previous response was rejected for this reason:\n"
                        f"{error}\n"
                        "Fix the JSON and return ONLY the corrected JSON object."
                    ),
                }
            ]
            continue
        return raw_text
    preview = sanitize_diagnostic(raw_text or "", max_chars=500)
    detail = f": {last_error}" if last_error else ""
    raise LongformError(
        f"beat {beat_key!r} produced no usable JSON after "
        f"{len(budgets)} attempts{detail}; model preview: {preview}"
    )


@dataclass
class LongformScene:
    title: str
    narration: str
    visual_direction: str
    lower_third: str | None = None

    @property
    def words(self) -> int:
        return _words(self.narration)


@dataclass
class LongformBeat:
    spec: BeatSpec
    scenes: list[LongformScene] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(scene.words for scene in self.scenes)


@dataclass
class LongformScript:
    title: str
    description: str
    topic: str
    target_minutes: float
    beats: list[LongformBeat] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    format_key: str | None = None

    @property
    def total_words(self) -> int:
        return sum(beat.words for beat in self.beats)

    @property
    def estimated_minutes(self) -> float:
        return self.total_words / WORDS_PER_MINUTE

    @property
    def scene_titles(self) -> list[str]:
        return [scene.title for beat in self.beats for scene in beat.scenes]

    def to_json(self) -> str:
        return json.dumps(
            {
                "title": self.title,
                "description": self.description,
                "topic": self.topic,
                "target_minutes": self.target_minutes,
                "total_words": self.total_words,
                "estimated_minutes": round(self.estimated_minutes, 2),
                "sources": self.sources,
                "generated_at": self.generated_at,
                "format_key": self.format_key,
                "beats": [
                    {
                        "key": beat.spec.key,
                        "label": beat.spec.label,
                        "words": beat.words,
                        "scenes": [
                            {
                                "title": scene.title,
                                "narration": scene.narration,
                                "visual_direction": scene.visual_direction,
                                "lower_third": scene.lower_third,
                            }
                            for scene in beat.scenes
                        ],
                    }
                    for beat in self.beats
                ],
            },
            indent=2,
        )


def _validate_beat(spec: BeatSpec, scenes: list[LongformScene], word_target: int) -> None:
    if not scenes:
        raise LongformError(f"beat {spec.key!r} came back with no scenes")
    words = sum(scene.words for scene in scenes)
    if words < word_target * 0.6:
        raise LongformError(
            f"beat {spec.key!r} is too thin: {words} words vs {word_target} target"
        )
    if words > word_target * 1.6:
        raise LongformError(
            f"beat {spec.key!r} is too long: {words} words vs {word_target} target"
        )
    for index, scene in enumerate(scenes):
        if not scene.title.strip():
            raise LongformError(f"beat {spec.key!r} scene {index} has no title")
        if scene.words < 40:
            raise LongformError(
                f"beat {spec.key!r} scene {index} narration is too short "
                f"({scene.words} words)"
            )


def _parse_beat_scenes(spec: BeatSpec, raw: dict[str, Any]) -> list[LongformScene]:
    scenes_raw = raw.get("scenes")
    if not isinstance(scenes_raw, list) or not scenes_raw:
        raise LongformError(f"beat {spec.key!r} has no scenes list")
    scenes: list[LongformScene] = []
    for index, item in enumerate(scenes_raw):
        if not isinstance(item, dict):
            raise LongformError(f"beat {spec.key!r} scene {index} is not an object")
        narration = item.get("narration")
        if not isinstance(narration, str) or not narration.strip():
            raise LongformError(f"beat {spec.key!r} scene {index} has no narration")
        lower_third = item.get("lower_third")
        scenes.append(
            LongformScene(
                title=str(item.get("title") or f"Scene {index + 1}"),
                narration=narration.strip(),
                visual_direction=str(item.get("visual_direction") or ""),
                lower_third=lower_third.strip() if isinstance(lower_third, str) and lower_third.strip() else None,
            )
        )
    return scenes


def _build_bible(
    topic: str,
    description: str,
    source_url: str,
    research_brief: str | None,
    title: str | None,
    preset: FormatPreset | None = None,
) -> str:
    lines = [
        f"Topic: {topic}",
        f"Angle: {description}",
        f"Primary source: {source_url}",
    ]
    if title:
        lines.append(f"Working title: {title}")
    if research_brief:
        brief = research_brief.strip()
        lines.append(f"Research brief (ground every claim in this):\n{brief[:4000]}")
    lines.append(_format_voice_line(preset))
    return "\n".join(lines)


def generate_longform_script(
    topic: str,
    description: str,
    source_url: str,
    *,
    target_minutes: float = DEFAULT_LONGFORM_MINUTES,
    research_brief: str | None = None,
    title: str | None = None,
    sources: list[str] | None = None,
    chat_fn: Callable[[list[dict[str, str]], int], str] | None = None,
    output_path: Path | None = None,
    format_key: str | None = None,
) -> LongformScript:
    """Generate a long-form documentary script beat by beat.

    Each beat is generated in its own model call against a shared series
    bible, so the model sustains coherence across thousands of words.
    Pass ``format_key`` (see ``ai_video_factory.formats``) to use a winning
    YouTube format's beat structure, hook plan, and narration register.
    Raises ``LongformError`` when any beat is missing, malformed, or too
    far from its word target, and when the assembled script misses the
    target duration by more than 15%.
    """
    if not (MIN_LONGFORM_MINUTES <= target_minutes <= MAX_LONGFORM_MINUTES):
        raise LongformError(
            f"target_minutes must be between {MIN_LONGFORM_MINUTES} and "
            f"{MAX_LONGFORM_MINUTES}; got {target_minutes}"
        )
    chat = chat_fn or _lm_studio_chat
    preset = get_preset(format_key) if format_key else None
    bible = _build_bible(topic, description, source_url, research_brief, title, preset)
    beats = _beats_for_format(preset)

    script = LongformScript(
        title=title or f"{topic}",
        description=description,
        topic=topic,
        target_minutes=target_minutes,
        sources=sources or [source_url],
        format_key=format_key,
    )
    previous_titles: list[str] = []
    for spec in beats:
        word_target = beat_word_target(spec, target_minutes)
        scene_target = max(2, round(word_target / 150))
        prompt = _beat_user_prompt(
            spec=spec,
            bible=bible,
            word_target=word_target,
            scene_target=scene_target,
            previous_beats=previous_titles,
        )
        def _validate_beat_json(
            raw: dict[str, Any], _spec: BeatSpec = spec, _target: int = word_target
        ) -> None:
            _validate_beat(_spec, _parse_beat_scenes(_spec, raw), _target)

        try:
            raw_text = _generate_beat_text(
                chat,
                [
                    {"role": "system", "content": LONGFORM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max(BEAT_MIN_TOKENS, word_target * BEAT_TOKEN_HEADROOM),
                spec.key,
                validate=_validate_beat_json,
            )
        except LongformError:
            raise
        except Exception as error:
            raise LongformError(f"beat {spec.key!r} generation failed: {error}") from error
        parsed = _extract_json_object(raw_text)
        scenes = _parse_beat_scenes(spec, parsed)
        _validate_beat(spec, scenes, word_target)
        script.beats.append(LongformBeat(spec=spec, scenes=scenes))
        previous_titles.extend(scene.title for scene in scenes)

    target_words = target_minutes * WORDS_PER_MINUTE
    if abs(script.total_words - target_words) / target_words > 0.25:
        raise LongformError(
            f"assembled script is {script.total_words} words vs {target_words:.0f} "
            f"target (estimated {script.estimated_minutes:.1f} min vs "
            f"{target_minutes:.0f} min target)"
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(script.to_json(), encoding="utf-8")
    return script


def longform_to_edit_document(
    script: LongformScript,
    *,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    created_at: str | None = None,
) -> EditDocument:
    """Convert a long-form script into a timed edit document.

    Scene durations derive from narration word counts at 150 wpm, so the
    timeline matches the spoken track. The first scene of each beat carries
    the beat label for the act title card; cold open renders as intro and
    the final outro scene as outro.
    """
    scenes: list[EditScene] = []
    cursor = 0
    total_beats = len(script.beats)
    for beat_index, beat in enumerate(script.beats):
        for scene_index, scene in enumerate(beat.scenes):
            seconds = max(3.0, scene.words / WORDS_PER_MINUTE * 60.0)
            duration_frames = round(seconds * fps)
            is_first_overall = beat_index == 0 and scene_index == 0
            is_last_overall = (
                beat_index == total_beats - 1 and scene_index == len(beat.scenes) - 1
            )
            kind = "intro" if is_first_overall else "outro" if is_last_overall else "normal"
            edit_scene = EditScene(
                id=f"{beat.spec.key}-{scene_index}",
                from_frame=cursor,
                duration_frames=duration_frames,
                title=scene.title,
                caption=scene.narration,
                kind=kind,  # type: ignore[arg-type]
                visual=scene.visual_direction or None,
                narration=scene.narration,
                subtitle=None,
                act=beat.spec.label if scene_index == 0 else None,
                lower_third=scene.lower_third,
            )
            scenes.append(edit_scene)
            cursor += duration_frames

    return EditDocument(
        schema_version=1,
        width=width,
        height=height,
        fps=fps,
        duration_frames=cursor,
        scenes=scenes,
        title=script.title,
        description=script.description,
        created_at=created_at or script.generated_at,
        sources=script.sources,
    )


_BEAT_SPECS_BY_KEY = {spec.key: spec for spec in LONGFORM_BEATS}


def longform_script_from_dict(data: dict[str, Any]) -> LongformScript:
    """Rebuild a LongformScript from its serialized JSON form (run resume)."""
    beats: list[LongformBeat] = []
    for beat_data in data.get("beats", []):
        spec = _BEAT_SPECS_BY_KEY.get(str(beat_data.get("key", "")))
        if spec is None:
            raise LongformError(
                f"unknown beat key in saved script: {beat_data.get('key')!r}"
            )
        scenes = [
            LongformScene(
                title=str(item.get("title", "")),
                narration=str(item.get("narration", "")),
                visual_direction=str(item.get("visual_direction") or ""),
                lower_third=item.get("lower_third"),
            )
            for item in beat_data.get("scenes", [])
        ]
        beats.append(LongformBeat(spec=spec, scenes=scenes))
    return LongformScript(
        title=str(data.get("title", "")),
        description=str(data.get("description", "")),
        topic=str(data.get("topic", "")),
        target_minutes=float(data.get("target_minutes", DEFAULT_LONGFORM_MINUTES)),
        beats=beats,
        sources=list(data.get("sources", [])),
        generated_at=str(data.get("generated_at", "")),
    )
