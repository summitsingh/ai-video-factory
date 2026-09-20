"""Duration enforcement loop for long-form scripts.

After script generation completes, the assembled narration is often
shorter than the target runtime (models undershoot word targets even
when told otherwise). ``enforce_duration`` closes the gap: it computes
total words vs the target (150 words/minute), finds the most underweight
beats relative to their word allocation, and extends them through the
same LLM beat-generation path used for the original beats (new scenes
appended to the beat, never rewriting what is already there). If every
beat is already at its cap and the script is still short, a new encore
beat is appended instead. The loop repeats up to ``max_iterations``
times until the script is within ``tolerance`` of the target, then fails
loud when the gap cannot be closed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_video_factory.longform import (
    BEAT_MIN_TOKENS,
    BEAT_TOKEN_HEADROOM,
    WORDS_PER_MINUTE,
    BeatSpec,
    LongformBeat,
    LongformError,
    LongformScene,
    LongformScript,
    _BEAT_SPECS_BY_KEY,
    _extract_json_object,
    _generate_beat_text,
    _parse_beat_scenes,
    _words,
    beat_word_target,
    LONGFORM_SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)

# A beat at or above this multiple of its word target is considered "at
# cap" and will not be extended further (mirrors _validate_beat's upper
# bound so enforcement can never push a beat into the rejected range).
_BEAT_CAP_MULTIPLE = 1.6

# Synthetic beat appended when every existing beat is at cap.
_ENCORE_SPEC = BeatSpec(
    key="encore",
    label="ENCORE",
    fraction=0.0,  # fraction set from the remaining deficit at append time
    purpose=(
        "One more movement the story needs: the strongest unused angle, "
        "the counter-argument, or the implication the viewer will carry "
        "out of the video."
    ),
    retention="End on a forward pull: a question, tease, or reversal.",
)


def _extension_user_prompt(
    *,
    spec: BeatSpec,
    scenes: list[LongformScene],
    deficit_words: int,
    scene_target: int,
    bible: str = "",
) -> str:
    existing = "\n".join(
        f"{index + 1}. {scene.title}: {scene.narration[:220].strip()}..."
        for index, scene in enumerate(scenes)
    )
    bible_block = f"Series bible (stay consistent with this):\n{bible}\n\n" if bible else ""
    return f"""{bible_block}You are extending the "{spec.label}" beat ({spec.key}) of a documentary script. \
The beat came back shorter than its word target and must be lengthened WITHOUT repeating existing material.

Current scenes in this beat (do not repeat or rephrase these):
{existing}

Beat purpose: {spec.purpose}
Retention device: {spec.retention}

Requirements:
- CRITICAL: write EXACTLY {scene_target} NEW scene(s) totaling AT LEAST {deficit_words} additional words of narration.
- New angles only: concrete details, numbers, competing explanations, or deepening stakes the current scenes do not cover.
- The final new scene must end on a forward pull (question, tease, or reversal) that leads into the next beat.

Output JSON only, exactly this shape:
{{"scenes": [{{"title": str, "narration": str, "visual_direction": str, "lower_third": str | null}}]}}"""


def _encore_user_prompt(
    *,
    deficit_words: int,
    scene_target: int,
    previous_titles: list[str],
) -> str:
    context = ""
    if previous_titles:
        context = (
            "Scene titles already used (do not repeat these angles):\n"
            + "\n".join(f"- {title}" for title in previous_titles[-30:])
            + "\n"
        )
    return f"""You are writing a bonus "ENCORE" beat for a documentary script that is still short of its target runtime.

Purpose: {_ENCORE_SPEC.purpose}
Retention device: {_ENCORE_SPEC.retention}

Requirements:
- CRITICAL: write EXACTLY {scene_target} scene(s) totaling AT LEAST {deficit_words} words of narration.
- Each scene: a short title, narration (110-190 words), a visual_direction, and optionally a lower_third.
{context}
Output JSON only, exactly this shape:
{{"scenes": [{{"title": str, "narration": str, "visual_direction": str, "lower_third": str | null}}]}}"""


def _validate_extension(raw: dict[str, Any], beat_key: str) -> list[LongformScene]:
    scenes = _parse_beat_scenes(
        BeatSpec(key=beat_key, label=beat_key, fraction=0.0, purpose="", retention=""),
        raw,
    )
    if not scenes:
        raise LongformError(f"extension for beat {beat_key!r} returned no scenes")
    for index, scene in enumerate(scenes):
        if scene.words < 20:
            raise LongformError(
                f"extension for beat {beat_key!r} scene {index} is too thin "
                f"({scene.words} words)"
            )
    return scenes


def _extend_beat(
    chat: Callable[[list[dict[str, str]], int], str],
    *,
    beat: LongformBeat,
    deficit_words: int,
    beat_key: str,
    bible: str = "",
) -> list[LongformScene]:
    """Generate additional scenes for an underweight beat.

    Uses the same LLM beat-generation path as the original beats
    (``_generate_beat_text`` with retry/feedback), but the prompt asks for
    NEW scenes on top of the beat's existing narration instead of a fresh
    beat. Returns the new scenes to append.
    """
    spec = beat.spec
    scene_target = max(1, round(deficit_words / 150))
    prompt = _extension_user_prompt(
        spec=spec,
        scenes=beat.scenes,
        deficit_words=deficit_words,
        scene_target=scene_target,
        bible=bible,
    )
    messages = [
        {"role": "system", "content": LONGFORM_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    raw_text = _generate_beat_text(
        chat,
        messages,
        max(BEAT_MIN_TOKENS, deficit_words * BEAT_TOKEN_HEADROOM),
        f"{beat_key}+ext",
    )
    parsed = _extract_json_object(raw_text)
    return _validate_extension(parsed, beat_key)


def _append_encore_beat(
    chat: Callable[[list[dict[str, str]], int], str],
    *,
    script: LongformScript,
    deficit_words: int,
) -> LongformBeat:
    """Append a synthetic encore beat when every beat is at cap."""
    target_words = script.target_minutes * WORDS_PER_MINUTE
    spec = BeatSpec(
        key="encore",
        label=_ENCORE_SPEC.label,
        fraction=round(deficit_words / target_words, 4) if target_words else 0.0,
        purpose=_ENCORE_SPEC.purpose,
        retention=_ENCORE_SPEC.retention,
    )
    scene_target = max(1, round(deficit_words / 150))
    prompt = _encore_user_prompt(
        deficit_words=deficit_words,
        scene_target=scene_target,
        previous_titles=script.scene_titles,
    )
    messages = [
        {"role": "system", "content": LONGFORM_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    raw_text = _generate_beat_text(
        chat,
        messages,
        max(BEAT_MIN_TOKENS, deficit_words * BEAT_TOKEN_HEADROOM),
        "encore",
    )
    parsed = _extract_json_object(raw_text)
    scenes = _validate_extension(parsed, "encore")
    beat = LongformBeat(spec=spec, scenes=scenes)
    # Register the synthetic spec so run-resume (longform_script_from_dict)
    # can rebuild the script from its serialized form.
    _BEAT_SPECS_BY_KEY[spec.key] = spec
    return beat


@dataclass
class BeatExtensionRecord:
    beat_key: str
    words_before: int
    words_added: int
    words_after: int
    kind: str = "extend"  # "extend" or "encore"

    def to_dict(self) -> dict[str, Any]:
        return {
            "beat_key": self.beat_key,
            "kind": self.kind,
            "words_before": self.words_before,
            "words_added": self.words_added,
            "words_after": self.words_after,
        }


@dataclass
class EnforcementIteration:
    iteration: int
    total_words_before: int
    total_words_after: int
    target_words: int
    extensions: list[BeatExtensionRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "total_words_before": self.total_words_before,
            "total_words_after": self.total_words_after,
            "target_words": self.target_words,
            "extensions": [ext.to_dict() for ext in self.extensions],
        }


@dataclass
class DurationEnforcementReport:
    target_minutes: float
    target_words: int
    tolerance: float
    iterations: list[EnforcementIteration] = field(default_factory=list)
    final_words: int = 0
    within_tolerance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_minutes": self.target_minutes,
            "target_words": self.target_words,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
            "final_words": self.final_words,
            "estimated_minutes": round(self.final_words / WORDS_PER_MINUTE, 2),
            "iterations": [it.to_dict() for it in self.iterations],
        }


def _deficit_for_beat(
    beat: LongformBeat, target_minutes: float
) -> tuple[int, int]:
    """Return (word_target, deficit) for a beat; deficit <= 0 means on target."""
    target = beat_word_target(beat.spec, target_minutes)
    return target, target - beat.words


def enforce_duration(
    script: LongformScript,
    target_minutes: float,
    chat: Callable[[list[dict[str, str]], int], str],
    *,
    bible: str = "",
    tolerance: float = 0.05,
    max_iterations: int = 3,
    max_beats_per_iteration: int = 2,
) -> DurationEnforcementReport:
    """Extend an underweight script until it reaches the target duration.

    Compares ``script.total_words`` against ``target_minutes * 150 wpm``.
    While the script is more than ``tolerance`` short, each iteration
    extends the most underweight beats (vs their word allocation) via the
    LLM beat-generation path, appending brand-new scenes. When every beat
    is at cap (1.6x its allocation) and the script is still short, a new
    encore beat is appended instead. Mutates ``script`` in place and
    returns a report of every iteration for the run log.
    """
    target_words = int(target_minutes * WORDS_PER_MINUTE)
    report = DurationEnforcementReport(
        target_minutes=target_minutes,
        target_words=target_words,
        tolerance=tolerance,
    )
    log.info(
        "duration enforcement: script has %d words vs %d target (%.1f min)",
        script.total_words,
        target_words,
        target_minutes,
    )
    for iteration in range(1, max_iterations + 1):
        total = script.total_words
        shortfall = target_words - total
        if shortfall <= target_words * tolerance:
            report.within_tolerance = True
            report.final_words = total
            log.info(
                "duration enforcement: within tolerance after %d iteration(s) "
                "(%d words vs %d target)",
                iteration - 1,
                total,
                target_words,
            )
            return report

        record = EnforcementIteration(
            iteration=iteration,
            total_words_before=total,
            total_words_after=total,
            target_words=target_words,
        )
        # Most underweight beats first.
        ranked: list[tuple[int, LongformBeat, int]] = []
        for beat in script.beats:
            word_target, deficit = _deficit_for_beat(beat, target_minutes)
            if deficit > 0 and beat.words < word_target * _BEAT_CAP_MULTIPLE:
                ranked.append((deficit, beat, word_target))
        ranked.sort(key=lambda item: item[0], reverse=True)

        extended_any = False
        if ranked:
            for deficit, beat, _word_target in ranked[:max_beats_per_iteration]:
                before = beat.words
                try:
                    new_scenes = _extend_beat(
                        chat,
                        beat=beat,
                        deficit_words=deficit,
                        beat_key=beat.spec.key,
                        bible=bible,
                    )
                except LongformError as error:
                    log.warning(
                        "duration enforcement: extension of beat %r failed: %s",
                        beat.spec.key,
                        error,
                    )
                    continue
                beat.scenes.extend(new_scenes)
                added = beat.words - before
                extended_any = True
                log.info(
                    "duration enforcement: extended beat %r: %d -> %d words "
                    "(+%d)",
                    beat.spec.key,
                    before,
                    beat.words,
                    added,
                )
                record.extensions.append(
                    BeatExtensionRecord(
                        beat_key=beat.spec.key,
                        words_before=before,
                        words_added=added,
                        words_after=beat.words,
                    )
                )
        else:
            # Every beat is at cap (or no beat is underweight) yet the
            # script is still short: append a synthetic encore beat.
            log.info(
                "duration enforcement: all beats at cap, appending encore beat "
                "(shortfall %d words)",
                shortfall,
            )
            try:
                encore = _append_encore_beat(
                    chat, script=script, deficit_words=shortfall
                )
            except LongformError as error:
                log.warning(
                    "duration enforcement: encore beat generation failed: %s",
                    error,
                )
            else:
                # Insert before the outro when one exists so the video
                # still closes properly.
                insert_at = len(script.beats)
                for index, beat in enumerate(script.beats):
                    if beat.spec.key == "outro":
                        insert_at = index
                        break
                script.beats.insert(insert_at, encore)
                extended_any = True
                log.info(
                    "duration enforcement: appended encore beat with %d words",
                    encore.words,
                )
                record.extensions.append(
                    BeatExtensionRecord(
                        beat_key="encore",
                        words_before=0,
                        words_added=encore.words,
                        words_after=encore.words,
                        kind="encore",
                    )
                )

        record.total_words_after = script.total_words
        report.iterations.append(record)
        if not extended_any:
            log.warning(
                "duration enforcement: iteration %d extended nothing; stopping",
                iteration,
            )
            break

    report.final_words = script.total_words
    report.within_tolerance = (
        target_words - report.final_words
    ) <= target_words * tolerance
    if not report.within_tolerance:
        raise LongformError(
            f"duration enforcement failed after {len(report.iterations)} "
            f"iteration(s): script is {report.final_words} words vs "
            f"{target_words} target ({report.final_words / WORDS_PER_MINUTE:.1f} "
            f"min vs {target_minutes:.0f} min target)"
        )
    return report


def words_per_minute() -> int:
    """Narration pace assumption used for duration math."""
    return WORDS_PER_MINUTE
