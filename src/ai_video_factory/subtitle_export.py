"""Generate styled subtitle files (.srt) and YouTube chapters from an edit document.

This module produces export artifacts for a finished documentary:

1. A WebVTT/SRT subtitle file with per-scene timing derived from each scene's
   ``from_frame`` / ``duration_frames`` and the document fps, so captions line up
   exactly with the rendered video regardless of intro/outro framing.
2. A YouTube chapters manifest (WebVTT CUE points) that lets viewers jump to each
   scene by name - a standard part of production-grade uploads.
3. A karaoke-style ASS subtitle file (``build_ass_karaoke``) with word-level
   ``{\\k}`` timing for burned-in word-highlight captions, plus
   ``burn_karaoke_captions`` to burn it onto the finished master with ffmpeg's
   libass ``subtitles`` filter.

Word-timing note: the TTS backends in narration.py (kokoro / piper / espeak)
emit no word-level timestamps, so per-word karaoke timing is ESTIMATED by
distributing each caption segment's words evenly across the segment duration
(weighted slightly by word length). This tracks the spoken narration well
enough for a highlight sweep; syllable-exact highlighting would need forced
alignment (e.g. whisper word timestamps) - a possible future upgrade.

All artifacts are written as plain files on disk; nothing is uploaded here.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from ai_video_factory.edit_schema import EditDocument, EditScene


def _frame_to_seconds(frame: int, fps: float) -> float:
    """Convert a frame index to seconds."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return frame / fps


def _seconds_to_srt_time(seconds: float) -> str:
    """Format seconds as an SRT timestamp ``HH:MM:SS,mmm``."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _seconds_to_vtt_time(seconds: float) -> str:
    """Format seconds as a WebVTT timestamp ``HH:MM:SS.mmm``."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _scene_text(scene: EditScene) -> str:
    """Return the fallback caption text for a scene.

    Prefers the explicit burned-in ``subtitle`` field, falling back to the
    scene ``caption`` so every scene has a line even when one is omitted.
    Used only when a scene carries no narration text; see ``scene_cues``.
    """
    return (scene.subtitle or scene.caption).strip()


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MIN_CUE_SECONDS = 1.0


def split_narration(text: str) -> list[str]:
    """Split narration into sentence-level caption chunks.

    Falls back to word-wrapped chunks for text without sentence punctuation
    so very long single sentences still produce readable captions.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return []
    chunks: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        # Keep captions readable: wrap sentences longer than ~14 words.
        while len(words) > 14:
            chunks.append(" ".join(words[:14]))
            words = words[14:]
        chunks.append(" ".join(words))
    return [c for c in chunks if c]


def distribute_cues(
    chunks: list[str], start: float, end: float
) -> list[tuple[float, float, str]]:
    """Spread caption chunks across [start, end), proportional to length.

    Each cue gets at least ``_MIN_CUE_SECONDS``; shortfalls are taken from
    the remaining time so cues stay sequential and never overlap.
    """
    if not chunks or end <= start:
        return []
    total_chars = sum(len(c) for c in chunks)
    duration = end - start
    raw = [max(_MIN_CUE_SECONDS, duration * len(c) / total_chars) for c in chunks]
    scale = duration / sum(raw)
    cues: list[tuple[float, float, str]] = []
    cursor = start
    for chunk, raw_seconds in zip(chunks, raw):
        cue_seconds = raw_seconds * scale
        cue_end = min(end, cursor + cue_seconds)
        cues.append((cursor, cue_end, chunk))
        cursor = cue_end
    return cues


def scene_cues(
    scene: EditScene, fps: float, narration_overrides: dict[str, str] | None = None
) -> tuple[list[tuple[float, float, str]], bool]:
    """Build timed caption cues for one scene.

    Returns (cues, used_title). Narration text is preferred: it is split into
    sentence-level chunks distributed across the scene duration. When the
    scene has no narration, a single cue from the scene title is used and
    ``used_title`` is True so callers can flag it.
    """
    start = _frame_to_seconds(scene.from_frame, fps)
    end = _frame_to_seconds(scene.from_frame + scene.duration_frames, fps)
    narration = None
    if narration_overrides is not None and scene.id in narration_overrides:
        narration = narration_overrides[scene.id]
    elif scene.narration:
        narration = scene.narration
    narration = (narration or "").strip()
    if narration:
        chunks = split_narration(narration)
        if chunks:
            return distribute_cues(chunks, start, end), False
    title_text = _scene_text(scene)
    if title_text:
        return [(start, end, title_text)], True
    return [], False


def subtitle_provenance(
    doc: EditDocument, narration_overrides: dict[str, str] | None = None
) -> dict[str, Any]:
    """Describe where the subtitle cues come from.

    ``captions_from_titles`` is True when at least one scene fell back to
    title text because it had no narration, so downstream consumers (and the
    pipeline metadata) can see that captions are not speech-faithful.
    """
    narration_cues = 0
    title_cues = 0
    for scene in doc.scenes:
        if scene.kind in ("intro", "outro"):
            continue
        cues, used_title = scene_cues(scene, doc.fps, narration_overrides)
        if used_title:
            title_cues += len(cues)
        else:
            narration_cues += len(cues)
    return {
        "captions_from_titles": title_cues > 0,
        "narration_cues": narration_cues,
        "title_cues": title_cues,
    }


def build_srt(
    doc: EditDocument, narration_overrides: dict[str, str] | None = None
) -> str:
    """Build an SRT subtitle string from the edit document.

    Captions come from each scene's narration text, split into sentence-level
    cues distributed across the scene's duration so they track the spoken
    audio. Scenes without narration fall back to a single title cue (see
    ``subtitle_provenance``). Intro/outro scenes are skipped (they carry no
    narration captions).
    """
    fps = doc.fps
    cues: list[str] = []
    index = 1
    for scene in doc.scenes:
        if scene.kind == "intro" or scene.kind == "outro":
            continue
        for cue_start, cue_end, text in scene_cues(scene, fps, narration_overrides)[0]:
            cues.append(
                f"{index}\n"
                f"{_seconds_to_srt_time(cue_start)} --> {_seconds_to_srt_time(cue_end)}\n"
                f"{text}\n"
            )
            index += 1
    return "\n".join(cues)


def build_webvtt(
    doc: EditDocument, narration_overrides: dict[str, str] | None = None
) -> str:
    """Build a WebVTT subtitle file (YouTube accepts WebVTT for captions)."""
    fps = doc.fps
    blocks: list[str] = ["WEBVTT\n"]
    index = 1
    for scene in doc.scenes:
        if scene.kind == "intro" or scene.kind == "outro":
            continue
        for cue_start, cue_end, text in scene_cues(scene, fps, narration_overrides)[0]:
            blocks.append(
                f"{index}\n"
                f"{_seconds_to_vtt_time(cue_start)} --> {_seconds_to_vtt_time(cue_end)}\n"
                f"{text}\n"
            )
            index += 1
    return "\n".join(blocks)


def build_chapters(doc: EditDocument) -> str:
    """Build a WebVTT file with CUE points for YouTube chapters.

    Each named scene becomes a ``CUE`` timestamp so viewers can jump to it in the
    player. Intro/outro scenes are included as chapter markers too.
    """
    fps = doc.fps
    lines = ["WEBVTT\n", "Kind: captions"]
    for i, scene in enumerate(doc.scenes):
        start = _frame_to_seconds(scene.from_frame, fps)
        label = (scene.title or f"Scene {i + 1}").strip()
        lines.append(f"CUE\nSTART {_seconds_to_vtt_time(start)}\nEND {_seconds_to_vtt_time(start + 0.001)}\nTEXT {label}")
    return "\n".join(lines)


def build_chapters_json(doc: EditDocument) -> str:
    """Build a JSON chapters manifest (title + start seconds), useful for APIs."""
    fps = doc.fps
    chapters = [
        {"title": scene.title or f"Scene {i + 1}", "start_seconds": _frame_to_seconds(scene.from_frame, fps)}
        for i, scene in enumerate(doc.scenes)
    ]
    return json.dumps({"chapters": chapters}, indent=2)


def write_subtitles(
    doc: EditDocument,
    output_dir: Path,
    *,
    base_name: str = "subtitles",
    narration_overrides: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Write SRT and WebVTT subtitle files plus a chapters manifest.

    Returns a mapping of artifact name to written path so callers can register
    them as pipeline artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    srt_path = output_dir / f"{base_name}.srt"
    vtt_path = output_dir / f"{base_name}.webvtt"
    chapters_vtt_path = output_dir / f"{base_name}-chapters.webvtt"
    chapters_json_path = output_dir / f"{base_name}-chapters.json"

    srt_path.write_text(build_srt(doc, narration_overrides), encoding="utf-8")
    vtt_path.write_text(build_webvtt(doc, narration_overrides), encoding="utf-8")
    chapters_vtt_path.write_text(build_chapters(doc), encoding="utf-8")
    chapters_json_path.write_text(build_chapters_json(doc), encoding="utf-8")

    return {
        "srt": srt_path,
        "webvtt": vtt_path,
        "chapters_webvtt": chapters_vtt_path,
        "chapters_json": chapters_json_path,
    }


def validate_subtitles(doc: EditDocument) -> bool:
    """Return True if the document has at least one subtitle-able scene."""
    return any(
        scene.kind not in ("intro", "outro")
        and ((scene.narration or "").strip() or _scene_text(scene))
        for scene in doc.scenes
    )
# ---------------------------------------------------------------------------
# Karaoke-style styled captions (#9)
# ---------------------------------------------------------------------------

_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,DejaVu Sans,64,&H0000D7FF,&H00FFFFFF,&H80000000,&H96000000,-1,0,0,0,100,100,0,0,4,2,1,2,40,40,56,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _seconds_to_ass_time(seconds: float) -> str:
    """Format seconds as an ASS timestamp ``H:MM:SS.cc``."""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    hours, remainder = divmod(total_cs, 3600_00)
    minutes, remainder = divmod(remainder, 60_00)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    """Neutralize ASS override-block braces in caption text."""
    return text.replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def distribute_word_timings(
    text: str, start: float, end: float
) -> list[tuple[float, float, str]]:
    """Evenly distribute words across [start, end), weighted by word length.

    The TTS backends emit no word-level timestamps (see the module docstring),
    so per-word karaoke timing is an ESTIMATE: each word gets a share of the
    segment proportional to its length, and the words tile the segment
    contiguously with no gaps or overlaps.
    """
    words = text.split()
    if not words or end <= start:
        return []
    weights = [max(2, len(word)) for word in words]
    total = sum(weights)
    duration = end - start
    cues: list[tuple[float, float, str]] = []
    cursor = start
    for index, (word, weight) in enumerate(zip(words, weights)):
        word_end = end if index == len(words) - 1 else cursor + duration * weight / total
        cues.append((cursor, word_end, word))
        cursor = word_end
    return cues


def _ass_dialogue(
    cue_start: float,
    cue_end: float,
    word_cues: list[tuple[float, float, str]],
    *,
    style: str = "Karaoke",
) -> str:
    """Build one ASS Dialogue line with ``{\\k}`` per-word karaoke timing."""
    parts = []
    for word_start, word_end, word in word_cues:
        centiseconds = max(1, int(round((word_end - word_start) * 100)))
        parts.append(f"{{\\k{centiseconds}}}{_ass_escape(word)}")
    text = " ".join(parts)
    return (
        f"Dialogue: 0,{_seconds_to_ass_time(cue_start)},"
        f"{_seconds_to_ass_time(cue_end)},{style},,0,0,0,,{text}"
    )


def build_ass_karaoke(
    doc: EditDocument, narration_overrides: dict[str, str] | None = None
) -> str:
    """Build a karaoke-style ASS subtitle string with word-level timing.

    One Dialogue event per caption cue, using the same cue source as the SRT
    export (narration split into sentence-level chunks distributed across
    each scene). Every word inside a cue gets a ``{\\k}`` sweep tag so the
    current word highlights (yellow fill sweeping over white on a dark
    translucent box, lower-third position) as it is spoken.

    Word timings are estimated by :func:`distribute_word_timings` - see the
    module docstring. Intro/outro scenes are skipped, matching the SRT.
    """
    lines = [_ASS_HEADER.rstrip("\n")]
    for scene in doc.scenes:
        if scene.kind == "intro" or scene.kind == "outro":
            continue
        for cue_start, cue_end, text in scene_cues(scene, doc.fps, narration_overrides)[0]:
            word_cues = distribute_word_timings(text, cue_start, cue_end)
            if not word_cues:
                continue
            lines.append(_ass_dialogue(cue_start, cue_end, word_cues))
    return "\n".join(lines) + "\n"


def write_ass_karaoke(
    doc: EditDocument,
    output_dir: Path,
    *,
    base_name: str = "subtitles",
    narration_overrides: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Write the karaoke ASS file next to the plain subtitle exports.

    Returns ``{"ass": path}`` so callers can register it as a pipeline
    artifact. The plain .srt/.vtt exports are untouched (kept for YouTube
    upload).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ass_path = output_dir / f"{base_name}.karaoke.ass"
    ass_path.write_text(
        build_ass_karaoke(doc, narration_overrides), encoding="utf-8"
    )
    return {"ass": ass_path}


def burn_karaoke_captions(
    source: Path,
    ass_path: Path,
    destination: Path,
    *,
    ffmpeg: str = "ffmpeg",
    timeout: int = 600,
) -> Path:
    """Burn a karaoke ASS file onto a video (post-render ffmpeg pass).

    Uses the libass ``subtitles`` filter. Video is re-encoded (libx264,
    crf 18, veryfast - matching the pipeline's polish pass) while audio
    is stream-copied (optional mapping, so silent test clips work too).
    Returns ``destination``.
    """
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ass = str(Path(ass_path).resolve())
    # Escape for the subtitles filter parser: backslashes, colons and
    # single quotes in the path need escaping, and the whole filename is
    # single-quoted for the filter.
    escaped = ass.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    vf = f"subtitles='{escaped}'"
    argv = [
        ffmpeg, "-y",
        "-i", str(source),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", "30",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:a", "copy",
        str(destination),
    ]
    subprocess.run(argv, check=True, capture_output=True, text=True, timeout=timeout)
    return destination
