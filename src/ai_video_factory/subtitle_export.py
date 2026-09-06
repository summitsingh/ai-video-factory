"""Generate styled subtitle files (.srt) and YouTube chapters from an edit document.

This module produces two export artifacts for a finished documentary:

1. A WebVTT/SRT subtitle file with per-scene timing derived from each scene's
   ``from_frame`` / ``duration_frames`` and the document fps, so captions line up
   exactly with the rendered video regardless of intro/outro framing.
2. A YouTube chapters manifest (WebVTT CUE points) that lets viewers jump to each
   scene by name — a standard part of production-grade uploads.

Both are written as plain files on disk; nothing is uploaded here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

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
    """Return the caption text to subtitle for a scene.

    Prefers the explicit burned-in ``subtitle`` field, falling back to the
    scene ``caption`` so every scene has a line even when one is omitted.
    """
    return (scene.subtitle or scene.caption).strip()


def build_srt(doc: EditDocument) -> str:
    """Build an SRT subtitle string from the edit document.

    Each non-intro/outro scene contributes one cue spanning its duration, timed
    from ``from_frame`` so it lines up with the rendered video. Intro/outro
    scenes are skipped (they carry no narration captions).
    """
    fps = doc.fps
    cues: list[str] = []
    index = 1
    for scene in doc.scenes:
        if scene.kind == "intro" or scene.kind == "outro":
            continue
        text = _scene_text(scene)
        if not text:
            continue
        start = _frame_to_seconds(scene.from_frame, fps)
        end = _frame_to_seconds(
            scene.from_frame + scene.duration_frames, fps
        )
        cues.append(
            f"{index}\n"
            f"{_seconds_to_srt_time(start)} --> {_seconds_to_srt_time(end)}\n"
            f"{text}\n"
        )
        index += 1
    return "\n".join(cues)


def build_webvtt(doc: EditDocument) -> str:
    """Build a WebVTT subtitle file (YouTube accepts WebVTT for captions)."""
    fps = doc.fps
    blocks: list[str] = ["WEBVTT\n"]
    index = 1
    for scene in doc.scenes:
        if scene.kind == "intro" or scene.kind == "outro":
            continue
        text = _scene_text(scene)
        if not text:
            continue
        start = _frame_to_seconds(scene.from_frame, fps)
        end = _frame_to_seconds(
            scene.from_frame + scene.duration_frames, fps
        )
        blocks.append(
            f"{index}\n"
            f"{_seconds_to_vtt_time(start)} --> {_seconds_to_vtt_time(end)}\n"
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

    srt_path.write_text(build_srt(doc), encoding="utf-8")
    vtt_path.write_text(build_webvtt(doc), encoding="utf-8")
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
        scene.kind not in ("intro", "outro") and _scene_text(scene)
        for scene in doc.scenes
    )
