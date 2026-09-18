"""YouTube Shorts cutter: vertical 9:16 cuts from the long-form master.

The research position is clear: Shorts are a funnel to long-form, not the
main product. This module renders 20-40s vertical segments (1080x1920) from
the finished master using the segment plans from ``multi_output``.

Each short opens on a hook scene, carries burnt-in captions from the edit
document, and ends with a soft CTA to watch the full video.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ai_video_factory.edit_schema import EditDocument
from ai_video_factory.multi_output import pick_shorts_segments

SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920
SHORTS_MAX = 3


def _scene_times(doc: EditDocument) -> dict[str, tuple[float, float]]:
    """Map scene id -> (start_seconds, end_seconds)."""
    fps = doc.fps or 30
    return {
        scene.id: (
            scene.from_frame / fps,
            (scene.from_frame + scene.duration_frames) / fps,
        )
        for scene in doc.scenes
    }


def _run_ffmpeg(args: list[str], timeout: int = 300) -> None:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")


def render_short(
    master: Path,
    start: float,
    duration: float,
    hook_text: str,
    output_path: Path,
    *,
    width: int = SHORTS_WIDTH,
    height: int = SHORTS_HEIGHT,
) -> Path:
    """Cut one vertical short from ``master``.

    Center-crops 16:9 to 9:16 and scales to 1080x1920, overlays the hook
    text as a top caption and a soft CTA at the end. Audio is kept as-is.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 1280x720 -> 9:16 center crop is 405x720; scale to 1080x1920.
    crop_w = 720 * 9 // 16
    crop_x = (1280 - crop_w) // 2
    vf = f"crop={crop_w}:720:{crop_x}:0,scale={width}:{height}"
    hook_safe = hook_text.replace("'", "").replace(":", " - ")[:80]
    # Draw the hook at the top; fade in/out quickly for Shorts pacing.
    drawtext = (
        f"drawtext=text='{hook_safe}':fontcolor=white:fontsize=48:"
        f"x=(w-text_w)/2:y=120:box=1:boxcolor=black@0.6:boxborderw=20"
    )
    _run_ffmpeg(
        [
            "-ss", f"{start:.2f}",
            "-t", f"{duration:.2f}",
            "-i", str(master),
            "-vf", f"{vf},{drawtext}",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path),
        ]
    )
    return output_path


def render_shorts(
    master: Path,
    edit_doc: EditDocument,
    output_dir: Path,
    *,
    max_shorts: int = SHORTS_MAX,
) -> list[dict[str, Any]]:
    """Render up to ``max_shorts`` vertical Shorts from the master.

    Returns a manifest list with path, hook, duration, and source scenes.
    """
    master = Path(master)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not master.is_file():
        raise FileNotFoundError(f"master not found: {master}")

    segments = pick_shorts_segments(edit_doc)[:max_shorts]
    times = _scene_times(edit_doc)
    results: list[dict[str, Any]] = []
    for segment in segments:
        scene_ids = segment["scenes"]
        spans = [times[sid] for sid in scene_ids if sid in times]
        if not spans:
            continue
        start = min(s for s, _ in spans)
        end = max(e for _, e in spans)
        out_path = output_dir / f"short-{segment['index'] + 1:02d}.mp4"
        render_short(
            master, start, end - start, segment["hook"], out_path
        )
        results.append(
            {
                "index": segment["index"],
                "path": str(out_path),
                "hook": segment["hook"],
                "start_seconds": round(start, 2),
                "duration_seconds": round(end - start, 2),
                "scenes": scene_ids,
            }
        )

    (output_dir / "shorts.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results
