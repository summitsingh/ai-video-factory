"""Per-output settings sidecars for generated media.

Every hero clip gets a ``<name>.json`` written next to it with the full
generation settings: prompt, seed, model, quant, timings. This kills
manifest archaeology when debugging a bad render: the answers are next to
the file, not three pipeline stages away.

The concept follows WanGP's ``record_file_metadata`` (a ``<media>.json``
sidecar carrying the full settings dict, creation timestamp, model
filename, and timings); the shape here is reimplemented for this
pipeline's fields.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SIDECAR_VERSION = 1


def sidecar_path(media_path: str | Path) -> Path:
    """Path of the sidecar for ``media_path``: same stem, ``.json`` suffix."""
    p = Path(media_path)
    return p.parent / (p.stem + ".json")


def build_sidecar(
    *,
    media_name: str,
    prompt: str,
    seed: int | None,
    model: str,
    quant: str,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    steps: int,
    variant: str | None = None,
    guidance: float | None = None,
    negative_prompt: str | None = None,
    timings: dict[str, Any] | None = None,
    backend: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the sidecar dict for one generated media file."""
    now = time.time()
    sidecar: dict[str, Any] = {
        "sidecar_version": SIDECAR_VERSION,
        "media": media_name,
        "creation_date": datetime.fromtimestamp(now, tz=timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
        "creation_timestamp": int(now),
        "prompt": prompt,
        "seed": seed,
        "model": model,
        "quant": quant,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "steps": steps,
    }
    if variant is not None:
        sidecar["variant"] = variant
    if guidance is not None:
        sidecar["guidance"] = guidance
    if negative_prompt is not None:
        sidecar["negative_prompt"] = negative_prompt
    if timings:
        sidecar["timings"] = dict(timings)
    if backend is not None:
        sidecar["backend"] = backend
    if extra:
        sidecar["extra"] = dict(extra)
    return sidecar


def write_sidecar(media_path: str | Path, settings: dict[str, Any]) -> Path:
    """Write ``settings`` as JSON next to ``media_path``. Returns the path."""
    path = sidecar_path(media_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log.info("wrote sidecar %s", path)
    return path


def record(
    media_path: str | Path,
    **kwargs: Any,
) -> Path | None:
    """Build and write a sidecar for ``media_path``; never raises.

    Returns the sidecar path, or None if writing failed (logged). A failed
    sidecar must not fail the render that produced the media.
    """
    try:
        settings = build_sidecar(
            media_name=Path(media_path).name, **kwargs
        )
        return write_sidecar(media_path, settings)
    except OSError as exc:
        log.warning("could not write sidecar for %s: %s", media_path, exc)
        return None


def read_sidecar(media_path: str | Path) -> dict[str, Any] | None:
    """Read the sidecar next to ``media_path``; None when absent/unreadable."""
    path = sidecar_path(media_path)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        log.debug("no readable sidecar at %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None
