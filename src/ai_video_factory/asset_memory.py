"""Persistent memory of rejected assets.

Records asset IDs/URLs that failed QC so the pipeline never re-downloads
them across runs. Backed by a small JSON file; writes are atomic
(write-temp-then-rename) so a crash mid-write cannot corrupt the store.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

_SCHEMA_VERSION = 1
_ENV_VAR = "AI_VIDEO_FACTORY_ASSET_MEMORY"

# Words that carry no matching value for the relevance scorer.
_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "with", "into",
    "from", "by", "at", "is", "are", "was", "were", "to", "be", "as",
})


def _default_path() -> Path:
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override)
    return Path("data") / "asset_memory.json"


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for w in words:
        if w in _STOP_WORDS or len(w) <= 1:
            continue
        # Light plural stemming so "clouds" matches "cloud".
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


class AssetMemory:
    """JSON-backed blocklist of rejected assets."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _default_path()
        self._entries: dict[str, dict] = {}
        self._load()

    # -- persistence ----------------------------------------------------

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._entries = {}
            return
        if not isinstance(raw, dict):
            self._entries = {}
            return
        entries = raw.get("rejections", {})
        self._entries = entries if isinstance(entries, dict) else {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": _SCHEMA_VERSION, "rejections": self._entries}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # -- public API -----------------------------------------------------

    def is_blocked(self, asset_id: str) -> bool:
        """True when this asset ID or URL was rejected before."""
        return asset_id in self._entries

    def record_rejection(self, asset_id: str, reason: str, source: str = "") -> None:
        """Block an asset, remembering why and bumping its reject count."""
        now = time.time()
        entry = self._entries.get(asset_id, {})
        self._entries[asset_id] = {
            "reason": reason,
            "source": source,
            "rejected_at": now,
            "first_rejected_at": entry.get("first_rejected_at", now),
            "count": int(entry.get("count", 0)) + 1,
        }
        self._save()

    def reason_for(self, asset_id: str) -> str | None:
        """The recorded rejection reason, or None when not blocked."""
        entry = self._entries.get(asset_id)
        return entry.get("reason") if entry else None

    def prune(self, days: int = 90) -> int:
        """Drop entries older than `days`. Returns the number removed."""
        cutoff = time.time() - days * 86400
        stale = [
            key for key, entry in self._entries.items()
            if float(entry.get("rejected_at", 0)) < cutoff
        ]
        for key in stale:
            del self._entries[key]
        if stale:
            self._save()
        return len(stale)

    def stats(self) -> dict:
        """Small summary: total blocked, distinct reasons, oldest entry age."""
        reasons: dict[str, int] = {}
        oldest: float | None = None
        for entry in self._entries.values():
            reasons[entry.get("reason", "unknown")] = reasons.get(entry.get("reason", "unknown"), 0) + 1
            first = entry.get("first_rejected_at")
            if first is not None:
                oldest = first if oldest is None else min(oldest, float(first))
        return {
            "blocked": len(self._entries),
            "reasons": reasons,
            "oldest_rejected_at": oldest,
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, asset_id: str) -> bool:
        return self.is_blocked(asset_id)


def score_relevance(asset_title: str, scene_visual: str) -> float:
    """Keyword-overlap relevance of an asset title to a scene, 0-1.

    Overlap coefficient: how much of the scene's visual vocabulary appears
    in the asset title. Cheap stub; replace with embeddings when a model
    is available in the pipeline environment.
    """
    scene_words = _tokenize(scene_visual)
    if not scene_words:
        return 0.0
    title_words = _tokenize(asset_title)
    if not title_words:
        return 0.0
    return len(scene_words & title_words) / len(scene_words)
