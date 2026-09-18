"""Overnight batch: research topics, render drafts, fill a morning review queue.

Runs unattended on summit-amd: picks fresh trending topics, generates one
script per topic, renders DRAFT masters only, and files everything in
``data/review_queue/<date>/queue.json`` for human approval. Nothing here
publishes or uploads.

Integration seams mirror scheduler.py: the research/script/pipeline steps
are injected callables so tests run without GPUs, LLMs, or the network.
The real wiring (CLI ``nightly`` command, cron) is documented in
INTEGRATION.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Callable seams. Real implementations are wired by the integrator; see
# INTEGRATION.md for the draft=True contract on PipelineFn.
ResearchFn = Callable[[int], list[dict[str, Any]]]
ScriptFn = Callable[[dict[str, Any]], dict[str, Any]]
PipelineFn = Callable[..., dict[str, Any]]

QUEUE_STATUSES = ("pending", "approved", "rejected", "failed")


@dataclass
class NightlyConfig:
    topics_per_night: int = 3
    trend_source: str = "all"
    theme: str = "space"
    target_duration_minutes: float = 25.0
    llm_url: str = "http://localhost:8080/v1/chat/completions"
    date: str | None = None

    def run_date(self, now: datetime | None = None) -> str:
        if self.date:
            return self.date
        return (now or datetime.now()).date().isoformat()


def _normalize_topic(topic: str) -> str:
    return " ".join(topic.lower().split())


def queue_dir(data_root: Path, run_date: str) -> Path:
    return Path(data_root) / "review_queue" / run_date


def load_queue_items(queue_path: Path) -> list[dict[str, Any]]:
    path = Path(queue_path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    items = data.get("items", [])
    return items if isinstance(items, list) else []


def save_queue_items(queue_path: Path, items: list[dict[str, Any]]) -> None:
    path = Path(queue_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"items": items}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _set_status(
    queue_path: Path, item_id: str, status: str
) -> dict[str, Any]:
    if status not in QUEUE_STATUSES:
        raise ValueError(f"status must be one of {QUEUE_STATUSES}")
    items = load_queue_items(queue_path)
    updated: dict[str, Any] | None = None
    for item in items:
        if str(item.get("id")) == str(item_id):
            item["status"] = status
            item["reviewed_at"] = datetime.now().isoformat()
            updated = item
    if updated is None:
        raise KeyError(f"no queue item with id {item_id!r}")
    save_queue_items(queue_path, items)
    return updated


def approve_item(queue_path: Path, item_id: str) -> dict[str, Any]:
    """Mark a review queue item approved for final render/upload."""
    return _set_status(queue_path, item_id, "approved")


def reject_item(queue_path: Path, item_id: str) -> dict[str, Any]:
    """Mark a review queue item rejected; it stays in history."""
    return _set_status(queue_path, item_id, "rejected")


def _script_summary(script_result: dict[str, Any]) -> str:
    for key in ("summary", "description", "title"):
        value = script_result.get(key)
        if value:
            return str(value)[:280]
    return ""


def run_nightly(
    data_root: Path,
    config: NightlyConfig,
    *,
    research_fn: ResearchFn,
    script_fn: ScriptFn,
    pipeline_fn: PipelineFn,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Research, draft, and queue one night of videos.

    Idempotent per date: topics already present in the day's queue (any
    status) are skipped, so a re-run never duplicates work. Returns a
    report dict with counts and the queue path.
    """
    data_root = Path(data_root)
    run_date = config.run_date(now)
    qdir = queue_dir(data_root, run_date)
    qpath = qdir / "queue.json"
    items = load_queue_items(qpath)
    queued_topics = {
        _normalize_topic(str(item.get("topic", ""))) for item in items
    }

    candidates = research_fn(config.topics_per_night * 2)
    fresh = [
        c
        for c in candidates
        if c.get("title") and _normalize_topic(c["title"]) not in queued_topics
    ][: config.topics_per_night]

    drafted = 0
    failed = 0
    skipped = 0
    for candidate in fresh:
        topic = str(candidate["title"])
        try:
            script_result = script_fn(candidate)
            # INTEGRATION CONTRACT: pipeline_fn must accept draft=True and
            # render a low-res draft master without uploading anything.
            pipeline_result = pipeline_fn(
                topic=candidate,
                script=script_result,
                draft=True,
                theme=config.theme,
                target_duration_minutes=config.target_duration_minutes,
                llm_url=config.llm_url,
            )
        except Exception as error:  # noqa: BLE001 - one bad topic must not kill the night
            failed += 1
            items.append(
                {
                    "id": f"{run_date}-{len(items) + 1:02d}",
                    "topic": topic,
                    "title": topic,
                    "status": "failed",
                    "error": str(error)[:500],
                    "generated_at": datetime.now().isoformat(),
                }
            )
            continue
        if pipeline_result.get("status") not in ("pass", "draft_ok", "ok"):
            failed += 1
            items.append(
                {
                    "id": f"{run_date}-{len(items) + 1:02d}",
                    "topic": topic,
                    "title": str(script_result.get("title", topic)),
                    "status": "failed",
                    "error": str(pipeline_result.get("error", "draft render failed"))[:500],
                    "generated_at": datetime.now().isoformat(),
                }
            )
            continue
        drafted += 1
        items.append(
            {
                "id": f"{run_date}-{len(items) + 1:02d}",
                "topic": topic,
                "title": str(script_result.get("title", topic)),
                "description": str(candidate.get("description", ""))[:280],
                "status": "pending",
                "draft_master_path": str(
                    pipeline_result.get("draft_master_path", "")
                ),
                "thumbnail_paths": list(
                    pipeline_result.get("thumbnail_paths", [])
                ),
                "script_summary": _script_summary(script_result),
                "run_id": str(pipeline_result.get("run_id", "")),
                "generated_at": datetime.now().isoformat(),
            }
        )

    save_queue_items(qpath, items)
    report = {
        "date": run_date,
        "queue_path": str(qpath),
        "topics_researched": len(candidates),
        "drafted": drafted,
        "failed": failed,
        "skipped": skipped,
        "queue_size": len(items),
    }
    (qdir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
