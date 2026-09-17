"""Daily channel scheduler for automated YouTube publishing.

Each channel has its own topic focus, schedule, and runtime target. The
scheduler:

- picks a fresh topic per channel (topic ledger with exact + near-duplicate
  detection, so channels never repeat themselves),
- runs at most one scheduler at a time (file lock),
- runs only channels that are due (once per day, after their scheduled time),
- uploads unlisted and files the video in a review queue for human approval
  before anything goes public.
"""

from __future__ import annotations

import fcntl
import json
import re
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo


class SchedulerError(RuntimeError):
    """Raised for scheduler misconfiguration or lock contention."""


class SchedulerBusy(SchedulerError):
    """Raised when another scheduler run holds the lock."""


VALID_PRIVACY = ("unlisted", "private", "public")


@dataclass
class ChannelConfig:
    name: str
    slug: str
    topic_focus: str
    theme: str = "space"
    target_duration_minutes: float = 25.0
    schedule_time: str = "09:00"
    timezone: str = "UTC"
    privacy: str = "unlisted"
    category_id: str = "28"
    tags: list[str] = field(default_factory=list)
    language: str = "en"
    research_queries: list[str] = field(default_factory=list)
    trend_source: str = "all"

    def __post_init__(self) -> None:
        if self.privacy not in VALID_PRIVACY:
            raise SchedulerError(
                f"channel {self.slug!r}: privacy must be one of {VALID_PRIVACY}"
            )
        if not 20 <= self.target_duration_minutes <= 30:
            raise SchedulerError(
                f"channel {self.slug!r}: target_duration_minutes must be 20-30"
            )
        try:
            parsed = dtime.fromisoformat(self.schedule_time)
        except ValueError as error:
            raise SchedulerError(
                f"channel {self.slug!r}: bad schedule_time {self.schedule_time!r}"
            ) from error
        self.schedule_time = parsed.strftime("%H:%M")
        try:
            ZoneInfo(self.timezone)
        except Exception as error:
            raise SchedulerError(
                f"channel {self.slug!r}: bad timezone {self.timezone!r}"
            ) from error


def load_channels_config(path: Path) -> list[ChannelConfig]:
    """Load channel definitions from a TOML file with [[channels]] entries."""
    path = Path(path)
    if not path.is_file():
        raise SchedulerError(f"channels config not found: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise SchedulerError(f"invalid TOML in {path}: {error}") from error
    entries = data.get("channels")
    if not isinstance(entries, list) or not entries:
        raise SchedulerError(f"{path} must define a non-empty [[channels]] list")
    configs: list[ChannelConfig] = []
    seen_slugs: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SchedulerError(f"channels[{index}] is not a table")
        try:
            config = ChannelConfig(
                name=str(entry["name"]),
                slug=str(entry["slug"]),
                topic_focus=str(entry["topic_focus"]),
                theme=str(entry.get("theme", "space")),
                target_duration_minutes=float(entry.get("target_duration_minutes", 25.0)),
                schedule_time=str(entry.get("schedule_time", "09:00")),
                timezone=str(entry.get("timezone", "UTC")),
                privacy=str(entry.get("privacy", "unlisted")),
                category_id=str(entry.get("category_id", "28")),
                tags=list(entry.get("tags", [])),
                language=str(entry.get("language", "en")),
                research_queries=list(entry.get("research_queries", [])),
                trend_source=str(entry.get("trend_source", "all")),
            )
        except KeyError as error:
            raise SchedulerError(
                f"channels[{index}] is missing required key {error}"
            ) from error
        if config.slug in seen_slugs:
            raise SchedulerError(f"duplicate channel slug: {config.slug!r}")
        seen_slugs.add(config.slug)
        configs.append(config)
    return configs


def normalize_topic(topic: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", "", topic.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _token_set(normalized: str) -> frozenset[str]:
    return frozenset(normalized.split())


def _near_duplicate(a: str, b: str, threshold: float = 0.85) -> bool:
    tokens_a, tokens_b = _token_set(a), _token_set(b)
    if not tokens_a or not tokens_b:
        return False
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b) >= threshold


class TopicLedger:
    """Append-only record of topics each channel has already covered.

    Dedup is exact on the normalized topic plus a token-set Jaccard check
    (>= 0.85) so "Water on Mars" and "water on mars!!" cannot both ship.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._entries.append(json.loads(line))

    def is_duplicate(self, channel_slug: str, topic: str) -> bool:
        normalized = normalize_topic(topic)
        if not normalized:
            return True
        for entry in self._entries:
            if entry.get("channel") != channel_slug:
                continue
            existing = str(entry.get("normalized", ""))
            if existing == normalized or _near_duplicate(existing, normalized):
                return True
        return False

    def add(
        self,
        channel_slug: str,
        topic: str,
        *,
        run_id: str | None = None,
        video_id: str | None = None,
    ) -> dict[str, Any]:
        entry = {
            "channel": channel_slug,
            "topic": topic,
            "normalized": normalize_topic(topic),
            "run_id": run_id,
            "video_id": video_id,
            "created_at": datetime.now().isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        self._entries.append(entry)
        return entry

    def topics_for(self, channel_slug: str) -> list[dict[str, Any]]:
        return [e for e in self._entries if e.get("channel") == channel_slug]


@contextmanager
def scheduler_lock(data_dir: Path) -> Iterator[None]:
    """Hold an exclusive, non-blocking lock for the duration of a run.

    Guarantees at most one scheduler runs at a time, even across cron
    overlaps or manual invocations.
    """
    lock_path = Path(data_dir) / "scheduler.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise SchedulerBusy(
                "another scheduler run is already holding the lock"
            ) from error
        try:
            handle.write(f"{datetime.now().isoformat()}\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _channel_state_path(state_dir: Path, slug: str) -> Path:
    return Path(state_dir) / f"{slug}.json"


def read_channel_state(state_dir: Path, slug: str) -> dict[str, Any]:
    path = _channel_state_path(state_dir, slug)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_channel_state(state_dir: Path, slug: str, state: dict[str, Any]) -> None:
    path = _channel_state_path(state_dir, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def is_channel_due(
    config: ChannelConfig, state_dir: Path, now: datetime | None = None
) -> bool:
    """A channel is due when it has not run today and its time has passed."""
    now = now or datetime.now(tz=ZoneInfo(config.timezone))
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo(config.timezone))
    local = now.astimezone(ZoneInfo(config.timezone))
    state = read_channel_state(state_dir, config.slug)
    if state.get("last_run_date") == local.date().isoformat():
        return False
    scheduled = dtime.fromisoformat(config.schedule_time)
    return local.time() >= scheduled


@dataclass
class ReviewEntry:
    video_id: str
    run_id: str
    channel: str
    title: str
    url: str
    uploaded_at: str
    status: str = "pending_review"


class ReviewQueue:
    """Human approval queue: uploads land unlisted, nothing goes public
    without an explicit review approve."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries

    def add(self, entry: ReviewEntry) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "video_id": entry.video_id,
                        "run_id": entry.run_id,
                        "channel": entry.channel,
                        "title": entry.title,
                        "url": entry.url,
                        "uploaded_at": entry.uploaded_at,
                        "status": entry.status,
                    }
                )
                + "\n"
            )

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        entries = self._read_all()
        if status is not None:
            entries = [e for e in entries if e.get("status") == status]
        return entries

    def set_status(self, video_id: str, status: str) -> dict[str, Any]:
        entries = self._read_all()
        updated: dict[str, Any] | None = None
        for entry in entries:
            if entry.get("video_id") == video_id:
                entry["status"] = status
                updated = entry
        if updated is None:
            raise SchedulerError(f"no review entry for video {video_id!r}")
        with self.path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry) + "\n")
        return updated


# Callable types for the orchestration seam (real implementations are wired
# in cli.py; tests inject fakes).
ResearchFn = Callable[[ChannelConfig], list[dict[str, str]]]
PipelineFn = Callable[[ChannelConfig, str, str], dict[str, Any]]
UploadFn = Callable[[ChannelConfig, dict[str, Any]], dict[str, str]]


def pick_topic(
    config: ChannelConfig,
    ledger: TopicLedger,
    research_fn: ResearchFn,
) -> dict[str, str] | None:
    """Return the first researched candidate the channel has not covered."""
    for candidate in research_fn(config):
        title = candidate.get("title", "")
        if title and not ledger.is_duplicate(config.slug, title):
            return candidate
    return None


def run_channel(
    config: ChannelConfig,
    *,
    data_dir: Path,
    ledger: TopicLedger,
    review_queue: ReviewQueue,
    research_fn: ResearchFn,
    pipeline_fn: PipelineFn,
    upload_fn: UploadFn,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one channel end to end: topic, pipeline, unlisted upload, review.

    The topic is recorded in the ledger only after a successful upload, so a
    failed run retries the same topic tomorrow instead of burning it.
    """
    data_dir = Path(data_dir)
    state_dir = data_dir / "state"
    candidate = pick_topic(config, ledger, research_fn)
    if candidate is None:
        return {"channel": config.slug, "status": "skipped", "reason": "no fresh topic"}

    pipeline_result = pipeline_fn(config, candidate["title"], candidate.get("description", ""))
    run_id = str(pipeline_result.get("run_id", ""))
    if pipeline_result.get("status") != "pass":
        write_channel_state(
            state_dir,
            config.slug,
            {
                "last_run_date": (now or datetime.now()).date().isoformat(),
                "last_run_id": run_id,
                "last_status": "failed",
                "last_error": str(pipeline_result.get("error", "pipeline failed")),
            },
        )
        return {"channel": config.slug, "status": "failed", "run_id": run_id}

    upload_result = upload_fn(config, pipeline_result)
    video_id = str(upload_result.get("video_id", ""))
    ledger.add(config.slug, candidate["title"], run_id=run_id, video_id=video_id)
    review_queue.add(
        ReviewEntry(
            video_id=video_id,
            run_id=run_id,
            channel=config.slug,
            title=str(upload_result.get("title", candidate["title"])),
            url=f"https://www.youtube.com/watch?v={video_id}",
            uploaded_at=datetime.now().isoformat(),
        )
    )
    write_channel_state(
        state_dir,
        config.slug,
        {
            "last_run_date": (now or datetime.now()).date().isoformat(),
            "last_run_id": run_id,
            "last_status": "uploaded_unlisted",
            "last_video_id": video_id,
            "last_topic": candidate["title"],
        },
    )
    return {
        "channel": config.slug,
        "status": "uploaded_unlisted",
        "run_id": run_id,
        "video_id": video_id,
        "topic": candidate["title"],
    }


def channel_status(
    configs: list[ChannelConfig], data_dir: Path, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Monitoring snapshot: last run, due state, and topics used per channel."""
    data_dir = Path(data_dir)
    state_dir = data_dir / "state"
    ledger = TopicLedger(data_dir / "topics.jsonl")
    review_queue = ReviewQueue(data_dir / "review.jsonl")
    pending = {
        entry["channel"]
        for entry in review_queue.list(status="pending_review")
        if entry.get("channel")
    }
    rows = []
    for config in configs:
        state = read_channel_state(state_dir, config.slug)
        rows.append(
            {
                "channel": config.slug,
                "name": config.name,
                "schedule_time": config.schedule_time,
                "timezone": config.timezone,
                "target_duration_minutes": config.target_duration_minutes,
                "due": is_channel_due(config, state_dir, now),
                "last_run_date": state.get("last_run_date"),
                "last_status": state.get("last_status"),
                "last_video_id": state.get("last_video_id"),
                "topics_used": len(ledger.topics_for(config.slug)),
                "pending_review": config.slug in pending,
            }
        )
    return rows
