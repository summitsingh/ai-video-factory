"""YouTube Data API v3 uploader (P0-12).

Publishes finished pipeline runs to YouTube with resumable uploads,
thumbnail and caption attachment, chapter metadata, a daily quota guard,
and an explicit approval gate before anything goes public.

Google client libraries are an optional extra (``youtube``); this module
imports them lazily so the rest of the package works without them.

Authentication is the installed-app OAuth flow: the user creates a desktop
OAuth client in Google Cloud Console once, downloads ``client_secrets.json``,
and runs ``ai-video-factory youtube auth``. The refresh token is stored at
``data/youtube/token.json`` with owner-only permissions.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from ai_video_factory.models import RunManifest, StageStatus
from ai_video_factory.sanitization import sanitize_diagnostic


SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# YouTube Data API v3 quota costs (units) for the calls we make.
QUOTA_COSTS = {
    "videos.insert": 1600,
    "thumbnails.set": 50,
    "captions.insert": 50,
}

DEFAULT_DAILY_QUOTA_BUDGET = 10_000
DEFAULT_CATEGORY_ID = "28"  # Science & Technology
DEFAULT_LANGUAGE = "en"

PRIVACY_CHOICES = ("unlisted", "private", "public")
DEFAULT_PRIVACY = "unlisted"

_CHUNK_SIZE = 8 * 1024 * 1024
_MAX_UPLOAD_ATTEMPTS = 6
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 60.0
_RETRIABLE_STATUS_CODES = {500, 502, 503, 504}

_TOKEN_FILE = "token.json"
_QUOTA_FILE = "quota.json"

_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


class YouTubeError(RuntimeError):
    """Base error for YouTube upload failures."""


class YouTubeAuthError(YouTubeError):
    """Raised when OAuth credentials are missing, invalid, or expired."""


class YouTubeQuotaError(YouTubeError):
    """Raised when the daily quota budget cannot cover the upload."""


class YouTubeUploadError(YouTubeError):
    """Raised when the resumable upload fails after retries."""


class MissingYouTubeDependencies(YouTubeError):
    """Raised when the optional ``youtube`` extra is not installed."""


def _require_google_libs() -> None:
    try:
        import googleapiclient.discovery  # noqa: F401
        import googleapiclient.errors  # noqa: F401
        import googleapiclient.http  # noqa: F401
        import google.auth.transport.requests  # noqa: F401
        import google.oauth2.credentials  # noqa: F401
    except ImportError as error:
        raise MissingYouTubeDependencies(
            "the YouTube uploader needs the optional 'youtube' extra: "
            "pip install 'ai-video-factory[youtube]'"
        ) from error


def _default_data_dir(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    override = os.environ.get("AVF_YOUTUBE_DATA_DIR")
    if override:
        return Path(override)
    return project_root / "data" / "youtube"


def _quota_budget() -> int:
    raw = os.environ.get("AVF_YOUTUBE_QUOTA_BUDGET")
    if raw is None:
        return DEFAULT_DAILY_QUOTA_BUDGET
    try:
        budget = int(raw)
    except ValueError as error:
        raise YouTubeError(f"AVF_YOUTUBE_QUOTA_BUDGET is not an integer: {raw!r}") from error
    if budget <= 0:
        raise YouTubeError("AVF_YOUTUBE_QUOTA_BUDGET must be positive")
    return budget


def token_path(data_dir: Path) -> Path:
    return Path(data_dir) / _TOKEN_FILE


def quota_path(data_dir: Path) -> Path:
    return Path(data_dir) / _QUOTA_FILE


def save_credentials(credentials: Any, data_dir: Path) -> Path:
    """Persist OAuth credentials with owner-only permissions.

    Only duck-typed credential attributes are used, so this works without
    the Google client libraries installed.
    """
    path = token_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": list(credentials.scopes or SCOPES),
    }
    if not payload["refresh_token"]:
        raise YouTubeAuthError(
            "OAuth flow did not return a refresh token; re-run "
            "'youtube auth' and approve offline access"
        )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)
    return path


def load_credentials(data_dir: Path) -> Any:
    """Load stored credentials, refreshing the access token when needed."""
    _require_google_libs()
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    path = token_path(data_dir)
    if not path.is_file():
        raise YouTubeAuthError(
            f"no YouTube credentials at {path}; run 'ai-video-factory youtube auth' first"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        credentials = Credentials(
            token=payload.get("token"),
            refresh_token=payload.get("refresh_token"),
            token_uri=payload.get("token_uri"),
            client_id=payload.get("client_id"),
            client_secret=payload.get("client_secret"),
            scopes=payload.get("scopes") or SCOPES,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise YouTubeAuthError(f"stored YouTube credentials are unreadable: {path}") from error
    if not credentials.refresh_token:
        raise YouTubeAuthError(
            "stored YouTube credentials have no refresh token; "
            "re-run 'ai-video-factory youtube auth'"
        )
    try:
        if not credentials.valid:
            credentials.refresh(Request())
            save_credentials(credentials, data_dir)
    except Exception as error:
        raise YouTubeAuthError(
            f"YouTube token refresh failed: {sanitize_diagnostic(error)}; "
            "re-run 'ai-video-factory youtube auth'"
        ) from error
    return credentials


def run_auth_flow(client_secrets: Path, data_dir: Path) -> Path:
    """Run the installed-app OAuth flow and store the refresh token."""
    _require_google_libs()
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets_path = Path(client_secrets)
    if not secrets_path.is_file():
        raise YouTubeAuthError(
            f"client secrets file not found: {secrets_path}; "
            "create a desktop OAuth client in Google Cloud Console and "
            "download client_secrets.json"
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
    credentials = flow.run_local_server(port=0)
    return save_credentials(credentials, data_dir)


def build_service(credentials: Any) -> Any:
    _require_google_libs()
    from googleapiclient.discovery import build

    return build("youtube", "v3", credentials=credentials)


class QuotaGuard:
    """Tracks YouTube API quota units per calendar day (UTC).

    The state file is best-effort local accounting, not a server-side
    guarantee; it keeps unattended runs from burning the whole daily budget
    on retries and duplicate attempts.
    """

    def __init__(self, data_dir: Path, daily_budget: int | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.daily_budget = daily_budget if daily_budget is not None else _quota_budget()
        self._path = quota_path(self.data_dir)

    def _today(self) -> str:
        return datetime.now(UTC).date().isoformat()

    def _read(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {"date": self._today(), "units_used": 0}
        try:
            state = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"date": self._today(), "units_used": 0}
        if state.get("date") != self._today():
            return {"date": self._today(), "units_used": 0}
        return {"date": state["date"], "units_used": int(state.get("units_used", 0))}

    def _write(self, state: dict[str, Any]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(self._path)

    def units_used_today(self) -> int:
        return self._read()["units_used"]

    def units_remaining(self) -> int:
        return max(0, self.daily_budget - self.units_used_today())

    def check(self, cost: int, operation: str) -> None:
        if self.units_used_today() + cost > self.daily_budget:
            raise YouTubeQuotaError(
                f"insufficient YouTube quota for {operation}: needs {cost} units, "
                f"{self.units_remaining()} of {self.daily_budget} remaining today"
            )

    def charge(self, cost: int) -> int:
        state = self._read()
        state["units_used"] = state["units_used"] + cost
        self._write(state)
        return state["units_used"]


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or H:MM:SS for YouTube chapter descriptions."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_chapter_lines(chapters: list[dict[str, Any]]) -> list[str]:
    """Render chapter lines YouTube recognizes (first timestamp must be 0:00).

    Returns an empty list when the chapters do not meet YouTube's minimum
    (at least 3 chapters, first starting at zero).
    """
    cleaned: list[tuple[float, str]] = []
    for chapter in chapters:
        title = str(chapter.get("title", "")).strip()
        try:
            start = float(chapter.get("start_seconds", 0))
        except (TypeError, ValueError):
            continue
        if title:
            cleaned.append((start, title))
    cleaned.sort(key=lambda item: item[0])
    if len(cleaned) < 3:
        return []
    if cleaned[0][0] > 1.0:
        return []
    return [f"{format_timestamp(start)} {title}" for start, title in cleaned]


def build_description(base: str, chapters: list[dict[str, Any]] | None) -> str:
    description = (base or "").strip()
    lines = build_chapter_lines(chapters or [])
    if lines:
        chapter_block = "Chapters:\n" + "\n".join(lines)
        description = f"{description}\n\n{chapter_block}" if description else chapter_block
    return description


def build_video_body(
    *,
    title: str,
    description: str,
    tags: list[str] | None = None,
    category_id: str = DEFAULT_CATEGORY_ID,
    privacy_status: str = DEFAULT_PRIVACY,
    language: str = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    title = title.strip()
    if not title:
        raise YouTubeError("video title must not be empty")
    if len(title) > 100:
        raise YouTubeError("video title must be 100 characters or fewer")
    if privacy_status not in PRIVACY_CHOICES:
        raise YouTubeError(f"privacy must be one of {PRIVACY_CHOICES}")
    body: dict[str, Any] = {
        "snippet": {
            "title": title,
            "description": description or "",
            "categoryId": category_id,
            "defaultLanguage": language,
        },
        "status": {
            "privacyStatus": privacy_status,
            "madeForKids": False,
            "selfDeclaredMadeForKids": False,
        },
    }
    if tags:
        body["snippet"]["tags"] = [tag for tag in (t.strip() for t in tags) if tag][:500]
    return body


def _is_retriable(error: BaseException) -> bool:
    status = getattr(getattr(error, "resp", None), "status", None)
    if isinstance(status, int) and status in _RETRIABLE_STATUS_CODES:
        return True
    return isinstance(error, (OSError, socket.timeout, TimeoutError, ConnectionError))


def _backoff_delay(attempt: int) -> float:
    delay = _BACKOFF_BASE_SECONDS * (2 ** attempt)
    delay = min(delay, _BACKOFF_MAX_SECONDS)
    return delay * (0.5 + random.random() * 0.5)


@dataclass
class UploadRequest:
    video_path: Path
    title: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    category_id: str = DEFAULT_CATEGORY_ID
    language: str = DEFAULT_LANGUAGE
    privacy_status: str = DEFAULT_PRIVACY
    confirm_public: bool = False
    thumbnail_path: Path | None = None
    captions_path: Path | None = None
    chapters: list[dict[str, Any]] | None = None
    dry_run: bool = False

    def validate(self) -> None:
        video = Path(self.video_path)
        if not video.is_file():
            raise YouTubeError(f"video file not found: {video}")
        if video.stat().st_size == 0:
            raise YouTubeError(f"video file is empty: {video}")
        if self.thumbnail_path is not None:
            thumb = Path(self.thumbnail_path)
            if not thumb.is_file():
                raise YouTubeError(f"thumbnail file not found: {thumb}")
            if thumb.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise YouTubeError("thumbnail must be a PNG or JPEG image")
            if thumb.stat().st_size > 2 * 1024 * 1024:
                raise YouTubeError("thumbnail must be 2 MB or smaller (YouTube limit)")
        if self.captions_path is not None and not Path(self.captions_path).is_file():
            raise YouTubeError(f"captions file not found: {self.captions_path}")
        if self.privacy_status == "public" and not self.confirm_public:
            raise YouTubeError(
                "publishing as public requires explicit confirmation; "
                "re-run with confirm_public=True (CLI: --confirm-public)"
            )
        build_video_body(
            title=self.title,
            description=self.description,
            tags=self.tags,
            category_id=self.category_id,
            privacy_status=self.privacy_status,
            language=self.language,
        )


@dataclass
class UploadResult:
    video_id: str | None
    url: str | None
    title: str
    privacy_status: str
    quota_units_used: int
    thumbnail_uploaded: bool = False
    captions_uploaded: bool = False
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "url": self.url,
            "title": self.title,
            "privacy_status": self.privacy_status,
            "quota_units_used": self.quota_units_used,
            "thumbnail_uploaded": self.thumbnail_uploaded,
            "captions_uploaded": self.captions_uploaded,
            "dry_run": self.dry_run,
        }


def _planned_cost(request: UploadRequest) -> int:
    cost = QUOTA_COSTS["videos.insert"]
    if request.thumbnail_path is not None:
        cost += QUOTA_COSTS["thumbnails.set"]
    if request.captions_path is not None:
        cost += QUOTA_COSTS["captions.insert"]
    return cost


def _upload_resumable(
    service: Any,
    body: dict[str, Any],
    video_path: Path,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """Resumable videos.insert with exponential-backoff retries."""
    _require_google_libs()
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True, chunksize=_CHUNK_SIZE)
    insert_request = service.videos().insert(
        part="snippet,status", body=body, media_body=media
    )
    attempt = 0
    total_size = video_path.stat().st_size
    while True:
        try:
            status, response = insert_request.next_chunk()
            if response is not None and "id" in response:
                return str(response["id"])
            if status is not None and progress is not None:
                progress(status.resumable_progress, total_size)
        except Exception as error:
            if not _is_retriable(error) or attempt >= _MAX_UPLOAD_ATTEMPTS:
                raise YouTubeUploadError(
                    f"video upload failed: {sanitize_diagnostic(error)}"
                ) from error
            time.sleep(_backoff_delay(attempt))
            attempt += 1


def upload_package(
    request: UploadRequest,
    service: Any,
    quota: QuotaGuard,
    progress: Callable[[int, int], None] | None = None,
) -> UploadResult:
    """Upload a video plus optional thumbnail and captions, quota-guarded."""
    request.validate()
    planned = _planned_cost(request)
    quota.check(planned, "upload package")

    description = build_description(request.description, request.chapters)
    body = build_video_body(
        title=request.title,
        description=description,
        tags=request.tags,
        category_id=request.category_id,
        privacy_status=request.privacy_status,
        language=request.language,
    )

    if request.dry_run:
        return UploadResult(
            video_id=None,
            url=None,
            title=request.title.strip(),
            privacy_status=request.privacy_status,
            quota_units_used=0,
            thumbnail_uploaded=False,
            captions_uploaded=False,
            dry_run=True,
        )

    video_id = _upload_resumable(service, body, Path(request.video_path), progress)
    quota.charge(QUOTA_COSTS["videos.insert"])

    thumbnail_uploaded = False
    if request.thumbnail_path is not None:
        _require_google_libs()
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(request.thumbnail_path), mimetype="image/png", resumable=False)
        try:
            service.thumbnails().set(videoId=video_id, media_body=media).execute()
        except Exception as error:
            raise YouTubeUploadError(
                f"thumbnail upload failed for video {video_id}: {sanitize_diagnostic(error)}"
            ) from error
        quota.charge(QUOTA_COSTS["thumbnails.set"])
        thumbnail_uploaded = True

    captions_uploaded = False
    if request.captions_path is not None:
        _require_google_libs()
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(request.captions_path), mimetype="text/plain", resumable=False)
        caption_body = {
            "snippet": {
                "videoId": video_id,
                "language": request.language,
                "name": "English",
                "isDraft": False,
            }
        }
        try:
            service.captions().insert(part="snippet", body=caption_body, media_body=media).execute()
        except Exception as error:
            raise YouTubeUploadError(
                f"captions upload failed for video {video_id}: {sanitize_diagnostic(error)}"
            ) from error
        quota.charge(QUOTA_COSTS["captions.insert"])
        captions_uploaded = True

    return UploadResult(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        title=request.title.strip(),
        privacy_status=request.privacy_status,
        quota_units_used=quota.units_used_today(),
        thumbnail_uploaded=thumbnail_uploaded,
        captions_uploaded=captions_uploaded,
    )


def find_run_manifest(data_root: Path, run_id: str) -> tuple[RunManifest, Path]:
    """Locate a pipeline run manifest under data/projects/<slug>/runs/<run-id>/."""
    if _RUN_ID.fullmatch(run_id) is None:
        raise YouTubeError("run_id must be exactly 32 lowercase hexadecimal characters")
    projects = Path(data_root) / "projects"
    matches: list[Path] = []
    if projects.is_dir():
        for slug_dir in projects.iterdir():
            candidate = slug_dir / "runs" / run_id / "manifest.json"
            if candidate.is_file():
                matches.append(candidate)
    if len(matches) != 1:
        raise YouTubeError(f"run {run_id} was not found under {projects}")
    manifest_path = matches[0]
    try:
        manifest = RunManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise YouTubeError(f"run manifest is unreadable: {manifest_path}") from error
    return manifest, manifest_path.parent


def _verify_artifact_digest(run_dir: Path, record_path: str, expected_sha256: str) -> Path:
    artifact = (run_dir / record_path).resolve()
    try:
        artifact.relative_to(run_dir.resolve())
    except ValueError as error:
        raise YouTubeError(f"artifact path escapes run directory: {record_path}") from error
    if not artifact.is_file():
        raise YouTubeError(f"artifact is missing: {record_path}")
    digest = hashlib.sha256()
    with artifact.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise YouTubeError(
            f"artifact digest changed since the run completed: {record_path}; "
            "refusing to upload a tampered artifact"
        )
    return artifact


def request_from_run(
    data_root: Path,
    run_id: str,
    *,
    title: str | None = None,
    description: str | None = None,
    thumbnail_choice: int = 1,
    privacy_status: str = DEFAULT_PRIVACY,
    confirm_public: bool = False,
    tags: list[str] | None = None,
    dry_run: bool = False,
) -> UploadRequest:
    """Build an UploadRequest from a completed pipeline run's artifacts."""
    manifest, run_dir = find_run_manifest(data_root, run_id)
    if manifest.status is not StageStatus.completed:
        raise YouTubeError(
            f"run {run_id} has status {manifest.status.value}; only completed runs can be uploaded"
        )
    integrity = manifest.artifact_integrity.get("master")
    if integrity is None:
        raise YouTubeError(f"run {run_id} has no integrity record for its master video")
    video_path = _verify_artifact_digest(run_dir, integrity.path, integrity.sha256)

    thumbnail_path: Path | None = None
    thumb_record = manifest.artifact_integrity.get(f"thumbnail_{thumbnail_choice}")
    if thumb_record is not None:
        thumbnail_path = _verify_artifact_digest(run_dir, thumb_record.path, thumb_record.sha256)

    captions_path: Path | None = None
    srt_record = manifest.artifact_integrity.get("srt")
    if srt_record is not None:
        captions_path = _verify_artifact_digest(run_dir, srt_record.path, srt_record.sha256)

    chapters: list[dict[str, Any]] | None = None
    chapters_record = manifest.artifact_integrity.get("chapters_json")
    if chapters_record is not None:
        chapters_path = _verify_artifact_digest(run_dir, chapters_record.path, chapters_record.sha256)
        try:
            chapters = json.loads(chapters_path.read_text(encoding="utf-8"))["chapters"]
        except (json.JSONDecodeError, KeyError, TypeError):
            chapters = None

    inputs = manifest.inputs or {}
    resolved_title = title or str(inputs.get("title") or inputs.get("topic") or run_id)
    resolved_description = description or str(
        inputs.get("description_override") or inputs.get("description") or ""
    )

    return UploadRequest(
        video_path=video_path,
        title=resolved_title,
        description=resolved_description,
        tags=tags or [],
        privacy_status=privacy_status,
        confirm_public=confirm_public,
        thumbnail_path=thumbnail_path,
        captions_path=captions_path,
        chapters=chapters,
        dry_run=dry_run,
    )
