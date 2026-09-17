"""Tests for the YouTube Data API v3 uploader.

The Google client libraries are optional, so these tests stub the
``googleapiclient`` modules and the API service instead of importing them.
"""

import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ai_video_factory import cli, youtube
from ai_video_factory.youtube import (
    DEFAULT_PRIVACY,
    QUOTA_COSTS,
    QuotaGuard,
    UploadRequest,
    YouTubeAuthError,
    YouTubeError,
    YouTubeQuotaError,
    YouTubeUploadError,
    _is_retriable,
    _planned_cost,
    _upload_resumable,
    build_chapter_lines,
    build_description,
    build_video_body,
    find_run_manifest,
    format_timestamp,
    request_from_run,
    save_credentials,
    token_path,
    upload_package,
)


# ---------------------------------------------------------------------------
# Google API stubs
# ---------------------------------------------------------------------------


class _StubHttpError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.resp = SimpleNamespace(status=status)


class _StubMediaFileUpload:
    def __init__(self, filename: str, mimetype=None, resumable=False, chunksize=None) -> None:
        self.filename = filename


@pytest.fixture()
def stub_google(monkeypatch):
    """Install stub googleapiclient modules and disable the dependency guard."""
    http_module = types.ModuleType("googleapiclient.http")
    http_module.MediaFileUpload = _StubMediaFileUpload
    package = types.ModuleType("googleapiclient")
    package.http = http_module
    monkeypatch.setitem(sys.modules, "googleapiclient", package)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http_module)
    monkeypatch.setattr(youtube, "_require_google_libs", lambda: None)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return http_module


class _StubInsertRequest:
    def __init__(self, script) -> None:
        self._script = list(script)
        self.calls = 0

    def next_chunk(self):
        self.calls += 1
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(item, BaseException):
            raise item
        return item


class _StubExecute:
    def __init__(self, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def execute(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class _StubService:
    """Minimal fake of the subset of the YouTube API the uploader uses."""

    def __init__(self, insert_request, thumbnail=None, captions=None) -> None:
        self._insert_request = insert_request
        self._thumbnail = thumbnail or _StubExecute({"items": [{}]})
        self._captions = captions or _StubExecute({"id": "cap1"})
        self.insert_bodies = []
        self.thumbnail_video_ids = []
        self.caption_video_ids = []

    def videos(self):
        return self

    def thumbnails(self):
        return self

    def captions(self):
        return self

    def insert(self, part=None, body=None, media_body=None):
        self.insert_bodies.append(body)
        if part == "snippet,status":
            return self._insert_request
        self.caption_video_ids.append(body["snippet"]["videoId"])
        return self._captions

    def set(self, videoId=None, media_body=None):
        self.thumbnail_video_ids.append(videoId)
        return self._thumbnail


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3599) == "59:59"
    assert format_timestamp(3661) == "1:01:01"


def test_build_chapter_lines_valid():
    chapters = [
        {"title": "Intro", "start_seconds": 0},
        {"title": "Middle", "start_seconds": 30},
        {"title": "End", "start_seconds": 90},
    ]
    assert build_chapter_lines(chapters) == ["00:00 Intro", "00:30 Middle", "01:30 End"]


def test_build_chapter_lines_too_few():
    chapters = [
        {"title": "Intro", "start_seconds": 0},
        {"title": "End", "start_seconds": 90},
    ]
    assert build_chapter_lines(chapters) == []


def test_build_chapter_lines_first_not_at_zero():
    chapters = [
        {"title": "Intro", "start_seconds": 12},
        {"title": "Middle", "start_seconds": 30},
        {"title": "End", "start_seconds": 90},
    ]
    assert build_chapter_lines(chapters) == []


def test_build_chapter_lines_sorted_and_cleaned():
    chapters = [
        {"title": "  End  ", "start_seconds": 90},
        {"title": "", "start_seconds": 45},
        {"title": "Intro", "start_seconds": 0},
        {"title": "Middle", "start_seconds": 30},
    ]
    assert build_chapter_lines(chapters) == ["00:00 Intro", "00:30 Middle", "01:30 End"]


def test_build_description_appends_chapters():
    chapters = [
        {"title": "Intro", "start_seconds": 0},
        {"title": "Middle", "start_seconds": 30},
        {"title": "End", "start_seconds": 90},
    ]
    description = build_description("A great video", chapters)
    assert description.startswith("A great video\n\nChapters:\n")
    assert "00:30 Middle" in description


def test_build_description_without_chapters_unchanged():
    assert build_description("Hello", None) == "Hello"
    assert build_description("Hello", [{"title": "Only", "start_seconds": 0}]) == "Hello"


def test_build_video_body_defaults():
    body = build_video_body(title="My video", description="desc")
    assert body["snippet"]["title"] == "My video"
    assert body["status"]["privacyStatus"] == DEFAULT_PRIVACY == "unlisted"
    assert body["status"]["madeForKids"] is False
    assert "tags" not in body["snippet"]


def test_build_video_body_tags_cleaned():
    body = build_video_body(title="T", description="", tags=[" a ", "", "b"])
    assert body["snippet"]["tags"] == ["a", "b"]


def test_build_video_body_rejects_bad_input():
    with pytest.raises(YouTubeError):
        build_video_body(title="   ", description="")
    with pytest.raises(YouTubeError):
        build_video_body(title="x" * 101, description="")
    with pytest.raises(YouTubeError):
        build_video_body(title="T", description="", privacy_status="everyone")


# ---------------------------------------------------------------------------
# Quota guard
# ---------------------------------------------------------------------------


def test_quota_guard_fresh_state(tmp_path):
    guard = QuotaGuard(tmp_path, daily_budget=1000)
    assert guard.units_used_today() == 0
    assert guard.units_remaining() == 1000


def test_quota_guard_charge_accumulates(tmp_path):
    guard = QuotaGuard(tmp_path, daily_budget=1000)
    guard.charge(100)
    guard.charge(50)
    assert guard.units_used_today() == 150
    assert guard.units_remaining() == 850


def test_quota_guard_check_rejects_over_budget(tmp_path):
    guard = QuotaGuard(tmp_path, daily_budget=100)
    with pytest.raises(YouTubeQuotaError):
        guard.check(101, "videos.insert")
    guard.check(100, "videos.insert")


def test_quota_guard_resets_each_day(tmp_path):
    guard = QuotaGuard(tmp_path, daily_budget=1000)
    guard.charge(900)
    state_path = tmp_path / "quota.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["date"] = "2000-01-01"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert guard.units_used_today() == 0


def test_quota_guard_survives_corrupt_state(tmp_path):
    (tmp_path / "quota.json").write_text("not json", encoding="utf-8")
    guard = QuotaGuard(tmp_path, daily_budget=1000)
    assert guard.units_used_today() == 0


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_planned_cost():
    video = UploadRequest(video_path=Path("v.mp4"), title="T")
    assert _planned_cost(video) == QUOTA_COSTS["videos.insert"]
    full = UploadRequest(
        video_path=Path("v.mp4"),
        title="T",
        thumbnail_path=Path("t.png"),
        captions_path=Path("c.srt"),
    )
    assert _planned_cost(full) == 1600 + 50 + 50


def test_validate_missing_video(tmp_path):
    request = UploadRequest(video_path=tmp_path / "nope.mp4", title="T")
    with pytest.raises(YouTubeError, match="video file not found"):
        request.validate()


def test_validate_empty_video(tmp_path):
    video = _write(tmp_path / "empty.mp4", b"")
    with pytest.raises(YouTubeError, match="empty"):
        UploadRequest(video_path=video, title="T").validate()


def test_validate_thumbnail_rules(tmp_path):
    video = _write(tmp_path / "v.mp4", b"data")
    bad_suffix = _write(tmp_path / "t.gif", b"data")
    with pytest.raises(YouTubeError, match="PNG or JPEG"):
        UploadRequest(video_path=video, title="T", thumbnail_path=bad_suffix).validate()
    big = _write(tmp_path / "big.png", b"x" * (2 * 1024 * 1024 + 1))
    with pytest.raises(YouTubeError, match="2 MB"):
        UploadRequest(video_path=video, title="T", thumbnail_path=big).validate()
    good = _write(tmp_path / "t.png", b"data")
    UploadRequest(video_path=video, title="T", thumbnail_path=good).validate()


def test_validate_public_needs_confirmation(tmp_path):
    video = _write(tmp_path / "v.mp4", b"data")
    request = UploadRequest(video_path=video, title="T", privacy_status="public")
    with pytest.raises(YouTubeError, match="explicit confirmation"):
        request.validate()
    request.confirm_public = True
    request.validate()


# ---------------------------------------------------------------------------
# Credential storage
# ---------------------------------------------------------------------------


def test_save_credentials_owner_only(tmp_path):
    creds = SimpleNamespace(
        token="access",
        refresh_token="refresh",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="cid",
        client_secret="csecret",
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    path = save_credentials(creds, tmp_path)
    assert path == token_path(tmp_path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["refresh_token"] == "refresh"


def test_save_credentials_requires_refresh_token(tmp_path):
    creds = SimpleNamespace(
        token="access",
        refresh_token=None,
        token_uri="u",
        client_id="c",
        client_secret="s",
        scopes=[],
    )
    with pytest.raises(YouTubeAuthError):
        save_credentials(creds, tmp_path)


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


def test_is_retriable():
    assert _is_retriable(_StubHttpError(500))
    assert _is_retriable(_StubHttpError(503))
    assert _is_retriable(OSError("boom"))
    assert not _is_retriable(_StubHttpError(400))
    assert not _is_retriable(_StubHttpError(403))
    assert not _is_retriable(ValueError("nope"))


def test_upload_resumable_success(stub_google, tmp_path):
    video = _write(tmp_path / "v.mp4", b"data")
    service = _StubService(_StubInsertRequest([(None, {"id": "vid123"})]))
    video_id = _upload_resumable(service, {"snippet": {}}, video)
    assert video_id == "vid123"


def test_upload_resumable_retries_then_succeeds(stub_google, tmp_path):
    video = _write(tmp_path / "v.mp4", b"data")
    script = [
        _StubHttpError(500),
        _StubHttpError(503),
        (None, {"id": "vid123"}),
    ]
    insert_request = _StubInsertRequest(script)
    service = _StubService(insert_request)
    assert _upload_resumable(service, {"snippet": {}}, video) == "vid123"
    assert insert_request.calls == 3


def test_upload_resumable_gives_up(stub_google, tmp_path):
    video = _write(tmp_path / "v.mp4", b"data")
    insert_request = _StubInsertRequest([_StubHttpError(500)])
    service = _StubService(insert_request)
    with pytest.raises(YouTubeUploadError):
        _upload_resumable(service, {"snippet": {}}, video)
    assert insert_request.calls == 7  # initial attempt + 6 retries


def test_upload_resumable_non_retriable_fails_fast(stub_google, tmp_path):
    video = _write(tmp_path / "v.mp4", b"data")
    insert_request = _StubInsertRequest([_StubHttpError(403)])
    service = _StubService(insert_request)
    with pytest.raises(YouTubeUploadError):
        _upload_resumable(service, {"snippet": {}}, video)
    assert insert_request.calls == 1


# ---------------------------------------------------------------------------
# upload_package orchestration
# ---------------------------------------------------------------------------


def _full_request(tmp_path: Path, **overrides) -> UploadRequest:
    video = _write(tmp_path / "v.mp4", b"video-bytes")
    thumb = _write(tmp_path / "t.png", b"thumb-bytes")
    captions = _write(tmp_path / "c.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHi\n")
    kwargs = {
        "video_path": video,
        "title": "Test title",
        "description": "Test description",
        "tags": ["space"],
        "thumbnail_path": thumb,
        "captions_path": captions,
        "chapters": [
            {"title": "Intro", "start_seconds": 0},
            {"title": "Middle", "start_seconds": 30},
            {"title": "End", "start_seconds": 90},
        ],
    }
    kwargs.update(overrides)
    return UploadRequest(**kwargs)


def test_upload_package_full(stub_google, tmp_path):
    request = _full_request(tmp_path)
    service = _StubService(_StubInsertRequest([(None, {"id": "vid123"})]))
    quota = QuotaGuard(tmp_path / "yt", daily_budget=10_000)
    result = upload_package(request, service, quota)
    assert result.video_id == "vid123"
    assert result.url == "https://www.youtube.com/watch?v=vid123"
    assert result.thumbnail_uploaded is True
    assert result.captions_uploaded is True
    assert result.quota_units_used == 1700
    assert service.thumbnail_video_ids == ["vid123"]
    assert service.caption_video_ids == ["vid123"]
    body = service.insert_bodies[0]
    assert body["status"]["privacyStatus"] == "unlisted"
    assert "00:30 Middle" in body["snippet"]["description"]


def test_upload_package_quota_exceeded(stub_google, tmp_path):
    request = _full_request(tmp_path)
    service = _StubService(_StubInsertRequest([(None, {"id": "vid123"})]))
    quota = QuotaGuard(tmp_path / "yt", daily_budget=100)
    with pytest.raises(YouTubeQuotaError):
        upload_package(request, service, quota)
    assert quota.units_used_today() == 0


def test_upload_package_dry_run(tmp_path):
    request = _full_request(tmp_path, dry_run=True)
    quota = QuotaGuard(tmp_path / "yt", daily_budget=10_000)
    result = upload_package(request, service=None, quota=quota)
    assert result.dry_run is True
    assert result.video_id is None
    assert result.quota_units_used == 0
    assert quota.units_used_today() == 0


def test_upload_package_thumbnail_failure(stub_google, tmp_path):
    request = _full_request(tmp_path)
    service = _StubService(
        _StubInsertRequest([(None, {"id": "vid123"})]),
        thumbnail=_StubExecute(error=_StubHttpError(400)),
    )
    quota = QuotaGuard(tmp_path / "yt", daily_budget=10_000)
    with pytest.raises(YouTubeUploadError, match="thumbnail"):
        upload_package(request, service, quota)
    # The video insert was still charged; the failed thumbnail was not.
    assert quota.units_used_today() == QUOTA_COSTS["videos.insert"]


# ---------------------------------------------------------------------------
# Run resolution
# ---------------------------------------------------------------------------


def _make_run(tmp_path: Path, run_id: str, status: str = "completed", tamper: bool = False):
    run_dir = tmp_path / "data" / "projects" / "my-topic" / "runs" / run_id
    video = _write(run_dir / "master.mp4", b"video-bytes")
    thumb = _write(run_dir / "thumb1.png", b"thumb-bytes")
    srt = _write(run_dir / "subs.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHi\n")
    chapters_path = _write(
        run_dir / "chapters.json",
        json.dumps(
            {"chapters": [
                {"title": "Intro", "start_seconds": 0},
                {"title": "Middle", "start_seconds": 30},
                {"title": "End", "start_seconds": 90},
            ]}
        ).encode(),
    )
    if tamper:
        video.write_bytes(b"different-bytes")

    def digest(path: Path) -> dict:
        return {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    # Digests are recorded before tampering, so tamper=True simulates drift.
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "video",
        "input_fingerprint": "abc",
        "status": status,
        "resumed": False,
        "created_at": "2026-09-18T00:00:00+00:00",
        "updated_at": "2026-09-18T00:00:00+00:00",
        "inputs": {"topic": "Water on Mars", "description": "A documentary"},
        "artifacts": {
            "master": str(video),
            "thumbnail_1": str(thumb),
            "srt": str(srt),
            "chapters_json": str(chapters_path),
        },
        "artifact_integrity": {
            "master": digest(run_dir / "master.mp4") if not tamper else {
                "path": "master.mp4",
                "sha256": hashlib.sha256(b"video-bytes").hexdigest(),
                "size_bytes": len(b"video-bytes"),
            },
            "thumbnail_1": digest(thumb),
            "srt": digest(srt),
            "chapters_json": digest(chapters_path),
        },
        "error": None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_find_run_manifest(tmp_path):
    run_id = "a" * 32
    run_dir = _make_run(tmp_path, run_id)
    manifest, found_dir = find_run_manifest(tmp_path / "data", run_id)
    assert found_dir == run_dir
    assert manifest.run_id == run_id


def test_find_run_manifest_bad_id(tmp_path):
    with pytest.raises(YouTubeError, match="32 lowercase hexadecimal"):
        find_run_manifest(tmp_path / "data", "nope")


def test_find_run_manifest_missing(tmp_path):
    with pytest.raises(YouTubeError, match="was not found"):
        find_run_manifest(tmp_path / "data", "b" * 32)


def test_request_from_run(tmp_path):
    run_id = "c" * 32
    _make_run(tmp_path, run_id)
    request = request_from_run(tmp_path / "data", run_id)
    assert request.video_path.name == "master.mp4"
    assert request.thumbnail_path is not None and request.thumbnail_path.name == "thumb1.png"
    assert request.captions_path is not None and request.captions_path.name == "subs.srt"
    assert request.title == "Water on Mars"
    assert request.description == "A documentary"
    assert request.chapters is not None and len(request.chapters) == 3
    assert request.privacy_status == "unlisted"
    request.validate()


def test_request_from_run_title_override(tmp_path):
    run_id = "d" * 32
    _make_run(tmp_path, run_id)
    request = request_from_run(tmp_path / "data", run_id, title="Custom title")
    assert request.title == "Custom title"


def test_request_from_run_rejects_non_completed(tmp_path):
    run_id = "e" * 32
    _make_run(tmp_path, run_id, status="failed")
    with pytest.raises(YouTubeError, match="only completed runs"):
        request_from_run(tmp_path / "data", run_id)


def test_request_from_run_rejects_tampered_artifact(tmp_path):
    run_id = "f" * 32
    _make_run(tmp_path, run_id, tamper=True)
    with pytest.raises(YouTubeError, match="digest changed"):
        request_from_run(tmp_path / "data", run_id)


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


def test_cli_youtube_status_unauthorized(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli.app, ["youtube", "status", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Authorized: no" in result.output


def test_cli_youtube_upload_missing_video(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["youtube", "upload", "--video", str(tmp_path / "nope.mp4"),
         "--title", "T", "--dry-run", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert "video file not found" in result.output


def test_cli_youtube_upload_dry_run_from_run(tmp_path):
    run_id = "1" * 32
    _make_run(tmp_path, run_id)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["youtube", "upload", "--run-id", run_id, "--dry-run",
         "--data-dir", str(tmp_path / "yt"),
         "--project-data", str(tmp_path / "data"), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["title"] == "Water on Mars"
    assert payload["privacy_status"] == "unlisted"


def test_cli_youtube_upload_public_needs_confirm(tmp_path):
    run_id = "2" * 32
    _make_run(tmp_path, run_id)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["youtube", "upload", "--run-id", run_id, "--dry-run",
         "--privacy", "public",
         "--data-dir", str(tmp_path / "yt"),
         "--project-data", str(tmp_path / "data")],
    )
    assert result.exit_code == 2
    assert "explicit confirmation" in result.output


# ---------------------------------------------------------------------------
# Privacy updates (review publish step)
# ---------------------------------------------------------------------------


class _StubUpdateService:
    def __init__(self, response):
        self._response = response
        self.update_calls = []

    def videos(self):
        return self

    def update(self, part=None, body=None):
        self.update_calls.append({"part": part, "body": body})
        return _StubExecute(self._response)


def test_set_video_privacy_public(stub_google, tmp_path):
    from ai_video_factory.youtube import set_video_privacy

    service = _StubUpdateService({"status": {"privacyStatus": "public"}})
    quota = QuotaGuard(tmp_path)
    result = set_video_privacy(service, "vid1", "public", quota)
    assert result == "public"
    call = service.update_calls[0]
    assert call["body"]["id"] == "vid1"
    assert call["body"]["status"]["privacyStatus"] == "public"
    assert quota.units_used_today() == QUOTA_COSTS["videos.update"]


def test_set_video_privacy_rejects_bad_status(stub_google, tmp_path):
    from ai_video_factory.youtube import set_video_privacy

    service = _StubUpdateService({})
    with pytest.raises(YouTubeError, match="unlisted, private, or public"):
        set_video_privacy(service, "vid1", "everyone", QuotaGuard(tmp_path))
    assert service.update_calls == []


def test_set_video_privacy_quota_guarded(stub_google, tmp_path):
    from ai_video_factory.youtube import set_video_privacy

    service = _StubUpdateService({})
    quota = QuotaGuard(tmp_path, daily_budget=10)
    with pytest.raises(YouTubeQuotaError):
        set_video_privacy(service, "vid1", "public", quota)
    assert service.update_calls == []
