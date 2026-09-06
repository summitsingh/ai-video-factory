"""Tests for the upload-ready package builder and disabled YouTube publisher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_video_factory.publisher import (
    ApprovalState,
    PublisherError,
    YouTubeConfig,
    YouTubePublisher,
    YouTubePublisherDisabled,
    approve_package,
    build_package,
    is_approved,
    load_approval,
    reject_package,
)


def _touch(path: Path, content: str = "x") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_build_package_assembles_and_initializes_pending(tmp_path: Path) -> None:
    master = _touch(tmp_path / "src.mp4", "video-bytes")
    thumb = _touch(tmp_path / "thumb.jpg", "img")
    rights = _touch(tmp_path / "rights.json", '{"assets": []}')

    package = build_package(
        tmp_path / "package",
        master_mp4=master,
        thumbnails=[thumb],
        title_candidates=["Title A", "Title B"],
        description="A documentary about space.",
        tags=["space", "documentary"],
        category_id="28",
        rights_manifest=rights,
    )

    pkg_root = Path(package.root)
    assert (pkg_root / "master.mp4").is_file()
    assert (pkg_root / "thumbnail-0.jpg").is_file()
    assert (pkg_root / "rights_manifest.json").is_file()
    manifest = json.loads((pkg_root / "package.json").read_text())
    assert len(manifest["package"]["title_candidates"]) == 2

    # Approval initialized as pending.
    approval = load_approval(pkg_root)
    assert approval.status == "pending"
    assert is_approved(pkg_root) is False


def test_build_package_handles_missing_optional_inputs(tmp_path: Path) -> None:
    package = build_package(tmp_path / "pkg", description="minimal")
    pkg_root = Path(package.root)
    assert package.master_mp4 is None
    assert package.rights_manifest is None
    # Still writes approval.json as pending.
    assert (pkg_root / "approval.json").is_file()


def test_approve_and_reject_package(tmp_path: Path) -> None:
    build_package(tmp_path / "pkg", description="x")
    pkg_root = tmp_path / "pkg"

    state = approve_package(pkg_root, approver="human-editor", notes="looks good")
    assert state.status == "approved"
    assert is_approved(pkg_root) is True

    # Re-approval after approval records the new approver.
    reject_package(pkg_root, approver="reviewer2", notes="needs work")
    assert load_approval(pkg_root).status == "rejected"
    assert is_approved(pkg_root) is False


def test_approve_missing_approval_raises(tmp_path: Path) -> None:
    with pytest.raises(PublisherError):
        approve_package(tmp_path / "nonexistent", approver="x")


# ========== YouTube publisher (disabled by default) ==========

def test_publisher_disabled_by_default() -> None:
    publisher = YouTubePublisher()
    assert publisher.can_publish() is False
    with pytest.raises(YouTubePublisherDisabled):
        publisher.upload(Path("/tmp/whatever"))


def test_publisher_rejects_upload_without_approval(tmp_path: Path) -> None:
    # Even when enabled+configured, an unapproved package cannot be uploaded.
    config = YouTubeConfig(enabled=True, client_id="id", client_secret="secret", refresh_token="tok")
    publisher = YouTubePublisher(config)
    build_package(tmp_path / "pkg", description="x")
    with pytest.raises(PublisherError):
        publisher.upload(tmp_path / "pkg")


def test_publisher_rejects_scheduling_without_approval(tmp_path: Path) -> None:
    config = YouTubeConfig(enabled=True, client_id="id", client_secret="secret", refresh_token="tok")
    publisher = YouTubePublisher(config)
    build_package(tmp_path / "pkg", description="x")
    with pytest.raises(PublisherError):
        publisher.schedule_publish(tmp_path / "pkg", scheduled_at="2026-09-07T00:00:00+00:00")


def test_publisher_upload_blocked_when_approved_but_not_configured(tmp_path: Path) -> None:
    # Approved package but publisher not configured -> still blocked.
    config = YouTubeConfig(enabled=True, client_id="id", client_secret="secret", refresh_token=None)
    publisher = YouTubePublisher(config)
    build_package(tmp_path / "pkg", description="x")
    approve_package(tmp_path / "pkg", approver="human")
    with pytest.raises(YouTubePublisherDisabled):
        publisher.upload(tmp_path / "pkg")


def test_approval_state_serialization() -> None:
    state = ApprovalState(status="pending")
    assert state.to_dict()["status"] == "pending"
