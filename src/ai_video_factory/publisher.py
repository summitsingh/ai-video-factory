"""Upload-ready package assembly for AI Video Factory (Phase 3).

This module builds a private-upload-ready package but NEVER uploads or publishes.
Every artifact is assembled into a directory with an ``approval.json`` initialized
as ``pending``. A human must explicitly approve before any upload/scheduling action
can occur, and even then the actual YouTube API publisher is disabled by default
and unconfigured until later phases wire up OAuth credentials stored outside Git.

Package contents:
    - master.mp4
    - selected thumbnail + alternates
    - title candidates
    - description with sources/attributions where needed
    - chapters (JSON)
    - tags / category / draft metadata
    - captions (SRT + WebVTT)
    - rights manifest
    - research brief
    - QC reports (technical + editorial)
    - approval.json (initialized pending)

The YouTube publisher interface is a thin, disabled stub. It cannot be enabled
without explicit configuration and credentials that live outside Git.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PublisherError(RuntimeError):
    """Raised when package assembly or an upload attempt fails."""


@dataclass
class ApprovalState:
    status: str = "pending"  # pending | approved | rejected
    approver: str | None = None
    approved_at: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PublisherPackage:
    root: str
    master_mp4: str | None
    thumbnails: list[str]
    title_candidates: list[str]
    description: str
    chapters_json: str | None
    tags: list[str]
    category_id: str | None
    captions_srt: str | None
    captions_webvtt: str | None
    rights_manifest: str | None
    research_brief: str | None
    qc_reports: list[str]
    approval_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ========== Package assembly ==========

def build_package(
    output_dir: Path,
    *,
    master_mp4: Path | None = None,
    thumbnails: list[Path] | None = None,
    title_candidates: list[str] | None = None,
    description: str = "",
    chapters_json: Path | None = None,
    tags: list[str] | None = None,
    category_id: str | None = None,
    captions_srt: Path | None = None,
    captions_webvtt: Path | None = None,
    rights_manifest: Path | None = None,
    research_brief: Path | None = None,
    qc_reports: list[Path] | None = None,
) -> PublisherPackage:
    """Assemble the upload-ready package directory.

    Nothing is uploaded. The function copies referenced artifacts into the package
    root (so the package is self-contained), writes a description file, and creates
    an ``approval.json`` initialized as pending. Missing optional inputs are simply
    omitted from the package but recorded so downstream stages know what's absent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path | None, dest_name: str) -> str | None:
        if src is None or not Path(src).is_file():
            return None
        dest = output_dir / dest_name
        try:
            dest.write_bytes(Path(src).read_bytes())
        except OSError as error:
            raise PublisherError(f"failed to copy {src} into package: {error}") from error
        return str(dest)

    master_path = _copy(master_mp4, "master.mp4")
    thumb_paths = [p for p in [_copy(t, f"thumbnail-{i}.jpg") for i, t in enumerate(thumbnails or [])] if p is not None]
    rights_path = _copy(rights_manifest, "rights_manifest.json")
    brief_path = _copy(research_brief, "research_brief.md")
    srt_path = _copy(captions_srt, "captions.srt")
    vtt_path = _copy(captions_webvtt, "captions.webvtt")
    chapters_path = _copy(chapters_json, "chapters.json")

    qc_paths: list[str] = []
    for i, report in enumerate(qc_reports or []):
        if Path(report).is_file():
            dest = output_dir / f"qc_report_{i}.json"
            try:
                dest.write_bytes(Path(report).read_bytes())
            except OSError as error:
                raise PublisherError(f"failed to copy {report} into package: {error}") from error
            qc_paths.append(str(dest))

    title_candidates = title_candidates or ["Untitled"]
    description = description or ""

    # Write the machine-readable package manifest.
    package = PublisherPackage(
        root=str(output_dir),
        master_mp4=master_path,
        thumbnails=thumb_paths,
        title_candidates=title_candidates,
        description=description,
        chapters_json=chapters_path,
        tags=list(tags or []),
        category_id=category_id,
        captions_srt=srt_path,
        captions_webvtt=vtt_path,
        rights_manifest=rights_path,
        research_brief=brief_path,
        qc_reports=qc_paths,
        approval_path=str(output_dir / "approval.json"),
    )

    _write_package_manifest(package)
    _write_approval(package.approval_path, ApprovalState())
    return package


def _write_package_manifest(package: PublisherPackage) -> None:
    path = Path(package.root) / "package.json"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "package": package.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_approval(path: str, state: ApprovalState) -> None:
    payload = {
        "status": state.status,
        "approver": state.approver,
        "approved_at": state.approved_at,
        "notes": state.notes,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_approval(package_root: Path) -> ApprovalState:
    path = Path(package_root) / "approval.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return ApprovalState(
        status=data.get("status", "pending"),
        approver=data.get("approver"),
        approved_at=data.get("approved_at"),
        notes=data.get("notes"),
    )


def approve_package(package_root: Path, *, approver: str, notes: str | None = None) -> ApprovalState:
    """Record explicit human approval. Returns the updated state."""
    path = Path(package_root) / "approval.json"
    if not path.is_file():
        raise PublisherError(f"no approval.json found in {package_root}")
    state = load_approval(package_root)
    state.status = "approved"
    state.approver = approver
    state.approved_at = datetime.now(UTC).isoformat()
    state.notes = notes
    _write_approval(str(path), state)
    return state


def reject_package(package_root: Path, *, approver: str, notes: str | None = None) -> ApprovalState:
    """Record explicit human rejection."""
    path = Path(package_root) / "approval.json"
    if not path.is_file():
        raise PublisherError(f"no approval.json found in {package_root}")
    state = load_approval(package_root)
    state.status = "rejected"
    state.approver = approver
    state.notes = notes
    _write_approval(str(path), state)
    return state


def is_approved(package_root: Path) -> bool:
    """True only when approval.json status is 'approved'."""
    return load_approval(package_root).status == "approved"


# ========== YouTube publisher interface (DISABLED / UNCONFIGURED) ==========

class YouTubePublisherDisabled(PublisherError):
    """Raised whenever an upload/publish action is attempted while disabled."""


@dataclass
class YouTubeConfig:
    enabled: bool = False
    client_id: str | None = None
    client_secret: str | None = None
    refresh_token: str | None = None
    default_privacy_status: str = "private"

    def is_configured(self) -> bool:
        return (
            self.enabled
            and bool(self.client_id)
            and bool(self.client_secret)
            and bool(self.refresh_token)
        )


class YouTubePublisher:
    """Stub publisher interface for a later phase.

    Disabled by default. Cannot upload, schedule, or publish anything until it is
    explicitly enabled AND configured with OAuth credentials that live outside Git.
    When eventually wired up, it must:
      - upload private first (default_privacy_status = "private")
      - use resumable transfer for large master files
      - require an approved approval.json before scheduling/publishing
    """

    def __init__(self, config: YouTubeConfig | None = None) -> None:
        self.config = config or YouTubeConfig()

    def can_publish(self) -> bool:
        return self.config.is_configured()

    def upload(self, package_root: Path) -> str:
        if not self.config.enabled:
            raise YouTubePublisherDisabled(
                "YouTube publisher is disabled; cannot upload. Enable only in a later phase with OAuth credentials stored outside Git."
            )
        if not self.config.is_configured():
            raise YouTubePublisherDisabled("YouTube publisher is unconfigured (missing client_id/secret/refresh_token).")
        if not is_approved(package_root):
            raise PublisherError(
                f"upload blocked: {package_root} has not been approved. Require explicit human approval before any publish action."
            )
        # Actual upload logic belongs in a later phase; the gate above enforces the
        # safety model regardless of implementation.
        raise YouTubePublisherDisabled("Upload implementation pending in a later phase.")

    def schedule_publish(self, package_root: Path, scheduled_at: str) -> str:
        if not self.config.is_configured():
            raise YouTubePublisherDisabled("YouTube publisher is unconfigured; cannot schedule.")
        if not is_approved(package_root):
            raise PublisherError(
                f"scheduling blocked: {package_root} has not been approved. Require explicit human approval before any publish action."
            )
        raise YouTubePublisherDisabled("Scheduling implementation pending in a later phase.")
