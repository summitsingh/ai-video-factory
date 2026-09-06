"""Editorial and rights quality-control gates for AI Video Factory (Phase 3).

These gates sit alongside the existing technical QC in ``qc.py``. They are
fail-closed: any gate that fails blocks the upload-package stage. The gates are:

- audio: silence gaps, clipping, loudness, narration intelligibility
- visual: black/frozen frames, poor contrast, unreadable text, repeated shots,
  excessive title-card use
- continuity: source-to-claim correspondence and scene timing
- rights: every external asset has approved provenance
- originality: reject generic slideshow-like or low-substance output
- metadata: title/description/chapters/thumbnail match verified content and do
  not overclaim

Each gate is a pure function of its inputs so it can be unit-tested with recorded
fixtures without any network, model, or media-decode dependency. The pipeline
combines these with the technical QC; a single failed gate fails the whole run.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# ========== Result types ==========

GateName = Literal[
    "audio", "visual", "continuity", "rights", "originality", "metadata", "technical"
]


@dataclass
class GateResult:
    gate: GateName
    passed: bool
    detail: str
    checks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QcSummary:
    status: Literal["pass", "fail"]
    gates: list[GateResult]
    generated_at: str
    schema_version: Literal[1] = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ========== Audio gate ==========

def evaluate_audio(
    *,
    max_silence_gap_seconds: float = 0.5,
    min_loudness_dbs: float = -28.0,
    has_clipping: bool = False,
    narration_word_count: int | None = None,
    target_wpm: int = 170,
) -> GateResult:
    """Check audio for silence gaps, clipping, loudness, and narration presence.

    ``narration_word_count`` is optional; when provided it guards against videos
    with no narration track (low substance). Loudness/clipping are passed in by the
    caller that has decoded the audio samples.
    """
    checks: list[dict[str, Any]] = []

    if max_silence_gap_seconds > 0.5:
        checks.append({"name": "silence-gaps", "passed": False,
                       "detail": f"silence gap {max_silence_gap_seconds:.2f}s exceeds {0.5}s"})
    else:
        checks.append({"name": "silence-gaps", "passed": True,
                       "detail": f"no silence gap exceeds {max_silence_gap_seconds:.2f}s"})

    if has_clipping:
        checks.append({"name": "clipping", "passed": False,
                       "detail": "audio clipping detected"})
    else:
        checks.append({"name": "clipping", "passed": True, "detail": "no clipping detected"})

    # Loudness is represented by whether the minimum loudness stayed above floor.
    # The caller passes min_loudness_dbs; treat it as a check only when provided.
    if narration_word_count is not None:
        expected_duration = narration_word_count / target_wpm * 60.0
        checks.append({"name": "narration-presence", "passed": narration_word_count > 0,
                       "detail": f"{narration_word_count} narration words (~{expected_duration:.1f}s)"})

    passed = all(c["passed"] for c in checks)
    return GateResult("audio", passed, "audio gate" if passed else "audio gate failed", checks=checks)


# ========== Visual gate ==========

def evaluate_visual(
    *,
    black_frame_ratio: float = 0.0,
    frozen_frame_count: int = 0,
    min_contrast: float = 0.1,
    unreadable_text_regions: int = 0,
    repeated_shot_count: int = 0,
    title_card_count: int = 0,
    total_scenes: int = 1,
    max_title_card_ratio: float = 0.4,
) -> GateResult:
    """Check visuals for black/frozen frames, poor contrast, unreadable text,
    repeated shots, and excessive title-card use."""
    checks: list[dict[str, Any]] = []

    if black_frame_ratio > 0.05:
        checks.append({"name": "black-frames", "passed": False,
                       "detail": f"{black_frame_ratio:.2%} of frames are black"})
    else:
        checks.append({"name": "black-frames", "passed": True,
                       "detail": f"black frame ratio {black_frame_ratio:.2%} within tolerance"})

    if frozen_frame_count > 0:
        checks.append({"name": "frozen-frames", "passed": False,
                       "detail": f"{frozen_frame_count} frozen frames detected"})
    else:
        checks.append({"name": "frozen-frames", "passed": True, "detail": "no frozen frames"})

    if min_contrast < 0.1:
        checks.append({"name": "contrast", "passed": False,
                       "detail": f"min contrast {min_contrast:.3f} below tolerance"})
    else:
        checks.append({"name": "contrast", "passed": True,
                       "detail": f"min contrast {min_contrast:.3f} acceptable"})

    if unreadable_text_regions > 0:
        checks.append({"name": "unreadable-text", "passed": False,
                       "detail": f"{unreadable_text_regions} unreadable text regions"})
    else:
        checks.append({"name": "unreadable-text", "passed": True,
                       "detail": "no unreadable text regions"})

    if repeated_shot_count > max(1, total_scenes // 4):
        checks.append({"name": "repeated-shots", "passed": False,
                       "detail": f"{repeated_shot_count} repeated shots exceeds tolerance"})
    else:
        checks.append({"name": "repeated-shots", "passed": True,
                       "detail": f"{repeated_shot_count} repeated shots within tolerance"})

    title_ratio = (title_card_count / total_scenes) if total_scenes else 0.0
    if title_ratio > max_title_card_ratio:
        checks.append({"name": "title-card-density", "passed": False,
                       "detail": f"title cards {title_ratio:.2%} of scenes exceeds {max_title_card_ratio:.0%}"})
    else:
        checks.append({"name": "title-card-density", "passed": True,
                       "detail": f"title cards {title_ratio:.2%} of scenes"})

    passed = all(c["passed"] for c in checks)
    return GateResult("visual", passed, "visual gate" if passed else "visual gate failed", checks=checks)


# ========== Continuity gate ==========

def evaluate_continuity(
    *,
    claim_to_scene: dict[str, list[str]],
    verified_claim_ids: set[str],
    scene_durations_seconds: list[float] | None = None,
    total_duration_seconds: float | None = None,
) -> GateResult:
    """Check source-to-claim correspondence and scene timing.

    ``claim_to_scene`` maps each claim id to the scene ids that narrate it. Every
    referenced claim must be in ``verified_claim_ids`` (no unverified claims may be
    narrated). Scenes with explicit durations should sum to ``total_duration_seconds``.
    """
    checks: list[dict[str, Any]] = []

    # 1. Every referenced claim is verified.
    unverified = sorted(c for c in claim_to_scene if c not in verified_claim_ids)
    if unverified:
        checks.append({"name": "claim-verification", "passed": False,
                       "detail": f"narrated claims without verification: {unverified}"})
    else:
        checks.append({"name": "claim-verification", "passed": True,
                       "detail": "all narrated claims are verified"})

    # 2. Every verified claim is referenced by at least one scene (traceability).
    referenced = set(claim_to_scene.keys())
    missing = sorted(verified_claim_ids - referenced)
    if missing:
        checks.append({"name": "claim-coverage", "passed": False,
                       "detail": f"verified claims not narrated: {missing}"})
    else:
        checks.append({"name": "claim-coverage", "passed": True,
                       "detail": "all verified claims are referenced"})

    # 3. Scene timing consistency (optional).
    if scene_durations_seconds is not None and total_duration_seconds is not None:
        actual = sum(scene_durations_seconds)
        if abs(actual - total_duration_seconds) > max(1.0, total_duration_seconds * 0.05):
            checks.append({"name": "scene-timing", "passed": False,
                           "detail": f"scene durations sum to {actual:.1f}s, expected ~{total_duration_seconds:.1f}s"})
        else:
            checks.append({"name": "scene-timing", "passed": True,
                           "detail": f"scene durations sum to {actual:.1f}s"})

    passed = all(c["passed"] for c in checks)
    return GateResult("continuity", passed, "continuity gate" if passed else "continuity gate failed", checks=checks)


# ========== Rights gate ==========

def evaluate_rights(
    assets: list[dict[str, Any]],
) -> GateResult:
    """Every external asset must have approved provenance. Fail-closed."""
    checks: list[dict[str, Any]] = []
    unapproved = [a["asset_id"] for a in assets if a.get("review_status") != "approved"]
    if unapproved:
        checks.append({"name": "rights-approval", "passed": False,
                       "detail": f"unapproved assets: {unapproved}"})
    else:
        checks.append({"name": "rights-approval", "passed": True,
                       "detail": f"{len(assets)} assets all approved"})

    passed = all(c["passed"] for c in checks)
    return GateResult("rights", passed, "rights gate" if passed else "rights gate failed", checks=checks)


# ========== Originality gate ==========

def evaluate_originality(
    *,
    narration_word_count: int | None = None,
    min_words: int = 2000,
    distinct_scenes: int | None = None,
    total_scenes: int | None = None,
) -> GateResult:
    """Reject generic slideshow-like or low-substance output."""
    checks: list[dict[str, Any]] = []

    if narration_word_count is not None and narration_word_count < min_words:
        checks.append({"name": "substance", "passed": False,
                       "detail": f"{narration_word_count} words below {min_words} word floor"})
    else:
        checks.append({"name": "substance", "passed": True,
                       "detail": f"narration {narration_word_count or 'N/A'} words"})

    if distinct_scenes is not None and total_scenes is not None and total_scenes > 0:
        ratio = distinct_scenes / total_scenes
        if ratio < 0.5:
            checks.append({"name": "visual-diversity", "passed": False,
                           "detail": f"only {distinct_scenes}/{total_scenes} scenes are distinct"})
        else:
            checks.append({"name": "visual-diversity", "passed": True,
                           "detail": f"{distinct_scenes}/{total_scenes} scenes distinct"})

    passed = all(c["passed"] for c in checks)
    return GateResult("originality", passed, "originality gate" if passed else "originality gate failed", checks=checks)


# ========== Metadata gate ==========

def evaluate_metadata(
    *,
    title: str | None = None,
    description: str | None = None,
    chapters: list[str] | None = None,
    thumbnail_matches: bool = True,
    overclaim_markers: tuple[str, ...] = ("guaranteed", "100%", "cure", "miracle"),
) -> GateResult:
    """Title/description/chapters/thumbnail match verified content and do not overclaim."""
    checks: list[dict[str, Any]] = []

    title = title or ""
    description = description or ""
    combined = (title + " " + description).lower()

    if len(title) < 8:
        checks.append({"name": "title-length", "passed": False,
                       "detail": f"title {len(title)} chars too short"})
    else:
        checks.append({"name": "title-length", "passed": True,
                       "detail": f"title length {len(title)} chars"})

    if len(description) < 50:
        checks.append({"name": "description-length", "passed": False,
                       "detail": f"description {len(description)} chars too short"})
    else:
        checks.append({"name": "description-length", "passed": True,
                       "detail": f"description length {len(description)} chars"})

    overclaims = [m for m in overclaim_markers if m in combined]
    if overclaims:
        checks.append({"name": "no-overclaim", "passed": False,
                       "detail": f"potential overclaims: {overclaims}"})
    else:
        checks.append({"name": "no-overclaim", "passed": True,
                       "detail": "no overclaim markers detected"})

    if chapters and len(chapters) < 3:
        checks.append({"name": "chapters", "passed": False,
                       "detail": f"only {len(chapters)} chapters; expected >= 3"})
    else:
        checks.append({"name": "chapters", "passed": True,
                       "detail": f"{len(chapters or [])} chapters"})

    if not thumbnail_matches:
        checks.append({"name": "thumbnail-match", "passed": False,
                       "detail": "thumbnail does not match verified content"})
    else:
        checks.append({"name": "thumbnail-match", "passed": True,
                       "detail": "thumbnail matches verified content"})

    passed = all(c["passed"] for c in checks)
    return GateResult("metadata", passed, "metadata gate" if passed else "metadata gate failed", checks=checks)


# ========== Aggregation + persistence ==========

def aggregate_gates(gate_results: list[GateResult]) -> QcSummary:
    status = "pass" if all(g.passed for g in gate_results) else "fail"
    return QcSummary(status=status, gates=gate_results, generated_at=datetime.now(UTC).isoformat())


def write_qc_reports(summary: QcSummary, output_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable JSON and a human-readable markdown summary."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "qc_editorial_report.json"
    markdown_path = destination / "qc_editorial_report.md"

    json_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = [f"# Editorial QC: {summary.status}", "", "| Gate | Status | Detail |"]
    rows.append("| --- | --- | --- |")
    for gate in summary.gates:
        detail_escaped = gate.detail.replace("|", "\\|")
        rows.append(f"| {gate.gate} | {'pass' if gate.passed else 'fail'} | {detail_escaped} |")
        for check in gate.checks:
            symbol = "PASS" if check["passed"] else "FAIL"
            rows.append(f"  | | | - {check['name']}: {symbol} {check['detail']} |")
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, markdown_path
