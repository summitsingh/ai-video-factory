"""Tests for editorial and rights quality-control gates (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_video_factory.qc_editorial import (
    evaluate_audio,
    evaluate_continuity,
    evaluate_metadata,
    evaluate_originality,
    evaluate_rights,
    evaluate_visual,
    aggregate_gates,
    write_qc_reports,
)


# ========== Audio gate ==========

def test_audio_gate_passes_when_clean() -> None:
    result = evaluate_audio(max_silence_gap_seconds=0.3, has_clipping=False,
                            narration_word_count=2500)
    assert result.passed is True
    assert all(c["passed"] for c in result.checks)


def test_audio_gate_fails_on_clipping_and_long_silence() -> None:
    result = evaluate_audio(max_silence_gap_seconds=1.2, has_clipping=True,
                            narration_word_count=0)
    assert result.passed is False
    names = {c["name"] for c in result.checks}
    assert {"clipping", "silence-gaps", "narration-presence"} <= names


# ========== Visual gate ==========

def test_visual_gate_passes_when_healthy() -> None:
    result = evaluate_visual(black_frame_ratio=0.0, frozen_frame_count=0,
                             min_contrast=0.3, unreadable_text_regions=0,
                             repeated_shot_count=1, title_card_count=2, total_scenes=10)
    assert result.passed is True


def test_visual_gate_fails_on_black_frames_and_repeated_shots() -> None:
    result = evaluate_visual(black_frame_ratio=0.2, frozen_frame_count=5,
                             min_contrast=0.05, unreadable_text_regions=3,
                             repeated_shot_count=8, title_card_count=9, total_scenes=10)
    assert result.passed is False
    names = {c["name"] for c in result.checks}
    assert {"black-frames", "frozen-frames", "contrast", "unreadable-text",
            "repeated-shots", "title-card-density"} <= names


# ========== Continuity gate ==========

def test_continuity_gate_passes_when_all_verified() -> None:
    claim_to_scene = {"claim-01": ["scene-01"], "claim-02": ["scene-02"]}
    verified = {"claim-01", "claim-02"}
    result = evaluate_continuity(claim_to_scene=claim_to_scene, verified_claim_ids=verified,
                                 scene_durations_seconds=[6.8, 8.4], total_duration_seconds=15.2)
    assert result.passed is True


def test_continuity_gate_fails_on_unverified_and_missing_claims() -> None:
    claim_to_scene = {"claim-01": ["scene-01"], "claim-99": ["scene-03"]}
    verified = {"claim-01"}  # claim-99 not verified; claim-02 missing from narration
    result = evaluate_continuity(claim_to_scene=claim_to_scene, verified_claim_ids=verified)
    assert result.passed is False
    names = {c["name"] for c in result.checks}
    assert {"claim-verification", "claim-coverage"} <= names


# ========== Rights gate ==========

def test_rights_gate_passes_when_all_approved() -> None:
    assets = [{"asset_id": "a1", "review_status": "approved"},
              {"asset_id": "a2", "review_status": "approved"}]
    result = evaluate_rights(assets)
    assert result.passed is True


def test_rights_gate_fails_on_unapproved() -> None:
    assets = [{"asset_id": "a1", "review_status": "approved"},
              {"asset_id": "a2", "review_status": "pending_review"}]
    result = evaluate_rights(assets)
    assert result.passed is False


# ========== Originality gate ==========

def test_originality_gate_passes_for_substantial_video() -> None:
    result = evaluate_originality(narration_word_count=2500, distinct_scenes=8, total_scenes=10)
    assert result.passed is True


def test_originality_gate_fails_on_low_substance() -> None:
    result = evaluate_originality(narration_word_count=500, distinct_scenes=2, total_scenes=10)
    assert result.passed is False


# ========== Metadata gate ==========

def test_metadata_gate_passes_when_clean() -> None:
    result = evaluate_metadata(
        title="The Deep Space Telescope's First Images",
        description="A documentary about the James Webb Space Telescope and its discoveries.",
        chapters=["Intro", "Launch", "First Light", "Findings"],
        thumbnail_matches=True,
    )
    assert result.passed is True


def test_metadata_gate_fails_on_overclaim_and_short_title() -> None:
    result = evaluate_metadata(
        title="Short",
        description="This will cure everything and guarantee 100% results.",
        chapters=["Only one"],
        thumbnail_matches=False,
    )
    assert result.passed is False
    names = {c["name"] for c in result.checks}
    assert {"title-length", "no-overclaim", "chapters", "thumbnail-match"} <= names


# ========== Aggregation + persistence ==========

def test_aggregate_gates_fails_when_any_gate_fails() -> None:
    good = evaluate_audio(max_silence_gap_seconds=0.1, has_clipping=False)
    bad = evaluate_rights([{"asset_id": "a", "review_status": "rejected"}])
    summary = aggregate_gates([good, bad])
    assert summary.status == "fail"


def test_write_qc_reports_roundtrip(tmp_path: Path) -> None:
    good = evaluate_audio(max_silence_gap_seconds=0.1, has_clipping=False)
    summary = aggregate_gates([good])
    json_path, md_path = write_qc_reports(summary, tmp_path)

    data = json.loads(json_path.read_text())
    assert data["status"] == "pass"
    assert len(data["gates"]) == 1
    assert md_path.is_file()
    assert "# Editorial QC: pass" in md_path.read_text()
