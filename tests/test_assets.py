"""Tests for the rights-cleared asset library (offline, fixture-based).

Every provider adapter accepts an injected transport so no test makes a real
network call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_video_factory.assets import (
    AssetError,
    LicenseStatus,
    MediaProperties,
    RightsRecord,
    approve_asset,
    acquire_asset,
    classify_license,
    gate_assets,
    load_rights_manifest,
    NASA_MEDIA_USAGE_TERMS_URL,
    nasa_eligible,
    render_cannot_proceed,
    score_asset,
    select_assets,
    write_rights_manifest,
)


# ========== License classification ==========

def test_classify_license_nasa_is_public_domain() -> None:
    name, url = classify_license("", provider="nasa")
    assert "Public Domain" in (name or "")
    assert url  # canonical URL present


def test_classify_license_wikimedia_accepted() -> None:
    cc0 = "https://creativecommons.org/publicdomain/zero/1.0/"
    name, url = classify_license(cc0, provider="wikimedia")
    assert name == "CC0 1.0"
    assert url == cc0


def test_classify_license_wikimedia_by_sa() -> None:
    url = "https://creativecommons.org/licenses/by-sa/4.0/"
    name, _ = classify_license(url, provider="wikimedia")
    assert name == "CC-BY-SA"


def test_classify_license_rejects_nc_nd() -> None:
    for bad in (
        "https://creativecommons.org/licenses/by-nc/4.0/",
        "https://creativecommons.org/licenses/by-nd/4.0/",
    ):
        name, url = classify_license(bad, provider="wikimedia")
        assert name is None and url == ""


def test_classify_license_rejects_unknown() -> None:
    name, _ = classify_license("https://example.com/some-random-license", provider="wikimedia")
    assert name is None


# ========== Approval gate (fail-closed) ==========

def _record(license_name: str, status: LicenseStatus = "pending_review") -> RightsRecord:
    return RightsRecord(
        asset_id="asset-nasa-abc", kind="image", provider="nasa",
        original_url="https://images-api.nasa.gov/item/x", download_url="https://cdn.example/x.jpg",
        creator="NASA", license_name=license_name, license_url="", attribution="",
        acquired_at="2026-09-06T00:00:00+00:00", checksum_sha256="deadbeef",
        properties=MediaProperties(width=1920, height=1080), review_status=status,
        # Per-asset evidence required by the operator asset policy (2026-09-08):
        # NASA public-domain approval is grounded on 17 U.S.C. § 105 and must be
        # recorded per item — an empty nasa_id or license_basis is ambiguous.
        nasa_id="nasa-123",
        canonical_url="https://images-api.nasa.gov/item/nasa-123",
        retrieved_metadata={"title": "Sample asset", "center": "NASA"},
        license_basis="Public domain — 17 U.S.C. § 105: works prepared by a U.S. government agency are not subject to copyright.",
        usage_terms_url=NASA_MEDIA_USAGE_TERMS_URL,
    )


def test_gate_approves_public_domain() -> None:
    record = _record("Public Domain (NASA)")
    approved = gate_assets([record])
    assert len(approved) == 1
    assert approved[0].review_status == "approved"


def test_gate_rejects_missing_license() -> None:
    record = _record("")
    with pytest.raises(AssetError):
        gate_assets([record])


def test_gate_rejects_explicitly_rejected_record() -> None:
    record = _record("CC-BY-NC", status="rejected")
    with pytest.raises(AssetError):
        gate_assets([record])


def test_render_cannot_proceed_detects_unapproved() -> None:
    assert render_cannot_proceed([_record("")]) is True
    assert render_cannot_proceed([_record("Public Domain (NASA)")]) is False

# ========== NASA item-level eligibility (operator asset policy) ==========


def _nasa_record(
    *,
    creator: str = "NASA",
    center: str = "Goddard Space Flight Center",
    nasa_id: str = "nasa-123",
    title: str = "Sample asset",
    license_name: str = "Public Domain (NASA)",
    usage_terms_url: str = NASA_MEDIA_USAGE_TERMS_URL,
    status: LicenseStatus = "pending_review",
    extra_metadata: dict | None = None,
) -> RightsRecord:
    meta = {"title": title, "center": center}
    if extra_metadata:
        meta.update(extra_metadata)
    return RightsRecord(
        asset_id="asset-nasa-xyz", kind="image", provider="nasa",
        original_url="https://images-api.nasa.gov/item/nasa-123",
        download_url="https://cdn.example/x.jpg",
        creator=creator, license_name=license_name, license_url="", attribution="",
        acquired_at="2026-09-06T00:00:00+00:00", checksum_sha256="deadbeef",
        properties=MediaProperties(width=1920, height=1080), review_status=status,
        nasa_id=nasa_id,
        canonical_url="https://images-api.nasa.gov/item/nasa-123",
        retrieved_metadata=meta,
        license_basis="Public domain — 17 U.S.C. § 105: works prepared by a U.S. "
                      "government agency are not subject to copyright.",
        usage_terms_url=usage_terms_url,
    )


def test_nasa_eligible_approves_operated_center_authorship() -> None:
    record = _nasa_record()  # default center is a NASA center
    assert nasa_eligible(record) == (True, "")
    assert gate_assets([record])[0].review_status == "approved"


def test_nasa_eligible_rejects_astronaut_portrait_without_release() -> None:
    # Human-subject material is rejected by keyword even with no model-release phrase.
    record = _nasa_record(
        creator="NASA",
        center="Johnson Space Center",
        extra_metadata={"description": "Portrait of an astronaut inside the ISS"},
    )
    ok, reason = nasa_eligible(record)
    assert ok is False and "human subject" in reason


def test_nasa_eligible_rejects_contractor_jpl_caltech() -> None:
    record = _nasa_record(
        creator="NASA",
        center="Jet Propulsion Laboratory, Caltech",
    )
    ok, reason = nasa_eligible(record)
    assert ok is False and "contractor" in reason


def test_nasa_eligible_rejects_mixed_third_party_credit() -> None:
    # creator=NASA but a second structured author is an unrecognized private entity.
    record = _nasa_record(
        creator="NASA",
        center="NASA",
        extra_metadata={"secondary_creator": "Acme Imaging LLC"},
    )
    ok, reason = nasa_eligible(record)
    assert ok is False and "author credit" in reason


def test_nasa_eligible_rejects_other_agency() -> None:
    # NASA-provider auto-clearance excludes other agencies (NOAA) as ambiguous.
    record = _nasa_record(creator="NOAA", center="NOAA")
    ok, reason = nasa_eligible(record)
    assert ok is False and "author" in reason


def test_nasa_eligible_rejects_missing_usage_terms() -> None:
    record = _nasa_record(usage_terms_url="")
    ok, reason = nasa_eligible(record)
    assert ok is False and "usage terms" in reason


def test_nasa_eligible_preserves_empty_creator() -> None:
    # A missing creator must NOT be manufactured into an agency credit.
    record = _nasa_record(creator="", center="")
    ok, reason = nasa_eligible(record)
    assert ok is False and "author" in reason


def test_gate_enforces_eligibility_even_when_preapproved() -> None:
    # A pre-approved astronaut portrait cannot bypass eligibility at the gate.
    record = _nasa_record(
        creator="NASA",
        extra_metadata={"description": "Portrait of an astronaut"},
    )
    preapproved = approve_asset(record)  # concrete license -> approved by approve_asset
    assert preapproved.review_status == "approved"
    with pytest.raises(AssetError):
        gate_assets([preapproved])

# ========== Acquisition with injected transport ==========

_NASA_SEARCH_JSON = json.dumps({
    "collection": {
        "items": [
            {
                "item_id": "123",
                "title": "Apollo 11 Moon Landing",
                "metadata": {"creator": "NASA"},
                "links": [{"rel": "media", "href": "https://cdn.nasa.gov/apollo.mp4"}],
            },
        ]
    }
})


def test_acquire_asset_builds_rights_record(tmp_path: Path) -> None:
    # Search returns JSON; download returns bytes.
    payloads = {"images-api.nasa.gov": _NASA_SEARCH_JSON}

    def transport(url: str) -> bytes:
        if "images-api.nasa.gov" in url:
            return _NASA_SEARCH_JSON.encode("utf-8")
        return b"\x00\x01\x02fake-bytes"

    dest = tmp_path / "apollo.mp4"
    record = acquire_asset(
        "nasa", "Apollo 11", scene_id="scene-01", destination=dest,
        mediatype="clip", transport=transport,
    )
    assert dest.is_file()
    assert record.provider == "nasa"
    assert record.kind == "clip"
    assert record.license_name and "Public Domain" in record.license_name
    assert record.checksum_sha256  # computed from file bytes
    assert record.candidate_scene_ids == ["scene-01"]
    # Starts pending; must be approved before render.
    assert record.review_status == "pending_review"


def test_acquire_asset_rejects_no_download_url(tmp_path: Path) -> None:
    empty = json.dumps({"collection": {"items": [{"item_id": "x", "title": "t", "links": []}]}})

    def transport(url: str) -> bytes:
        return empty.encode("utf-8")

    with pytest.raises(AssetError):
        acquire_asset("nasa", "empty", scene_id="scene-01", destination=tmp_path / "x.mp4",
                      mediatype="clip", transport=transport)


# ========== Selection scoring ==========

def test_score_asset_rewards_relevance_and_rights() -> None:
    relevant = RightsRecord(
        asset_id="a1", kind="image", provider="nasa",
        original_url="https://cdn.example/rocket-launch-high-res.jpg", download_url="u",
        creator="NASA", license_name="Public Domain (NASA)", license_url="", attribution="",
        acquired_at="2026-09-06T00:00:00+00:00", checksum_sha256="x",
        properties=MediaProperties(width=3840, height=2160, codec="h264"),
    )
    score = score_asset(approve_asset(relevant), ["rocket", "launch"])
    assert score.relevance > 0.0
    assert score.rights_confidence == 1.0  # public domain


def test_select_assets_dedupes_and_ranks() -> None:
    records = [
        RightsRecord(
            asset_id=f"a{i}", kind="image", provider="nasa",
            original_url=f"https://cdn.example/{topic}.jpg" if i else "https://cdn.example/rocket-launch.jpg",
            download_url="u", creator="NASA", license_name="Public Domain (NASA)",
            license_url="", attribution="", acquired_at="2026-09-06T00:00:00+00:00",
            checksum_sha256="x", properties=MediaProperties(width=1920, height=1080),
        )
        for i, topic in enumerate(["rocket-launch", "generic", "brand-logo"])
    ]
    selected = select_assets(records, ["rocket", "launch"], limit=3)
    assert len(selected) >= 1
    # Sorted descending by score.
    scores = [s.score for s in selected]
    assert scores == sorted(scores, reverse=True)


def test_select_assets_penalizes_reuse() -> None:
    reused = RightsRecord(
        asset_id="same", kind="image", provider="nasa",
        original_url="https://cdn.example/rocket.jpg", download_url="u", creator="NASA",
        license_name="Public Domain (NASA)", license_url="", attribution="",
        acquired_at="2026-09-06T00:00:00+00:00", checksum_sha256="x",
        properties=MediaProperties(width=1920, height=1080),
    )
    fresh = RightsRecord(
        asset_id="other", kind="image", provider="nasa",
        original_url="https://cdn.example/telescope.jpg", download_url="u", creator="NASA",
        license_name="Public Domain (NASA)", license_url="", attribution="",
        acquired_at="2026-09-06T00:00:00+00:00", checksum_sha256="x",
        properties=MediaProperties(width=1920, height=1080),
    )
    with_reuse = score_asset(reused, ["topic"], used_ids=["same"])
    without_reuse = score_asset(fresh, ["topic"], used_ids=[])
    assert with_reuse.score < without_reuse.score


# ========== Persistence ==========

def test_write_and_load_rights_manifest_roundtrip(tmp_path: Path) -> None:
    records = [
        approve_asset(_record("Public Domain (NASA)")),
        _record("", status="rejected"),
    ]
    path = write_rights_manifest(records, tmp_path / "rights.json")
    loaded = load_rights_manifest(path)
    assert len(loaded) == 2
    assert loaded[0].review_status == "approved"
    assert loaded[1].rejection_reason is None  # rejected stays rejected on reload


def test_approve_asset_rejects_bad_license() -> None:
    record = _record("")
    approved = approve_asset(record)
    assert approved.review_status == "rejected"
