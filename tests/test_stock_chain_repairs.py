"""Regression tests for the no-NASA stock chain, QC reason/source fixes,
strict duration QC (hard-cut concat), and karaoke caption placement."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from ai_video_factory.edit_schema import load_edit
from ai_video_factory.media_probe import MediaInfo, parse_ffprobe
from ai_video_factory.qc import evaluate_qc
from ai_video_factory.stock_media import StockAsset
from ai_video_factory.stock_providers import StockFootageProvider
from ai_video_factory.subtitle_export import _ASS_HEADER
from ai_video_factory.video_pipeline import _rejection_reasons


class _StubClient:
    def __init__(self, name, available=True):
        self.name = name
        self.available = available


def _provider_with(pexels=True, pixabay=True):
    provider = StockFootageProvider.__new__(StockFootageProvider)
    provider.pexels = _StubClient("pexels", pexels)
    provider.pixabay = _StubClient("pixabay", pixabay)
    return provider


def test_provider_chain_excludes_nasa():
    provider = StockFootageProvider()
    assert not hasattr(provider, "nasa"), "NASA must not be in the provider chain"
    assert [c.name for c in _provider_with()._ordered_clients("space cosmos galaxy")] == [
        "pexels",
        "pixabay",
    ]


def test_provider_chain_skips_unavailable_clients():
    ordered = _provider_with(pexels=False, pixabay=True)._ordered_clients("anything")
    assert [c.name for c in ordered] == ["pixabay"]
    assert _provider_with(pexels=False, pixabay=False)._ordered_clients("x") == []


def _decoded_media() -> MediaInfo:
    payload = json.loads(Path("tests/fixtures/ffprobe-video.json").read_text())
    return replace(parse_ffprobe(payload), decode_succeeded=True)


def test_duration_qc_rejects_short_master_strictly():
    # Hard-cut concat is duration-preserving, so the master must match the
    # edit timeline exactly. A master shortened by the old xfade join
    # overlap (19.2s on the 25-chunk Fermi v3 render) is a real defect: it
    # would desync the edit-timeline narration, captions, and chapters.
    # No overlap may be subtracted to make a short master pass.
    edit = load_edit(Path("fixtures/synthetic-edit.json"))
    edit_total = edit.duration_frames / edit.fps
    short = replace(_decoded_media(), duration_seconds=edit_total - 19.2)
    failing = next(
        check for check in evaluate_qc(short, edit).checks if check.name == "duration"
    )
    assert failing.passed is False
    assert "expected" in failing.detail and "got" in failing.detail


def test_duration_qc_without_overlap_is_unchanged():
    edit = load_edit(Path("fixtures/synthetic-edit.json"))
    edit_total = edit.duration_frames / edit.fps
    media = replace(_decoded_media(), duration_seconds=edit_total)
    assert next(
        check for check in evaluate_qc(media, edit).checks if check.name == "duration"
    ).passed is True


def test_stock_asset_carries_provider():
    asset = StockAsset(
        kind="clip", path="x.mp4", title="t", artist="a", license="l",
        license_url="u", source_url="s", query="q",
    )
    assert asset.provider == "unknown"
    assert asset.to_dict()["provider"] == "unknown"
    asset2 = replace(asset, provider="pexels")
    assert asset2.to_dict()["provider"] == "pexels"


def test_rejection_reasons_never_empty():
    assert _rejection_reasons({"ok": False, "reasons": ["black_frame"]}) == ["black_frame"]
    fallback = _rejection_reasons({"ok": False, "scores": {"min_brightness": 0.4}})
    assert len(fallback) == 1
    assert fallback[0].startswith("qc_failed_undiagnosed(")
    assert "min_brightness=0.400" in fallback[0]
    assert _rejection_reasons({"ok": False})[0].startswith("qc_failed_undiagnosed(")


def test_karaoke_caption_margin_clears_lower_thirds():
    # Karaoke band raised (MarginV=120) so it clears burned-in source
    # lower-thirds, which sit in the bottom ~100px of the frame.
    assert ",2,40,40,120,1" in _ASS_HEADER
