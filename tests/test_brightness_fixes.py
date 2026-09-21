"""Brightness-fix regressions: generated.png fallback, content-hash blocklist.

Covers the root cause of the Fermi v3 brightness QC failure (38/50 dark
frames): scenes with no stock clip got a bright AI still at
``scene-NN/generated.png``, but ``attach_scene_assets`` excluded it from
the candidate list, so scenes rendered with the near-black procedural
fallback. Also covers the AssetMemory slot-poisoning bug: rejections were
keyed by slot filename (``scene-05-clip.mp4``), so a rejected clip blocked
a *different* good clip that later landed at the same slot.
"""

import json
from pathlib import Path

import pytest

from ai_video_factory import video_pipeline
from ai_video_factory.asset_memory import AssetMemory
from ai_video_factory.edit_schema import (
    EditDocument,
    EditScene,
    scene_asset_slots,
)
from ai_video_factory.stock_media import StockAsset


def _make_scene(scene_id: str, kind: str = "normal") -> EditScene:
    return EditScene(
        id=scene_id,
        from_frame=0,
        duration_frames=900,
        title=f"Title {scene_id}",
        caption=f"Caption {scene_id}",
        kind=kind,  # type: ignore[arg-type]
        visual="a spiral galaxy slowly rotating",
        narration="Some narration text for the scene.",
    )


def _two_scene_edit() -> EditDocument:
    scenes = [_make_scene("intro-0", kind="intro"), _make_scene("body-0"), _make_scene("outro-0", kind="outro")]
    total = sum(s.duration_frames for s in scenes)
    return EditDocument(
        schema_version=1,
        width=1280,
        height=536,
        fps=30,
        duration_frames=total,
        scenes=scenes,
    )


def test_attach_prefers_stock_image_over_generated_fallback(tmp_path: Path):
    """When both a stock image and generated.png exist, stock wins."""
    edit = _two_scene_edit()
    scene_dir = tmp_path / "scene-00"
    scene_dir.mkdir()
    stock = scene_dir / "stock-photo-pexels.jpg"
    stock.write_bytes(b"stock-bytes")
    generated = scene_dir / "generated.png"
    generated.write_bytes(b"generated-bytes")

    attached = video_pipeline.attach_scene_assets(edit, tmp_path)
    body = [s for s in attached.scenes if s.id == "body-0"][0]
    assert body.image == "scene-00-image.jpg"
    assert (scene_dir / "scene-00-image.jpg").is_file()
    # The fallback must not be consumed or renamed away.
    assert generated.is_file()


def test_attach_falls_back_to_generated_png_when_no_stock(tmp_path: Path):
    """generated.png is attached (not dropped) when no stock image exists."""
    edit = _two_scene_edit()
    scene_dir = tmp_path / "scene-00"
    scene_dir.mkdir()
    generated = scene_dir / "generated.png"
    generated.write_bytes(b"generated-bytes")

    attached = video_pipeline.attach_scene_assets(edit, tmp_path)
    body = [s for s in attached.scenes if s.id == "body-0"][0]
    assert body.image == "scene-00-image.png"
    assert (scene_dir / "scene-00-image.png").is_file()


def test_attach_leaves_scene_media_less_without_any_image(tmp_path: Path):
    """No image at all: scene keeps the procedural title-card look."""
    edit = _two_scene_edit()
    (tmp_path / "scene-00").mkdir()
    attached = video_pipeline.attach_scene_assets(edit, tmp_path)
    body = [s for s in attached.scenes if s.id == "body-0"][0]
    assert body.image is None
    assert body.clip is None


def test_asset_identity_is_content_hash_not_slot_name(tmp_path: Path):
    """A rejected file must not block a different file at the same slot."""
    memory = AssetMemory(path=tmp_path / "memory.json")
    bad = tmp_path / "scene-05-clip.mp4"
    bad.write_bytes(b"bad clip bytes")
    bad_id = video_pipeline._asset_identity(bad, "scene-05-clip.mp4")
    assert bad_id.startswith("sha256:")

    memory.record_rejection(bad_id, "black_frame", source="pexels")
    assert memory.is_blocked(bad_id)

    # A different clip later lands at the same slot filename: not blocked.
    good = tmp_path / "scene-05-clip.mp4"
    good.write_bytes(b"completely different good clip bytes")
    good_id = video_pipeline._asset_identity(good, "scene-05-clip.mp4")
    assert good_id != bad_id
    assert not memory.is_blocked(good_id)


def test_asset_identity_falls_back_to_slot_name_when_file_missing(tmp_path: Path):
    missing = tmp_path / "no-such-file.mp4"
    assert video_pipeline._asset_identity(missing, "scene-05-clip.mp4") == "scene-05-clip.mp4"


def test_provider_attribution_survives_attach_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """stock-providers.json still resolves after attach renames clips."""
    import ai_video_factory.stock_providers as providers_module

    fake_clip = tmp_path / "provider-cache-clip.mp4"
    fake_clip.write_bytes(b"\x00" * 1024)

    class FakeProvider:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_clip(self, description: str, min_duration_sec: float = 5.0):
            return StockAsset(
                kind="clip",
                path=str(fake_clip),
                title="Fake galaxy clip",
                artist="Fake artist",
                license="Fake license",
                license_url="https://example.invalid/license",
                source_url="https://example.invalid/video",
                query="galaxy",
                provider="pexels",
            )

    monkeypatch.setattr(providers_module, "StockFootageProvider", FakeProvider)

    edit = _two_scene_edit()
    assets_dir = tmp_path / "assets"
    video_pipeline._populate_stock_clips(edit, assets_dir)
    attached = video_pipeline.attach_scene_assets(edit, assets_dir)

    # The clip was renamed from stock-clip-pexels.mp4 to the scene slot name.
    body = [s for s in attached.scenes if s.id == "body-0"][0]
    assert body.clip == "scene-00-clip.mp4"

    # ...but the provider sidecar is keyed by scene dir, so QC can still
    # attribute the true source after the rename.
    providers_map = json.loads((assets_dir / "stock-providers.json").read_text())
    assert providers_map.get("scene-00") == "pexels"
