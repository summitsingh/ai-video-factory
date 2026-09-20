"""Stock-footage visual path: scene asset selection for longform beat ids.

Regression tests for the Fermi v2 failure: every stage of the visual chain
(NASA populate, stock fetch, attach, Remotion staging) gated on scene ids
starting with ``scene-``, while longform edits use beat-prefixed ids
(``discovery-0``, ``mechanism-3``, ...). Result: 26 scenes, zero assets,
91.9% near-black master. All selection now goes through
``scene_asset_slots`` (positional among ``kind == "normal"`` scenes).

Every provider/HTTP call is mocked; no network, no downloads, no render.
"""

from pathlib import Path

import pytest

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


def _fermi_like_edit() -> EditDocument:
    """Mimics the Fermi v2 edit: beat-prefixed ids, intro/outro kinds."""
    scenes = [
        _make_scene("cold_open-0", kind="intro"),
        _make_scene("discovery-0"),
        _make_scene("discovery-1"),
        _make_scene("mechanism-0"),
        _make_scene("meaning-0"),
        _make_scene("open_question-0", kind="outro"),
    ]
    total = sum(s.duration_frames for s in scenes)
    return EditDocument(
        schema_version=1,
        width=1280,
        height=536,
        fps=30,
        duration_frames=total,
        scenes=scenes,
    )


def test_scene_asset_slots_maps_beat_ids_positionally():
    edit = _fermi_like_edit()
    slots = scene_asset_slots(edit.scenes)
    # Intro/outro get no slot; normal scenes are slotted in order.
    assert slots == {
        "discovery-0": 0,
        "discovery-1": 1,
        "mechanism-0": 2,
        "meaning-0": 3,
    }
    assert "cold_open-0" not in slots
    assert "open_question-0" not in slots


def test_scene_asset_slots_still_supports_legacy_scene_ids():
    scenes = [_make_scene(f"scene-{i}") for i in range(3)]
    assert scene_asset_slots(scenes) == {"scene-0": 0, "scene-1": 1, "scene-2": 2}


def test_populate_stock_clips_resolves_stock_for_beat_id_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The Fermi v2 path: mocked provider must resolve clips for beat ids."""
    import ai_video_factory.stock_providers as providers_module
    from ai_video_factory import video_pipeline

    fake_clip = tmp_path / "provider-cache-clip.mp4"
    fake_clip.write_bytes(b"\x00" * 1024)

    calls: list[str] = []

    class FakeProvider:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_clip(self, description: str, min_duration_sec: float = 5.0):
            calls.append(description)
            return StockAsset(
                kind="clip",
                path=str(fake_clip),
                title="Fake galaxy clip",
                artist="Fake artist",
                license="Fake license",
                license_url="https://example.invalid/license",
                source_url="https://example.invalid/video",
                query="galaxy",
            )

    monkeypatch.setattr(providers_module, "StockFootageProvider", FakeProvider)

    edit = _fermi_like_edit()
    assets_dir = tmp_path / "assets"
    assets = video_pipeline._populate_stock_clips(edit, assets_dir)

    # 4 normal scenes -> 4 fetched clips, one per scene slot dir.
    assert len(assets) == 4
    assert len(calls) == 4
    for slot in range(4):
        copied = assets_dir / f"scene-{slot:02d}" / "stock-clip.mp4"
        assert copied.is_file(), f"missing {copied}"
    # Asset paths are rewritten to the per-scene copies.
    assert assets[0].path == str(assets_dir / "scene-00" / "stock-clip.mp4")

    # And attach binds them onto the beat-id scenes.
    attached = video_pipeline.attach_scene_assets(edit, assets_dir)
    by_id = {s.id: s for s in attached.scenes}
    assert by_id["discovery-0"].clip == "scene-00-clip.mp4"
    assert by_id["discovery-1"].clip == "scene-01-clip.mp4"
    assert by_id["mechanism-0"].clip == "scene-02-clip.mp4"
    assert by_id["meaning-0"].clip == "scene-03-clip.mp4"
    assert by_id["cold_open-0"].clip is None  # intro keeps title card
    assert by_id["open_question-0"].clip is None  # outro keeps title card


def test_attach_scene_assets_skips_scenes_with_no_slot_dir(tmp_path: Path):
    from ai_video_factory import video_pipeline

    edit = _fermi_like_edit()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    attached = video_pipeline.attach_scene_assets(edit, assets_dir)
    assert all(s.clip is None and s.image is None for s in attached.scenes)


def test_cache_assets_for_remotion_keeps_slot_names_with_gaps(tmp_path: Path):
    """A missing slot dir must not shift later scenes onto wrong filenames."""
    from ai_video_factory import video_pipeline

    assets_dir = tmp_path / "assets"
    (assets_dir / "scene-00").mkdir(parents=True)
    (assets_dir / "scene-02").mkdir(parents=True)  # gap: scene-01 missing
    (assets_dir / "scene-00" / "scene-00-clip.mp4").write_bytes(b"\x00" * 64)
    (assets_dir / "scene-02" / "scene-02-clip.mp4").write_bytes(b"\x00" * 64)

    public = tmp_path / "public"
    video_pipeline.cache_assets_for_remotion(assets_dir, public)

    assert (public / "scene-00-clip.mp4").is_file()
    assert (public / "scene-02-clip.mp4").is_file()
    assert not (public / "scene-01-clip.mp4").exists()


def test_populate_assets_from_nasa_queries_beat_id_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """NASA pass must run for beat-id scenes, not silently skip them."""
    import ai_video_factory.nasa_media as nasa_module

    calls: list[tuple[Path, list[str]]] = []

    def fake_fetch(queries, scene_dir, **kwargs):
        calls.append((Path(scene_dir), list(queries)))
        scene_dir = Path(scene_dir)
        scene_dir.mkdir(parents=True, exist_ok=True)
        img = scene_dir / "nasa-image.jpg"
        img.write_bytes(b"\x00" * 64)
        return [
            StockAsset(
                kind="image",
                path=str(img),
                title="Fake NASA image",
                artist="NASA",
                license="Public Domain",
                license_url="https://example.invalid/nasa",
                source_url="https://example.invalid/nasa-img",
                query=queries[0],
            )
        ]

    monkeypatch.setattr(nasa_module, "fetch_nasa_for_scene", fake_fetch)

    edit = _fermi_like_edit()
    assets_dir = tmp_path / "nasa_assets"
    summary = nasa_module.populate_assets_from_nasa(edit, assets_dir)

    assert summary["scenes"] == 4
    assert summary["images"] == 4
    called_dirs = sorted(p.name for p, _ in calls)
    assert called_dirs == ["scene-00", "scene-01", "scene-02", "scene-03"]
