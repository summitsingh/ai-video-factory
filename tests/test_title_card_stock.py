"""Regression tests for stock motion clips on intro/outro title cards."""

import json
from pathlib import Path

import pytest

from ai_video_factory.edit_schema import EditDocument, EditScene
from ai_video_factory.stock_media import StockAsset


def _scene(scene_id: str, kind: str, *, frames: int = 90) -> EditScene:
    return EditScene(
        id=scene_id,
        from_frame=0,
        duration_frames=frames,
        title=f"Title {scene_id}",
        caption="Caption",
        kind=kind,  # type: ignore[arg-type]
        visual=f"Visual {scene_id}",
    )


def _edit(*scenes: EditScene) -> EditDocument:
    return EditDocument(
        schema_version=1,
        width=1280,
        height=720,
        fps=30,
        duration_frames=max(s.duration_frames for s in scenes),
        scenes=list(scenes),
    )


def test_populate_fetches_intro_and_outro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ai_video_factory.stock_providers as providers_module
    from ai_video_factory import video_pipeline

    source = tmp_path / "source.mp4"
    source.write_bytes(b"motion")
    calls: list[tuple[str, float]] = []

    class FakeProvider:
        def fetch_clip(self, description: str, min_duration_sec: float = 5.0):
            calls.append((description, min_duration_sec))
            return StockAsset(
                kind="clip", path=str(source), title="Clip", artist="Artist",
                license="License", license_url="https://example.invalid/license",
                source_url="https://example.invalid/source", query=description,
                provider="pexels",
            )

    monkeypatch.setattr(providers_module, "StockFootageProvider", FakeProvider)
    assets_dir = tmp_path / "assets"
    assets = video_pipeline._populate_stock_clips(
        _edit(_scene("intro", "intro"), _scene("outro", "outro")), assets_dir
    )

    assert calls == [("Visual intro", 3.0), ("Visual outro", 3.0)]
    assert len(assets) == 2
    assert (assets_dir / "card-intro" / "stock-clip-pexels.mp4").is_file()
    assert (assets_dir / "card-outro" / "stock-clip-pexels.mp4").is_file()
    providers = json.loads((assets_dir / "stock-providers.json").read_text())
    assert providers == {"card-intro": "pexels", "card-outro": "pexels"}


def test_populate_skips_title_card_with_existing_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ai_video_factory.stock_providers as providers_module
    from ai_video_factory import video_pipeline

    class UnexpectedProvider:
        def fetch_clip(self, description: str, min_duration_sec: float = 5.0):
            raise AssertionError("existing title-card clip must not be overridden")

    monkeypatch.setattr(
        providers_module, "StockFootageProvider", UnexpectedProvider
    )
    assets_dir = tmp_path / "assets"
    (assets_dir / "card-intro").mkdir(parents=True)
    supplied = assets_dir / "card-intro" / "user-clip.mp4"
    supplied.write_bytes(b"user")

    assets = video_pipeline._populate_stock_clips(
        _edit(_scene("intro", "intro")), assets_dir
    )

    assert assets == []
    assert supplied.read_bytes() == b"user"


def test_attach_title_cards_uses_clips_and_never_images(tmp_path: Path) -> None:
    from ai_video_factory import video_pipeline

    assets_dir = tmp_path / "assets"
    for kind in ("intro", "outro"):
        card_dir = assets_dir / f"card-{kind}"
        card_dir.mkdir(parents=True)
        (card_dir / "stock-clip-pixabay.mp4").write_bytes(b"motion")
        (card_dir / "still.jpg").write_bytes(b"still")

    attached = video_pipeline.attach_scene_assets(
        _edit(_scene("intro", "intro"), _scene("outro", "outro")), assets_dir
    )

    assert [scene.clip for scene in attached.scenes] == [
        "card-intro-clip.mp4", "card-outro-clip.mp4"
    ]
    assert all(scene.image is None for scene in attached.scenes)


def test_cache_title_card_clips_and_clean_stale_card_media(tmp_path: Path) -> None:
    from ai_video_factory import video_pipeline

    assets_dir = tmp_path / "assets"
    for kind in ("intro", "outro"):
        card_dir = assets_dir / f"card-{kind}"
        card_dir.mkdir(parents=True)
        (card_dir / f"card-{kind}-clip.mp4").write_bytes(kind.encode())
    public = tmp_path / "public"
    public.mkdir()
    (public / "card-old-clip.mp4").write_bytes(b"stale")
    (public / "card-old-image.png").write_bytes(b"stale")

    video_pipeline.cache_assets_for_remotion(assets_dir, public)

    assert (public / "card-intro-clip.mp4").read_bytes() == b"intro"
    assert (public / "card-outro-clip.mp4").read_bytes() == b"outro"
    assert not (public / "card-old-clip.mp4").exists()
    assert not (public / "card-old-image.png").exists()


def test_fail_closed_gate_rejects_intro_without_clip() -> None:
    from ai_video_factory.video_pipeline import _enforce_title_card_clip_gate

    edit = _edit(_scene("intro", "intro"))
    with pytest.raises(RuntimeError, match=r"intro.*refusing to render"):
        _enforce_title_card_clip_gate(edit)
