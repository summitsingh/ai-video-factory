"""Tests for the real production renderer (RemotionKokoroRenderEngine).

These exercise configuration resolution, asset wiring, media-kind detection, and
fail-closed behavior without invoking a headless Remotion render (which needs
Chrome + node). The audio-mux path is covered by an integration test that stubs
only the browser render step and drives ffmpeg directly.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ai_video_factory.edit_schema import EditDocument, EditScene
from ai_video_factory.production import (
    RenderEngine,
    ProductionError,
    RemotionKokoroRenderEngine,
    audio_duration_seconds,
    build_edit_document,
    build_candidate,
    mix_scenes_to_track,
    _media_kind,
    _write_tone_wav,
    _run_ffmpeg,
)
from ai_video_factory.research_pipeline import Claim, ResearchResult, SourceContent


def _make_engine(tmp_path: Path) -> RemotionKokoroRenderEngine:
    return RemotionKokoroRenderEngine(
        remotion_root=tmp_path / "remotion",
        public_dir=tmp_path / "public",
        browser_executable=tmp_path / "chrome-fake",
        npm_executable="npm",
    )


def _scene(scene_id: str, *, frames: int = 30) -> EditScene:
    return EditScene(id=scene_id, from_frame=0, duration_frames=frames, title="T", caption="C")


def _edit(*scenes: EditScene) -> EditDocument:
    total = sum(s.duration_frames for s in scenes) or 1
    return EditDocument(
        schema_version=1, width=1920, height=1080, fps=30,
        duration_frames=total, scenes=list(scenes),
    )


# ===== Configuration resolution =====


def test_engine_resolves_default_remotion_root():
    engine = RemotionKokoroRenderEngine()
    assert engine._root.name == "remotion"
    assert (engine._root / "package.json").is_file()
    assert engine._public_dir == engine._root / "public"


def test_engine_honors_injected_paths_and_binaries(tmp_path):
    root = tmp_path / "remotion"
    public = tmp_path / "public"
    browser = tmp_path / "chrome"
    engine = RemotionKokoroRenderEngine(
        remotion_root=root, public_dir=public, browser_executable=browser, npm_executable="npm",
    )
    assert engine._root == root
    assert engine._public_dir == public
    assert engine._browser_executable == browser


# ===== Media-kind detection =====


def test_media_kind_by_extension(tmp_path):
    img = tmp_path / "a.jpg"
    vid = tmp_path / "b.mp4"
    img.write_bytes(b"x")
    vid.write_bytes(b"x")
    assert _media_kind(img) == "image"
    assert _media_kind(vid) == "clip"


# ===== NASA still/image wiring =====


def test_wire_assets_stages_image_and_sets_scene_image(tmp_path):
    engine = _make_engine(tmp_path)
    still = tmp_path / "nasa-still.jpg"
    still.write_bytes(b"\x89png-fake-jpeg-bytes")  # content irrelevant; only size matters
    scene = _scene("scene-1")
    edit = _edit(scene)

    staged = engine._wire_assets(edit, {"scene-1": still})

    assert len(staged) == 1
    assert scene.image is not None
    assert scene.clip is None
    resolved = engine._public_dir / scene.image
    assert resolved.is_file() and resolved.stat().st_size > 0
    assert scene.image.startswith("assets/")


def test_wire_assets_sets_scene_clip_for_video(tmp_path):
    engine = _make_engine(tmp_path)
    clip = tmp_path / "nasa-clip.mp4"
    clip.write_bytes(b"fakemp4")
    scene = _scene("scene-2", frames=60)
    edit = _edit(scene)

    engine._wire_assets(edit, {"scene-2": clip})

    assert scene.clip is not None
    assert scene.image is None
    assert (engine._public_dir / scene.clip).is_file()


def test_wire_assets_ignores_scenes_without_assets(tmp_path):
    engine = _make_engine(tmp_path)
    a = _scene("scene-1")
    b = _scene("scene-2")
    edit = _edit(a, b)

    still = tmp_path / "only.jpg"
    still.write_bytes(b"\xff\xd8\xff-fake-jpeg")
    engine._wire_assets(edit, {"scene-1": still})

    assert a.image is not None
    assert b.image is None and b.clip is None


# ===== Missing / invalid real artifacts (fail-closed) =====


def test_stage_asset_missing_raises(tmp_path):
    engine = _make_engine(tmp_path)
    with pytest.raises(ProductionError, match="missing"):
        engine._stage_asset(tmp_path / "does-not-exist.jpg", tmp_path / "run")


def test_stage_asset_empty_raises(tmp_path):
    engine = _make_engine(tmp_path)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(ProductionError, match="empty"):
        engine._stage_asset(empty, tmp_path / "run")


def test_render_master_fails_closed_on_missing_asset_before_remotion(tmp_path):
    # A missing asset must raise before any browser/node/remotion is invoked.
    engine = _make_engine(tmp_path)
    scene = _scene("scene-1")
    edit = _edit(scene)
    dest = tmp_path / "master.mp4"

    with pytest.raises(ProductionError, match="missing"):
        engine.render_master(
            edit,
            narration_segments=[(tmp_path / "n.wav", 0.0)],
            assets_by_scene_id={"scene-1": tmp_path / "nope.jpg"},
            destination=dest,
        )

    assert not dest.exists()


# ===== Full render_master path (ffmpeg, no Chrome) =====


def test_render_master_muxes_real_narration_with_ffmpeg(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path)
    still = tmp_path / "still.jpg"
    still.write_bytes(b"\xff\xd8\xff-fake-jpeg")

    def fake_remotion(props_path: Path, output: Path, browser: Path) -> None:
        # Stand in for headless Remotion: a real 2s muted video (no audio).
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i",
                "testsrc=size=640x360:rate=30:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
            ],
            check=True, capture_output=True, text=True, timeout=120,
        )

    monkeypatch.setattr(engine, "_run_remotion", fake_remotion)

    narr = tmp_path / "narr.wav"
    _write_tone_wav(narr, seconds=2)

    scene = _scene("scene-1", frames=60)  # 60 frames @ 30 fps = 2s
    edit = _edit(scene)
    dest = tmp_path / "master.mp4"

    out = engine.render_master(
        edit,
        narration_segments=[(narr, 0.0)],
        assets_by_scene_id={"scene-1": still},
        destination=dest,
    )

    assert out == dest and dest.is_file() and dest.stat().st_size > 0
    # Asset was wired into the edit and staged under the public root.
    assert scene.image is not None
    assert (engine._public_dir / scene.image).is_file()

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(dest),
        ],
        capture_output=True, text=True, check=True, timeout=30,
    )
    types = {s["codec_type"] for s in json.loads(probe.stdout)["streams"]}
    assert {"video", "audio"} <= types


# ===== Timeline layout: bookends occupy real space, scenes never overlap =====


def test_build_edit_document_lays_out_bookends_without_overlap():
    """Intro/outro occupy real timeline space; every scene begins exactly where the
    previous ended (exact integer-frame cursor) and the total duration includes both
    four-second bookends plus all normal scenes."""
    plan = [
        {"id": "scene-01", "duration_seconds": 2.0, "title": "A", "caption": "a", "narration": "one"},
        {"id": "scene-02", "duration_seconds": 2.0, "title": "B", "caption": "b", "narration": "two"},
        {"id": "scene-03", "duration_seconds": 2.0, "title": "C", "caption": "c", "narration": "three"},
    ]
    fps = 30
    doc = build_edit_document(plan, fps=fps)

    # intro + three normal scenes + outro, in that order.
    assert [s.kind for s in doc.scenes] == ["intro", "normal", "normal", "normal", "outro"]

    # Exact integer-frame starts: intro@0, normals at 4s/6s/8s (120/180/240 frames),
    # outro at 10s (300). No gaps and no overlaps between any two scenes.
    assert [s.from_frame for s in doc.scenes] == [0, 120, 180, 240, 300]

    cursor = 0
    for scene in doc.scenes:
        assert scene.from_frame == cursor
        cursor += scene.duration_frames

    # 4s intro + 3x2s scenes + 4s outro = 14s == 420 frames, all accounted for.
    assert doc.duration_frames == cursor == 420


def test_build_edit_document_layout_stays_on_exact_frames():
    """Non-integer scene durations still land on exact frame boundaries with their
    neighbours -- the integer cursor avoids sub-frame gaps/overlaps from float math."""
    plan = [
        {"id": "scene-01", "duration_seconds": 1.5, "title": "A", "caption": "a", "narration": "one"},
        {"id": "scene-02", "duration_seconds": 2.25, "title": "B", "caption": "b", "narration": "two"},
    ]
    fps = 30
    doc = build_edit_document(plan, fps=fps)

    cursor = 0
    for scene in doc.scenes:
        assert scene.from_frame == cursor
        assert scene.from_frame % 1 == 0
        cursor += scene.duration_frames
    assert doc.duration_frames == cursor


# ===== Narration audio sync: segments placed at absolute timeline offsets =====


def test_mix_scenes_to_track_places_segments_at_absolute_offsets(tmp_path):
    """Each narration WAV is placed at its absolute start offset, so a segment after
    the intro lands later instead of being concatenated at 0:00. A control with both
    offsets at 0 proves the extra length comes from the offset, not mere concatenation."""
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    _write_tone_wav(wav_a, seconds=1.0, hertz=440)
    _write_tone_wav(wav_b, seconds=1.0, hertz=880)

    # segB starts at 4.0s -> ~5s track (1s tone + 3s silence + 1s tone).
    offset_out = mix_scenes_to_track([(wav_a, 0.0), (wav_b, 4.0)], tmp_path / "offsets.wav")
    assert 4.8 <= (audio_duration_seconds(offset_out) or 0.0) <= 5.2

    # Baseline: a single segment -> ~1s. The ~5s offset track above is therefore the
    # extra 4s of placement delay, proving absolute-offset positioning (amix mixes to
    # the longest input's end, so two zero-offset clips would be ~1s, not ~2s).
    one = mix_scenes_to_track([(wav_a, 0.0)], tmp_path / "one.wav")
    assert 0.8 <= (audio_duration_seconds(one) or 0.0) <= 1.2


def test_build_candidate_maps_narration_offsets_to_scene_frames(tmp_path, monkeypatch):
    """build_candidate passes each plan scene's WAV at its edit scene's from_frame/fps
    offset (not a hard-coded 0.0), so the real engine syncs narration to visuals."""
    import tempfile

    # Bypass the 15-minute runtime gate and real asset acquisition; this test asserts
    # offset wiring only, not long-form gating or rights clearing.
    monkeypatch.setattr("ai_video_factory.production.enforce_runtime", lambda durations: None)
    monkeypatch.setattr("ai_video_factory.production.acquire_assets", lambda *a, **k: ([], {}))

    captured_edit = None
    captured_segs = None

    class _CaptureEngine(RenderEngine):
        def synthesize_narration(self, text, voice=None):  # pragma: no cover - test double
            wav = tempfile.mkstemp(suffix=".wav")[1]
            _write_tone_wav(wav, seconds=1.0)
            return Path(wav)

        def render_master(self, edit, *, narration_segments, assets_by_scene_id, destination):
            nonlocal captured_edit, captured_segs
            captured_edit = edit
            captured_segs = list(narration_segments)
            # A real (tiny) MP4 so build_candidate's downstream ffprobe QC metrics work.
            _run_ffmpeg((
                "-y", "-f", "lavfi",
                "-i", "testsrc=size=320x240:rate=30:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination),
            ))

    # Three distinct sentences (>=12 chars each) totalling >= the 120-word floor.
    sentences = [" ".join(f"topic{i}_{j}" for j in range(50)) + "." for i in range(3)]
    content = SourceContent(
        url="https://example.com/water", title="Water", quality="reliable",
        content=" ".join(sentences), content_hash="0" * 64,
    )
    research = ResearchResult(
        topic="water test topic",
        verified_facts=[
            Claim(claim_id=f"c{i}", text=f"Fact number {i} about water.", source_ids=[f"s{i}"],
                  provisional_classification="confirmed fact")
            for i in range(3)
        ],
        source_contents={"https://example.com/water": content},
    )

    build_candidate(research, engine=_CaptureEngine(), min_words=120, workdir=tmp_path / "out")

    assert captured_segs is not None and captured_edit is not None
    normal_scenes = [s for s in captured_edit.scenes if s.kind == "normal"]
    # One narration segment per normal scene, each placed at its scene's start offset.
    assert len(captured_segs) == len(normal_scenes) == 3
    for (wav, offset), scene in zip(captured_segs, normal_scenes):
        assert wav.suffix == ".wav"
        assert offset == scene.from_frame / 30
    # First normal scene starts after the 4s intro bookend (offset 4.0s).
    assert captured_segs[0][1] == 4.0
