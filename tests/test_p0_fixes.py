"""Mocked unit tests for the P0 audit fixes.

Every HTTP call is mocked; no network, FFmpeg, Remotion, TTS, or
end-to-end render is exercised here.
"""

import io
import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_video_factory import research as research_module
from ai_video_factory.edit_schema import EditDocument, EditScene
from ai_video_factory.narration import _espeak_voice
from ai_video_factory.research import (
    ResearchError,
    get_trending_topics,
    research_trending_topics,
)
from ai_video_factory.run_store import RunStore
from ai_video_factory.script_generator import (
    ScriptGenerationError,
    _extract_json,
    generate_script_with_lm_studio,
)
from ai_video_factory.subtitle_export import (
    build_srt,
    distribute_cues,
    scene_cues,
    split_narration,
    subtitle_provenance,
)
from ai_video_factory.theme import (
    GENERIC_THEME,
    SPACE_THEME,
    ThemeConfig,
    load_theme_config,
    resolve_theme,
)
from ai_video_factory.video_pipeline import (
    _resolve_ffprobe,
    remotion_public_lock,
    slugify_topic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal context-manager stand-in for urllib responses."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_urlopen(monkeypatch, handler):
    monkeypatch.setattr(
        research_module.urllib.request, "urlopen", handler
    )
    # Skip the retry sleep so failure tests stay fast.
    monkeypatch.setattr(research_module.time, "sleep", lambda seconds: None)


def _reddit_payload(title="Webb spots oldest galaxy yet", score=5200):
    return json.dumps(
        {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": title,
                            "stickied": False,
                            "permalink": "/r/Documentaries/comments/x/test/",
                            "created_utc": datetime.now(timezone.utc).timestamp(),
                            "score": score,
                        }
                    }
                ]
            }
        }
    ).encode("utf-8")


def _gnews_payload(title="Breakthrough in fusion energy announced"):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<rss><channel>"
        f"<item><title>{title}</title>"
        "<link>https://news.example.com/fusion</link>"
        "<pubDate>Mon, 14 Sep 2026 08:00:00 GMT</pubDate>"
        "<source>Example News</source></item>"
        "</channel></rss>"
    ).encode("utf-8")


def _make_scene(scene_id="scene-0", narration=None, **kwargs):
    params = {
        "id": scene_id,
        "from_frame": 0,
        "duration_frames": 300,
        "title": "Test Scene",
        "caption": "Test caption",
        "kind": "normal",
    }
    params.update(kwargs)
    if narration is not None:
        params["narration"] = narration
    return EditScene(**params)


def _make_doc(scenes):
    return EditDocument(
        schema_version=1,
        width=1920,
        height=1080,
        fps=30,
        duration_frames=sum(s.duration_frames for s in scenes),
        scenes=scenes,
    )


# ---------------------------------------------------------------------------
# P0-2: real trend research (mocked HTTP)
# ---------------------------------------------------------------------------


def test_research_reddit_success(monkeypatch):
    def fake_urlopen(request, timeout=None):
        assert "reddit.com" in request.full_url
        return _FakeResponse(_reddit_payload())

    _patch_urlopen(monkeypatch, fake_urlopen)
    topics = get_trending_topics(
        max_topics=1, min_engagement=100, trend_source="reddit"
    )
    assert len(topics) == 1
    assert topics[0].title == "Webb spots oldest galaxy yet"
    assert topics[0].source == "reddit"
    assert topics[0].url.startswith("https://www.reddit.com")


def test_research_gnews_success(monkeypatch):
    def fake_urlopen(request, timeout=None):
        assert "news.google.com" in request.full_url
        return _FakeResponse(_gnews_payload())

    _patch_urlopen(monkeypatch, fake_urlopen)
    topics = get_trending_topics(
        max_topics=1, min_engagement=100, trend_source="gnews"
    )
    assert len(topics) == 1
    assert topics[0].title == "Breakthrough in fusion energy announced"
    assert topics[0].source == "google_news"


def test_research_all_providers_fail_raises(monkeypatch):
    monkeypatch.delenv("AVF_ALLOW_SYNTHETIC_RESEARCH", raising=False)

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    _patch_urlopen(monkeypatch, fake_urlopen)
    with pytest.raises(ResearchError, match="all trend providers failed"):
        get_trending_topics(max_topics=1, min_engagement=100, trend_source="all")


def test_research_synthetic_fallback_requires_opt_in(monkeypatch):
    monkeypatch.setenv("AVF_ALLOW_SYNTHETIC_RESEARCH", "1")

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    _patch_urlopen(monkeypatch, fake_urlopen)
    result = research_trending_topics(max_topics=2, min_engagement=100)
    assert result.synthetic is True
    assert len(result.topics) == 2
    assert "synthetic" in result.methodology


def test_research_respects_min_engagement(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_reddit_payload(score=50))

    _patch_urlopen(monkeypatch, fake_urlopen)
    with pytest.raises(ResearchError, match="no usable topics"):
        # score 50 < min_engagement 100 -> filtered out -> empty
        from ai_video_factory.research import pick_trending_topic

        pick_trending_topic(max_topics=1, min_engagement=100, trend_source="reddit")


# ---------------------------------------------------------------------------
# P0-3: fail-loud script generation
# ---------------------------------------------------------------------------


def _lm_studio_payload(script_dict):
    return json.dumps(
        {
            "choices": [
                {"message": {"content": json.dumps(script_dict)}}
            ]
        }
    ).encode("utf-8")


def _lm_studio_payload_text(content: str):
    """Raw LM Studio response envelope with literal message content."""
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")


def _valid_script_dict():
    return {
        "title": "Test",
        "narration": "Hello world.",
        "scenes": [
            {
                "id": "scene-0",
                "title": "Intro",
                "caption": "cap",
                "narration": "Hello world.",
                "duration_frames": 90,
            }
        ],
        "sources": ["https://example.com"],
        "captions": [],
    }


def test_extract_json_handles_fenced_response():
    fenced = "```json\n" + json.dumps(_valid_script_dict()) + "\n```"
    assert _extract_json(fenced)["title"] == "Test"


def test_extract_json_handles_surrounding_text():
    wrapped = "Here is your script:\n" + json.dumps(_valid_script_dict()) + "\nDone."
    assert _extract_json(wrapped)["title"] == "Test"


def test_script_generation_raises_on_http_failure(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("LM Studio not running")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ScriptGenerationError):
        generate_script_with_lm_studio("T", "D", "https://example.com")


def test_script_generation_raises_on_invalid_shape(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_lm_studio_payload({"nope": "bad shape"}))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ScriptGenerationError):
        generate_script_with_lm_studio("T", "D", "https://example.com")


def test_script_generation_explicit_fallback_covers_parse_errors(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_lm_studio_payload({"nope": "bad shape"}))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    fallback = generate_script_with_lm_studio(
        "T", "D", "https://example.com", allow_fallback=True
    )
    assert fallback["title"].startswith("Trending:")


def test_script_generation_valid_response(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_lm_studio_payload(_valid_script_dict()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    parsed = generate_script_with_lm_studio("T", "D", "https://example.com")
    assert parsed["title"] == "Test"
    assert len(parsed["scenes"]) == 1


# ---------------------------------------------------------------------------
# P0-8: topic slugging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,expected",
    [
        ("Webb Telescope's Deepest Field!", "webb-telescope-s-deepest-field"),
        ("  Multiple   Spaces  ", "multiple-spaces"),
        ("UPPER CASE", "upper-case"),
        ("", "untitled"),
        ("!!!", "untitled"),
        ("a" * 100, "a" * 60),
        ("../traversal", "traversal"),
    ],
)
def test_slugify_topic(topic, expected):
    assert slugify_topic(topic) == expected


def test_slugify_topic_no_path_separators():
    slug = slugify_topic("a/b\\c:d*e?f\"g<h>i|j")
    assert "/" not in slug and "\\" not in slug


# ---------------------------------------------------------------------------
# P0-6: theme resolution
# ---------------------------------------------------------------------------


def test_default_theme_is_space():
    theme = resolve_theme()
    assert theme.name == "space"
    assert theme is SPACE_THEME


def test_space_theme_preserves_original_values():
    assert SPACE_THEME.outro_caption == "The Eyes That See Everything"
    assert "golden eye" in SPACE_THEME.thumbnail_power_words
    assert "telescope" in SPACE_THEME.nasa_stop_words


def test_generic_theme_differs():
    theme = resolve_theme("generic")
    assert theme is GENERIC_THEME
    assert theme.outro_caption != SPACE_THEME.outro_caption


def test_resolve_theme_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown theme"):
        resolve_theme("nope")


def test_load_theme_from_json(tmp_path):
    path = tmp_path / "theme.json"
    path.write_text(
        json.dumps(
            {
                "name": "custom",
                "outro_caption": "Thanks for watching",
                "thumbnail_power_words": ["wow"],
                "nasa_stop_words": ["the"],
                "channel_name": "Test",
            }
        )
    )
    theme = load_theme_config(path)
    assert theme.name == "custom"
    assert theme.outro_caption == "Thanks for watching"


def test_theme_json_overrides_builtin_name(tmp_path):
    path = tmp_path / "theme.json"
    path.write_text(
        json.dumps(
            {
                "name": "custom",
                "outro_caption": "Thanks for watching",
                "thumbnail_power_words": ["wow"],
                "nasa_stop_words": ["the"],
                "channel_name": "Test",
            }
        )
    )
    assert resolve_theme("space", path).name == "custom"


def test_theme_config_rejects_bad_values():
    with pytest.raises(Exception):
        ThemeConfig(
            name="",
            outro_caption="x",
            thumbnail_power_words=["x"],
            nasa_stop_words=["x"],
            channel_name="x",
        )


# ---------------------------------------------------------------------------
# P0-7: espeak voice resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested,expected",
    [
        ("af_heart", "en-us"),  # Kokoro default -> espeak default
        ("bf_emma", "en-us"),  # Kokoro voice -> espeak default
        ("am_adam", "en-us"),
        ("", "en-us"),  # empty -> espeak default
        (None, "en-us"),
        ("en-us", "en-us"),  # explicit valid espeak voice preserved
        ("en+f3", "en+f3"),
        ("english", "english"),
    ],
)
def test_espeak_voice_resolution(requested, expected):
    assert _espeak_voice(requested) == expected


# ---------------------------------------------------------------------------
# P0-5: ffprobe derivation
# ---------------------------------------------------------------------------


def test_resolve_ffprobe_sibling_of_ffmpeg(tmp_path):
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffmpeg.write_text("#!/bin/sh\n")
    ffprobe.write_text("#!/bin/sh\n")
    ffmpeg.chmod(0o755)
    ffprobe.chmod(0o755)
    assert _resolve_ffprobe(str(ffmpeg)) == str(ffprobe)


def test_resolve_ffprobe_windows_exe_suffix(tmp_path, monkeypatch):
    import ai_video_factory.video_pipeline as video_pipeline

    monkeypatch.setattr(video_pipeline, "_FFPROBE_EXE", "ffprobe.exe")
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    ffmpeg.write_text("x")
    ffprobe.write_text("x")
    ffprobe.chmod(0o755)
    assert _resolve_ffprobe(str(ffmpeg)) == str(ffprobe)


def test_resolve_ffprobe_falls_back_to_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "shutil.which", lambda name: str(tmp_path / "ffprobe") if name == "ffprobe" else None
    )
    assert _resolve_ffprobe("/nonexistent/ffmpeg") == str(tmp_path / "ffprobe")


def test_resolve_ffprobe_bare_name_last_resort(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert _resolve_ffprobe("/nonexistent/ffmpeg") == "ffprobe"


# ---------------------------------------------------------------------------
# P0-11: narration-based subtitles
# ---------------------------------------------------------------------------


def test_split_narration_sentences():
    chunks = split_narration("First sentence. Second sentence! Third?")
    assert chunks == ["First sentence.", "Second sentence!", "Third?"]


def test_split_narration_wraps_long_sentences():
    long_sentence = " ".join(f"word{i}" for i in range(40)) + "."
    chunks = split_narration(long_sentence)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 14 for c in chunks)


def test_distribute_cues_covers_duration():
    cues = distribute_cues(["a", "b", "c"], 0.0, 9.0)
    assert len(cues) == 3
    assert cues[0][0] == 0.0
    assert cues[-1][1] == pytest.approx(9.0)
    # sequential, non-overlapping
    for (_, end_a, _), (start_b, _, _) in zip(cues, cues[1:]):
        assert start_b >= end_a


def test_scene_cues_prefer_narration():
    scene = _make_scene(
        narration="The cosmos is vast. It holds ancient light.",
        subtitle="BURNED IN TITLE",
        caption="Fallback caption",
    )
    cues, used_title = scene_cues(scene, fps=30.0)
    assert used_title is False
    assert len(cues) == 2
    assert all("BURNED IN TITLE" not in text for _, _, text in cues)
    assert cues[0][2] == "The cosmos is vast."


def test_scene_cues_fall_back_to_title():
    scene = _make_scene(subtitle="On-screen Title", caption="Caption text")
    cues, used_title = scene_cues(scene, fps=30.0)
    assert used_title is True
    assert len(cues) == 1
    assert cues[0][2] == "On-screen Title"


def test_build_srt_uses_narration_not_titles():
    scenes = [
        _make_scene("scene-0", narration="Spoken line one. Spoken line two.", subtitle="TITLE A"),
        _make_scene("scene-1", narration="Another spoken line.", subtitle="TITLE B"),
    ]
    scenes[1] = EditScene(
        **{**scenes[1].model_dump(), "from_frame": 300, "duration_frames": 300}
    )
    doc = _make_doc(scenes)
    srt = build_srt(doc)
    assert "Spoken line one." in srt
    assert "TITLE A" not in srt
    assert "TITLE B" not in srt


def test_subtitle_provenance_flags_title_fallback():
    scenes = [
        _make_scene("scene-0", narration="Spoken words here."),
        _make_scene("scene-1", subtitle="Title Only Scene"),
    ]
    scenes[1] = EditScene(
        **{**scenes[1].model_dump(), "from_frame": 300, "duration_frames": 300}
    )
    doc = _make_doc(scenes)
    provenance = subtitle_provenance(doc)
    assert provenance["captions_from_titles"] is True
    assert provenance["title_cues"] == 1
    assert provenance["narration_cues"] >= 1


def test_subtitle_provenance_clean_when_all_narrated():
    scenes = [
        _make_scene("scene-0", narration="Spoken words here."),
        _make_scene("scene-1", narration="More spoken words."),
    ]
    scenes[1] = EditScene(
        **{**scenes[1].model_dump(), "from_frame": 300, "duration_frames": 300}
    )
    doc = _make_doc(scenes)
    assert subtitle_provenance(doc)["captions_from_titles"] is False


def test_intro_outro_excluded_from_subtitles():
    scenes = [
        _make_scene("intro", narration="Intro narration.", kind="intro"),
        _make_scene("scene-0", narration="Body narration."),
        _make_scene("outro", narration="Outro narration.", kind="outro"),
    ]
    doc = _make_doc(scenes)
    srt = build_srt(doc)
    assert "Body narration." in srt
    assert "Intro narration." not in srt
    assert "Outro narration." not in srt


# ---------------------------------------------------------------------------
# P0-10: RunStore TTL / heartbeat / stale reaping
# ---------------------------------------------------------------------------


def _manifest_path(store: RunStore, run_id: str, stage: str = "video-render") -> Path:
    return store.root / stage / run_id / "manifest.json"


def _write_legacy_manifest(root: Path, run_id: str, created_at: str):
    manifest_dir = root / "video-render" / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    # Legacy shape: no started_at / last_heartbeat_at / ttl_seconds fields.
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "stage": "video-render",
                "input_fingerprint": "abc",
                "status": "running",
                "resumed": False,
                "created_at": created_at,
                "updated_at": created_at,
                "inputs": {},
                "artifacts": {},
                "artifact_integrity": {},
                "error": None,
            }
        )
    )
    return manifest_path


def test_heartbeat_refreshes_timestamp(tmp_path):
    store = RunStore(tmp_path / "state", artifact_root=tmp_path / "runs")
    run = store.start("video-render", {"x": 1})
    first = run.last_heartbeat_at
    time.sleep(0.01)
    refreshed = store.heartbeat(run.run_id)
    assert refreshed.last_heartbeat_at >= first


def test_reap_stale_runs_marks_superseded(tmp_path):
    store = RunStore(
        tmp_path / "state", artifact_root=tmp_path / "runs", default_ttl_seconds=60
    )
    run = store.start("video-render", {"x": 1})
    # Backdate the heartbeat beyond the TTL.
    manifest_path = _manifest_path(store, run.run_id)
    data = json.loads(manifest_path.read_text())
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    data["last_heartbeat_at"] = old
    data["started_at"] = old
    manifest_path.write_text(json.dumps(data))
    reaped = store.reap_stale_runs()
    assert run.run_id in reaped
    assert store._load_run(run.run_id).status.value == "superseded"


def test_reap_ignores_fresh_runs(tmp_path):
    store = RunStore(
        tmp_path / "state", artifact_root=tmp_path / "runs", default_ttl_seconds=3600
    )
    run = store.start("video-render", {"x": 1})
    assert store.reap_stale_runs() == []
    assert store._load_run(run.run_id).status.value == "running"


def test_reap_legacy_manifest_without_heartbeat(tmp_path):
    store = RunStore(
        tmp_path / "state", artifact_root=tmp_path / "runs", default_ttl_seconds=60
    )
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    legacy_id = "a" * 32  # hex run ids only
    _write_legacy_manifest(tmp_path / "state", legacy_id, old)
    reaped = store.reap_stale_runs()
    assert legacy_id in reaped


def test_start_records_ttl(tmp_path):
    store = RunStore(tmp_path / "state", artifact_root=tmp_path / "runs")
    run = store.start("video-render", {}, ttl_seconds=123)
    assert run.ttl_seconds == 123
    assert run.started_at is not None
    assert run.last_heartbeat_at is not None


# ---------------------------------------------------------------------------
# P0-9: remotion/public lock
# ---------------------------------------------------------------------------


def test_public_lock_serializes_access(tmp_path):
    order = []

    def worker(name, hold):
        with remotion_public_lock(tmp_path):
            order.append(f"{name}-enter")
            time.sleep(hold)
            order.append(f"{name}-exit")

    threads = [
        threading.Thread(target=worker, args=("a", 0.2)),
        threading.Thread(target=worker, args=("b", 0.0)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    # No interleaving: one thread's enter/exit pair is never split.
    assert order.index("a-enter") < order.index("a-exit")
    assert order.index("b-enter") < order.index("b-exit")
    flat = "".join(order)
    assert "a-entera-exit" in flat or "b-enterb-exit" in flat
    assert (tmp_path / ".avf-public.lock").is_file()


def test_public_lock_releasable_and_reacquirable(tmp_path):
    with remotion_public_lock(tmp_path):
        pass
    with remotion_public_lock(tmp_path):
        pass


def test_script_generation_retries_empty_thinking_output(monkeypatch):
    """A reasoning model returning empty content once must be retried."""
    calls: list[int] = []

    def fake_urlopen(request, timeout=None):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append(payload["max_tokens"])
        if len(calls) == 1:
            return _FakeResponse(_lm_studio_payload_text(""))
        return _FakeResponse(_lm_studio_payload(_valid_script_dict()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    parsed = generate_script_with_lm_studio("T", "D", "https://example.com")
    assert parsed["title"] == "Test"
    assert len(calls) == 2
    assert calls[1] == calls[0] * 2  # retry doubles the token budget


def test_script_generation_retry_exhaustion_fails_loud(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_lm_studio_payload_text(""))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ScriptGenerationError, match="2 attempts"):
        generate_script_with_lm_studio("T", "D", "https://example.com")
