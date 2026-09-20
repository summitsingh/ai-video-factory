"""Tests for the music provider fallback chain.

Covers music_providers (license policy, IA/Openverse/Freesound parsing with
a mocked HTTP layer) and the music_bed.resolve_music_track fallback order:
explicit path -> MUSIC_BED_PATH -> local mood library -> network providers
(IA -> Openverse -> Freesound) -> None (procedural drone).
"""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_video_factory import music_bed, music_providers
from ai_video_factory.music_providers import (
    MusicTrack,
    fetch_from_providers,
    normalize_commercial_license,
    search_freesound,
    search_internet_archive,
    search_openverse,
)


# ---------------------------------------------------------------------------
# License policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://creativecommons.org/licenses/by/4.0/", "by"),
    ("http://creativecommons.org/licenses/by/3.0/", "by"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
    ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
    ("https://creativecommons.org/publicdomain/mark/1.0/", "pdm"),
    ("https://creativecommons.org/licenses/by-nc/4.0/", None),
    ("https://creativecommons.org/licenses/by-nc-nd/4.0/", None),
    ("https://creativecommons.org/licenses/by-nd/4.0/", None),
    ("https://creativecommons.org/licenses/by-nc-sa/4.0/", None),
    ("http://creativecommons.org/licenses/by-nc-nd/2.5/", None),
    (None, None),
    ("", None),
    ("https://example.com/some-license", None),
])
def test_normalize_commercial_license(url, expected):
    assert normalize_commercial_license(url) == expected


# ---------------------------------------------------------------------------
# Internet Archive parsing (mocked HTTP)
# ---------------------------------------------------------------------------

_IA_DOCS = {
    "response": {"docs": [
        {"identifier": "good-track",
         "title": "Dark Space Drone",
         "creator": "Some Artist",
         "licenseurl": "https://creativecommons.org/licenses/by/4.0/"},
        {"identifier": "nc-track",
         "title": "Nice But NC",
         "creator": "Other",
         "licenseurl": "https://creativecommons.org/licenses/by-nc/4.0/"},
        {"identifier": "nolicense-track",
         "title": "No License Listed",
         "creator": "Anon"},
    ]}
}

_IA_META_GOOD = {
    "files": [
        {"name": "good-track_meta.sqlite", "bitrate": "0", "length": "600"},
        {"name": "good-track_128kb.mp3", "bitrate": "128", "length": "601.5"},
        {"name": "good-track_64kb.mp3", "bitrate": "64", "length": "601.5"},
        {"name": "short_128kb.mp3", "bitrate": "128", "length": "45"},
    ]
}


def _fake_ia_get_json(url, params=None):
    if "advancedsearch" in url:
        return _IA_DOCS
    if "/metadata/good-track" in url:
        return _IA_META_GOOD
    raise AssertionError(f"unexpected IA url: {url}")


def test_ia_search_skips_nc_and_picks_best_mp3():
    with patch.object(music_providers, "_get_json",
                      side_effect=_fake_ia_get_json):
        tracks = search_internet_archive("cosmic")
    assert len(tracks) == 1
    track = tracks[0]
    assert track.identifier == "good-track"
    assert track.license == "by"
    assert track.download_url.endswith("good-track_128kb.mp3")
    assert track.download_url.startswith(
        "https://archive.org/download/good-track/")
    assert track.duration_sec == pytest.approx(601.5)


def test_ia_search_skips_spoken_word():
    docs = {"response": {"docs": [
        {"identifier": "librivox-story",
         "title": "Danger in Deep Space (Dramatic Reading)",
         "creator": "LibriVox",
         "licenseurl": "https://creativecommons.org/publicdomain/mark/1.0/"},
    ]}}
    with patch.object(music_providers, "_get_json", return_value=docs):
        assert search_internet_archive("cosmic") == []


def test_ia_search_all_unusable_returns_empty():
    docs = {"response": {"docs": [
        {"identifier": "nc-only",
         "licenseurl": "https://creativecommons.org/licenses/by-nc/4.0/"},
    ]}}
    with patch.object(music_providers, "_get_json", return_value=docs):
        assert search_internet_archive("cosmic") == []


# ---------------------------------------------------------------------------
# Openverse parsing (mocked HTTP)
# ---------------------------------------------------------------------------

_OV_RESPONSE = {
    "results": [
        {"id": "1", "title": "Good Bed", "creator": "A",
         "license": "by", "duration": 240000,
         "url": "https://example.com/good.mp3",
         "foreign_landing_url": "https://example.com/good"},
        {"id": "2", "title": "ShareAlike", "creator": "B",
         "license": "by-sa", "duration": 240000,
         "url": "https://example.com/sa.mp3",
         "foreign_landing_url": "https://example.com/sa"},
        {"id": "3", "title": "Too Short", "creator": "C",
         "license": "cc0", "duration": 30000,
         "url": "https://example.com/short.mp3",
         "foreign_landing_url": "https://example.com/short"},
        {"id": "4", "title": "CC0 Long", "creator": "D",
         "license": "cc0", "duration": 300000,
         "url": "https://example.com/cc0.mp3",
         "foreign_landing_url": "https://example.com/cc0"},
    ]
}


def test_openverse_filters_license_and_duration():
    with patch.object(music_providers, "_get_json",
                      return_value=_OV_RESPONSE):
        tracks = search_openverse("calm")
    titles = [t.title for t in tracks]
    assert titles == ["Good Bed", "CC0 Long"]
    assert tracks[0].provider == "openverse"
    assert tracks[0].duration_sec == pytest.approx(240.0)


# ---------------------------------------------------------------------------
# Freesound gating
# ---------------------------------------------------------------------------

def test_freesound_skipped_without_key(monkeypatch):
    monkeypatch.delenv("FREESOUND_API_KEY", raising=False)
    assert search_freesound("tech") == []


# ---------------------------------------------------------------------------
# Provider orchestration order
# ---------------------------------------------------------------------------

def _track(provider, identifier="abc"):
    return MusicTrack(provider=provider, identifier=identifier,
                      title=f"{provider} track", artist="X", license="by",
                      download_url=f"https://example.com/{identifier}.mp3",
                      duration_sec=300.0, source_url="https://example.com/")


def test_fetch_tries_providers_in_order(tmp_path):
    ov_track = _track("openverse", "ov1")
    fs_track = _track("freesound", "fs1")
    with patch.object(music_providers, "search_internet_archive",
                      side_effect=RuntimeError("ia down")), \
         patch.object(music_providers, "search_openverse",
                      return_value=[ov_track]) as ov_search, \
         patch.object(music_providers, "search_freesound",
                      return_value=[fs_track]) as fs_search, \
         patch.object(music_providers, "_download",
                      side_effect=lambda url, dest: Path(dest).write_bytes(
                          b"x" * 60000)):
        dest, provider, track = fetch_from_providers("cosmic",
                                                     cache_dir=tmp_path)
    assert provider == "openverse"
    assert track is ov_track
    assert dest.name.startswith("cosmic_openverse_ov1")
    assert dest.is_file()
    fs_search.assert_not_called()
    assert ov_search.called


def test_fetch_skips_empty_provider_and_uses_next(tmp_path):
    fs_track = _track("freesound", "fs9")
    with patch.object(music_providers, "search_internet_archive",
                      return_value=[]), \
         patch.object(music_providers, "search_openverse",
                      return_value=[]), \
         patch.object(music_providers, "search_freesound",
                      return_value=[fs_track]), \
         patch.object(music_providers, "_download",
                      side_effect=lambda url, dest: Path(dest).write_bytes(
                          b"x" * 60000)):
        dest, provider, _ = fetch_from_providers("epic", cache_dir=tmp_path)
    assert provider == "freesound"
    assert dest.is_file()


def test_fetch_uses_cache_without_downloading(tmp_path):
    cached = tmp_path / "calm_openverse_cached1.mp3"
    cached.write_bytes(b"x" * 60000)
    track = _track("openverse", "cached1")
    with patch.object(music_providers, "search_internet_archive",
                      return_value=[]), \
         patch.object(music_providers, "search_openverse",
                      return_value=[track]), \
         patch.object(music_providers, "_download") as dl:
        dest, provider, _ = fetch_from_providers("calm", cache_dir=tmp_path)
    assert dest == cached
    dl.assert_not_called()


def test_fetch_all_fail_returns_nones(tmp_path):
    with patch.object(music_providers, "search_internet_archive",
                      side_effect=RuntimeError("nope")), \
         patch.object(music_providers, "search_openverse",
                      return_value=[]), \
         patch.object(music_providers, "search_freesound",
                      return_value=[]):
        assert fetch_from_providers("tech",
                                    cache_dir=tmp_path) == (None, None, None)


# ---------------------------------------------------------------------------
# resolve_music_track fallback chain
# ---------------------------------------------------------------------------

def _write(path: Path, size: int = 60000) -> Path:
    path.write_bytes(b"x" * size)
    return path


def test_resolve_explicit_path_wins(tmp_path, monkeypatch):
    explicit = _write(tmp_path / "explicit.mp3")
    env_track = _write(tmp_path / "env.mp3")
    monkeypatch.setenv("MUSIC_BED_PATH", str(env_track))
    assert music_bed.resolve_music_track(
        explicit, mood="cosmic", allow_network=False) == explicit


def test_resolve_env_before_library(tmp_path, monkeypatch):
    env_track = _write(tmp_path / "env.mp3")
    monkeypatch.setenv("MUSIC_BED_PATH", str(env_track))
    lib = tmp_path / "lib"
    lib.mkdir()
    _write(lib / "cosmic-bed.mp3")
    monkeypatch.setenv("MUSIC_BED_DIR", str(lib))
    assert music_bed.resolve_music_track(
        mood="cosmic", allow_network=False) == env_track


def test_resolve_library_before_network(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSIC_BED_PATH", raising=False)
    lib = tmp_path / "lib"
    lib.mkdir()
    lib_track = _write(lib / "mystery-bed.mp3")
    monkeypatch.setenv("MUSIC_BED_DIR", str(lib))
    with patch("ai_video_factory.music_providers.fetch_from_providers") as f:
        result = music_bed.resolve_music_track(topic="unsolved mystery",
                                               allow_network=True)
    assert result == lib_track
    f.assert_not_called()


def test_resolve_network_when_library_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSIC_BED_PATH", raising=False)
    lib = tmp_path / "lib"
    lib.mkdir()
    monkeypatch.setenv("MUSIC_BED_DIR", str(lib))
    net_track = _write(tmp_path / "net.mp3")
    with patch("ai_video_factory.music_providers.fetch_from_providers",
               return_value=(net_track, "openverse",
                             _track("openverse"))) as f:
        result = music_bed.resolve_music_track(topic="black holes",
                                               allow_network=True)
    assert result == net_track
    f.assert_called_once()
    assert f.call_args[0][0] == "cosmic"


def test_resolve_none_when_everything_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSIC_BED_PATH", raising=False)
    lib = tmp_path / "lib"
    lib.mkdir()
    monkeypatch.setenv("MUSIC_BED_DIR", str(lib))
    with patch("ai_video_factory.music_providers.fetch_from_providers",
               return_value=(None, None, None)):
        assert music_bed.resolve_music_track(topic="black holes",
                                             allow_network=True) is None
    assert music_bed.resolve_music_track(allow_network=False) is None


# ---------------------------------------------------------------------------
# Mood mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topic,expected", [
    ("The Fermi Paradox", "cosmic"),
    ("Black holes explained", "cosmic"),
    ("How AI will change medicine", "tech"),
    ("Startup ideas for 2026", "tech"),
    ("The Fall of Rome", "epic"),
    ("Ancient Egypt pyramids", "epic"),
    ("The unsolved disappearance of flight 19", "mystery"),
    ("True crime: the cold case", "mystery"),
    ("The deep ocean", "calm"),
    ("Why we dream", "calm"),
    ("The global warming crisis", "calm"),  # 'warming' must not hit 'war'
    ("Something completely random", "calm"),  # default mood
])
def test_mood_for_topic(topic, expected):
    assert music_bed.mood_for_topic(topic) == expected
