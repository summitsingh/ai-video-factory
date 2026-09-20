"""Unit tests for ai_video_factory.music_bed (item #8).

Fast, hermetic tests: no network, no ffmpeg invocations, no audio fixtures.
The full duck + mix + loudnorm chain is exercised by the synthetic 20s test
kept outside the repo (see the item-8 implementation notes).
"""

import os

import pytest

from ai_video_factory import music_bed
from ai_video_factory.music_bed import (
    MUSIC_BED_PATH_ENV,
    _parse_loudnorm_json,
    duck_filter,
    resolve_music_track,
)


def test_duck_filter_wires_sidechain_labels() -> None:
    filt = duck_filter("3:a", "1:a", "ducked")
    assert "[3:a]" in filt
    assert "[1:a]" in filt
    assert filt.endswith("[ducked]")
    assert "sidechaincompress" in filt


def test_resolve_music_track_prefers_explicit_path(tmp_path, monkeypatch) -> None:
    track = tmp_path / "bed.mp3"
    track.write_bytes(b"fake")
    monkeypatch.setenv(MUSIC_BED_PATH_ENV, str(tmp_path / "missing.mp3"))
    assert resolve_music_track(track) == track


def test_resolve_music_track_reads_env(tmp_path, monkeypatch) -> None:
    track = tmp_path / "bed.wav"
    track.write_bytes(b"fake")
    monkeypatch.setenv(MUSIC_BED_PATH_ENV, str(track))
    assert resolve_music_track() == track


def test_resolve_music_track_missing_file_falls_back(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(MUSIC_BED_PATH_ENV, str(tmp_path / "nope.mp3"))
    assert resolve_music_track() is None


def test_resolve_music_track_no_env(monkeypatch) -> None:
    monkeypatch.delenv(MUSIC_BED_PATH_ENV, raising=False)
    assert resolve_music_track() is None


def test_parse_loudnorm_json_extracts_measured_values() -> None:
    stderr = (
        "[Parsed_loudnorm_0 @ 0x1234] {\n"
        '\t"input_i" : "-23.45",\n'
        '\t"input_tp" : "-3.10",\n'
        '\t"input_lra" : "5.20",\n'
        '\t"input_thresh" : "-34.10",\n'
        '\t"output_i" : "-23.00",\n'
        '\t"target_offset" : "0.44"\n'
        "}\n"
    )
    parsed = _parse_loudnorm_json(stderr)
    assert parsed is not None
    assert parsed["input_i"] == "-23.45"
    assert parsed["target_offset"] == "0.44"


def test_parse_loudnorm_json_rejects_garbage() -> None:
    assert _parse_loudnorm_json("no json here") is None
    assert _parse_loudnorm_json('{"input_i": "-20"}') is None


def test_fit_track_rejects_missing_source(tmp_path) -> None:
    with pytest.raises(music_bed.MusicBedError):
        music_bed.fit_track_to_duration(
            tmp_path / "missing.wav", 10.0, tmp_path / "out.wav"
        )
