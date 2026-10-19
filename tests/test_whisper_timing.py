"""Tests for whisper_timing (word-level karaoke timing) and its wiring into
subtitle_export.build_ass_karaoke."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from ai_video_factory.edit_schema import EditDocument, EditScene
from ai_video_factory.subtitle_export import build_ass_karaoke, write_ass_karaoke
from ai_video_factory import whisper_timing
from ai_video_factory.whisper_timing import (
    SegmentTiming,
    Transcription,
    WordTiming,
    align_words_to_cues,
    detect_speech_intervals,
    extract_audio,
    karaoke_timings_for_cues,
    load_transcription,
    save_transcription,
    segments_to_srt,
    tighten_bounds,
    transcribe_media,
)

FPS = 30


def _scene(
    scene_id: str = "s1",
    from_frame: int = 0,
    duration_frames: int = 90,
    narration: str = "Hello world test.",
) -> EditScene:
    return EditScene(
        id=scene_id,
        from_frame=from_frame,
        duration_frames=duration_frames,
        title="T",
        caption="C",
        kind="normal",
        narration=narration,
    )


def _doc() -> EditDocument:
    return EditDocument(
        schema_version=1,
        width=1280,
        height=720,
        fps=FPS,
        duration_frames=90,
        scenes=[_scene()],
    )


def _make_tone_silence_tone(path: Path) -> Path:
    """2s tone, 1s silence, 2s tone -> 5s wav (speech-like envelope)."""
    argv = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=duration=1:r=16000",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-filter_complex",
        "[0:a][1:a][2:a]concat=n=3:v=0:a=1",
        str(path),
    ]
    subprocess.run(argv, check=True, timeout=120)
    return path


# ---------------------------------------------------------------------------
# tighten_bounds
# ---------------------------------------------------------------------------


def test_tighten_bounds_moves_past_pause() -> None:
    speech = [(1.0, 5.0)]
    assert tighten_bounds(0.0, 6.0, speech) == (1.0, 5.0)


def test_tighten_bounds_ignores_short_pause() -> None:
    speech = [(0.1, 5.0)]
    # 0.1s leading gap < min_pause 0.15 -> start stays
    assert tighten_bounds(0.0, 5.0, speech) == (0.0, 5.0)


def test_tighten_bounds_no_speech_unchanged() -> None:
    assert tighten_bounds(0.5, 2.5, []) == (0.5, 2.5)
    assert tighten_bounds(None, 2.5, [(0.0, 3.0)]) == (None, 2.5)


def test_tighten_bounds_custom_min_pause() -> None:
    speech = [(0.5, 5.0)]
    assert tighten_bounds(0.0, 5.0, speech, min_pause=0.6) == (0.0, 5.0)
    assert tighten_bounds(0.0, 5.0, speech, min_pause=0.4) == (0.5, 5.0)


# ---------------------------------------------------------------------------
# segments_to_srt
# ---------------------------------------------------------------------------


def test_segments_to_srt_format() -> None:
    segments = [
        SegmentTiming(start=1.5, end=3.0, text="Hello world"),
        SegmentTiming(start=4.0, end=5.25, text="Second cue"),
        {"start": None, "end": 6.0, "text": "skipped"},  # type: ignore[dict-item]
    ]
    srt = segments_to_srt(segments)
    assert "1\n00:00:01,500 --> 00:00:03,000\nHello world\n" in srt
    assert "2\n00:00:04,000 --> 00:00:05,250\nSecond cue\n" in srt
    assert "skipped" not in srt


# ---------------------------------------------------------------------------
# detect_speech_intervals (ffmpeg)
# ---------------------------------------------------------------------------


def test_detect_speech_intervals_tone_silence_tone(tmp_path: Path) -> None:
    wav = _make_tone_silence_tone(tmp_path / "in.wav")
    speech = detect_speech_intervals(wav)
    assert len(speech) == 2
    first, second = speech
    assert first[0] == pytest.approx(0.0, abs=0.15)
    assert first[1] == pytest.approx(2.0, abs=0.15)
    assert second[0] == pytest.approx(3.0, abs=0.15)
    assert second[1] == pytest.approx(5.0, abs=0.15)


def test_detect_speech_intervals_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        detect_speech_intervals(tmp_path / "nope.wav")


# ---------------------------------------------------------------------------
# extract_audio
# ---------------------------------------------------------------------------


def test_extract_audio_produces_mono_16k_wav(tmp_path: Path) -> None:
    src = _make_tone_silence_tone(tmp_path / "src.wav")
    dest = tmp_path / "out.wav"
    result = extract_audio(src, dest)
    assert result == dest
    header = dest.read_bytes()[:16]
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"


def test_extract_audio_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "missing.mp4", tmp_path / "out.wav")


# ---------------------------------------------------------------------------
# align_words_to_cues
# ---------------------------------------------------------------------------


def test_align_words_to_cues_exact_match() -> None:
    cues = [(0.0, 3.0, "Hello world")]
    words = [
        WordTiming("hello", 0.5, 0.9),
        WordTiming("world", 1.0, 1.6),
    ]
    aligned = align_words_to_cues(cues, words)
    assert aligned == [[(0.5, 0.9, "Hello"), (1.0, 1.6, "world")]]


def test_align_words_to_cues_case_and_punct_insensitive() -> None:
    cues = [(0.0, 3.0, "Hello, WORLD!")]
    words = [
        WordTiming("HELLO", 0.5, 0.9),
        WordTiming("world.", 1.0, 1.6),
    ]
    aligned = align_words_to_cues(cues, words)
    # Script text kept verbatim; timings from whisper.
    assert aligned == [[(0.5, 0.9, "Hello,"), (1.0, 1.6, "WORLD!")]]


def test_align_words_to_cues_interpolates_missing() -> None:
    cues = [(0.0, 4.0, "one two three four")]
    words = [
        WordTiming("one", 0.2, 0.6),
        WordTiming("four", 3.0, 3.5),
    ]
    aligned = align_words_to_cues(cues, words)
    assert len(aligned) == 1
    timed = aligned[0]
    assert timed[0] == (0.2, 0.6, "one")
    assert timed[3] == (3.0, 3.5, "four")
    # "two" and "three" fill the gap evenly, tiled with no overlap.
    assert timed[1][0] == pytest.approx(0.6)
    assert timed[2][1] == pytest.approx(3.0)
    assert timed[1][1] == pytest.approx(timed[2][0])
    assert timed[1][2] == "two"
    assert timed[2][2] == "three"


def test_align_words_to_cues_consumes_in_order() -> None:
    cues = [(0.0, 2.0, "go"), (2.0, 4.0, "go")]
    words = [WordTiming("go", 0.5, 0.8), WordTiming("go", 2.5, 2.9)]
    aligned = align_words_to_cues(cues, words)
    assert aligned[0] == [(0.5, 0.8, "go")]
    assert aligned[1] == [(2.5, 2.9, "go")]


def test_karaoke_timings_for_cues_flattens_segments() -> None:
    transcription = Transcription(
        text="hi there",
        language="en",
        segments=[
            SegmentTiming(
                start=0.0,
                end=2.0,
                text="hi there",
                words=[WordTiming("hi", 0.1, 0.4), WordTiming("there", 0.5, 1.0)],
            )
        ],
    )
    aligned = karaoke_timings_for_cues([(0.0, 2.0, "hi there")], transcription)
    assert aligned == [[(0.1, 0.4, "hi"), (0.5, 1.0, "there")]]


# ---------------------------------------------------------------------------
# transcribe_media error paths (no model download in unit tests)
# ---------------------------------------------------------------------------


def test_transcribe_media_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        transcribe_media(tmp_path / "missing.wav")


def test_transcribe_media_no_backend_errors_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = _make_tone_silence_tone(tmp_path / "in.wav")
    monkeypatch.setattr(whisper_timing, "whisper_available", lambda: False)
    monkeypatch.setattr(whisper_timing, "default_whisper_python", lambda: None)
    with pytest.raises(RuntimeError, match="openai-whisper is not importable"):
        transcribe_media(wav, detect_speech=False)


# ---------------------------------------------------------------------------
# JSON sidecar roundtrip
# ---------------------------------------------------------------------------


def test_save_load_transcription_roundtrip(tmp_path: Path) -> None:
    transcription = Transcription(
        text="hello world",
        language="en",
        segments=[
            SegmentTiming(
                start=0.5,
                end=1.6,
                text="hello world",
                words=[
                    WordTiming("hello", 0.5, 0.9, 0.98),
                    WordTiming("world", 1.0, 1.6, 0.97),
                ],
            )
        ],
        speech_intervals=[(0.4, 1.7)],
        model_name="medium",
    )
    path = tmp_path / "t.json"
    save_transcription(transcription, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "medium"
    assert payload["segments"][0]["words"][0]["probability"] == 0.98
    loaded = load_transcription(path)
    assert loaded == transcription


# ---------------------------------------------------------------------------
# subtitle_export wiring: build_ass_karaoke(word_timings=...)
# ---------------------------------------------------------------------------


_DIALOGUE_RE = re.compile(r"\{\\k(\d+)\}")


def test_build_ass_karaoke_uses_exact_word_timings() -> None:
    doc = _doc()
    word_timings = [[(0.0, 0.3, "Hello"), (0.3, 0.7, "world"), (0.7, 1.0, "test.")]]
    ass = build_ass_karaoke(doc, word_timings=word_timings)
    tags = _DIALOGUE_RE.findall(ass)
    # Exact centisecond durations from whisper timings, not the heuristic.
    assert tags == ["30", "40", "30"]


def test_build_ass_karaoke_falls_back_to_heuristic() -> None:
    doc = _doc()
    ass_plain = build_ass_karaoke(doc)
    ass_empty_entry = build_ass_karaoke(doc, word_timings=[[]])
    assert ass_plain == ass_empty_entry
    assert _DIALOGUE_RE.search(ass_plain)


def test_write_ass_karaoke_accepts_word_timings(tmp_path: Path) -> None:
    doc = _doc()
    word_timings = [[(0.0, 0.5, "Hello"), (0.5, 1.0, "world"), (1.0, 1.5, "test.")]]
    paths = write_ass_karaoke(doc, tmp_path, word_timings=word_timings)
    content = paths["ass"].read_text(encoding="utf-8")
    assert _DIALOGUE_RE.findall(content) == ["50", "50", "50"]
