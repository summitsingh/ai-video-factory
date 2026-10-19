"""Tests for envelope ducking (WanGP Q5): speech_detect + music_bed envelope mode.

Speech fixtures are synthesized with the stdlib `wave` module (deterministic
voiced-burst "talk" segments and silence gaps); mix/render tests shell out
to ffmpeg via music_bed and scripts/ab_ducking.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from ai_video_factory import music_bed
from ai_video_factory.music_bed import (
    duck_envelope_filter,
    duck_envelope_points,
    envelope_gain_expression,
    merge_duck_gaps,
    mix_ducked_audio,
)
from ai_video_factory.speech_detect import speech_intervals

REPO_ROOT = Path(__file__).resolve().parents[1]
AB_SCRIPT = REPO_ROOT / "scripts" / "ab_ducking.py"

NARR_SR = 16000
MUSIC_SR = 48000


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def _burst(dur_s: float, sr: int = NARR_SR) -> np.ndarray:
    """Voiced-burst 'talk': harmonics with syllabic amplitude wobble."""
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    return (
        0.40 * np.sin(2 * np.pi * 220 * t)
        + 0.25 * np.sin(2 * np.pi * 440 * t)
        + 0.15 * np.sin(2 * np.pi * 660 * t)
    ) * am


def _make_narration(path: Path, pattern: list[tuple[str, float]]) -> Path:
    """pattern: list of ('talk'|'gap', seconds)."""
    sig = np.concatenate(
        [
            _burst(dur) if kind == "talk" else np.zeros(int(dur * NARR_SR))
            for kind, dur in pattern
        ]
    )
    return _write_wav(path, sig, NARR_SR)


def _make_music(path: Path, dur_s: float = 6.0) -> Path:
    n = int(dur_s * MUSIC_SR)
    t = np.arange(n) / MUSIC_SR
    sig = 0.30 * np.sin(2 * np.pi * 110 * t) + 0.18 * np.sin(2 * np.pi * 165 * t)
    return _write_wav(path, sig, MUSIC_SR)


@pytest.fixture()
def narration_bursts(tmp_path: Path) -> Path:
    return _make_narration(
        tmp_path / "narr.wav",
        [("talk", 1.0), ("gap", 1.0), ("talk", 1.0), ("gap", 1.0), ("talk", 1.0)],
    )


@pytest.fixture()
def narration_long(tmp_path: Path) -> Path:
    return _make_narration(
        tmp_path / "narr_long.wav", [("talk", 2.0), ("gap", 2.0), ("talk", 2.0)]
    )


@pytest.fixture()
def music_bed_wav(tmp_path: Path) -> Path:
    return _make_music(tmp_path / "music.wav")


# --- speech_intervals -------------------------------------------------------


def test_speech_intervals_detects_burst_pattern(narration_bursts: Path) -> None:
    spans = speech_intervals(narration_bursts, isolated=True)
    assert len(spans) == 3
    for (start, end), expected_start in zip(spans, (0.0, 2.0, 4.0)):
        assert abs(start - expected_start) < 0.12
        assert abs(end - (expected_start + 1.0)) < 0.12


def test_speech_intervals_auto_detects_isolated_voice(narration_bursts: Path) -> None:
    # Generous silence (40%) -> isolated-voice mode kicks in automatically.
    spans = speech_intervals(narration_bursts)
    assert len(spans) == 3


def test_speech_intervals_silence_returns_empty(tmp_path: Path) -> None:
    silent = _write_wav(tmp_path / "sil.wav", np.zeros(2 * NARR_SR), NARR_SR)
    assert speech_intervals(silent) == []


def test_speech_intervals_merges_short_gaps(tmp_path: Path) -> None:
    tight = _make_narration(
        tmp_path / "tight.wav", [("talk", 1.0), ("gap", 0.10), ("talk", 1.0)]
    )
    assert len(speech_intervals(tight, isolated=True)) == 1
    wide = _make_narration(
        tmp_path / "wide.wav", [("talk", 1.0), ("gap", 0.50), ("talk", 1.0)]
    )
    assert len(speech_intervals(wide, isolated=True)) == 2


def test_speech_intervals_drops_sub_min_speech_blips(tmp_path: Path) -> None:
    for blip in (0.02, 0.03):
        path = _make_narration(
            tmp_path / f"blip{blip}.wav", [("gap", 0.5), ("talk", blip), ("gap", 0.5)]
        )
        assert speech_intervals(path, isolated=True) == []


def test_speech_intervals_mixture_mode_over_music_bed(tmp_path: Path) -> None:
    # Continuous -37 dBFS bed under the bursts: mixture mode, not isolated.
    n = 6 * NARR_SR
    t = np.arange(n) / NARR_SR
    bed = 0.02 * np.sin(2 * np.pi * 330 * t)
    sig = bed.copy()
    sig[1 * NARR_SR : 2 * NARR_SR] += _burst(1.0)
    sig[4 * NARR_SR : 5 * NARR_SR] += _burst(1.0)
    mixed = _write_wav(tmp_path / "mixed.wav", sig, NARR_SR)
    spans = speech_intervals(mixed)
    assert len(spans) == 2
    assert abs(spans[0][0] - 1.0) < 0.15
    assert abs(spans[1][0] - 4.0) < 0.15
    bed_only = _write_wav(tmp_path / "bed.wav", bed, NARR_SR)
    assert speech_intervals(bed_only) == []


def test_speech_intervals_rejects_missing_file(tmp_path: Path) -> None:
    from ai_video_factory.speech_detect import SpeechDetectError

    with pytest.raises(SpeechDetectError):
        speech_intervals(tmp_path / "nope.wav")


# --- envelope math (pure unit tests, no ffmpeg) ------------------------------


def test_merge_duck_gaps() -> None:
    # Pause 0.2 s < attack+release 0.8 s -> merged; pause 1.0 s stays split.
    assert merge_duck_gaps([(0.0, 1.0), (1.2, 2.0)], 0.2, 0.6) == [(0.0, 2.0)]
    assert merge_duck_gaps([(0.0, 1.0), (2.0, 3.0)], 0.2, 0.6) == [
        (0.0, 1.0),
        (2.0, 3.0),
    ]
    # Unsorted input and degenerate spans are handled.
    assert merge_duck_gaps([(2.0, 3.0), (0.0, 1.0), (5.0, 5.0)], 0.2, 0.6) == [
        (0.0, 1.0),
        (2.0, 3.0),
    ]


def test_duck_envelope_points() -> None:
    points = duck_envelope_points([(1.0, 3.0)], depth_db=10.0, attack_s=0.2, release_s=0.6)
    assert points == [(0.8, 0.0), (1.0, -10.0), (3.0, -10.0), (3.6, 0.0)]
    # Attack clamped at t=0 for a dip starting at 0.
    clamped = duck_envelope_points([(0.0, 1.0)], attack_s=0.2, release_s=0.6)
    assert clamped[0] == (0.0, 0.0)
    assert all(t >= 0.0 for t, _ in clamped)


def test_envelope_gain_expression_structure() -> None:
    expr = envelope_gain_expression([(1.0, 3.0)])
    assert expr.startswith("pow(10,(")
    assert expr.endswith("/20)")
    assert "if(lt(t," in expr
    assert envelope_gain_expression([]) == "1"


def test_duck_envelope_filter_wiring() -> None:
    filt = duck_envelope_filter("music", "ducked", [(1.0, 3.0)])
    assert filt.startswith("[music]asetnsamples=n=480:p=0,volume='pow(10,(")
    assert filt.endswith(":eval=frame[ducked]")
    assert "'" in filt  # expression is single-quoted for the filtergraph


# --- render tests (ffmpeg) ----------------------------------------------------


def test_envelope_render_achieves_configured_depth(
    tmp_path: Path, narration_long: Path, music_bed_wav: Path
) -> None:
    out = tmp_path / "mix.wav"
    stem = tmp_path / "stem.wav"
    mix_ducked_audio(
        narration_long,
        music_bed_wav,
        out,
        duck_mode="envelope",
        intervals=[(0.0, 2.0), (4.0, 6.0)],
        ducked_stem=stem,
    )
    assert out.is_file() and stem.is_file()
    # Deep in the first dip (past the 0.2 s attack) vs mid-gap (past release).
    speech_lufs = music_bed.segment_loudness_lufs(stem, 0.8, 0.8)
    gap_lufs = music_bed.segment_loudness_lufs(stem, 2.8, 0.8)
    assert speech_lufs is not None and gap_lufs is not None
    depth = gap_lufs - speech_lufs
    assert depth == pytest.approx(10.0, abs=1.5), f"depth {depth:.1f} dB"


def test_sidechain_render_completes(
    tmp_path: Path, narration_long: Path, music_bed_wav: Path
) -> None:
    out = tmp_path / "mix_sc.wav"
    mix_ducked_audio(narration_long, music_bed_wav, out, duck_mode="sidechain")
    assert out.is_file()
    lufs = music_bed.measure_integrated_loudness(out)
    assert lufs == pytest.approx(-16.0, abs=2.0)
    assert music_bed.measure_true_peak_dbfs(out) <= -1.0


def test_envelope_mix_meets_loudness_targets(
    tmp_path: Path, narration_long: Path, music_bed_wav: Path
) -> None:
    out = tmp_path / "mix_env.wav"
    mix_ducked_audio(
        narration_long, music_bed_wav, out, duck_mode="envelope", intervals=[(0.0, 2.0)]
    )
    assert music_bed.measure_integrated_loudness(out) == pytest.approx(-16.0, abs=1.5)
    assert music_bed.measure_true_peak_dbfs(out) <= -1.0


def test_mix_ducked_audio_rejects_bad_mode(tmp_path: Path) -> None:
    with pytest.raises(music_bed.MusicBedError):
        mix_ducked_audio(tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "o.wav",
                         duck_mode="bogus")


def test_ab_script_end_to_end(
    tmp_path: Path, narration_long: Path, music_bed_wav: Path
) -> None:
    out_dir = tmp_path / "ab_out"
    completed = subprocess.run(
        [sys.executable, str(AB_SCRIPT), str(narration_long), str(music_bed_wav),
         "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "sidechain" in completed.stdout
    assert "envelope" in completed.stdout
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.is_file()
    payload = json.loads(metrics_path.read_text())
    assert set(payload["modes"]) == {"sidechain", "envelope"}
    for mode in ("sidechain", "envelope"):
        assert (out_dir / f"{mode}_mix.wav").is_file()
        assert (out_dir / f"{mode}_stem.wav").is_file()
