"""Speech-interval detection for AI Video Factory.

Finds spans of speech in an audio file from speech-band (200-4000 Hz)
energy in 20 ms windows. Used by envelope-based music ducking
(:mod:`ai_video_factory.music_bed`, ``duck_mode="envelope"``) and
available for narration-presence QC ("speech present where the script
expects narration").

Detection rule
--------------
A window counts as speech when its 200-4000 Hz band energy rises at
least ``margin_db`` (default 9 dB) above the local background, defined as
the 10th percentile of band energy over the surrounding 10 seconds. A
steady music bed or room tone therefore raises the threshold instead of
counting as speech. Recordings that are mostly an isolated voice (5th
percentile of band energy below -60 dBFS) use a single threshold of peak
band energy minus 40 dB instead, since an isolated voice may not pause
long enough to measure a background. Energy below -55 dBFS never counts.

Spans are post-processed with ``min_speech`` (drop spans shorter than
0.06 s) and ``min_gap`` (merge spans separated by less than 0.15 s).

This is a clean-room reimplementation of the algorithm documented in the
WanGP feature inventory (Q5 / section 2.4: band corners, window size,
margin, percentiles, floors). No WanGP source was copied.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class SpeechDetectError(RuntimeError):
    """Raised when speech detection cannot run (bad input, ffmpeg failure)."""


# Analysis runs at 16 kHz with 20 ms windows (320 samples per frame).
ANALYSIS_SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
FRAME_S = FRAME_SAMPLES / ANALYSIS_SAMPLE_RATE

# Speech band-pass corners (Hz).
BAND_LOW_HZ = 200
BAND_HIGH_HZ = 4000

# Speech must exceed the local background by this margin (dB).
DEFAULT_MARGIN_DB = 9.0

# Background estimate: this percentile of band energy over a window of
# BACKGROUND_WINDOW_S seconds centered on each 0.5 s block.
BACKGROUND_WINDOW_S = 10.0
BACKGROUND_PERCENTILE = 10.0
BACKGROUND_BLOCK_S = 0.5

# Band energy below this (dBFS) never counts as speech.
ENERGY_FLOOR_DBFS = -55.0

# Auto-detect isolated voice: this percentile of band energy below this
# (dBFS) means the recording is mostly a lone voice with real silence.
ISOLATED_SILENCE_PERCENTILE = 5.0
ISOLATED_SILENCE_DBFS = -60.0

# Isolated-voice threshold: peak band energy minus this (dB).
ISOLATED_PEAK_MARGIN_DB = 40.0

# Post-processing: drop spans shorter than this; merge spans separated by
# gaps shorter than this (seconds).
DEFAULT_MIN_SPEECH_S = 0.06
DEFAULT_MIN_GAP_S = 0.15

_EPS = 1e-12


def _resolve_ffmpeg(ffmpeg: str) -> str:
    """Return a usable ffmpeg path or raise SpeechDetectError."""
    if "/" in ffmpeg:
        if Path(ffmpeg).is_file():
            return ffmpeg
        raise SpeechDetectError(f"ffmpeg not found: {ffmpeg}")
    resolved = shutil.which(ffmpeg)
    if resolved is None:
        raise SpeechDetectError(f"ffmpeg not found on PATH: {ffmpeg}")
    return resolved


def _band_energy_db(
    path: Path,
    *,
    ffmpeg: str,
    start_time: float,
    duration: float | None,
) -> np.ndarray:
    """Decode *path* through a 200-4000 Hz band-pass; return per-frame energy in dBFS."""
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start_time > 0:
        cmd += ["-ss", f"{start_time:.3f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-vn",
        "-af",
        f"highpass=f={BAND_LOW_HZ},lowpass=f={BAND_HIGH_HZ}",
        "-ac", "1",
        "-ar", str(ANALYSIS_SAMPLE_RATE),
        "-f", "f32le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SpeechDetectError(
            f"speech detection could not decode {path}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()[-800:]
        raise SpeechDetectError(
            f"speech detection could not decode {path}: {detail or 'ffmpeg failed'}"
        )
    samples = np.frombuffer(completed.stdout, dtype="<f4")
    n_frames = len(samples) // FRAME_SAMPLES
    if n_frames == 0:
        return np.empty(0, dtype=np.float64)
    frames = (
        samples[: n_frames * FRAME_SAMPLES]
        .reshape(n_frames, FRAME_SAMPLES)
        .astype(np.float64)
    )
    return 10.0 * np.log10(np.square(frames).mean(axis=1) + _EPS)


def _isolated_threshold(energy: np.ndarray) -> np.ndarray:
    """Flat threshold for isolated-voice recordings: peak minus 40 dB, floored at -55."""
    level = max(ENERGY_FLOOR_DBFS, float(energy.max()) - ISOLATED_PEAK_MARGIN_DB)
    return np.full(len(energy), level)


def _mixture_threshold(energy: np.ndarray, margin_db: float) -> np.ndarray:
    """Per-frame threshold: 10th-percentile background over surrounding 10 s + margin."""
    n = len(energy)
    block = max(1, int(round(BACKGROUND_BLOCK_S / FRAME_S)))
    context = max(1, int(round((BACKGROUND_WINDOW_S / 2) / FRAME_S)))
    n_blocks = (n + block - 1) // block
    floors = np.empty(n_blocks)
    for index in range(n_blocks):
        center = min(index * block + block // 2, n - 1)
        lo = max(0, center - context)
        hi = min(n, center + context)
        floors[index] = np.percentile(energy[lo:hi], BACKGROUND_PERCENTILE)
    per_frame = np.repeat(floors, block)[:n]
    return np.maximum(per_frame + margin_db, ENERGY_FLOOR_DBFS)


def _spans_from_active(
    active: np.ndarray, *, min_speech: float, min_gap: float
) -> list[tuple[float, float]]:
    """Turn a per-frame speech mask into merged [(start, end)] seconds."""
    padded = np.concatenate([[False], active, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1]) * FRAME_S
    spans: list[list[float]] = []
    for start, end in zip(edges[::2], edges[1::2]):
        start_f, end_f = float(start), float(end)
        if spans and start_f - spans[-1][1] < min_gap:
            spans[-1][1] = end_f
        else:
            spans.append([start_f, end_f])
    return [
        (round(start, 3), round(end, 3))
        for start, end in spans
        if end - start >= min_speech
    ]


def speech_intervals(
    path: str | Path,
    *,
    start_time: float = 0.0,
    duration: float | None = None,
    isolated: bool | None = None,
    margin_db: float = DEFAULT_MARGIN_DB,
    min_speech: float = DEFAULT_MIN_SPEECH_S,
    min_gap: float = DEFAULT_MIN_GAP_S,
    ffmpeg: str = "ffmpeg",
) -> list[tuple[float, float]]:
    """Detect spans of speech in *path*; return [(start_s, end_s)] sorted by time.

    Times are seconds relative to the analyzed segment (``start_time`` seeks
    the input first, so pass ``start_time=0`` for source-file coordinates).
    ``isolated=None`` auto-detects isolated-voice recordings (5th percentile
    of band energy below -60 dBFS); pass True/False to force the mode.
    """
    src = Path(path)
    if not src.is_file():
        raise SpeechDetectError(f"audio file not found: {src}")
    ffmpeg_bin = _resolve_ffmpeg(ffmpeg)
    energy = _band_energy_db(
        src, ffmpeg=ffmpeg_bin, start_time=start_time, duration=duration
    )
    if len(energy) == 0:
        return []
    if isolated is None:
        isolated = (
            float(np.percentile(energy, ISOLATED_SILENCE_PERCENTILE))
            < ISOLATED_SILENCE_DBFS
        )
    threshold = (
        _isolated_threshold(energy)
        if isolated
        else _mixture_threshold(energy, margin_db)
    )
    return _spans_from_active(energy > threshold, min_speech=min_speech, min_gap=min_gap)


__all__ = [
    "ANALYSIS_SAMPLE_RATE",
    "BAND_HIGH_HZ",
    "BAND_LOW_HZ",
    "DEFAULT_MARGIN_DB",
    "DEFAULT_MIN_GAP_S",
    "DEFAULT_MIN_SPEECH_S",
    "ENERGY_FLOOR_DBFS",
    "SpeechDetectError",
    "speech_intervals",
]
