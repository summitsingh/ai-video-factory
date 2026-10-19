"""Whisper word-timing for karaoke captions.

Deterministic pipeline piece: given the finished narration audio (or the
master's audio track), produce exact per-word timings so karaoke caption
events highlight each word as it is actually spoken, instead of the
length-weighted heuristic in :mod:`ai_video_factory.subtitle_export`.

Design (concepts reimplemented from the WanGP transcription reference,
own code throughout):

* ``transcribe_media`` runs openai-whisper with word timestamps and returns
  serializable segment/word timings.
* ``detect_speech_intervals`` finds speech spans with ffmpeg
  ``silencedetect`` (inverted), so timestamps can be tightened to detected
  speech: Whisper attaches inter-word pauses to the following word, so
  ``tighten_bounds`` pulls each bound inward past any pause of at least
  ``min_pause`` seconds (default 0.15).
* ``align_words_to_cues`` maps whisper word timings onto the validated
  script's caption cues by word order (the script stays authoritative;
  whisper only supplies timing). Unmatched words are interpolated between
  their timed neighbours.
* ``segments_to_srt`` renders plain SubRip from timed segments.

Whisper backend: ``transcribe_media`` uses an in-process ``whisper``
import when available, otherwise it shells out to a whisper-capable
Python (``WHISPER_PYTHON`` env var, defaulting to the WanGP conda env on
summit-amd where openai-whisper is installed). This keeps the module
importable and unit-testable in the lean pipeline venv.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

DEFAULT_MODEL = "medium"
MIN_PAUSE_SECONDS = 0.15
DEFAULT_LANGUAGE = "en"

_NORMALIZE_WORD_RE = re.compile(r"[^a-z0-9]")


@dataclass
class WordTiming:
    """One word with exact start/end seconds."""

    word: str
    start: float
    end: float
    probability: float | None = None


@dataclass
class SegmentTiming:
    """One transcription segment with optional word timings."""

    start: float
    end: float
    text: str
    words: list[WordTiming] = field(default_factory=list)


@dataclass
class Transcription:
    """Result of :func:`transcribe_media`."""

    text: str
    language: str
    segments: list[SegmentTiming] = field(default_factory=list)
    speech_intervals: list[tuple[float, float]] = field(default_factory=list)
    model_name: str = ""


# ---------------------------------------------------------------------------
# Audio helpers (ffmpeg only)
# ---------------------------------------------------------------------------


def _probe_duration(audio_path: Path, ffmpeg_bin: str) -> float:
    """Return media duration in seconds via ffprobe (falls back to ffmpeg)."""
    ffprobe = "ffprobe"
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return float(proc.stdout.strip())
    except (OSError, ValueError):
        pass
    # Fallback: parse "Duration: HH:MM:SS.xx" from ffmpeg stderr.
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", str(audio_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", proc.stderr)
    if not match:
        raise RuntimeError(f"could not determine duration of {audio_path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def extract_audio(
    source: Path | str,
    dest: Path | str,
    *,
    sample_rate: int = 16000,
    ffmpeg: str = "ffmpeg",
    timeout: int = 600,
) -> Path:
    """Extract ``source``'s audio to 16 kHz mono WAV for transcription."""
    source = Path(source)
    dest = Path(dest)
    if not source.is_file():
        raise FileNotFoundError(f"media file not found: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"audio extraction failed: {proc.stderr[-1500:]}")
    return dest


def detect_speech_intervals(
    audio_path: Path | str,
    *,
    threshold_db: float = -45.0,
    min_silence_s: float = 0.3,
    ffmpeg: str = "ffmpeg",
    timeout: int = 600,
) -> list[tuple[float, float]]:
    """Return ``[(start, end), ...]`` speech spans in seconds.

    Runs ffmpeg ``silencedetect`` and inverts the silence ranges, so the
    result is the complement: every non-silent span. A trailing silence
    that runs to end-of-file is closed at the media duration.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")
    duration = _probe_duration(audio_path, ffmpeg)
    argv = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={threshold_db}dB:d={min_silence_s}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    silence: list[tuple[float, float]] = []
    open_start: float | None = None
    for line in proc.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            open_start = float(start_match.group(1))
            continue
        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and open_start is not None:
            silence.append((open_start, float(end_match.group(1))))
            open_start = None
    if open_start is not None:
        silence.append((open_start, duration))
    speech: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted(silence):
        if start > cursor:
            speech.append((round(cursor, 3), round(start, 3)))
        cursor = max(cursor, end)
    if cursor < duration:
        speech.append((round(cursor, 3), round(duration, 3)))
    return [(s, e) for s, e in speech if e - s > 0.001]


# ---------------------------------------------------------------------------
# Timestamp tightening
# ---------------------------------------------------------------------------


def tighten_bounds(
    start: float | None,
    end: float | None,
    speech_intervals: Sequence[tuple[float, float]],
    *,
    min_pause: float = MIN_PAUSE_SECONDS,
) -> tuple[float | None, float | None]:
    """Pull ``(start, end)`` inward past leading/trailing pauses.

    Whisper attaches an inter-word pause to the following word, so a word
    (or segment) bound can sit inside silence. When the gap between the
    bound and the nearest detected-speech edge inside the span is at least
    ``min_pause`` seconds, the bound moves to that edge; smaller gaps are
    left alone so tight speech keeps its original timing.
    """
    if start is None or end is None:
        return start, end
    inside = [
        (max(s, start), min(e, end))
        for s, e in speech_intervals
        if e > start and s < end
    ]
    if not inside:
        return start, end
    new_start = inside[0][0] if inside[0][0] - start >= min_pause else start
    new_end = inside[-1][1] if end - inside[-1][1] >= min_pause else end
    return round(new_start, 3), round(new_end, 3)


# ---------------------------------------------------------------------------
# SRT export
# ---------------------------------------------------------------------------


def segments_to_srt(segments: Sequence[SegmentTiming | dict[str, Any]]) -> str:
    """Render timed segments as a SubRip (``.srt``) string."""

    def stamp(seconds: float) -> str:
        total_ms = int(round(float(seconds) * 1000))
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, ms = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def as_tuple(
        segment: SegmentTiming | dict[str, Any],
    ) -> tuple[float | None, float | None, str]:
        if isinstance(segment, dict):
            return segment.get("start"), segment.get("end"), str(
                segment.get("text", "") or ""
            ).strip()
        return segment.start, segment.end, segment.text.strip()

    cues = []
    for segment in segments:
        start, end, text = as_tuple(segment)
        if text and start is not None and end is not None:
            cues.append((start, end, text))
    return "\n".join(
        f"{index}\n{stamp(start)} --> {stamp(end)}\n{text}\n"
        for index, (start, end, text) in enumerate(cues, 1)
    )


# ---------------------------------------------------------------------------
# Word <-> cue alignment (script stays authoritative; whisper gives timing)
# ---------------------------------------------------------------------------


def _normalize_word(word: str) -> str:
    return _NORMALIZE_WORD_RE.sub("", word.lower())


def _fill_timing_gaps(
    cue_start: float,
    cue_end: float,
    cue_words: list[str],
    assigned: list[tuple[float, float] | None],
) -> list[tuple[float, float, str]]:
    """Interpolate timings for cue words with no whisper match.

    Each run of unmatched words is spread evenly across the gap between
    its timed neighbours (or the cue bounds), so the words still tile the
    cue contiguously with no overlaps.
    """
    result: list[tuple[float, float, str]] = []
    index = 0
    total = len(cue_words)
    while index < total:
        timing = assigned[index]
        if timing is not None:
            result.append((timing[0], timing[1], cue_words[index]))
            index += 1
            continue
        run_start = index
        while index < total and assigned[index] is None:
            index += 1
        prev_end = assigned[run_start - 1][1] if run_start > 0 else cue_start
        next_start = assigned[index][0] if index < total else cue_end
        gap_start = min(max(prev_end, cue_start), cue_end)
        gap_end = min(max(next_start, gap_start), cue_end)
        run_len = index - run_start
        for offset in range(run_len):
            word_start = gap_start + (gap_end - gap_start) * offset / run_len
            word_end = gap_start + (gap_end - gap_start) * (offset + 1) / run_len
            result.append(
                (
                    round(word_start, 3),
                    round(word_end, 3),
                    cue_words[run_start + offset],
                )
            )
    return result


def align_words_to_cues(
    cues: Sequence[tuple[float, float, str]],
    words: Sequence[WordTiming],
    *,
    tolerance: float = 0.75,
    max_lookahead: int = 64,
) -> list[list[tuple[float, float, str]]]:
    """Map whisper word timings onto script caption cues.

    ``cues`` are ``(start, end, text)`` from the validated script;
    ``words`` are whisper :class:`WordTiming` in audio order. Each cue word
    takes the timing of the next whisper word (within ``tolerance`` seconds
    of the cue span) whose normalized text matches; matching is
    case- and punctuation-insensitive. Words are consumed in order, so a
    repeated word matches its next occurrence. Unmatched cue words are
    interpolated between their timed neighbours. Cue word text is kept
    verbatim: the script is authoritative, whisper only supplies timing.
    """
    word_list = list(words)
    count = len(word_list)
    aligned: list[list[tuple[float, float, str]]] = []
    pos = 0
    for cue_start, cue_end, text in cues:
        cue_words = text.split()
        assigned: list[tuple[float, float] | None] = [None] * len(cue_words)
        for word_index, cue_word in enumerate(cue_words):
            target = _normalize_word(cue_word)
            if not target:
                continue
            cursor = pos
            while cursor < count and cursor < pos + max_lookahead:
                candidate = word_list[cursor]
                if candidate.start > cue_end + tolerance:
                    break
                if (
                    candidate.start >= cue_start - tolerance
                    and _normalize_word(candidate.word) == target
                ):
                    assigned[word_index] = (
                        round(candidate.start, 3),
                        round(candidate.end, 3),
                    )
                    pos = cursor + 1
                    break
                cursor += 1
        aligned.append(_fill_timing_gaps(cue_start, cue_end, cue_words, assigned))
    return aligned


def karaoke_timings_for_cues(
    cues: Sequence[tuple[float, float, str]],
    transcription: Transcription,
) -> list[list[tuple[float, float, str]]]:
    """Convenience: flatten a transcription's words and align to cues."""
    words = [word for segment in transcription.segments for word in segment.words]
    return align_words_to_cues(cues, words)


# ---------------------------------------------------------------------------
# Whisper transcription backends
# ---------------------------------------------------------------------------


def whisper_available() -> bool:
    """True when openai-whisper imports in this process."""
    try:
        import whisper  # noqa: F401

        return True
    except ImportError:
        return False


def default_whisper_python() -> str | None:
    """Locate a whisper-capable Python for the subprocess backend.

    ``WHISPER_PYTHON`` wins; otherwise the WanGP conda env on summit-amd,
    where openai-whisper is installed.
    """
    override = os.environ.get("WHISPER_PYTHON", "").strip()
    if override and Path(override).exists():
        return override
    candidate = Path("/home/summit/miniconda3/envs/wangp/bin/python")
    if candidate.exists():
        return str(candidate)
    return None


_RUNNER_SCRIPT = r'''
"""Out-of-process whisper runner: prints transcription JSON to stdout."""
import json
import sys

import torch
import whisper


def main() -> None:
    audio_path, model_name, device, language, word_timestamps = sys.argv[1:6]
    if device in ("", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    try:
        result = model.transcribe(
            audio_path,
            language=language or None,
            word_timestamps=(word_timestamps == "1"),
            verbose=None,
        )
    finally:
        del model
    segments = []
    for segment in result.get("segments", []) or []:
        words = []
        for item in segment.get("words", []) or []:
            words.append(
                {
                    "word": item.get("word", ""),
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "probability": item.get("probability"),
                }
            )
        segments.append(
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", ""),
                "words": words,
            }
        )
    print(
        json.dumps(
            {
                "text": result.get("text", ""),
                "language": result.get("language", ""),
                "segments": segments,
            }
        )
    )


main()
'''


def _coerce_word(payload: dict[str, Any]) -> WordTiming | None:
    start = payload.get("start")
    end = payload.get("end")
    if start is None or end is None:
        return None
    probability = payload.get("probability")
    return WordTiming(
        word=str(payload.get("word", "") or ""),
        start=round(float(start), 3),
        end=round(float(end), 3),
        probability=None if probability is None else round(float(probability), 4),
    )


def _coerce_result(payload: dict[str, Any], model_name: str) -> Transcription:
    segments: list[SegmentTiming] = []
    for item in payload.get("segments", []) or []:
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        words = [
            word
            for raw in (item.get("words", []) or [])
            if (word := _coerce_word(raw)) is not None
        ]
        segments.append(
            SegmentTiming(
                start=round(float(start), 3),
                end=round(float(end), 3),
                text=str(item.get("text", "") or "").strip(),
                words=words,
            )
        )
    return Transcription(
        text=str(payload.get("text", "") or "").strip(),
        language=str(payload.get("language", "") or "").strip(),
        segments=segments,
        model_name=model_name,
    )


def _transcribe_in_process(
    audio_path: Path,
    model_name: str,
    device: str | None,
    language: str | None,
    word_timestamps: bool,
    timeout: int,
) -> Transcription:
    import torch
    import whisper

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(model_name, device=resolved_device)
    try:
        result = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=word_timestamps,
            verbose=None,
        )
    finally:
        del model
        if resolved_device == "cuda":
            torch.cuda.empty_cache()
    return _coerce_result(result, model_name)


def _transcribe_via_subprocess(
    audio_path: Path,
    model_name: str,
    device: str | None,
    language: str | None,
    word_timestamps: bool,
    python: str,
    timeout: int,
) -> Transcription:
    with tempfile.TemporaryDirectory(prefix="whisper_runner_") as tmp_dir:
        runner = Path(tmp_dir) / "whisper_runner.py"
        runner.write_text(_RUNNER_SCRIPT, encoding="utf-8")
        argv = [
            python,
            str(runner),
            str(audio_path),
            model_name,
            device or "auto",
            language or "",
            "1" if word_timestamps else "0",
        ]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"whisper transcription failed (exit {proc.returncode}): "
            f"{proc.stderr[-2000:].strip()}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"whisper runner returned non-JSON output: {proc.stdout[-500:]}"
        ) from exc
    return _coerce_result(payload, model_name)


def transcribe_media(
    audio_path: Path | str,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    language: str | None = DEFAULT_LANGUAGE,
    word_timestamps: bool = True,
    detect_speech: bool = True,
    min_pause: float = MIN_PAUSE_SECONDS,
    timeout: int = 3600,
    python: str | None = None,
) -> Transcription:
    """Transcribe audio with word-level timestamps, tightened to speech.

    Uses an in-process ``whisper`` import when available, otherwise shells
    out to ``python`` (or ``WHISPER_PYTHON`` / the default conda env). Word
    and segment bounds are tightened past pauses of at least ``min_pause``
    seconds using ffmpeg-detected speech intervals. Raises
    :class:`FileNotFoundError` for a missing input and :class:`RuntimeError`
    when no whisper backend is available.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    backend_python = python or default_whisper_python()
    if whisper_available():
        transcription = _transcribe_in_process(
            audio_path, model_name, device, language, word_timestamps, timeout
        )
    elif backend_python:
        transcription = _transcribe_via_subprocess(
            audio_path, model_name, device, language, word_timestamps,
            backend_python, timeout,
        )
    else:
        raise RuntimeError(
            "openai-whisper is not importable here and no whisper-capable "
            "Python was found. Install it (`pip install openai-whisper`) or "
            "set WHISPER_PYTHON to a Python that has it."
        )

    if detect_speech:
        speech = detect_speech_intervals(audio_path)
        transcription.speech_intervals = speech
        for segment in transcription.segments:
            segment.start, segment.end = tighten_bounds(
                segment.start, segment.end, speech, min_pause=min_pause
            )
            for word in segment.words:
                word.start, word.end = tighten_bounds(
                    word.start, word.end, speech, min_pause=min_pause
                )
    return transcription


# ---------------------------------------------------------------------------
# JSON sidecar (cacheable transcription artifacts)
# ---------------------------------------------------------------------------


def transcription_to_dict(transcription: Transcription) -> dict[str, Any]:
    """Serialize a transcription to plain JSON-compatible dicts."""
    return {
        "text": transcription.text,
        "language": transcription.language,
        "model_name": transcription.model_name,
        "speech_intervals": [list(span) for span in transcription.speech_intervals],
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ],
            }
            for segment in transcription.segments
        ],
    }


def save_transcription(transcription: Transcription, path: Path | str) -> Path:
    """Write a transcription sidecar (deterministic, re-loadable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(transcription_to_dict(transcription), indent=2), encoding="utf-8"
    )
    return path


def load_transcription(path: Path | str) -> Transcription:
    """Read a transcription sidecar written by :func:`save_transcription`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = [
        SegmentTiming(
            start=item["start"],
            end=item["end"],
            text=item.get("text", ""),
            words=[
                WordTiming(
                    word=raw.get("word", ""),
                    start=raw["start"],
                    end=raw["end"],
                    probability=raw.get("probability"),
                )
                for raw in item.get("words", [])
            ],
        )
        for item in payload.get("segments", [])
    ]
    return Transcription(
        text=payload.get("text", ""),
        language=payload.get("language", ""),
        segments=segments,
        speech_intervals=[
            (span[0], span[1]) for span in payload.get("speech_intervals", [])
        ],
        model_name=payload.get("model_name", ""),
    )
