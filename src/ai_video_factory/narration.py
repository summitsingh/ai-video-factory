"""Local text-to-speech narration using espeak-ng (formant synthesis).

No model downloads: espeak-ng is a compiled formant synthesizer shipped by
the OS package manager. Output is normalized to 48 kHz stereo WAV so the
pipeline's FFmpeg stage can mux it directly.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ai_video_factory.sanitization import sanitize_diagnostic


class NarrationError(RuntimeError):
    """Raised when local speech synthesis fails."""


_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BARE_URL = re.compile(r"https?://\S+")
_MARKDOWN_CHARS = re.compile(r"[*_#>`~]")

_PIPER_VENV_BIN = (
    Path(__file__).resolve().parents[2] / ".tts-venv" / "bin" / "piper"
)
_PIPER_MODEL = (
    Path(__file__).resolve().parents[2] / "models" / "tts" / "en_US-lessac-high.onnx"
)


def clean_for_speech(text: str) -> str:
    """Strip markdown citations/URLs so they are speakable prose."""
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _BARE_URL.sub("", text)
    text = _MARKDOWN_CHARS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def synthesize_to_wav(
    text: str,
    output: Path,
    *,
    engine: str = "piper",
    voice: str = "en",
    speed_wpm: int = 170,
    pitch: int = 50,
    espeak: str = "espeak-ng",
    timeout: int = 300,
) -> Path:
    """Synthesize speakable text to a WAV file with a local engine.

    Engines: 'piper' (neural, preferred when installed) or 'espeak'
    (formant fallback, always available). Falls back to espeak when the
    Piper binary or voice model is absent.
    """
    speakable = clean_for_speech(text)
    if not speakable:
        raise NarrationError("no speakable text after cleaning narration")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if engine == "piper" and _PIPER_VENV_BIN.is_file() and _PIPER_MODEL.is_file():
        return _synthesize_piper(speakable, output, timeout=timeout)
    return _synthesize_espeak(
        speakable, output, voice=voice, speed_wpm=speed_wpm, pitch=pitch,
        espeak=espeak, timeout=timeout,
    )


def _synthesize_piper(text: str, output: Path, *, timeout: int) -> Path:
    try:
        completed = subprocess.run(
            (str(_PIPER_VENV_BIN), "--model", str(_PIPER_MODEL),
             "--output_file", str(output)),
            input=text.encode("utf-8"),
            shell=False,
            timeout=timeout,
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NarrationError(sanitize_diagnostic(f"piper could not run: {error}")) from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "ignore").strip()[-500:] or f"exit {completed.returncode}"
        raise NarrationError(sanitize_diagnostic(f"piper failed: {detail}"))
    if not output.is_file() or output.stat().st_size == 0:
        raise NarrationError("piper produced no audio output")
    return output


def _synthesize_espeak(
    speakable: str,
    output: Path,
    *,
    voice: str,
    speed_wpm: int,
    pitch: int,
    espeak: str,
    timeout: int,
) -> Path:
    try:
        completed = subprocess.run(
            (espeak, "--stdout", "-v", voice, "-s", str(speed_wpm), "-p", str(pitch), speakable),
            shell=False,
            timeout=timeout,
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NarrationError(sanitize_diagnostic(f"espeak-ng could not run: {error}")) from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "ignore").strip() or f"exit {completed.returncode}"
        raise NarrationError(sanitize_diagnostic(f"espeak-ng failed: {detail}"))
    if not completed.stdout.startswith(b"RIFF"):
        raise NarrationError("espeak-ng did not return WAV audio")
    output.write_bytes(completed.stdout)
    return output


def mix_scenes_to_track(
    segments: list[tuple[Path, float]],
    output: Path,
    *,
    sample_rate: int = 48000,
    ffmpeg: str = "ffmpeg",
    timeout: int = 600,
) -> Path:
    """Mix per-scene WAVs at start offsets into one padded stereo track.

    Args:
        segments: (wav path, start offset in seconds) pairs.
        output: destination WAV path (48 kHz stereo).
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not segments:
        raise NarrationError("no narration segments to mix")
    argv: list[str] = [ffmpeg, "-y"]
    for wav, _ in segments:
        argv += ("-i", str(wav))
    n = len(segments)
    filters = "".join(
        f"[{i}:a]aresample={sample_rate},adelay={int(start_ms)}|{int(start_ms)}[a{i}];"
        for i, (_, start) in enumerate(segments)
        for start_ms in (round(start * 1000),)
    )
    inputs = "".join(f"[a{i}]" for i in range(n))
    filters += f"{inputs}amix=inputs={n}:normalize=0,aformat=sample_rates={sample_rate}:channel_layouts=stereo[mix]"
    argv += ("-filter_complex", filters, "-map", "[mix]", str(output))
    try:
        completed = subprocess.run(
            argv, shell=False, timeout=timeout, capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NarrationError(sanitize_diagnostic(f"narration mix could not run: {error}")) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:] or f"exit {completed.returncode}"
        raise NarrationError(sanitize_diagnostic(f"narration mix failed: {detail}"))
    return output
