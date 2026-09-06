"""Local text-to-speech narration.

Default engine is Kokoro (neural, ONNX runtime via `kokoro-onnx`), which runs
on CPU and produces broadcast-quality English narration for long-form content.
Piper (`piper-tts`) and espeak-ng remain available as fallback engines. All
engines normalize output to 48 kHz stereo WAV so the pipeline's FFmpeg stage
can mux it directly.

Model files live under `models/tts/` (downloaded once, not fetched at runtime):
  - kokoro-v1.0.onnx        (Kokoro neural model)
  - voices-v1.0.bin         (Kokoro voice weights)
  - en_US-lessac-high.onnx  (Piper voice model)
"""

from __future__ import annotations

import io
import re
import subprocess
import wave
from pathlib import Path
from typing import Any

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

_KOKORO_MODEL = (
    Path(__file__).resolve().parents[2] / "models" / "tts" / "kokoro-v1.0.onnx"
)
_KOKORO_VOICES = (
    Path(__file__).resolve().parents[2] / "models" / "tts" / "voices-v1.0.bin"
)

# Default narrator voice for Kokoro (a clear, warm English voice). Override via
# the `voice` argument to synthesize_to_wav() or per-scene in the edit doc.
_KOKORO_DEFAULT_VOICE = "af_heart"

# Module-level cache so we don't reload the ~325 MB ONNX model once per scene.
_kokoro_model_cache: dict[str, object] = {}


def _load_kokoro() -> Any:
    """Load (and cache) the Kokoro ONNX model + voice weights."""
    import kokoro_onnx  # local import; only needed when Kokoro is used

    key = f"{_KOKORO_VOICES}|{_KOKORO_VOICES}"
    if key not in _kokoro_model_cache:
        model = kokoro_onnx.Kokoro(str(_KOKORO_MODEL), str(_KOKORO_VOICES))
        _kokoro_model_cache[key] = model
    return _kokoro_model_cache[key]


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
    engine: str = "kokoro",
    voice: str = "af_heart",
    speed_wpm: int = 170,
    pitch: int = 50,
    espeak: str = "espeak-ng",
    timeout: int = 300,
) -> Path:
    """Synthesize speakable text to a WAV file with a local engine.

    Default engine is 'kokoro' (neural ONNX TTS, broadcast-quality English for
    long-form narration). 'piper' and 'espeak' remain available as fallbacks;
    the pipeline falls back to espeak only when neither Kokoro nor Piper can run.
    """
    speakable = clean_for_speech(text)
    if not speakable:
        raise NarrationError("no speakable text after cleaning narration")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if engine == "kokoro" and _KOKORO_MODEL.is_file() and _KOKORO_VOICES.is_file():
        return _synthesize_kokoro(
            speakable, output, voice=voice or _KOKORO_DEFAULT_VOICE, timeout=timeout
        )
    if engine == "piper" and _PIPER_VENV_BIN.is_file() and _PIPER_MODEL.is_file():
        return _synthesize_piper(speakable, output, timeout=timeout)
    return _synthesize_espeak(
        speakable, output, voice=voice, speed_wpm=speed_wpm, pitch=pitch,
        espeak=espeak, timeout=timeout,
    )


def _synthesize_kokoro(
    speakable: str,
    output: Path,
    *,
    voice: str = "af_heart",
    speed: float = 1.0,
    timeout: int = 300,
) -> Path:
    """Synthesize text with the Kokoro ONNX model and normalize to 48 kHz stereo WAV.

    Kokoro emits a mono float32 array at 24 kHz. We write that as an intermediate
    WAV, then use FFmpeg to resample to 48 kHz and upmix to stereo so the result
    muxes directly into the pipeline's narration track. The model is loaded once
    per process (module-level cache) since each scene synthesizes separately.
    """
    import array
    import wave

    kokoro = _load_kokoro()

    try:
        audio, sample_rate = kokoro.create(speakable, voice=voice, speed=speed)
    except Exception as error:  # noqa: BLE001 - surface any synthesis failure
        raise NarrationError(
            sanitize_diagnostic(f"kokoro could not synthesize: {error}")
        ) from error

    if audio is None or len(audio) == 0:
        raise NarrationError("kokoro produced no audio output")

    # Normalize float32 in [-1, 1] to int16 PCM using stdlib only (the pipeline
    # venv has no numpy). Tolerate a numpy array, list, or array.array.
    samples = getattr(audio, "tolist", None)
    if callable(samples):
        floats = samples()
    else:
        floats = list(audio)

    pcm = array.array("h")  # signed short (int16)
    for value in floats:
        clipped = -1.0 if value < -1.0 else (1.0 if value > 1.0 else value)
        pcm.append(int(clipped * 32767.0))

    # Write intermediate mono WAV at the model's native sample rate.
    tmp_wav = output.with_suffix(".raw.wav")
    with wave.open(str(tmp_wav), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)  # int16
        wav_out.setframerate(int(sample_rate))
        wav_out.writeframes(pcm.tobytes())

    try:
        completed = subprocess.run(
            (
                "ffmpeg", "-y", "-i", str(tmp_wav),
                "-ar", "48000", "-ac", "2",
                "-c:a", "pcm_s16le", str(output),
            ),
            shell=False, timeout=timeout, capture_output=True, text=True, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NarrationError(
            sanitize_diagnostic(f"kokoro audio could not be normalized: {error}")
        ) from error
    finally:
        if tmp_wav.is_file():
            tmp_wav.unlink()

    if completed.returncode != 0:
        detail = completed.stderr.strip()[-500:] or f"exit {completed.returncode}"
        raise NarrationError(
            sanitize_diagnostic(f"kokoro audio normalization failed: {detail}")
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise NarrationError("kokoro produced no normalized audio output")
    return output


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


def generate_sfx(
    kind: str,
    output: Path,
    *,
    duration_seconds: float = 0.35,
    sample_rate: int = 48000,
    ffmpeg: str = "ffmpeg",
    timeout: int = 120,
) -> Path:
    """Generate a short sound-effect clip (no downloads) for scene transitions.

    Kinds: 'whoosh' (filtered noise sweep used on scene cuts), 'ping' (a soft
    tonal blip used when motion graphics reveal), and 'drone' (a low swell used
    at intro/outro boundaries). Output is a 48 kHz stereo WAV with fade edges so
    it never clicks.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = str(ffmpeg) if ffmpeg else "ffmpeg"
    kind = (kind or "whoosh").lower()

    # Base noise source + bandpass shape the effect; fades prevent clicks.
    if kind == "ping":
        src = f"sine=frequency=660:duration={duration_seconds}"
        chain = (
            f"volume=0.18,"
            f"afade=t=in:st=0:d=0.02,"
            f"afade=t=out:st={max(0.05, duration_seconds - 0.06):.3f}:d=0.06"
        )
    elif kind == "drone":
        src = f"sine=frequency=75:duration={duration_seconds}"
        chain = (
            f"volume=0.14,"
            f"afade=t=in:st=0:d=0.08,"
            f"afade=t=out:st={max(0.05, duration_seconds - 0.08):.3f}:d=0.08"
        )
    else:  # whoosh (default)
        src = "anoisesrc=d=0.3:c=pink"
        chain = (
            f"bandpass=f=1200,volume=0.12,"
            f"afade=t=in:st=0:d=0.05,"
            f"afade=t=out:st={max(0.05, duration_seconds - 0.05):.3f}:d=0.05"
        )

    argv = [ffmpeg_bin, "-y", "-f", "lavfi", "-i", src,
            "-filter_complex", chain,
            "-c:a", "pcm_s16le", "-ar", str(sample_rate), "-ac", "2", str(output)]
    try:
        completed = subprocess.run(
            argv, shell=False, timeout=timeout, capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NarrationError(sanitize_diagnostic(f"sfx {kind} could not run: {error}")) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:] or f"exit {completed.returncode}"
        raise NarrationError(sanitize_diagnostic(f"sfx {kind} failed: {detail}"))
    return output


def apply_voice_variation(
    wav: Path,
    output: Path,
    *,
    speed_wpm: int = 170,
    pitch_shift_semitones: float = 0.0,
    emphasis_boost_db: float = 2.0,
    ffmpeg: str = "ffmpeg",
    timeout: int = 120,
) -> Path:
    """Apply subtle per-sentence voice variation to a narration segment.

    Real narrators vary pace sentence-to-sentence; flat synthesis sounds
    robotic over long-form content. This applies a small speed modulation
    (±13%) and a light presence boost for clarity. Output stays 48 kHz stereo
    so it muxes directly into the pipeline. Pitch shifting via varispeed is
    unavailable in this ffmpeg build, so variation comes from pace + EQ.
    """
    wav = Path(wav)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = str(ffmpeg) if ffmpeg else "ffmpeg"

    # Speed: atempo modulates pace (±13%) so a faster reading shortens the clip.
    speed_ratio = max(0.87, min(1.13, speed_wpm / 170.0))
    filters = [f"[0:a]atempo={speed_ratio:.4f}"]

    # Presence boost around speech fundamentals for clarity; then pad to a
    # minimum length so downstream timing math is unaffected by the small speed
    # change. (Pitch shifting via varispeed is unavailable in this ffmpeg build,
    # so variation comes from pace + EQ rather than pitch.)
    filters.append(f"equalizer=f=2500:w=0.9:g={emphasis_boost_db:.1f}")
    filters.append("apad=pad_len=60000[aout]")

    argv = [ffmpeg_bin, "-y", "-i", str(wav),
            "-filter_complex", ",".join(filters),
            "-map", "[aout]",
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(output)]
    try:
        completed = subprocess.run(
            argv, shell=False, timeout=timeout, capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NarrationError(sanitize_diagnostic(f"voice variation could not run: {error}")) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:] or f"exit {completed.returncode}"
        raise NarrationError(sanitize_diagnostic(f"voice variation failed: {detail}"))
    return output
