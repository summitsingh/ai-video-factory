"""Background music bed for AI Video Factory (pipeline item #8).

Takes a real royalty-free music track, fits it to the full video duration
(loop or trim), loudness-normalizes the bed, ducks it under the narration
with sidechain compression, and loudness-normalizes the complete final mix
(dual-pass EBU R128 when feasible, single-pass fallback otherwise).

Track sourcing
--------------
Resolution order in :func:`resolve_music_track`:

1. the curated local mood library ``assets/music/<mood>-bed.mp3``
   (``mood`` argument, else :func:`mood_for_topic` on the video topic), then
2. the ``MUSIC_BED_PATH`` environment variable pointing at an audio file
   (mp3/wav/ogg/m4a/flac) - explicit operator override, then
3. network providers via :mod:`ai_video_factory.music_providers`
   (Internet Archive -> Openverse -> Freesound), cached under
   ``data/cache/music/`` so repeats never re-download.

If no track resolves, the caller falls back to the procedural ambient drone
(``video_pipeline._generate_music_bed``) so the mix never goes out silent;
either way the bed is fitted and normalized here before the mux.

Why there is no Pixabay music fetch
-----------------------------------
The project's ``PIXABAY_API_KEY`` only grants Pixabay's public image/video
endpoints. Tested 2026-09-20:

- ``GET https://pixabay.com/api/audio/?key=<key>&q=ambient`` ->
  ``403 [ERROR 403] Access denied`` (endpoint exists, key not granted)
- ``GET https://pixabay.com/api/music/`` -> 404
- ``GET https://pixabay.com/api/music/search/`` -> 404
- ``GET https://pixabay.com/api/?key=<key>&q=ambient`` -> 200 (images work)

Pixabay's published API docs (https://pixabay.com/api/docs/) cover images
and videos only; music search is a website feature that needs a separate
grant on the Pixabay account. When the user obtains music API access, add a
``fetch_pixabay_music()`` helper here that searches, picks an
instrumental/calm track by tags and duration, and downloads it to
``data/cache/music/`` - the rest of this module (fit, duck, normalize)
already accepts any local file.

Level plan
----------
- Bed pre-normalized to ``DEFAULT_BED_TARGET_I`` (-23 LUFS integrated) so
  its level is deterministic regardless of source mastering.
- In the mux the bed runs at unity gain and is ducked by the narration via
  :func:`duck_filter` (sidechain compression, ~10-14 dB of gain reduction
  while speech is present), landing roughly 20 dB under the narration.
- The complete mix is dual-pass loudness-normalized to YouTube/EBU R128
  (-16 LUFS integrated, -2 dBTP, 7 LU) before the final mux.

Envelope ducking (experimental)
-------------------------------
:func:`mix_ducked_audio` supports ``duck_mode="envelope"`` as a
deterministic alternative to sidechain compression: speech spans from
:mod:`ai_video_factory.speech_detect` drive a piecewise-linear gain
envelope (see :func:`duck_envelope_filter`; defaults 10 dB depth, 0.2 s
attack, 0.6 s release) applied with ``volume`` on 10 ms frames, mixed
with ``amix=normalize=0`` under a true-peak limiter. Compare the two
modes with ``scripts/ab_ducking.py``. The pipeline default is unchanged
(sidechain).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from ai_video_factory.sanitization import sanitize_diagnostic


# Name of the environment variable carrying the operator-supplied music file.
MUSIC_BED_PATH_ENV = "MUSIC_BED_PATH"

# Optional override for the curated mood-library directory.
MUSIC_BED_DIR_ENV = "MUSIC_BED_DIR"

# Optional override for the network-provider download cache.
MUSIC_CACHE_DIR_ENV = "MUSIC_CACHE_DIR"

# Curated royalty-free mood library. Every track is CC-BY 4.0, commercial
# YouTube use OK with attribution in the video description - see
# assets/music/ATTRIBUTION.md. The MP3s are gitignored; each machine
# re-downloads them (or they arrive via the network fallback below).
MOOD_TRACKS: dict[str, str] = {
    "cosmic": "cosmic-bed.mp3",    # 'Decoherence' - Scott Buckley
    "mystery": "mystery-bed.mp3",  # 'Intervention' - Scott Buckley
    "epic": "epic-bed.mp3",        # 'Emergent' - Scott Buckley
    "calm": "calm-bed.mp3",        # 'Ephemera' - Scott Buckley
    "tech": "tech-bed.mp3",        # 'Machina' - Scott Buckley
}

# Fallback mood when the topic matches no keyword.
DEFAULT_MOOD = "calm"

# (mood, keywords) in priority order. Keywords become case-insensitive
# word-prefix regexes, so "astronom" hits "astronomy"/"astronomical".
# FULL_WORD_KEYWORDS need a trailing boundary too, so "ai" never matches
# inside "said"/"mountain" and "war" never matches "warm"/"warning".
_MOOD_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("cosmic", ("space", "cosmos", "universe", "galaxy", "nasa", "astronom",
                "planet", "exoplanet", "alien", "fermi", "black hole",
                "nebula", "cosmolog", "astrophys", "telescope", "stars")),
    ("mystery", ("myster", "unsolved", "secret", "conspiracy", "paradox",
                 "disappear", "cold case", "detective", "crime", "enigma",
                 "cover-up", "coverup", "thriller")),
    ("epic", ("histor", "ancient", "rome", "roman", "egypt", "empire",
              "civiliz", "medieval", "viking", "battle", "legend", "myth",
              "kingdom", "sparta", "world war", "warfare", "gladiator")),
    ("tech", ("artificial intelligence", "ai", "tech", "future", "robot",
              "computer", "quantum", "digital", "cyber", "software", "neural",
              "silicon", "machine", "startup", "internet")),
    ("calm", ("nature", "ocean", "forest", "wildlife", "mountain", "river",
              "meditat", "mindful", "garden")),
]
_FULL_WORD_KEYWORDS = frozenset({"ai", "war", "star", "mars"})


def _keyword_pattern(keyword: str) -> str:
    pattern = r"\b" + re.escape(keyword)
    if keyword in _FULL_WORD_KEYWORDS:
        pattern += r"\b"
    return pattern


def mood_for_topic(topic: str | None) -> str:
    """Pick a music mood for a documentary topic via keyword matching."""
    text = (topic or "").lower()
    for mood, keywords in _MOOD_KEYWORDS:
        for keyword in keywords:
            if re.search(_keyword_pattern(keyword), text):
                return mood
    return DEFAULT_MOOD


def music_library_dir() -> Path:
    """Directory holding the curated mood-library tracks."""
    override = os.environ.get(MUSIC_BED_DIR_ENV, "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "assets" / "music"


def music_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Directory for network-provider downloads (already gitignored)."""
    if cache_dir:
        return Path(cache_dir)
    override = os.environ.get(MUSIC_CACHE_DIR_ENV, "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "cache" / "music"


def library_track_for_mood(mood: str) -> Path | None:
    """Return the local library track for *mood*, or None if missing."""
    filename = MOOD_TRACKS.get((mood or "").strip().lower())
    if not filename:
        return None
    candidate = music_library_dir() / filename
    return candidate if candidate.is_file() else None


def is_library_track(path: str | Path) -> str | None:
    """Return the mood if *path* is a track from the mood library."""
    try:
        name = Path(path).name
    except Exception:
        return None
    for mood, filename in MOOD_TRACKS.items():
        if filename == name:
            return mood
    return None

# Integrated loudness target for the pre-normalized music bed. -23 LUFS keeps
# the bed clearly audible in narration gaps while leaving headroom for the
# final mix normalization and the ducking stage.
DEFAULT_BED_TARGET_I = -23.0

# Final mix targets (EBU R128 / YouTube): -16 LUFS integrated, -2 dBTP, 7 LU.
FINAL_MIX_TARGET_I = -16.0
FINAL_MIX_TARGET_TP = -2.0
FINAL_MIX_TARGET_LRA = 7.0


class MusicBedError(RuntimeError):
    """Raised when a music bed cannot be built."""


def resolve_music_track(
    music_bed_path: str | Path | None = None,
    *,
    mood: str | None = None,
    topic: str | None = None,
    allow_network: bool = True,
    cache_dir: str | Path | None = None,
) -> Path | None:
    """Resolve the music bed track for a run, walking the fallback chain.

    Order: explicit ``music_bed_path`` -> ``MUSIC_BED_PATH`` env var ->
    local mood library (``mood``, else :func:`mood_for_topic` on ``topic``)
    -> network providers (Internet Archive, Openverse, Freesound; results
    cached under ``data/cache/music/``) -> ``None`` (the caller falls back
    to the procedural drone).

    A missing file or a failing provider logs a warning and moves to the
    next source rather than failing the run. ``allow_network=False``
    disables the network leg (offline runs, tests).
    """
    from ai_video_factory import music_providers

    wanted_mood = (mood or "").strip().lower()
    if not wanted_mood and topic:
        wanted_mood = mood_for_topic(topic)

    def _note(source: str, path: Path, extra: str = "") -> Path:
        print(f"[music_bed] source={source} mood={wanted_mood or '-'} "
              f"path={path}" + (f" {extra}" if extra else ""))
        return path

    if music_bed_path:
        candidate = Path(music_bed_path)
        if candidate.is_file():
            return _note("explicit", candidate)
        print(f"[music_bed] WARNING: explicit track {candidate} missing; "
              "continuing down the fallback chain.")
    env_path = os.environ.get(MUSIC_BED_PATH_ENV, "").strip()
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return _note("env", candidate)
        print(f"[music_bed] WARNING: {MUSIC_BED_PATH_ENV}={candidate} "
              "missing; continuing down the fallback chain.")
    if wanted_mood:
        track = library_track_for_mood(wanted_mood)
        if track is not None:
            return _note("library", track)
        print(f"[music_bed] WARNING: no local track for mood "
              f"'{wanted_mood}' in {music_library_dir()}; trying network.")
    if wanted_mood and allow_network:
        dest, provider_name, net_track = \
            music_providers.fetch_from_providers(
                wanted_mood,
                cache_dir=music_cache_dir(cache_dir),
            )
        if dest is not None:
            return _note(provider_name, dest,
                         extra=f"title={net_track.title!r} "
                               f"license={net_track.license}")
    elif not allow_network:
        print("[music_bed] network providers disabled "
              "(allow_network=False).")
    return None


def _resolve_ffprobe(ffmpeg_bin: str) -> str:
    """Find ffprobe next to the ffmpeg binary, else rely on PATH."""
    ffmpeg_path = Path(ffmpeg_bin)
    if ffmpeg_path.name != ffmpeg_path.as_posix() or "/" in ffmpeg_bin:
        sibling = ffmpeg_path.parent / "ffprobe"
        if sibling.is_file():
            return str(sibling)
    return "ffprobe"


def _run(
    argv: list[str],
    *,
    name: str,
    cwd: Path,
    timeout: int,
    retries: int = 0,
) -> None:
    """Run a subprocess, retrying transient failures (mirrors the pipeline)."""
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                shell=False,
                timeout=timeout,
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            last_error = sanitize_diagnostic(f"{name} could not run: {error}")
        else:
            if completed.returncode == 0:
                return
            detail = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"exit code {completed.returncode}"
            )
            last_error = sanitize_diagnostic(f"{name} failed: {detail}")
        if attempt < retries:
            print(f"[music_bed] {name}: attempt {attempt + 1} failed, retrying...")
            time.sleep(5)
    raise MusicBedError(last_error or f"{name} failed")


def probe_audio_duration(path: str | Path, *, ffprobe: str | None = None) -> float:
    """Return the duration of an audio file in seconds via ffprobe."""
    path = Path(path)
    ffprobe_bin = ffprobe or _resolve_ffprobe("ffmpeg")
    try:
        completed = subprocess.run(
            [
                ffprobe_bin,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MusicBedError(f"ffprobe could not run on {path}: {error}") from error
    if completed.returncode != 0:
        raise MusicBedError(
            f"ffprobe failed on {path}: "
            f"{sanitize_diagnostic(completed.stderr.strip() or 'unknown error')}"
        )
    try:
        return float(completed.stdout.strip())
    except ValueError as error:
        raise MusicBedError(f"ffprobe returned no duration for {path}") from error


def fit_track_to_duration(
    src: str | Path,
    duration_seconds: float,
    output: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str | None = None,
    fade_seconds: float = 2.0,
    timeout: int = 600,
) -> Path:
    """Fit a music track to exactly ``duration_seconds``.

    Shorter tracks are looped (``-stream_loop``) and longer tracks are
    trimmed; both get a short fade-in and fade-out so loop points and the
    video tail never click. Output is 48 kHz stereo PCM WAV.

    Note: loop seams are not crossfaded. For ambient/cinematic beds played
    quietly under ducked narration this is inaudible in practice; if a seam
    ever becomes audible, prefer a track whose own duration already covers
    the video.
    """
    src = Path(src)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if duration_seconds <= 0:
        raise MusicBedError(f"invalid target duration: {duration_seconds}")
    if not src.is_file():
        raise MusicBedError(f"music source not found: {src}")

    src_duration = probe_audio_duration(src, ffprobe=ffprobe)
    if src_duration <= 0:
        raise MusicBedError(f"music source has no measurable duration: {src}")

    fade = min(fade_seconds, duration_seconds / 4.0, src_duration / 4.0)
    fade_in = min(1.0, fade)
    fade_out_start = max(0.0, duration_seconds - fade)
    base_filter = (
        f"afade=t=in:st=0:d={fade_in:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade:.3f},"
        "aresample=48000,aformat=channel_layouts=stereo"
    )

    if src_duration >= duration_seconds:
        argv = [
            ffmpeg, "-y",
            "-i", str(src),
            "-af", f"atrim=0:{duration_seconds:.3f},{base_filter}",
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
            "-t", f"{duration_seconds:.3f}",
            str(output),
        ]
        name = "FFmpeg music bed trim"
    else:
        loops = math.ceil(duration_seconds / src_duration)
        argv = [
            ffmpeg, "-y",
            "-stream_loop", str(loops),
            "-i", str(src),
            "-af", base_filter,
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
            "-t", f"{duration_seconds:.3f}",
            str(output),
        ]
        name = "FFmpeg music bed loop"
    _run(argv, name=name, cwd=output.parent, timeout=timeout, retries=1)
    return output


def _parse_loudnorm_json(stderr: str) -> dict[str, str] | None:
    """Extract the loudnorm measurement JSON ffmpeg prints to stderr."""
    match = re.search(r"\{[^}]*\"input_i\"[^}]*\}", stderr, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    required = (
        "input_i", "input_tp", "input_lra", "input_thresh", "target_offset",
    )
    if not all(key in data for key in required):
        return None
    return {key: str(data[key]) for key in required}


def loudnorm_dual_pass(
    src: str | Path,
    dst: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    target_i: float = FINAL_MIX_TARGET_I,
    target_tp: float = FINAL_MIX_TARGET_TP,
    target_lra: float = FINAL_MIX_TARGET_LRA,
    timeout: int = 900,
    cwd: Path | None = None,
) -> Path:
    """Loudness-normalize audio with true dual-pass EBU R128.

    Pass 1 measures the input (``print_format=json`` to null); pass 2
    applies the measured values with ``linear=true``. If measurement parsing
    fails for any reason, falls back to single-pass loudnorm so the pipeline
    never breaks on an ffmpeg output-format quirk.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    workdir = cwd or dst.parent
    base = f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"

    measured: dict[str, str] | None = None
    try:
        probe = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(src), "-af",
             f"{base}:print_format=json", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if probe.returncode == 0:
            measured = _parse_loudnorm_json(probe.stderr)
    except (OSError, subprocess.TimeoutExpired):
        measured = None

    if measured is None:
        print("[music_bed] loudnorm measurement parse failed; "
              "falling back to single-pass loudnorm.")
        af = f"{base},aresample=48000"
    else:
        af = (
            f"{base}"
            f":measured_I={measured['input_i']}"
            f":measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}"
            f":measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}"
            ":linear=true,aresample=48000"
        )
    _run(
        [ffmpeg, "-y", "-i", str(src), "-af", af,
         "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(dst)],
        name="FFmpeg loudnorm",
        cwd=workdir,
        timeout=timeout,
        retries=1,
    )
    return dst


def prepare_music_bed(
    src: str | Path,
    duration_seconds: float,
    output: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str | None = None,
    target_i: float = DEFAULT_BED_TARGET_I,
    timeout: int = 900,
) -> Path:
    """Fit a track to the video duration and normalize its loudness.

    Convenience wrapper: :func:`fit_track_to_duration` then dual-pass
    loudnorm to ``target_i`` LUFS. Returns ``output`` (48 kHz stereo WAV)
    ready to be ducked under narration at unity gain.
    """
    output = Path(output)
    fitted = output.parent / f"{output.stem}.fitted.wav"
    try:
        fit_track_to_duration(
            src, duration_seconds, fitted,
            ffmpeg=ffmpeg, ffprobe=ffprobe, timeout=timeout,
        )
        loudnorm_dual_pass(
            fitted, output,
            ffmpeg=ffmpeg,
            target_i=target_i,
            target_tp=-2.0,
            target_lra=7.0,
            timeout=timeout,
            cwd=output.parent,
        )
    finally:
        fitted.unlink(missing_ok=True)
    return output


def duck_filter(music_label: str, narr_label: str, out_label: str) -> str:
    """Build the ffmpeg filter that ducks music under narration.

    ``music_label`` is the filter input for the (pre-normalized, unity-gain)
    music bed, ``narr_label`` the narration sidechain trigger. Uses
    ``sidechaincompress``: while speech is present the bed dips hard with
    a smooth 400 ms release, then returns to its full level in speech
    gaps (25 dB of reduction measured with -3 dBFS-peak gated narration;
    ~10-14 dB with quieter speech). With the bed pre-normalized to
    -23 LUFS, the ducked bed sits roughly 20 dB (or more) under active
    narration.
    """
    return (
        f"[{music_label}]volume=1.0[music_base];"
        f"[music_base][{narr_label}]"
        "sidechaincompress=threshold=0.02:ratio=12:attack=20:release=400"
        f"[{out_label}]"
    )

# ---------------------------------------------------------------------------
# Envelope ducking: deterministic alternative to sidechain compression
# (WanGP Q5 experiment). The pipeline default stays sidechain
# (DEFAULT_DUCK_MODE); envelope mode is opt-in via duck_mode="envelope".
#
# Speech spans come from ai_video_factory.speech_detect.speech_intervals
# (a pre-analysis pass over the narration). Per dip the gain envelope is:
#   (start-attack, 0 dB) -> (start, -depth) -> (end, -depth)
#   -> (end+release, 0 dB)
# Dips separated by pauses shorter than attack+release merge into one dip.
# The envelope drives ffmpeg volume='<piecewise-linear expr>':eval=frame on
# 10 ms frames (asetnsamples=n=480 at 48 kHz); the ducked bed is then mixed
# with the narration via amix=normalize=0 under a true-peak limiter.
# Compare the two modes with scripts/ab_ducking.py.
# ---------------------------------------------------------------------------

DUCK_MODE_SIDECHAIN = "sidechain"
DUCK_MODE_ENVELOPE = "envelope"
DEFAULT_DUCK_MODE = DUCK_MODE_SIDECHAIN

# Envelope ducking defaults (match the WanGP Q5 reference values).
ENVELOPE_DUCK_DEPTH_DB = 10.0
ENVELOPE_DUCK_ATTACK_S = 0.2
ENVELOPE_DUCK_RELEASE_S = 0.6

# 10 ms frames at the 48 kHz mix rate: the volume filter re-evaluates the
# envelope expression once per frame with eval=frame.
ENVELOPE_FRAME_SAMPLES = 480
MIX_SAMPLE_RATE = 48000

# The true-peak limiter runs at 4x the mix rate so inter-sample peaks are
# caught before the final loudness normalization.
LIMITER_OVERSAMPLE = 4


def merge_duck_gaps(
    intervals: list[tuple[float, float]],
    attack_s: float = ENVELOPE_DUCK_ATTACK_S,
    release_s: float = ENVELOPE_DUCK_RELEASE_S,
) -> list[tuple[float, float]]:
    """Merge speech spans separated by pauses shorter than attack+release.

    A pause shorter than the attack+release cycle would let the envelope
    rise and fall pointlessly between words; merging keeps one smooth dip.
    """
    cleaned = sorted(
        ((float(start), float(end)) for start, end in intervals if float(end) > float(start)),
        key=lambda span: span[0],
    )
    merged: list[list[float]] = []
    for start, end in cleaned:
        if merged and start - merged[-1][1] < attack_s + release_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def duck_envelope_points(
    intervals: list[tuple[float, float]],
    *,
    depth_db: float = ENVELOPE_DUCK_DEPTH_DB,
    attack_s: float = ENVELOPE_DUCK_ATTACK_S,
    release_s: float = ENVELOPE_DUCK_RELEASE_S,
) -> list[tuple[float, float]]:
    """Piecewise-linear (time_s, gain_db) envelope points for the merged dips.

    Per dip: (start-attack, 0 dB) -> (start, -depth) -> (end, -depth) ->
    (end+release, 0 dB). Attack ramps that would start before t=0 are
    clamped to 0.
    """
    points: list[tuple[float, float]] = []
    for start, end in merge_duck_gaps(intervals, attack_s, release_s):
        points += [
            (max(0.0, start - attack_s), 0.0),
            (start, -depth_db),
            (end, -depth_db),
            (end + release_s, 0.0),
        ]
    points.sort(key=lambda point: point[0])
    return points


def _fmt_expr(value: float) -> str:
    """Format a float for an ffmpeg expression: fixed decimals, no exponents."""
    return f"{value:.6f}"


def envelope_gain_expression(
    intervals: list[tuple[float, float]],
    *,
    depth_db: float = ENVELOPE_DUCK_DEPTH_DB,
    attack_s: float = ENVELOPE_DUCK_ATTACK_S,
    release_s: float = ENVELOPE_DUCK_RELEASE_S,
) -> str:
    """Build the ffmpeg expression evaluating the ducking envelope as linear gain.

    Piecewise-linear interpolation in dB between the envelope points,
    converted with pow(10, dB/20). Before the first point the first gain
    holds; after the last point the last gain holds. Empty intervals give
    unity gain ("1"). The caller single-quotes the expression in the
    filter graph so its commas survive filtergraph parsing.
    """
    points = duck_envelope_points(
        intervals, depth_db=depth_db, attack_s=attack_s, release_s=release_s
    )
    if not points:
        return "1"
    nested = _fmt_expr(points[-1][1])  # holds after the last point
    for (t0, g0), (t1, g1) in reversed(list(zip(points, points[1:]))):
        span = max(t1 - t0, 1e-6)
        nested = (
            f"if(lt(t,{_fmt_expr(t1)}),"
            f"{_fmt_expr(g0)}+({_fmt_expr(g1 - g0)})*(t-{_fmt_expr(t0)})/{_fmt_expr(span)},"
            f"{nested})"
        )
    nested = f"if(lt(t,{_fmt_expr(points[0][0])}),{_fmt_expr(points[0][1])},{nested})"
    return f"pow(10,({nested})/20)"


def duck_envelope_filter(
    music_label: str,
    out_label: str,
    intervals: list[tuple[float, float]],
    *,
    depth_db: float = ENVELOPE_DUCK_DEPTH_DB,
    attack_s: float = ENVELOPE_DUCK_ATTACK_S,
    release_s: float = ENVELOPE_DUCK_RELEASE_S,
) -> str:
    """Build the ffmpeg filter that applies the speech envelope to the music bed.

    ``music_label`` is the filter input for the (48 kHz, pre-normalized,
    unity-gain) music bed. The bed is framed into exact 10 ms chunks
    (``asetnsamples``) so ``volume`` with ``eval=frame`` re-evaluates the
    envelope expression on every chunk, giving smooth attack/release ramps.
    """
    expression = envelope_gain_expression(
        intervals, depth_db=depth_db, attack_s=attack_s, release_s=release_s
    )
    return (
        f"[{music_label}]asetnsamples=n={ENVELOPE_FRAME_SAMPLES}:p=0,"
        f"volume='{expression}':eval=frame"
        f"[{out_label}]"
    )


def mix_ducked_audio(
    narration: str | Path,
    music: str | Path,
    output: str | Path,
    *,
    duck_mode: str = DEFAULT_DUCK_MODE,
    intervals: list[tuple[float, float]] | None = None,
    depth_db: float = ENVELOPE_DUCK_DEPTH_DB,
    attack_s: float = ENVELOPE_DUCK_ATTACK_S,
    release_s: float = ENVELOPE_DUCK_RELEASE_S,
    ducked_stem: str | Path | None = None,
    target_i: float = FINAL_MIX_TARGET_I,
    target_tp: float = FINAL_MIX_TARGET_TP,
    target_lra: float = FINAL_MIX_TARGET_LRA,
    ffmpeg: str = "ffmpeg",
    ffprobe: str | None = None,
    timeout: int = 900,
    cwd: Path | None = None,
) -> Path:
    """Mix narration over a ducked music bed; return ``output`` (48 kHz stereo WAV).

    ``duck_mode="sidechain"`` reuses :func:`duck_filter` (the pipeline
    default); ``duck_mode="envelope"`` applies the deterministic speech
    envelope from :func:`duck_envelope_filter`. Envelope mode detects
    speech spans with :mod:`ai_video_factory.speech_detect` unless
    ``intervals`` are supplied. Both modes then mix with
    ``amix=normalize=0``, run under a true-peak limiter (4x oversampled
    ``alimiter``), and finish with dual-pass loudnorm to the ``target_*``
    levels. When ``ducked_stem`` is given, the ducked music bed alone is
    also written there (pre-mix, for A/B measurement).
    """
    if duck_mode not in (DUCK_MODE_SIDECHAIN, DUCK_MODE_ENVELOPE):
        raise MusicBedError(
            f"unknown duck_mode {duck_mode!r}: want 'sidechain' or 'envelope'"
        )
    narr_path = Path(narration)
    music_path = Path(music)
    if not narr_path.is_file():
        raise MusicBedError(f"narration file not found: {narr_path}")
    if not music_path.is_file():
        raise MusicBedError(f"music file not found: {music_path}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    workdir = cwd or output.parent

    duration = probe_audio_duration(narr_path, ffprobe=ffprobe)

    if duck_mode == DUCK_MODE_ENVELOPE and intervals is None:
        try:
            from ai_video_factory.speech_detect import speech_intervals
        except ImportError as exc:
            raise MusicBedError(
                "envelope ducking needs ai_video_factory.speech_detect (numpy): "
                f"{exc}"
            ) from exc
        intervals = speech_intervals(narr_path, ffmpeg=ffmpeg)
        speech_s = sum(end - start for start, end in intervals)
        print(
            f"[music_bed] envelope ducking: {len(intervals)} speech spans, "
            f"{speech_s:.1f}s of {duration:.1f}s"
        )

    ceiling = 10 ** (target_tp / 20)
    parts = [
        f"[0:a]aresample={MIX_SAMPLE_RATE},aformat=channel_layouts=stereo,"
        f"apad,atrim=0:{duration:.3f}[narr_fit]",
        f"[1:a]aresample={MIX_SAMPLE_RATE},aformat=channel_layouts=stereo,"
        f"apad,atrim=0:{duration:.3f}[music_fit]",
    ]
    if duck_mode == DUCK_MODE_SIDECHAIN:
        parts.append(duck_filter("music_fit", "narr_fit", "music_ducked_raw"))
    else:
        parts.append(
            duck_envelope_filter(
                "music_fit",
                "music_ducked_raw",
                intervals or [],
                depth_db=depth_db,
                attack_s=attack_s,
                release_s=release_s,
            )
        )
    # The ducked bed feeds both the mix and the optional stem map; split it
    # when the stem is mapped because a filtergraph label can only be
    # consumed once (an unconnected split output is a filtergraph error).
    duck_mix_label = "music_ducked_raw"
    if ducked_stem is not None:
        parts.append("[music_ducked_raw]asplit=2[music_ducked][music_stem]")
        duck_mix_label = "music_ducked"
    parts.append(
        f"[narr_fit][{duck_mix_label}]amix=inputs=2:normalize=0[mixed];"
        f"[mixed]aresample={MIX_SAMPLE_RATE * LIMITER_OVERSAMPLE},"
        f"alimiter=limit={ceiling:.4f}:level=0,"
        f"aresample={MIX_SAMPLE_RATE}[limited]"
    )

    limited_wav = workdir / f"{output.stem}.limited.wav"
    cmd = [
        ffmpeg, "-y", "-nostdin",
        "-i", str(narr_path),
        "-i", str(music_path),
        "-filter_complex", ";".join(parts),
        "-map", "[limited]", "-c:a", "pcm_s16le",
        "-ar", str(MIX_SAMPLE_RATE), "-ac", "2", str(limited_wav),
    ]
    if ducked_stem is not None:
        stem_path = Path(ducked_stem)
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += [
            "-map", "[music_stem]", "-c:a", "pcm_s16le",
            "-ar", str(MIX_SAMPLE_RATE), "-ac", "2", str(stem_path),
        ]
    try:
        _run(cmd, name="FFmpeg ducked mix", cwd=workdir, timeout=timeout, retries=1)
        loudnorm_dual_pass(
            limited_wav,
            output,
            ffmpeg=ffmpeg,
            target_i=target_i,
            target_tp=target_tp,
            target_lra=target_lra,
            timeout=timeout,
            cwd=workdir,
        )
    finally:
        limited_wav.unlink(missing_ok=True)
    return output


def _ebur128_stderr(
    path: str | Path,
    *,
    ffmpeg: str,
    start_s: float | None = None,
    duration_s: float | None = None,
    framelog_verbose: bool = False,
) -> str:
    """Run the ebur128 filter over *path* (or a segment); return ffmpeg stderr."""
    cmd = [ffmpeg, "-nostdin", "-hide_banner"]
    # The ebur128 Summary (integrated loudness, true peak) prints at info
    # level; per-frame logs need verbose.
    cmd += ["-loglevel", "verbose"] if framelog_verbose else ["-loglevel", "info"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(path)]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    framelog = "framelog=verbose" if framelog_verbose else "framelog=quiet"
    cmd += ["-af", f"ebur128={framelog}:peak=true", "-f", "null", "-"]
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MusicBedError(f"ebur128 measurement failed on {path}: {exc}") from exc
    if completed.returncode != 0:
        raise MusicBedError(
            "ebur128 measurement failed on "
            f"{path}: {sanitize_diagnostic(completed.stderr.strip()[-500:] or 'ffmpeg failed')}"
        )
    return completed.stderr


def measure_integrated_loudness(path: str | Path, *, ffmpeg: str = "ffmpeg") -> float:
    """Integrated loudness (LUFS) of *path* via the ebur128 filter."""
    stderr = _ebur128_stderr(path, ffmpeg=ffmpeg)
    match = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", stderr)
    if not match:
        raise MusicBedError(f"could not parse integrated loudness for {path}")
    return float(match.group(1))


def measure_true_peak_dbfs(path: str | Path, *, ffmpeg: str = "ffmpeg") -> float:
    """True peak (dBFS) of *path* via ebur128 peak=true (max over channels)."""
    stderr = _ebur128_stderr(path, ffmpeg=ffmpeg)
    peaks = [float(value) for value in re.findall(r"Peak:\s*(-?[\d.]+)\s*dBFS", stderr)]
    if not peaks:
        raise MusicBedError(f"could not parse true peak for {path}")
    return max(peaks)


def segment_loudness_lufs(
    path: str | Path,
    start_s: float,
    duration_s: float,
    *,
    ffmpeg: str = "ffmpeg",
) -> float | None:
    """Integrated loudness (LUFS) of the ``[start_s, start_s+duration_s)`` segment."""
    stderr = _ebur128_stderr(path, ffmpeg=ffmpeg, start_s=start_s, duration_s=duration_s)
    match = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", stderr)
    return float(match.group(1)) if match else None


def momentary_loudness_series(
    path: str | Path,
    *,
    start_s: float = 0.0,
    duration_s: float | None = None,
    ffmpeg: str = "ffmpeg",
) -> list[float]:
    """400 ms-momentary loudness (LUFS) sampled per 100 ms frame; -inf frames dropped."""
    stderr = _ebur128_stderr(
        path, ffmpeg=ffmpeg, start_s=start_s, duration_s=duration_s, framelog_verbose=True
    )
    values: list[float] = []
    for line in stderr.splitlines():
        match = re.search(r"\bM:\s*(-?(?:[\d.]+|inf))", line)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values
