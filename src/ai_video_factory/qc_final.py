"""Post-render final QC gate for the video pipeline.

Runs hard-fail technical checks against the finished (polished) master and
writes a machine-readable ``qc-report.json`` into the run directory. Any
failing check flips the pipeline run status to ``failed``.

Checks:
  * ``black-frames``: more than 15% of frames sampled at ~1 fps are
    near-black (mean luma < 16).
  * ``audio-silence``: any continuous run of digital silence longer than
    3 seconds in the mix (ffmpeg ``silencedetect``).
  * ``audio-peak``: overall audio peak below -30 dBFS (ffmpeg
    ``volumedetect``); missing/unmeasurable audio also fails.
  * ``resolution``: output resolution must exactly match the target (the
    pipeline's 2.39:1 letterboxed variant is accepted, mirroring
    :func:`ai_video_factory.qc.evaluate_qc`).
  * ``duration``: output duration within 5% of the target.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ai_video_factory.edit_schema import EditDocument
from ai_video_factory.subtitle_export import ACT_CARD_TOTAL_FRAMES

SCHEMA_VERSION = 1

# Check thresholds.
BLACK_LUMA_THRESHOLD = 16.0
BLACK_FRAME_RATIO_LIMIT = 0.15
BLACK_SAMPLE_FPS = 1.0
#: Brightest-1% pixel mean below which a low-luma frame counts as truly
#: empty. Mirrors asset_qc.BLACK_CONTENT_THRESHOLD: separates empty black
#: frames (missing/failed assets, the defect this gate exists to catch)
#: from legitimate dark cinematography such as starfields and night
#: scenes, which asset QC deliberately accepts.
BLACK_CONTENT_THRESHOLD = 15.0
SILENCE_NOISE_DB = -70.0
SILENCE_MAX_SECONDS = 3.0
AUDIO_PEAK_MIN_DBFS = -30.0
DURATION_TOLERANCE = 0.05
_LETTERBOX_ASPECT = 2.39
_LETTERBOX_ASPECT_TOL = 0.02


@dataclass
class FinalQcCheck:
    name: str
    passed: bool
    detail: str
    value: float | str | None = None


@dataclass
class FinalQcReport:
    master: str
    target: dict
    checks: list[FinalQcCheck] = field(default_factory=list)
    report_path: str | None = None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def status(self) -> str:
        return "pass" if self.passed else "fail"

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "tool": "qc_final",
            "master": self.master,
            "target": self.target,
            "status": self.status,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "detail": check.detail,
                    "value": check.value,
                }
                for check in self.checks
            ],
        }

    def write_json(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        self.report_path = str(destination)
        return destination


def _resolve_ffprobe(ffmpeg_bin: str) -> str:
    """Find an ffprobe binary: next to ffmpeg, else on PATH."""
    if ffmpeg_bin and Path(ffmpeg_bin).name != "ffmpeg":
        candidate = Path(ffmpeg_bin).parent / "ffprobe"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffprobe") or "ffprobe"


def _probe(ffprobe_bin: str, master: Path) -> dict:
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "stream=index,codec_type,width,height,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(master),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    try:
        return json.loads(completed.stdout or "{}")
    except ValueError:
        return {}


def _first_stream(payload: dict, codec_type: str) -> dict | None:
    for stream in payload.get("streams") or []:
        if stream.get("codec_type") == codec_type:
            return stream
    return None


def _stream_or_format_duration(stream: dict | None, payload: dict) -> float | None:
    for source in (stream, payload.get("format")):
        if source:
            try:
                return float(source.get("duration"))
            except (TypeError, ValueError):
                continue
    return None


def check_resolution(
    payload: dict, target_width: int, target_height: int
) -> FinalQcCheck:
    video = _first_stream(payload, "video")
    width = int(video.get("width") or 0) if video else 0
    height = int(video.get("height") or 0) if video else 0
    aspect = width / height if height else 0.0
    # Accept the exact canvas or the pipeline's 2.39:1 letterboxed variant,
    # mirroring ai_video_factory.qc.evaluate_qc. Either way the pixel
    # dimensions must match exactly: no percentage tolerance.
    accepted = (width == target_width and height == target_height) or (
        width == target_width
        and abs(aspect - _LETTERBOX_ASPECT) <= _LETTERBOX_ASPECT_TOL
    )
    return FinalQcCheck(
        name="resolution",
        passed=accepted,
        detail=(
            f"expected {target_width}x{target_height} (or {_LETTERBOX_ASPECT}:1 "
            f"letterbox at width {target_width}); got {width}x{height}"
        ),
        value=f"{width}x{height}",
    )


def check_duration(
    payload: dict, target_duration_seconds: float
) -> FinalQcCheck:
    video = _first_stream(payload, "video")
    actual = _stream_or_format_duration(video, payload)
    if actual is None or target_duration_seconds <= 0:
        return FinalQcCheck(
            name="duration",
            passed=False,
            detail=f"duration unavailable (target {target_duration_seconds:.2f}s)",
            value=actual,
        )
    drift = abs(actual - target_duration_seconds) / target_duration_seconds
    return FinalQcCheck(
        name="duration",
        passed=drift <= DURATION_TOLERANCE,
        detail=(
            f"target {target_duration_seconds:.2f}s; got {actual:.2f}s "
            f"(drift {drift * 100:.2f}%, limit {DURATION_TOLERANCE * 100:.0f}%)"
        ),
        value=round(actual, 3),
    )


def black_frame_exclude_windows(doc: EditDocument) -> list[tuple[float, float]]:
    """Dark-by-design windows to exclude from the near-black check.

    Intro/outro scenes render as dark branded sequences, and act-card
    scenes open with an 84-frame dark-scrim card (mirroring the Remotion
    renderer's ACT_CARD_TOTAL_FRAMES). Sampling those windows punishes the
    video for intended darkness: a dark-branded outro alone can trip the
    15% gate even though nothing is defective. Returns (start, end) seconds
    for every window where darkness is by design rather than a defect.
    """
    windows: list[tuple[float, float]] = []
    fps = float(doc.fps)
    card_seconds = ACT_CARD_TOTAL_FRAMES / fps
    for scene in doc.scenes:
        start = scene.from_frame / fps
        end = (scene.from_frame + scene.duration_frames) / fps
        if scene.kind in ("intro", "outro"):
            windows.append((start, end))
        elif scene.act:
            windows.append((start, min(end, start + card_seconds)))
    return windows


def _excluded_sample_mask(
    num_frames: int, exclude_windows: list[tuple[float, float]] | None
) -> "np.ndarray":
    """Boolean mask of 1fps samples whose timestamp falls in an exclude window."""
    mask = np.zeros(num_frames, dtype=bool)
    if not exclude_windows:
        return mask
    times = np.arange(num_frames, dtype=float) / BLACK_SAMPLE_FPS
    for start, end in exclude_windows:
        mask |= (times >= start) & (times < end)
    return mask


def check_black_frames(
    ffmpeg_bin: str,
    master: Path,
    exclude_windows: list[tuple[float, float]] | None = None,
) -> FinalQcCheck:
    """Fail when >15% of ~1fps sampled frames are truly empty.

    ``exclude_windows`` is a list of (start, end) seconds where dark frames
    are by design (intro/outro branded cards, act-card scrim windows - see
    :func:`black_frame_exclude_windows`). Samples inside those windows are
    ignored so intended darkness never counts as a defect.

    A frame counts as empty only when BOTH its mean luma is below
    BLACK_LUMA_THRESHOLD AND the mean of its brightest 1% of pixels is
    below BLACK_CONTENT_THRESHOLD. Legitimate dark cinematography
    (starfields, night skies, night-city footage) holds bright content and
    never counts, matching asset QC's deliberate acceptance of such footage
    (see asset_qc.BLACK_CONTENT_THRESHOLD). Genuinely empty frames from
    missing or failed assets still fail, as before.
    """
    completed = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(master),
            "-vf",
            f"fps={BLACK_SAMPLE_FPS},scale=160:90,format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        capture_output=True,
        timeout=600,
        check=False,
    )
    frame_bytes = 160 * 90
    raw = completed.stdout
    num_frames = len(raw) // frame_bytes
    if num_frames == 0:
        return FinalQcCheck(
            name="black-frames",
            passed=False,
            detail="no frames could be sampled from the master",
            value=0,
        )
    frames = (
        np.frombuffer(raw[: num_frames * frame_bytes], dtype=np.uint8)
        .reshape(num_frames, frame_bytes)
        .astype(float)
    )
    excluded = _excluded_sample_mask(num_frames, exclude_windows)
    measured_frames = frames[~excluded]
    if measured_frames.shape[0] == 0:
        return FinalQcCheck(
            name="black-frames",
            passed=False,
            detail="no content frames outside excluded windows could be sampled",
            value=0,
        )
    mean_luma = measured_frames.mean(axis=1)
    top1_count = max(frame_bytes // 100, 1)
    brightest1 = np.partition(
        measured_frames, frame_bytes - top1_count, axis=1
    )[:, frame_bytes - top1_count :].mean(axis=1)
    black = int(
        ((mean_luma < BLACK_LUMA_THRESHOLD) & (brightest1 < BLACK_CONTENT_THRESHOLD)).sum()
    )
    ratio = black / measured_frames.shape[0]
    excluded_note = (
        f"; {int(excluded.sum())} dark-by-design samples excluded"
        if exclude_windows
        else ""
    )
    return FinalQcCheck(
        name="black-frames",
        passed=ratio <= BLACK_FRAME_RATIO_LIMIT,
        detail=(
            f"{black}/{measured_frames.shape[0]} content frames truly empty "
            f"(mean luma < {BLACK_LUMA_THRESHOLD:g} with brightest-1% < {BLACK_CONTENT_THRESHOLD:g}; ratio {ratio:.2%}, "
            f"limit {BLACK_FRAME_RATIO_LIMIT:.0%}{excluded_note})"
        ),
        value=round(ratio, 4),
    )


_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(
    r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)"
)


def _parse_silence_gaps(stderr_text: str, total_duration: float | None) -> list[float]:
    """Parse silencedetect stderr into a list of silence-run durations."""
    gaps: list[float] = []
    pending_start: float | None = None
    for line in stderr_text.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            gaps.append(float(end_match.group(2)))
            pending_start = None
    if pending_start is not None and total_duration is not None:
        # Trailing silence runs to end of file.
        gaps.append(max(0.0, total_duration - pending_start))
    return gaps


def check_audio_silence(
    ffmpeg_bin: str, master: Path, total_duration: float | None
) -> FinalQcCheck:
    """Fail on any continuous silence longer than 3s in the mix."""
    completed = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-i",
            str(master),
            "-vn",
            "-af",
            f"silencedetect=noise={SILENCE_NOISE_DB:g}dB:d={SILENCE_MAX_SECONDS:g}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    gaps = _parse_silence_gaps(completed.stderr, total_duration)
    longest = max(gaps) if gaps else 0.0
    failed = any(gap > SILENCE_MAX_SECONDS for gap in gaps)
    return FinalQcCheck(
        name="audio-silence",
        passed=not failed,
        detail=(
            f"{len(gaps)} silence run(s) detected; longest "
            f"{longest:.2f}s (limit {SILENCE_MAX_SECONDS:g}s continuous)"
        ),
        value=round(longest, 3),
    )


_PEAK_RE = re.compile(r"max_volume:\s*(-?[0-9.]+|n/a)\s*dB")


def _parse_peak_db(stderr_text: str) -> float | None:
    """Parse volumedetect stderr into the max_volume peak in dBFS."""
    peak: float | None = None
    for line in stderr_text.splitlines():
        match = _PEAK_RE.search(line)
        if match and match.group(1) != "n/a":
            peak = float(match.group(1))
    return peak


def check_audio_peak(
    ffmpeg_bin: str, master: Path, has_audio: bool
) -> FinalQcCheck:
    """Fail when the mix peak is below -30 dBFS (or audio is missing)."""
    if not has_audio:
        return FinalQcCheck(
            name="audio-peak",
            passed=False,
            detail="no audio stream in the master",
            value=None,
        )
    completed = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-i",
            str(master),
            "-vn",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    peak = _parse_peak_db(completed.stderr)
    if peak is None:
        return FinalQcCheck(
            name="audio-peak",
            passed=False,
            detail="audio peak could not be measured",
            value=None,
        )
    return FinalQcCheck(
        name="audio-peak",
        passed=peak >= AUDIO_PEAK_MIN_DBFS,
        detail=f"peak {peak:.1f} dBFS (minimum {AUDIO_PEAK_MIN_DBFS:g} dBFS)",
        value=round(peak, 2),
    )


def run_final_qc(
    master_path: str | Path,
    *,
    target_width: int,
    target_height: int,
    target_duration_seconds: float,
    ffmpeg: str = "ffmpeg",
    ffprobe: str | None = None,
    report_path: str | Path | None = None,
    exclude_windows: list[tuple[float, float]] | None = None,
) -> FinalQcReport:
    """Run the post-render QC gate on a finished master file.

    Writes ``qc-report.json`` (default: next to the master) and returns the
    report. ``report.passed`` is False when any check fails; the pipeline
    hook turns that into ``status="fail"``.

    ``exclude_windows`` is forwarded to the black-frames check: (start, end)
    seconds where dark frames are by design (see
    :func:`black_frame_exclude_windows`).
    """
    master = Path(master_path)
    ffprobe_bin = ffprobe or _resolve_ffprobe(ffmpeg)
    payload = _probe(ffprobe_bin, master)
    video = _first_stream(payload, "video")
    audio = _first_stream(payload, "audio")
    total_duration = _stream_or_format_duration(video, payload)

    report = FinalQcReport(
        master=str(master),
        target={
            "width": target_width,
            "height": target_height,
            "duration_seconds": target_duration_seconds,
        },
    )
    report.checks.append(check_resolution(payload, target_width, target_height))
    report.checks.append(check_duration(payload, target_duration_seconds))
    report.checks.append(check_black_frames(ffmpeg, master, exclude_windows))
    report.checks.append(check_audio_silence(ffmpeg, master, total_duration))
    report.checks.append(
        check_audio_peak(ffmpeg, master, has_audio=audio is not None)
    )

    destination = (
        Path(report_path) if report_path is not None else master.parent / "qc-report.json"
    )
    report.write_json(destination)
    return report
