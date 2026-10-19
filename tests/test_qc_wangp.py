"""Tests for the WanGP-adopted QC gates: Q1 cut detection, Q2 audio/video
skew, Q3 warning-tier silence, Q7 clipping + loudness.

Pure unit tests (synthetic scores/payloads/stderr) always run; the
ffmpeg-backed fixture tests skip when ffmpeg is unavailable.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from ai_video_factory.asset_qc import (
    CUT_POLICY_ADVISORY,
    CUT_POLICY_HARD_FAIL,
    _cut_frame_indexes,
    detect_cuts,
    verify_video_asset,
)
from ai_video_factory.qc_final import (
    _parse_ebur128_summary,
    _parse_silence_ranges,
    check_audio_clipping,
    check_audio_loudness,
    check_audio_video_skew,
    check_mid_shot_cuts,
    run_final_qc,
)

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
        capture_output=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic fixtures generated with ffmpeg (skipped without ffmpeg)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    work = tmp_path_factory.mktemp("wangp_qc_media")

    cut = work / "cut.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=red:s=320x180:r=24:d=2",
        "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=24:d=2",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cut),
    )

    plain = work / "plain.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=0x336699:s=320x180:r=24:d=4",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-af", "loudnorm=I=-16:TP=-2:LRA=11",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(plain),
    )

    skew = work / "skew.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=0x336699:s=320x180:r=24:d=5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-map", "0:v", "-map", "1:a", "-t", "5", str(skew),
    )

    aligned = work / "aligned.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=0x336699:s=320x180:r=24:d=5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-map", "0:v", "-map", "1:a", "-t", "5", str(aligned),
    )

    clipped = work / "clipped.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "aevalsrc=1.5*sin(2*PI*440*t):d=2",
        "-c:a", "pcm_f32le", str(clipped),
    )

    clean = work / "clean.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2,volume=0.5",
        "-c:a", "pcm_f32le", str(clean),
    )

    norm = work / "norm.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-af", "loudnorm=I=-16:TP=-2:LRA=11",
        "-c:a", "pcm_s16le", str(norm),
    )

    hot = work / "hot.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2,volume=0.9",
        "-c:a", "pcm_f32le", str(hot),
    )

    gap = work / "gap.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-af", r"volume=if(between(t\,1\,2.5)\,0.001\,1):eval=frame",
        "-c:a", "pcm_s16le", str(gap),
    )

    return work


# ---------------------------------------------------------------------------
# Q1: cut detection
# ---------------------------------------------------------------------------


def test_cut_frame_indexes_detects_isolated_spike() -> None:
    scores = [0.01] * 6 + [0.5] + [0.01] * 6
    assert _cut_frame_indexes(scores) == [6]


def test_cut_frame_indexes_ignores_sub_threshold_score() -> None:
    assert _cut_frame_indexes([0.01] * 6 + [0.07] + [0.01] * 6) == []


def test_cut_frame_indexes_ignores_fast_motion() -> None:
    # High neighbours: 0.5 < 3 * 0.3, so not a cut.
    assert _cut_frame_indexes([0.3] * 6 + [0.5] + [0.3] * 6) == []


def test_detect_cuts_finds_hard_cut(media_dir: Path) -> None:
    cuts = detect_cuts(media_dir / "cut.mp4")
    assert len(cuts) == 1
    assert abs(cuts[0] - 2.0) < 0.15


def test_detect_cuts_clean_clip_has_none(media_dir: Path) -> None:
    assert detect_cuts(media_dir / "plain.mp4") == []


def test_verify_video_asset_hard_fail_on_cuts(media_dir: Path) -> None:
    verdict = verify_video_asset(
        media_dir / "cut.mp4", cut_policy=CUT_POLICY_HARD_FAIL
    )
    assert not verdict["ok"]
    assert "mid_shot_cuts" in verdict["reasons"]
    assert verdict["scores"]["cut_count"] == 1.0


def test_verify_video_asset_advisory_on_cuts(media_dir: Path) -> None:
    verdict = verify_video_asset(
        media_dir / "cut.mp4", cut_policy=CUT_POLICY_ADVISORY
    )
    assert verdict["ok"]
    assert "mid_shot_cuts" not in verdict["reasons"]
    assert verdict["scores"]["cut_count"] == 1.0
    assert len(verdict["scores"]["cut_times_seconds"]) == 1


def test_verify_video_asset_rejects_bad_cut_policy(media_dir: Path) -> None:
    with pytest.raises(ValueError):
        verify_video_asset(media_dir / "cut.mp4", cut_policy="bogus")


def test_check_mid_shot_cuts_policies(media_dir: Path) -> None:
    hard = check_mid_shot_cuts(
        FFMPEG, media_dir / "cut.mp4", cut_policy=CUT_POLICY_HARD_FAIL
    )
    assert not hard.passed
    assert hard.name == "mid-shot-cuts"
    advisory = check_mid_shot_cuts(
        FFMPEG, media_dir / "cut.mp4", cut_policy=CUT_POLICY_ADVISORY
    )
    assert advisory.passed
    assert "2.00s" in advisory.detail
    with pytest.raises(ValueError):
        check_mid_shot_cuts(FFMPEG, media_dir / "cut.mp4", cut_policy="bogus")


# ---------------------------------------------------------------------------
# Q2: audio ends before video
# ---------------------------------------------------------------------------


def _skew_payload(video_s: float, audio_s: float | None) -> dict:
    streams = [{"codec_type": "video", "duration": str(video_s)}]
    if audio_s is not None:
        streams.append({"codec_type": "audio", "duration": str(audio_s)})
    return {"streams": streams, "format": {"duration": str(video_s)}}


def test_audio_video_skew_boundary() -> None:
    assert not check_audio_video_skew(_skew_payload(10.0, 9.0)).passed
    assert check_audio_video_skew(_skew_payload(10.0, 9.01)).passed
    assert check_audio_video_skew(_skew_payload(10.0, 10.0)).passed
    # Audio outrunning the video is not the skew defect class.
    assert check_audio_video_skew(_skew_payload(10.0, 10.5)).passed


def test_audio_video_skew_missing_audio_fails() -> None:
    check = check_audio_video_skew(_skew_payload(10.0, None))
    assert not check.passed
    assert "no audio stream" in check.detail


def test_audio_video_skew_real_file(media_dir: Path) -> None:
    from ai_video_factory.qc_final import _probe, _resolve_ffprobe

    ffprobe = _resolve_ffprobe(FFMPEG)
    skewed = check_audio_video_skew(_probe(ffprobe, media_dir / "skew.mp4"))
    assert not skewed.passed
    assert abs(skewed.value - 2.0) < 0.15
    aligned = check_audio_video_skew(_probe(ffprobe, media_dir / "aligned.mp4"))
    assert aligned.passed


# ---------------------------------------------------------------------------
# Q3: warning-tier silence
# ---------------------------------------------------------------------------


def test_parse_silence_ranges() -> None:
    stderr = (
        "[silencedetect @ 0x1] silence_start: 1.0\n"
        "[silencedetect @ 0x1] silence_end: 2.5 | silence_duration: 1.5\n"
        "[silencedetect @ 0x1] silence_start: 3.0\n"
    )
    assert _parse_silence_ranges(stderr, 4.0) == [(1.0, 2.5), (3.0, 4.0)]


def test_silence_warning_tier_records_near_silence(media_dir: Path) -> None:
    from ai_video_factory.qc_final import (
        check_audio_silence,
        check_audio_silence_warning,
    )

    target = media_dir / "gap.wav"
    hard = check_audio_silence(FFMPEG, target, 4.0)
    assert hard.passed  # -60 dB gap is invisible to the -70 dB hard tier
    warning = check_audio_silence_warning(FFMPEG, target, 4.0)
    assert warning.passed  # the warning tier never fails
    assert warning.name == "audio-silence-warning"
    assert abs(warning.value - 1.5) < 0.15
    assert "1.0-2.5s" in warning.detail


def test_run_final_qc_includes_warning_tier(media_dir: Path) -> None:
    report = run_final_qc(
        media_dir / "plain.mp4",
        target_width=320,
        target_height=180,
        target_duration_seconds=4.0,
        ffmpeg=FFMPEG,
    )
    names = [check.name for check in report.checks]
    assert "audio-silence-warning" in names
    warning = next(c for c in report.checks if c.name == "audio-silence-warning")
    assert warning.passed


# ---------------------------------------------------------------------------
# Q7: clipping + loudness
# ---------------------------------------------------------------------------


def test_parse_ebur128_summary() -> None:
    stderr = (
        "[Parsed_ebur128_0 @ 0xabc] Summary:\n"
        "[Parsed_ebur128_0 @ 0xabc]   Integrated loudness:\n"
        "[Parsed_ebur128_0 @ 0xabc]     I:         -16.3 LUFS\n"
        "[Parsed_ebur128_0 @ 0xabc]     Threshold:   -26.6 LUFS\n"
        "[Parsed_ebur128_0 @ 0xabc]   True peak:\n"
        "[Parsed_ebur128_0 @ 0xabc]     Peak:        -1.8 dBFS\n"
    )
    assert _parse_ebur128_summary(stderr) == (-16.3, -1.8)


def test_parse_ebur128_summary_missing() -> None:
    assert _parse_ebur128_summary("no summary here") == (None, None)


def test_parse_ebur128_summary_minus_inf() -> None:
    stderr = "Summary:\n    I:         -inf LUFS\n    Peak:        -91.0 dBFS\n"
    assert _parse_ebur128_summary(stderr) == (None, -91.0)


def test_clipping_fails_on_overs(media_dir: Path) -> None:
    check = check_audio_clipping(FFMPEG, media_dir / "clipped.wav", True)
    assert not check.passed
    assert check.value > 0
    assert "clipping" in check.detail


def test_clipping_passes_clean(media_dir: Path) -> None:
    check = check_audio_clipping(FFMPEG, media_dir / "clean.wav", True)
    assert check.passed
    assert check.value == 0


def test_clipping_no_audio_fails(media_dir: Path) -> None:
    check = check_audio_clipping(FFMPEG, media_dir / "clean.wav", False)
    assert not check.passed


def test_loudness_passes_normalized(media_dir: Path) -> None:
    check = check_audio_loudness(FFMPEG, media_dir / "norm.wav", True)
    assert check.passed, check.detail


def test_loudness_fails_hot_mix(media_dir: Path) -> None:
    check = check_audio_loudness(FFMPEG, media_dir / "hot.wav", True)
    assert not check.passed
    assert "integrated" in check.detail


def test_loudness_fails_clipped(media_dir: Path) -> None:
    check = check_audio_loudness(FFMPEG, media_dir / "clipped.wav", True)
    assert not check.passed
    assert "true peak" in check.detail


# ---------------------------------------------------------------------------
# Full run_final_qc integration
# ---------------------------------------------------------------------------


def test_run_final_qc_full_report(media_dir: Path) -> None:
    report = run_final_qc(
        media_dir / "plain.mp4",
        target_width=320,
        target_height=180,
        target_duration_seconds=4.0,
        ffmpeg=FFMPEG,
        report_path=media_dir / "qc-report.json",
    )
    assert report.passed, [
        (c.name, c.passed, c.detail) for c in report.checks if not c.passed
    ]
    names = {check.name for check in report.checks}
    assert names == {
        "resolution",
        "duration",
        "black-frames",
        "audio-silence",
        "audio-silence-warning",
        "audio-peak",
        "audio-clipping",
        "audio-loudness",
        "audio-video-skew",
        "mid-shot-cuts",
    }
    assert (media_dir / "qc-report.json").is_file()


def test_run_final_qc_cut_policy_hard_fail(media_dir: Path) -> None:
    report = run_final_qc(
        media_dir / "cut.mp4",
        target_width=320,
        target_height=180,
        target_duration_seconds=4.0,
        ffmpeg=FFMPEG,
        cut_policy=CUT_POLICY_HARD_FAIL,
    )
    cuts = next(c for c in report.checks if c.name == "mid-shot-cuts")
    assert not cuts.passed
