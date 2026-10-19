#!/usr/bin/env python3
"""A/B test harness: sidechain vs envelope music ducking (WanGP Q5).

Takes a narration WAV and a music bed WAV, renders the identical mix twice
(once per ducking mode) through music_bed.mix_ducked_audio, then prints
measured loudness and pumping metrics for both.

Experiment harness only: the pipeline default (sidechain) is unchanged.
Nothing here is imported by the pipeline.

Usage:
    python scripts/ab_ducking.py narration.wav music.wav --out-dir /tmp/ab_out
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_video_factory import music_bed  # noqa: E402
from ai_video_factory.music_bed import (  # noqa: E402
    DUCK_MODE_ENVELOPE,
    DUCK_MODE_SIDECHAIN,
    ENVELOPE_DUCK_ATTACK_S,
    ENVELOPE_DUCK_DEPTH_DB,
    ENVELOPE_DUCK_RELEASE_S,
)
from ai_video_factory.speech_detect import speech_intervals  # noqa: E402

# Segments shorter than this are skipped by the dip-depth measurement (need
# a settled middle far from attack/release edges).
MIN_MEASURE_SEGMENT_S = 1.5


def _gap_edges(spans: list[tuple[float, float]], duration: float):
    """Yield (start, end) of the gaps between spans, clamped to [0, duration]."""
    cursor = 0.0
    for start, end in spans:
        if start > cursor:
            yield (cursor, start)
        cursor = max(cursor, end)
    if cursor < duration:
        yield (cursor, duration)


def _middle_lufs(stem: Path, start: float, end: float, *, ffmpeg: str) -> float | None:
    """Loudness of the middle half of [start, end) (avoids edge transients)."""
    pad = (end - start) * 0.25
    return music_bed.segment_loudness_lufs(
        stem, start + pad, (end - start) - 2 * pad, ffmpeg=ffmpeg
    )


def dip_depth_stats(
    stem: Path,
    spans: list[tuple[float, float]],
    duration: float,
    *,
    ffmpeg: str,
) -> dict:
    """Achieved dip depth (dB) per speech span: gap loudness minus deep-speech loudness.

    Compares the middle of each long speech span against the mean level of
    the long gaps, on the ducked music stem (pre-mix, so the measurement
    reflects the ducking alone).
    """
    gap_levels = [
        level
        for start, end in _gap_edges(spans, duration)
        if end - start >= MIN_MEASURE_SEGMENT_S
        for level in [_middle_lufs(stem, start, end, ffmpeg=ffmpeg)]
        if level is not None
    ]
    speech_levels = [
        (index, level)
        for index, (start, end) in enumerate(spans)
        if end - start >= MIN_MEASURE_SEGMENT_S
        for level in [_middle_lufs(stem, start, end, ffmpeg=ffmpeg)]
        if level is not None
    ]
    if not gap_levels or not speech_levels:
        return {"n_spans": 0, "note": "not enough long speech/gap segments to measure"}
    gap_mean = statistics.fmean(gap_levels)
    depths = [gap_mean - level for _, level in speech_levels]
    return {
        "n_spans": len(depths),
        "gap_lufs": round(gap_mean, 1),
        "speech_lufs": round(statistics.fmean([lvl for _, lvl in speech_levels]), 1),
        "depth_mean_db": round(statistics.fmean(depths), 1),
        "depth_std_db": round(statistics.pstdev(depths), 1) if len(depths) > 1 else 0.0,
        "depth_min_db": round(min(depths), 1),
        "depth_max_db": round(max(depths), 1),
    }


def pumping_std(
    stem: Path,
    spans: list[tuple[float, float]],
    *,
    ffmpeg: str,
) -> dict:
    """Gain-pumping metric: std of 400 ms-momentary loudness over the longest span.

    A flat envelope dip gives ~0; a compressor riding the gain (pumping /
    breathing) shows higher variance. Measured on the ducked stem, skipping
    0.5 s after span start and 0.2 s before span end to avoid the ramps.
    """
    if not spans:
        return {"lu": None, "note": "no speech spans"}
    start, end = max(spans, key=lambda span: span[1] - span[0])
    window_start = start + 0.5
    window_end = end - 0.2
    if window_end - window_start < 1.0:
        return {"lu": None, "note": "longest span too short to measure pumping"}
    series = music_bed.momentary_loudness_series(
        stem, start_s=window_start, duration_s=window_end - window_start, ffmpeg=ffmpeg
    )
    # Drop the first 0.5 s: ebur128's 400 ms momentary window starts empty
    # after an input seek, so the first frames read as silence (-120 LUFS).
    series = series[5:]
    if len(series) < 5:
        return {"lu": None, "note": "too few momentary frames"}
    return {"lu": round(statistics.pstdev(series), 2), "n_frames": len(series)}


def render_mode(
    mode: str,
    narration: Path,
    music: Path,
    out_dir: Path,
    intervals: list[tuple[float, float]],
    args,
) -> dict:
    """Render one mix + ducked stem; return measured metrics."""
    mix_path = out_dir / f"{mode}_mix.wav"
    stem_path = out_dir / f"{mode}_stem.wav"
    music_bed.mix_ducked_audio(
        narration,
        music,
        mix_path,
        duck_mode=mode,
        intervals=intervals,
        depth_db=args.depth,
        attack_s=args.attack,
        release_s=args.release,
        ducked_stem=stem_path,
        ffmpeg=args.ffmpeg,
        timeout=args.timeout,
    )
    duration = music_bed.probe_audio_duration(narration)
    return {
        "mix": str(mix_path),
        "stem": str(stem_path),
        "integrated_lufs": round(music_bed.measure_integrated_loudness(mix_path, ffmpeg=args.ffmpeg), 1),
        "true_peak_dbfs": round(music_bed.measure_true_peak_dbfs(mix_path, ffmpeg=args.ffmpeg), 1),
        "dip": dip_depth_stats(stem_path, intervals, duration, ffmpeg=args.ffmpeg),
        "pumping": pumping_std(stem_path, intervals, ffmpeg=args.ffmpeg),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="A/B test sidechain vs envelope music ducking on a narration+music pair."
    )
    parser.add_argument("narration", type=Path, help="Narration WAV (the ducking trigger).")
    parser.add_argument("music", type=Path, help="Music bed WAV.")
    parser.add_argument("--out-dir", type=Path, default=Path("ab_ducking_out"),
                        help="Where to write mixes, stems, and metrics.json.")
    parser.add_argument("--depth", type=float, default=ENVELOPE_DUCK_DEPTH_DB,
                        help="Envelope dip depth in dB (default 10).")
    parser.add_argument("--attack", type=float, default=ENVELOPE_DUCK_ATTACK_S,
                        help="Envelope attack in seconds (default 0.2).")
    parser.add_argument("--release", type=float, default=ENVELOPE_DUCK_RELEASE_S,
                        help="Envelope release in seconds (default 0.6).")
    parser.add_argument("--intervals-json", type=Path, default=None,
                        help="Optional precomputed [[start, end], ...] speech spans; "
                             "detected automatically when omitted.")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.narration.is_file():
        print(f"narration not found: {args.narration}", file=sys.stderr)
        return 2
    if not args.music.is_file():
        print(f"music not found: {args.music}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.intervals_json is not None:
        intervals = [
            (float(start), float(end)) for start, end in json.loads(args.intervals_json.read_text())
        ]
        print(f"loaded {len(intervals)} speech spans from {args.intervals_json}")
    else:
        print("detecting speech spans in narration...")
        intervals = speech_intervals(args.narration, ffmpeg=args.ffmpeg)
    speech_s = sum(end - start for start, end in intervals)
    duration = music_bed.probe_audio_duration(args.narration)
    print(f"speech: {len(intervals)} spans, {speech_s:.1f}s of {duration:.1f}s")

    results = {}
    for mode in (DUCK_MODE_SIDECHAIN, DUCK_MODE_ENVELOPE):
        print(f"\nrendering {mode}...")
        results[mode] = render_mode(mode, args.narration, args.music, args.out_dir, intervals, args)

    metrics_path = args.out_dir / "metrics.json"
    payload = {
        "narration": str(args.narration),
        "music": str(args.music),
        "duration_s": round(duration, 2),
        "speech_spans": [[round(s, 3), round(e, 3)] for s, e in intervals],
        "envelope_params": {"depth_db": args.depth, "attack_s": args.attack, "release_s": args.release},
        "modes": results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nmetrics written to {metrics_path}")

    print("\n=== Ducking A/B results ===")
    print(f"narration: {args.narration} ({duration:.1f}s, {len(intervals)} speech spans)")
    for mode in (DUCK_MODE_SIDECHAIN, DUCK_MODE_ENVELOPE):
        r = results[mode]
        dip = r["dip"]
        pump = r["pumping"]
        print(f"\n[{mode}]")
        print(f"  mix integrated loudness: {r['integrated_lufs']} LUFS")
        print(f"  mix true peak:           {r['true_peak_dbfs']} dBFS")
        if dip.get("n_spans"):
            print(f"  stem dip depth:  mean {dip['depth_mean_db']} dB "
                  f"(std {dip['depth_std_db']}, min {dip['depth_min_db']}, "
                  f"max {dip['depth_max_db']}, n={dip['n_spans']})")
            print(f"    gap level {dip['gap_lufs']} LUFS vs speech level {dip['speech_lufs']} LUFS")
        else:
            print(f"  stem dip depth:  n/a ({dip.get('note')})")
        if pump.get("lu") is not None:
            print(f"  stem pumping (momentary std over longest span): {pump['lu']} LU")
        else:
            print(f"  stem pumping: n/a ({pump.get('note')})")
        print(f"  mix:  {r['mix']}")
        print(f"  stem: {r['stem']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
