from ai_video_factory.benchmark import run_benchmarks


def test_benchmark_records_success_and_failure_duration() -> None:
    times = iter([1.0, 1.125, 2.0, 2.250])
    results = {
        ("ffmpeg", "-version"): (0, "ffmpeg version 8", ""),
        ("vulkaninfo", "--summary"): (1, "", "device unavailable"),
    }
    probes = run_benchmarks(lambda argv: results[tuple(argv)], lambda: next(times))
    assert [(p.name, p.status, p.duration_ms) for p in probes] == [
        ("ffmpeg-startup", "ready", 125),
        ("vulkan-enumeration", "not_ready", 250),
    ]
