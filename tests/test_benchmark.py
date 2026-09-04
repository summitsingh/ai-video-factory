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


def test_benchmark_redacts_sensitive_http_detail_lines() -> None:
    sensitive_lines = (
        "Authorization: Bearer secret-token",
        "Cookie: session=secret-cookie",
        "Set-Cookie: session=secret-cookie",
    )
    for line in sensitive_lines:
        probes = run_benchmarks(
            lambda argv, line=line: (0, line, ""),
            iter((1.0, 1.001, 2.0, 2.001)).__next__,
        )
        assert probes[0].detail.endswith("[REDACTED]")
        assert "secret" not in probes[0].detail
