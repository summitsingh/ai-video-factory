import subprocess

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


def test_benchmark_uses_only_the_two_exact_fixed_argv_vectors() -> None:
    """Changing benchmark argv could expand its local-only probe surface."""
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> tuple[int, str, str]:
        calls.append(tuple(argv))
        return 0, "version", ""

    run_benchmarks(runner, iter((1.0, 1.001, 2.0, 2.001)).__next__)

    assert calls == [
        ("ffmpeg", "-version"),
        ("vulkaninfo", "--summary"),
    ]


def test_benchmark_reports_missing_executable_without_raising() -> None:
    """A clean checkout on a partial host must get a not-ready machine report."""
    def runner(_argv: tuple[str, ...]) -> tuple[int, str, str]:
        raise FileNotFoundError

    probes = run_benchmarks(runner, iter((1.0, 1.001, 2.0, 2.001)).__next__)

    assert [probe.status for probe in probes] == ["not_ready", "not_ready"]
    assert [probe.detail for probe in probes] == [
        "executable not found",
        "executable not found",
    ]


def test_benchmark_reports_timeout_without_raising() -> None:
    """A stalled probe must produce bounded output and continue to the next probe."""
    def runner(argv: tuple[str, ...]) -> tuple[int, str, str]:
        raise subprocess.TimeoutExpired(argv, 15)

    probes = run_benchmarks(runner, iter((1.0, 1.001, 2.0, 2.001)).__next__)

    assert [probe.status for probe in probes] == ["not_ready", "not_ready"]
    assert all(probe.detail == "timed out after 15 seconds" for probe in probes)
