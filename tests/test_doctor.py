from ai_video_factory.doctor import collect_doctor_report


def test_missing_optional_gpu_tools_are_not_ready() -> None:
    results = {
        ("ffmpeg", "-version"): (0, "ffmpeg version 8.0.1", ""),
        ("ffprobe", "-version"): (0, "ffprobe version 8.0.1", ""),
        ("node", "--version"): (0, "v22.23.1", ""),
        ("npm", "--version"): (0, "10.9.2", ""),
        ("git", "--version"): (0, "git version 2.50.1", ""),
        ("vulkaninfo", "--summary"): (1, "", "no display or DRM device"),
        ("rocminfo",): (127, "", "not found"),
        ("rocm-smi", "--showproductname"): (127, "", "not found"),
    }

    report = collect_doctor_report(lambda argv: results[tuple(argv)])
    by_name = {check.name: check for check in report.checks}
    assert by_name["ffmpeg"].status == "ready"
    assert by_name["vulkan"].status == "not_ready"
    assert by_name["rocm"].status == "not_ready"
    assert report.overall_status == "degraded"


def test_doctor_redacts_and_limits_public_tool_diagnostics() -> None:
    """Doctor output must use the shared public diagnostic boundary."""
    secret = "doctor-secret"

    def runner(argv: tuple[str, ...]) -> tuple[int, str, str]:
        if argv[0] == "ffmpeg":
            return 1, "", f"password={secret}\n" + ("x" * 5_000)
        return 0, "version", ""

    report = collect_doctor_report(runner)
    detail = next(check.detail for check in report.checks if check.name == "ffmpeg")

    assert detail is not None
    assert secret not in detail
    assert len(detail) <= 512
