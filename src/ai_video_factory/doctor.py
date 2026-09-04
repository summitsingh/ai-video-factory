from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel

from ai_video_factory.sanitization import first_diagnostic_line, sanitize_diagnostic

CommandResult = tuple[int, str, str]
CommandRunner = Callable[[Sequence[str]], CommandResult]


class ToolCheck(BaseModel):
    name: str
    status: Literal["ready", "not_ready"]
    version: str | None
    detail: str | None


class DoctorReport(BaseModel):
    schema_version: Literal[1]
    overall_status: Literal["ready", "degraded"]
    checks: list[ToolCheck]


_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ffmpeg", ("ffmpeg", "-version")),
    ("ffprobe", ("ffprobe", "-version")),
    ("node", ("node", "--version")),
    ("npm", ("npm", "--version")),
    ("git", ("git", "--version")),
    ("vulkan", ("vulkaninfo", "--summary")),
    ("rocm", ("rocminfo",)),
    ("rocm_smi", ("rocm-smi", "--showproductname")),
)
def _subprocess_runner(argv: Sequence[str]) -> CommandResult:
    completed = subprocess.run(
        argv,
        shell=False,
        timeout=15,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _check_probe(name: str, argv: Sequence[str], runner: CommandRunner) -> ToolCheck:
    try:
        returncode, stdout, stderr = runner(argv)
    except FileNotFoundError:
        return ToolCheck(name=name, status="not_ready", version=None, detail="executable not found")
    except PermissionError:
        return ToolCheck(name=name, status="not_ready", version=None, detail="permission denied")
    except subprocess.TimeoutExpired:
        return ToolCheck(name=name, status="not_ready", version=None, detail="timed out after 15 seconds")
    except OSError as error:
        return ToolCheck(
            name=name,
            status="not_ready",
            version=None,
            detail=sanitize_diagnostic(error, max_chars=512),
        )

    if returncode != 0:
        detail = (
            first_diagnostic_line(stderr)
            or first_diagnostic_line(stdout)
            or f"exit code {returncode}"
        )
        return ToolCheck(name=name, status="not_ready", version=None, detail=detail)

    return ToolCheck(
        name=name,
        status="ready",
        version=first_diagnostic_line(stdout),
        detail=None,
    )


def collect_doctor_report(runner: CommandRunner = _subprocess_runner) -> DoctorReport:
    checks = [_check_probe(name, argv, runner) for name, argv in _PROBES]
    overall_status: Literal["ready", "degraded"] = (
        "ready" if all(check.status == "ready" for check in checks) else "degraded"
    )
    return DoctorReport(schema_version=1, overall_status=overall_status, checks=checks)
