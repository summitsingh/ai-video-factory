from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel

CommandResult = tuple[int, str, str]
CommandRunner = Callable[[Sequence[str]], CommandResult]
Clock = Callable[[], float]


class BenchmarkProbe(BaseModel):
    name: str
    status: Literal["ready", "not_ready"]
    duration_ms: int
    detail: str


class BenchmarkReport(BaseModel):
    schema_version: Literal[1]
    probes: list[BenchmarkProbe]


_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ffmpeg-startup", ("ffmpeg", "-version")),
    ("vulkan-enumeration", ("vulkaninfo", "--summary")),
)
_SENSITIVE_LINE = re.compile(r"^([^:=]+)([:=])(.*)$")
_SENSITIVE_KEY_PARTS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "KEY",
    "AUTHORIZATION",
    "COOKIE",
)


def _detail_line(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _SENSITIVE_LINE.match(line)
        if match and any(part in match.group(1).upper() for part in _SENSITIVE_KEY_PARTS):
            line = f"{match.group(1)}{match.group(2)}[REDACTED]"
        return line
    return ""


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


def _run_probe(
    name: str, argv: Sequence[str], runner: CommandRunner, clock: Clock
) -> BenchmarkProbe:
    started = clock()
    try:
        returncode, stdout, stderr = runner(argv)
        if returncode == 0:
            status: Literal["ready", "not_ready"] = "ready"
            detail = _detail_line(stdout) or "command completed successfully"
        else:
            status = "not_ready"
            detail = _detail_line(stderr) or _detail_line(stdout) or f"exit code {returncode}"
    except FileNotFoundError:
        status, detail = "not_ready", "executable not found"
    except PermissionError:
        status, detail = "not_ready", "permission denied"
    except subprocess.TimeoutExpired:
        status, detail = "not_ready", "timed out after 15 seconds"
    except OSError as error:
        status, detail = "not_ready", _detail_line(str(error)) or "operating system error"
    duration_ms = max(0, round((clock() - started) * 1000))
    return BenchmarkProbe(name=name, status=status, duration_ms=duration_ms, detail=detail)


def run_benchmarks(
    runner: CommandRunner = _subprocess_runner, clock: Clock = time.monotonic
) -> list[BenchmarkProbe]:
    return [_run_probe(name, argv, runner, clock) for name, argv in _PROBES]
