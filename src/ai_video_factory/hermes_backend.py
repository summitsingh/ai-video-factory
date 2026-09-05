"""Read-only inspection boundary for a locally installed Hermes CLI."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from ai_video_factory.hermes_config import HermesConfig
from ai_video_factory.hermes_models import HermesAuxiliaryRoute, HermesSnapshot
from ai_video_factory.sanitization import sanitize_diagnostic


_COMMAND_TIMEOUT_SECONDS = 15.0
_MAX_OUTPUT_BYTES = 64 * 1024
_VERSION = re.compile(
    r"^Hermes Agent v(?P<version>\d+\.\d+\.\d+) \([^\n]+\) · upstream "
    r"(?P<commit>[0-9a-f]{8})$",
    re.MULTILINE,
)

ALLOWED_COMMANDS = frozenset({
    ("hermes", "--version"),
    ("hermes", "config", "path"),
    ("hermes", "config", "check"),
    ("hermes", "config", "get", "model.provider"),
    ("hermes", "config", "get", "model.default"),
    ("hermes", "config", "get", "model.base_url"),
    ("hermes", "config", "get", "delegation.model"),
    ("hermes", "config", "get", "delegation.base_url"),
    ("hermes", "config", "get", "delegation.api_mode"),
    ("hermes", "config", "get", "delegation.max_iterations"),
    ("hermes", "config", "get", "delegation.max_concurrent_children"),
    ("hermes", "config", "get", "delegation.max_spawn_depth"),
    ("hermes", "config", "get", "delegation.orchestrator_enabled"),
    ("hermes", "config", "get", "delegation.subagent_auto_approve"),
    ("hermes", "config", "get", "delegation.inherit_mcp_toolsets"),
})

_DELEGATION_COMMANDS = frozenset(
    command for command in ALLOWED_COMMANDS if command[:3] == ("hermes", "config", "get")
    and command[3].startswith("delegation.")
)
_AUXILIARY_ROUTE_NAMES = (
    "vision", "web_extract", "compression", "title_generation", "background_review",
)
_SECRET_VALUE = re.compile(
    r"(?i)(?:api[._-]?key|access[._-]?key|authorization|credential|password|"
    r"secret|token)\s*(?:=|:)|bearer\s+"
)


class HermesError(RuntimeError):
    """A sanitized error from the fixed Hermes inspection boundary."""

    def __init__(self, detail: object) -> None:
        super().__init__(sanitize_diagnostic(detail))


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    executable_path: str


HermesCommandRunner = Callable[..., ProcessResult]


class HermesFileOps(Protocol):
    """Narrow injectable seam reserved for the later configuration transaction."""

    def copy2(self, source: Path, destination: Path) -> str | Path: ...


@lru_cache(maxsize=1)
def _resolved_hermes() -> str:
    """Return the canonical regular executable found on the user PATH."""
    resolved = shutil.which("hermes")
    if resolved is None:
        raise HermesError("Hermes executable not found")
    try:
        candidate = Path(resolved).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HermesError("Hermes executable could not be resolved") from exc
    if (
        not candidate.is_absolute()
        or not candidate.is_file()
        or candidate.is_symlink()
        or not os.access(candidate, os.X_OK)
    ):
        raise HermesError("Hermes executable is not a canonical executable regular file")
    return str(candidate)


def _validate_timeout(timeout: float) -> None:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise HermesError("Hermes command timeout is invalid")
    value = float(timeout)
    if not math.isfinite(value) or value <= 0 or value > _COMMAND_TIMEOUT_SECONDS:
        raise HermesError("Hermes command timeout exceeds the fixed bound")


def _bounded_result(completed: subprocess.CompletedProcess[str], executable: str) -> ProcessResult:
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    stderr = completed.stderr if isinstance(completed.stderr, str) else ""
    if len(stdout.encode("utf-8")) > _MAX_OUTPUT_BYTES or len(stderr.encode("utf-8")) > _MAX_OUTPUT_BYTES:
        raise HermesError("Hermes command output exceeds 64 KiB")
    if completed.returncode != 0:
        raise HermesError("Hermes command exited unsuccessfully")
    return ProcessResult(completed.returncode, stdout, stderr, executable)


def run_process(argv: tuple[str, ...], *, timeout: float) -> ProcessResult:
    """Run exactly one allowlisted read-only Hermes vector without a shell."""
    if argv not in ALLOWED_COMMANDS:
        raise HermesError("Hermes command is not allowlisted")
    _validate_timeout(timeout)
    executable = _resolved_hermes()
    try:
        completed = subprocess.run(
            (executable, *argv[1:]),
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise HermesError("Hermes executable not found") from exc
    except PermissionError as exc:
        raise HermesError("Hermes executable permission denied") from exc
    except subprocess.TimeoutExpired as exc:
        raise HermesError("Hermes command timed out") from exc
    except OSError as exc:
        raise HermesError("Hermes command could not run") from exc
    return _bounded_result(completed, executable)


def _canonical_executable(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute() or str(candidate.resolve(strict=False)) != str(candidate):
        raise HermesError("Hermes executable path is not canonical and absolute")
    return str(candidate)


def _single_safe_value(raw: str) -> str:
    if raw.endswith("\n"):
        raw = raw[:-1]
    if not raw or "\n" in raw or "\r" in raw or any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise HermesError("Hermes configuration value is invalid")
    if _SECRET_VALUE.search(raw):
        raise HermesError("Hermes configuration value is invalid")
    return raw


def _parse_version(raw: str) -> tuple[str, str]:
    value = _single_safe_value(raw)
    matched = _VERSION.fullmatch(value)
    if matched is None:
        raise HermesError("Hermes version text is invalid")
    return matched.group("version"), matched.group("commit")


def _parse_int(raw: str) -> int:
    value = _single_safe_value(raw)
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise HermesError("Hermes configuration value is invalid")
    return int(value)


def _parse_bool(raw: str) -> bool:
    value = _single_safe_value(raw)
    if value == "true":
        return True
    if value == "false":
        return False
    raise HermesError("Hermes configuration value is invalid")


class HermesBackend:
    """Read non-secret resolved Hermes fields using only fixed command vectors."""

    def __init__(
        self,
        config: HermesConfig,
        *,
        runner: HermesCommandRunner = run_process,
        file_ops: HermesFileOps | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.file_ops = file_ops

    def _command(self, argv: tuple[str, ...], *, allow_missing: bool = False) -> ProcessResult | None:
        if argv not in ALLOWED_COMMANDS:
            raise HermesError("Hermes command is not allowlisted")
        try:
            result = self.runner(argv, timeout=_COMMAND_TIMEOUT_SECONDS)
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired) as exc:
            raise HermesError("Hermes executable is not ready") from exc
        except HermesError:
            raise
        except OSError as exc:
            raise HermesError("Hermes command could not run") from exc
        except Exception as exc:
            raise HermesError("Hermes command runner failed") from exc
        if result.returncode != 0:
            if allow_missing:
                return None
            raise HermesError("Hermes command exited unsuccessfully")
        if len(result.stdout.encode("utf-8")) > _MAX_OUTPUT_BYTES or len(result.stderr.encode("utf-8")) > _MAX_OUTPUT_BYTES:
            raise HermesError("Hermes command output exceeds 64 KiB")
        _canonical_executable(result.executable_path)
        return result

    def _value(self, key: str, *, optional: bool = False) -> str | None:
        result = self._command(("hermes", "config", "get", key), allow_missing=optional)
        return None if result is None else _single_safe_value(result.stdout)

    @staticmethod
    def _unconfigured_auxiliaries() -> dict[str, HermesAuxiliaryRoute]:
        return {
            name: HermesAuxiliaryRoute(enabled=False, base_url=None, model=None)
            for name in _AUXILIARY_ROUTE_NAMES
        }

    def snapshot(self) -> HermesSnapshot:
        """Return a typed route snapshot without reading any Hermes config file."""
        version_result = self._command(("hermes", "--version"))
        assert version_result is not None
        version, commit = _parse_version(version_result.stdout)
        config_path_result = self._command(("hermes", "config", "path"))
        assert config_path_result is not None
        config_path = _single_safe_value(config_path_result.stdout)
        if not Path(config_path).is_absolute():
            raise HermesError("Hermes configuration path is invalid")
        config_check = self._command(("hermes", "config", "check"))
        assert config_check is not None

        parent_provider = self._value("model.provider")
        parent_model = self._value("model.default")
        parent_base_url = self._value("model.base_url")
        assert parent_provider is not None and parent_model is not None and parent_base_url is not None

        return HermesSnapshot(
            hermes_path=_canonical_executable(version_result.executable_path),
            version=version,
            commit=commit,
            config_path=config_path,
            profile="default",
            config_valid=True,
            parent_provider=parent_provider,
            parent_model=parent_model,
            parent_base_url=parent_base_url,
            delegation_model=self._value("delegation.model", optional=True),
            delegation_base_url=self._value("delegation.base_url", optional=True),
            delegation_api_mode=self._value("delegation.api_mode", optional=True),
            delegation_max_iterations=(
                None if (value := self._value("delegation.max_iterations", optional=True)) is None else _parse_int(value)
            ),
            delegation_max_concurrent_children=(
                None if (value := self._value("delegation.max_concurrent_children", optional=True)) is None else _parse_int(value)
            ),
            delegation_max_spawn_depth=(
                None if (value := self._value("delegation.max_spawn_depth", optional=True)) is None else _parse_int(value)
            ),
            delegation_orchestrator_enabled=(
                None if (value := self._value("delegation.orchestrator_enabled", optional=True)) is None else _parse_bool(value)
            ),
            delegation_subagent_auto_approve=(
                None if (value := self._value("delegation.subagent_auto_approve", optional=True)) is None else _parse_bool(value)
            ),
            delegation_inherit_mcp_toolsets=(
                None if (value := self._value("delegation.inherit_mcp_toolsets", optional=True)) is None else _parse_bool(value)
            ),
            auxiliary_routes=self._unconfigured_auxiliaries(),
        )
