"""Read-only, typed discovery helpers for the local LM Studio installation."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ai_video_factory.inference_config import InferenceConfig
from ai_video_factory.inference_models import InferenceCheck, MemoryEstimate, ModelIdentity
from ai_video_factory.sanitization import first_diagnostic_line, sanitize_diagnostic


_COMMAND_TIMEOUT_SECONDS = 15.0
_MAX_HTTP_RESPONSE_BYTES = 2 * 1024 * 1024
_LOOPBACK_ORIGIN = ("http", "127.0.0.1", 1234)
_READ_ONLY_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("lms", "--help"),
    ("lms", "runtime", "ls"),
    ("lms", "runtime", "survey"),
    ("lms", "server", "status"),
    ("lms", "ls", "--json"),
    ("lms", "ps", "--json"),
)
_ESTIMATE_COMMAND = (
    "lms", "load", "qwen3.6-35b-a3b-udt-mtp", "--gpu", "max",
    "--context-length", "65536", "--no-speculative-draft-mtp",
    "--estimate-only", "-y",
)
_SERVER_START_COMMAND = (
    "lms", "server", "start", "--port", "1234", "--bind", "127.0.0.1",
)
_MODEL_START_COMMAND = (
    "lms", "load", "qwen3.6-35b-a3b-udt-mtp", "--gpu", "max",
    "--context-length", "65536", "--parallel", "1", "--ttl", "3600",
    "--no-speculative-draft-mtp", "--identifier", "avf-qwen36-executor", "-y",
)
_MODEL_STOP_COMMAND = ("lms", "unload", "avf-qwen36-executor")
_APPROVED_COMMANDS = _READ_ONLY_COMMANDS + (
    _ESTIMATE_COMMAND,
    _SERVER_START_COMMAND,
    _MODEL_START_COMMAND,
    _MODEL_STOP_COMMAND,
)
_ESTIMATE_TIMEOUT_SECONDS = 60.0
_MODEL_LOAD_TIMEOUT_SECONDS = 600.0
_ESTIMATE_PATTERNS = {
    "gpu_gib": re.compile(r"^Estimated GPU Memory:\s*(\S+)\s+GiB\s*$"),
    "total_gib": re.compile(r"^Estimated Total Memory:\s*(\S+)\s+GiB\s*$"),
    "confidence": re.compile(r"^Confidence:\s*(LOW|MEDIUM|HIGH|UNKNOWN)\s*$"),
}


class LmStudioError(RuntimeError):
    """A sanitized error raised at the LM Studio process and HTTP boundary."""

    def __init__(self, detail: object) -> None:
        super().__init__(sanitize_diagnostic(detail))


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], float], ProcessResult]
HttpTransport = Callable[
    [str, str, dict[str, object] | None, Mapping[str, str], float], dict[str, object]
]


@dataclass(frozen=True)
class LmStudioModel:
    """An inventory model whose path was contained beneath the configured root."""

    model_key: str
    path: Path
    relative_path: str
    size_bytes: int

    def identity(self, identifier: str) -> ModelIdentity:
        """Return the unhashed identity used as input to later provenance work."""
        return ModelIdentity(
            model_key=self.model_key,
            identifier=identifier,
            relative_path=self.relative_path,
            size_bytes=self.size_bytes,
        )


@dataclass(frozen=True)
class LmStudioSnapshot:
    """A read-only view of CLI availability, model inventory, and residency."""

    cli_help: str
    runtimes: str
    runtime_survey: str
    server_status: str
    server_running: bool
    models: tuple[LmStudioModel, ...]
    loaded_identifiers: tuple[str, ...]
    configured_model: LmStudioModel
    configured_model_loaded: bool
    checks: Mapping[str, InferenceCheck]


class _InventoryModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    model_key: str = Field(validation_alias=AliasChoices("key", "modelKey"))
    path: str
    size_bytes: int = Field(validation_alias=AliasChoices("sizeBytes", "size_bytes"), ge=0)


class _LoadedModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    identifier: str


class _ApiModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    id: str


_INVENTORY_ADAPTER = TypeAdapter(list[_InventoryModel])
_LOADED_ADAPTER = TypeAdapter(list[_LoadedModel])
_API_MODELS_ADAPTER = TypeAdapter(list[_ApiModel])


@lru_cache(maxsize=1)
def _resolved_lms() -> str:
    """Find the CLI once, accepting only an executable regular file."""
    resolved = shutil.which("lms")
    if resolved is None:
        raise LmStudioError("LM Studio executable not found")
    candidate = Path(resolved)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise LmStudioError("LM Studio executable is not an executable regular file")
    return str(candidate)


def run_process(argv: Sequence[str], timeout: float) -> ProcessResult:
    """Run a fixed LM Studio command without a shell or inherited stdin."""
    if tuple(argv) not in _APPROVED_COMMANDS:
        raise LmStudioError(
            "LM Studio command is not an approved read-only or configured lifecycle command"
        )
    _resolved_lms()
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise LmStudioError("LM Studio executable not found") from exc
    except PermissionError as exc:
        raise LmStudioError("LM Studio executable permission denied") from exc
    except subprocess.TimeoutExpired as exc:
        raise LmStudioError(f"LM Studio command timed out after {timeout:g} seconds") from exc
    except OSError as exc:
        raise LmStudioError(f"LM Studio command could not run: {exc}") from exc
    return ProcessResult(completed.returncode, completed.stdout, completed.stderr)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, request: Request, fp: Any, code: int, message: str, headers: Any, newurl: str
    ) -> None:
        return None


def _validate_loopback_url(url: str) -> None:
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise LmStudioError("LM Studio URL must use the configured loopback origin") from exc
    if (
        (parsed.scheme, parsed.hostname, port) != _LOOPBACK_ORIGIN
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not (parsed.path == "/v1" or parsed.path.startswith("/v1/"))
    ):
        raise LmStudioError("LM Studio URL must use the configured loopback origin")


def request_json(
    method: str,
    url: str,
    body: dict[str, object] | None,
    headers: Mapping[str, str],
    timeout: float,
) -> dict[str, object]:
    """Issue one bounded JSON request to the fixed LM Studio loopback origin."""
    _validate_loopback_url(url)
    if set(headers) != {"Content-Type", "Accept"} or any(
        value != "application/json" for value in headers.values()
    ):
        raise LmStudioError("LM Studio requests require JSON Content-Type and Accept headers only")
    payload = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = Request(url, data=payload, headers=dict(headers), method=method)
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            status = response.getcode()
            raw = response.read(_MAX_HTTP_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise LmStudioError(f"LM Studio HTTP request failed with status {exc.code}: {exc.read(_MAX_HTTP_RESPONSE_BYTES)}") from exc
    except URLError as exc:
        raise LmStudioError(f"LM Studio HTTP request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LmStudioError(f"LM Studio HTTP request timed out after {timeout:g} seconds") from exc
    except OSError as exc:
        raise LmStudioError(f"LM Studio HTTP request could not run: {exc}") from exc
    if not 200 <= status < 300:
        raise LmStudioError(f"LM Studio HTTP request failed with status {status}")
    if len(raw) > _MAX_HTTP_RESPONSE_BYTES:
        raise LmStudioError("LM Studio HTTP response exceeds 2 MiB limit")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LmStudioError("LM Studio HTTP response was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise LmStudioError("LM Studio HTTP response JSON must be an object")
    return parsed


class LmStudioBackend:
    """Typed boundary around fixed LM Studio CLI and loopback HTTP calls."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        runner: CommandRunner = run_process,
        http: HttpTransport = request_json,
    ) -> None:
        self.config = config
        self.runner = runner
        self.http = http

    def _command(
        self,
        argv: tuple[str, ...],
        *,
        allow_failure: bool = False,
        timeout: float = _COMMAND_TIMEOUT_SECONDS,
    ) -> ProcessResult:
        try:
            result = self.runner(argv, timeout)
        except FileNotFoundError as exc:
            raise LmStudioError("LM Studio executable not found") from exc
        except PermissionError as exc:
            raise LmStudioError("LM Studio executable permission denied") from exc
        except subprocess.TimeoutExpired as exc:
            raise LmStudioError(f"LM Studio command timed out after {timeout:g} seconds") from exc
        except LmStudioError:
            raise
        except OSError as exc:
            raise LmStudioError(f"LM Studio command could not run: {exc}") from exc
        if result.returncode != 0 and not allow_failure:
            detail = first_diagnostic_line(result.stderr) or first_diagnostic_line(result.stdout)
            if detail is None:
                detail = f"exit code {result.returncode}"
            raise LmStudioError(f"LM Studio command {' '.join(argv)} failed: {detail}")
        return result

    @staticmethod
    def _parse_estimate(raw_estimate: str) -> MemoryEstimate:
        matched: dict[str, list[str]] = {name: [] for name in _ESTIMATE_PATTERNS}
        prefixes = {
            "gpu_gib": "Estimated GPU Memory:",
            "total_gib": "Estimated Total Memory:",
            "confidence": "Confidence:",
        }
        for line in raw_estimate.splitlines():
            for name, pattern in _ESTIMATE_PATTERNS.items():
                match = pattern.fullmatch(line)
                if match is not None:
                    matched[name].append(match.group(1))
                elif line.startswith(prefixes[name]):
                    raise LmStudioError("LM Studio estimate output is invalid")

        if any(len(values) != 1 for values in matched.values()):
            raise LmStudioError("LM Studio estimate output is invalid")
        try:
            gpu_gib = float(matched["gpu_gib"][0])
            total_gib = float(matched["total_gib"][0])
        except ValueError as exc:
            raise LmStudioError("LM Studio estimate output is invalid") from exc
        if not all(math.isfinite(value) and value >= 0 for value in (gpu_gib, total_gib)):
            raise LmStudioError("LM Studio estimate output is invalid")
        return MemoryEstimate(
            gpu_gib=gpu_gib,
            total_gib=total_gib,
            confidence=matched["confidence"][0],
            allowed=True,
        )

    def estimate(self) -> MemoryEstimate:
        """Estimate configured model memory without loading the model."""
        argv = (
            self.config.lms_binary,
            "load",
            self.config.model_key,
            "--gpu",
            self.config.gpu,
            "--context-length",
            str(self.config.context_length),
            "--no-speculative-draft-mtp",
            "--estimate-only",
            "-y",
        )
        result = self._command(argv, timeout=_ESTIMATE_TIMEOUT_SECONDS)
        return self._parse_estimate(result.stdout)

    def start_server(self) -> None:
        """Start only the configured loopback LM Studio server."""
        self._command(
            (
                self.config.lms_binary,
                "server",
                "start",
                "--port",
                "1234",
                "--bind",
                "127.0.0.1",
            )
        )

    def start(self) -> None:
        """Load only the configured model under its stable identifier."""
        self._command(
            (
                self.config.lms_binary,
                "load",
                self.config.model_key,
                "--gpu",
                self.config.gpu,
                "--context-length",
                str(self.config.context_length),
                "--parallel",
                str(self.config.parallel),
                "--ttl",
                str(self.config.ttl_seconds),
                "--no-speculative-draft-mtp",
                "--identifier",
                self.config.identifier,
                "-y",
            ),
            timeout=_MODEL_LOAD_TIMEOUT_SECONDS,
        )

    def stop(self) -> None:
        """Unload only the configured stable identifier."""
        self._command((self.config.lms_binary, "unload", self.config.identifier))

    def api_model_identifiers(self) -> tuple[str, ...]:
        """Return model identifiers visible from the fixed loopback API."""
        response = self.http(
            "GET",
            f"{self.config.base_url}/models",
            None,
            {"Content-Type": "application/json", "Accept": "application/json"},
            _COMMAND_TIMEOUT_SECONDS,
        )
        try:
            records = _API_MODELS_ADAPTER.validate_python(response.get("data"), strict=True)
        except ValidationError as exc:
            raise LmStudioError("LM Studio API model list JSON is invalid") from exc
        return tuple(record.id for record in records)

    def _inventory(self, raw_inventory: str) -> tuple[LmStudioModel, ...]:
        try:
            parsed = json.loads(raw_inventory)
            records = _INVENTORY_ADAPTER.validate_python(parsed, strict=True)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LmStudioError("LM Studio inventory JSON is invalid") from exc
        root = Path(self.config.models_directory).resolve(strict=False)
        models: list[LmStudioModel] = []
        for record in records:
            candidate = (root / record.path).resolve(strict=False)
            try:
                relative_path = candidate.relative_to(root)
            except ValueError as exc:
                raise LmStudioError("LM Studio inventory model path must be contained by models_directory") from exc
            models.append(
                LmStudioModel(
                    model_key=record.model_key,
                    path=candidate,
                    relative_path=relative_path.as_posix(),
                    size_bytes=record.size_bytes,
                )
            )
        return tuple(models)

    @staticmethod
    def _loaded(raw_loaded: str) -> tuple[str, ...]:
        try:
            parsed = json.loads(raw_loaded)
            records = _LOADED_ADAPTER.validate_python(parsed, strict=True)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LmStudioError("LM Studio loaded-model JSON is invalid") from exc
        return tuple(record.identifier for record in records)

    def snapshot(self) -> LmStudioSnapshot:
        """Inspect fixed CLI state without loading models or changing LM Studio."""
        help_result = self._command(_READ_ONLY_COMMANDS[0])
        runtimes_result = self._command(_READ_ONLY_COMMANDS[1])
        survey_result = self._command(_READ_ONLY_COMMANDS[2])
        server_result = self._command(_READ_ONLY_COMMANDS[3], allow_failure=True)
        inventory_result = self._command(_READ_ONLY_COMMANDS[4])
        loaded_result = self._command(_READ_ONLY_COMMANDS[5])

        models = self._inventory(inventory_result.stdout)
        configured_models = [model for model in models if model.model_key == self.config.model_key]
        if not configured_models:
            raise LmStudioError("configured model is missing from LM Studio inventory")
        if len(configured_models) != 1:
            raise LmStudioError("duplicate configured model keys in LM Studio inventory")
        loaded_identifiers = self._loaded(loaded_result.stdout)
        server_detail = (
            first_diagnostic_line(server_result.stdout)
            or first_diagnostic_line(server_result.stderr)
            or f"exit code {server_result.returncode}"
        )
        status_text = f"{server_result.stdout}\n{server_result.stderr}".casefold()
        server_running = server_result.returncode == 0 and "running" in status_text and all(
            marker not in status_text for marker in ("not running", "stopped")
        )
        checks: dict[str, InferenceCheck] = {
            "cli": InferenceCheck(status="ready", detail=first_diagnostic_line(help_result.stdout)),
            "runtime": InferenceCheck(status="ready", detail=first_diagnostic_line(runtimes_result.stdout)),
            "survey": InferenceCheck(status="ready", detail=first_diagnostic_line(survey_result.stdout)),
            "server": InferenceCheck(
                status="ready" if server_running else "not_ready",
                detail=None if server_running else server_detail,
            ),
            "configured_model": InferenceCheck(status="ready", detail=self.config.model_key),
        }
        configured_model_loaded = self.config.identifier in loaded_identifiers
        return LmStudioSnapshot(
            cli_help=help_result.stdout,
            runtimes=runtimes_result.stdout,
            runtime_survey=survey_result.stdout,
            server_status=server_detail,
            server_running=server_running,
            models=models,
            loaded_identifiers=loaded_identifiers,
            configured_model=configured_models[0],
            configured_model_loaded=configured_model_loaded,
            checks=checks,
        )
