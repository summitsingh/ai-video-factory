"""Safe, memory-gated lifecycle policy for local LM Studio inference."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from ai_video_factory.inference_config import InferenceConfig
from ai_video_factory.inference_models import InferenceCheck, InferenceResult, MemoryEstimate
from ai_video_factory.lm_studio import LmStudioBackend, LmStudioError, LmStudioSnapshot
from ai_video_factory.model_provenance import ModelDigestCache, capability_inputs as build_capability_inputs
from ai_video_factory.sanitization import sanitize_diagnostic


Clock = Callable[[], float]
Sleeper = Callable[[float], None]
MemoryReader = Callable[[], float]

_LIFECYCLE_TIMEOUT_SECONDS = 600.0
_POLL_INTERVAL_SECONDS = 1.0
_MEM_AVAILABLE = re.compile(r"^MemAvailable:\s*(\d+)\s+kB\s*$")
_CAPABILITY_CORPUS_VERSION = "lm-studio-capability-v4"


def available_memory_gib(path: Path = Path("/proc/meminfo")) -> float:
    """Read Linux MemAvailable and convert KiB to 1024-based GiB."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LmStudioError(f"could not read MemAvailable: {exc}") from exc
    matches = [match for line in lines if (match := _MEM_AVAILABLE.fullmatch(line))]
    if len(matches) != 1:
        raise LmStudioError("MemAvailable in /proc/meminfo is malformed")
    return int(matches[0].group(1)) / (1024 * 1024)


class InferenceService:
    """Backend-neutral policy for inspecting and changing configured residency."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        backend: LmStudioBackend | None = None,
        memory_reader: MemoryReader = available_memory_gib,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        self.config = config
        self.backend = backend or LmStudioBackend(config)
        self.memory_reader = memory_reader
        self.clock = clock
        self.sleeper = sleeper

    def _result(
        self,
        command: Literal["doctor", "estimate", "start", "status", "stop"],
        *,
        status: Literal["pass", "fail", "not_ready"],
        retryable: bool,
        checks: dict[str, InferenceCheck] | None = None,
        metrics: dict[str, int | float | str | bool | None] | None = None,
        error: str | None = None,
    ) -> InferenceResult:
        return InferenceResult(
            command=command,
            status=status,
            retryable=retryable,
            model_identifier=self.config.identifier,
            checks=checks or {},
            metrics=metrics or {},
            artifacts={},
            error=error,
        )

    def _failure(
        self,
        command: Literal["doctor", "estimate", "start", "status", "stop"],
        error: BaseException,
    ) -> InferenceResult:
        expected = isinstance(error, LmStudioError)
        return self._result(
            command,
            status="not_ready" if expected else "fail",
            retryable=expected,
            checks={"operation": InferenceCheck(status="not_ready", detail=None)},
            error=sanitize_diagnostic(error),
        )

    def _estimate_and_gate(self) -> tuple[MemoryEstimate, float, bool]:
        estimate = self.backend.estimate()
        available = self.memory_reader()
        if isinstance(available, bool) or not isinstance(available, (int, float)):
            raise LmStudioError("MemAvailable reader returned an invalid value")
        available = float(available)
        if not math.isfinite(available) or available < 0:
            raise LmStudioError("MemAvailable reader returned an invalid value")
        allowed = estimate.allowed and available >= self.config.minimum_available_memory_gib
        return estimate.model_copy(update={"allowed": allowed}), available, allowed

    def _estimate_metrics(
        self, estimate: MemoryEstimate, available: float,
    ) -> dict[str, int | float | str | bool | None]:
        return {
            "estimated_gpu_gib": estimate.gpu_gib,
            "estimated_total_gib": estimate.total_gib,
            "estimate_confidence": estimate.confidence,
            "available_memory_gib": available,
            "minimum_available_memory_gib": self.config.minimum_available_memory_gib,
            "allowed": estimate.allowed,
        }

    def _poll_for_identifier(
        self, *, present: bool,
    ) -> LmStudioSnapshot:
        deadline = self.clock() + _LIFECYCLE_TIMEOUT_SECONDS
        while True:
            snapshot = self.backend.snapshot()
            resident = snapshot.configured_model_loaded
            if resident is present:
                return snapshot
            if self.clock() >= deadline:
                state = "appear" if present else "disappear"
                raise LmStudioError(
                    f"LM Studio model lifecycle timed out waiting for identifier to {state}"
                )
            self.sleeper(_POLL_INTERVAL_SECONDS)

    def _capability_inputs(
        self, data_root: Path, *, require_resident: bool, use_cache: bool,
    ) -> dict[str, object]:
        snapshot = self.backend.snapshot()
        if require_resident:
            if not snapshot.server_running:
                raise LmStudioError("LM Studio server is not running")
            if not snapshot.configured_model_loaded:
                raise LmStudioError("configured LM Studio model is not loaded")
        cache = ModelDigestCache(
            Path(data_root) / "system" / "model-digests",
            model_root=Path(self.config.models_directory),
            identifier=self.config.identifier,
            use_cache=use_cache,
            write_cache=use_cache,
        )
        identity = cache.identity(snapshot.configured_model)
        return build_capability_inputs(
            self.config,
            snapshot,
            identity,
            _CAPABILITY_CORPUS_VERSION,
        )

    def capability_inputs(self, data_root: Path) -> dict[str, object]:
        """Build cache-backed benchmark provenance only while the model is resident."""
        return self._capability_inputs(
            data_root, require_resident=True, use_cache=True,
        )

    def current_capability_inputs(self, data_root: Path) -> dict[str, object]:
        """Recompute provenance from inventory without requiring a loaded model."""
        return self._capability_inputs(
            data_root, require_resident=False, use_cache=False,
        )

    def chat_completion(
        self, body: dict[str, object], *, timeout: float
    ) -> dict[str, object]:
        """Send one JSON completion through the backend's injected transport."""
        return self.backend.http(
            "POST",
            f"{self.config.base_url}/chat/completions",
            body,
            {"Content-Type": "application/json", "Accept": "application/json"},
            timeout,
        )

    def doctor(self) -> InferenceResult:
        """Inspect the configured backend without changing runtime state."""
        try:
            snapshot = self.backend.snapshot()
            return self._result(
                "doctor",
                status="pass",
                retryable=False,
                checks=dict(snapshot.checks),
                metrics={
                    "server_running": snapshot.server_running,
                    "configured_model_loaded": snapshot.configured_model_loaded,
                    "loaded_model_count": len(snapshot.loaded_identifiers),
                },
            )
        except Exception as exc:
            return self._failure("doctor", exc)

    def estimate(self) -> InferenceResult:
        """Run the non-loading estimate and apply the available-memory gate."""
        try:
            estimate, available, allowed = self._estimate_and_gate()
            return self._result(
                "estimate",
                status="pass" if allowed else "not_ready",
                retryable=not allowed,
                checks={
                    "estimate": InferenceCheck(status="ready", detail=estimate.confidence),
                    "available_memory": InferenceCheck(
                        status="ready" if allowed else "not_ready",
                        detail=available,
                    ),
                },
                metrics=self._estimate_metrics(estimate, available),
                error=None if allowed else "available memory is below the configured 40 GiB gate",
            )
        except Exception as exc:
            return self._failure("estimate", exc)

    def start(self) -> InferenceResult:
        """Memory-gate and load only the configured identifier when needed."""
        try:
            before = self.backend.snapshot()
            unrelated_before = {
                identifier
                for identifier in before.loaded_identifiers
                if identifier != self.config.identifier
            }
            server_started = False
            if before.configured_model_loaded:
                if not before.server_running:
                    self.backend.start_server()
                    server_started = True
                    after = self.backend.snapshot()
                    if not after.server_running or not after.configured_model_loaded:
                        raise LmStudioError(
                            "configured LM Studio model is not ready after server start"
                        )
                else:
                    after = before
                metrics: dict[str, int | float | str | bool | None] = {
                    "server_started": server_started,
                    "configured_model_loaded": True,
                    "api_visible": False,
                    "memory_gate_skipped": True,
                }
                memory_gate_detail: int | float | str | bool | None = (
                    "already resident"
                )
            else:
                estimate, available, allowed = self._estimate_and_gate()
                metrics = self._estimate_metrics(estimate, available)
                memory_gate_detail = available
                if not allowed:
                    return self._result(
                        "start",
                        status="not_ready",
                        retryable=True,
                        checks={
                            "memory_gate": InferenceCheck(
                                status="not_ready", detail=available
                            )
                        },
                        metrics=metrics,
                        error="available memory is below the configured 40 GiB gate",
                    )
                if not before.server_running:
                    self.backend.start_server()
                    server_started = True
                    server_snapshot = self.backend.snapshot()
                    if not server_snapshot.server_running:
                        raise LmStudioError(
                            "LM Studio server did not become safely ready after start"
                        )
                    unrelated_after_server = {
                        identifier
                        for identifier in server_snapshot.loaded_identifiers
                        if identifier != self.config.identifier
                    }
                    if unrelated_after_server != unrelated_before:
                        raise LmStudioError(
                            "LM Studio start changed unrelated loaded identifiers"
                        )
                    if server_snapshot.configured_model_loaded:
                        after = server_snapshot
                    else:
                        self.backend.start()
                        after = self._poll_for_identifier(present=True)
                else:
                    self.backend.start()
                    after = self._poll_for_identifier(present=True)

            unrelated_after = {
                identifier
                for identifier in after.loaded_identifiers
                if identifier != self.config.identifier
            }
            if unrelated_after != unrelated_before:
                raise LmStudioError("LM Studio start changed unrelated loaded identifiers")
            api_identifiers = self.backend.api_model_identifiers()
            if self.config.identifier not in api_identifiers:
                raise LmStudioError(
                    "configured model identifier is resident in the CLI but absent from the API"
                )

            metrics.update(
                {
                    "server_started": server_started,
                    "configured_model_loaded": True,
                    "api_visible": True,
                }
            )
            return self._result(
                "start",
                status="pass",
                retryable=False,
                checks={
                    "memory_gate": InferenceCheck(
                        status="ready", detail=memory_gate_detail
                    ),
                    "cli_residency": InferenceCheck(
                        status="ready", detail=self.config.identifier
                    ),
                    "api_visibility": InferenceCheck(
                        status="ready", detail=self.config.identifier
                    ),
                },
                metrics=metrics,
            )
        except Exception as exc:
            return self._failure("start", exc)

    def status(self) -> InferenceResult:
        """Report exact configured residency without changing LM Studio."""
        try:
            snapshot = self.backend.snapshot()
            ready = snapshot.server_running and snapshot.configured_model_loaded
            return self._result(
                "status",
                status="pass" if ready else "not_ready",
                retryable=not ready,
                checks={
                    "server": InferenceCheck(
                        status="ready" if snapshot.server_running else "not_ready",
                        detail=snapshot.server_running,
                    ),
                    "configured_model": InferenceCheck(
                        status="ready" if snapshot.configured_model_loaded else "not_ready",
                        detail=self.config.identifier,
                    ),
                },
                metrics={
                    "server_running": snapshot.server_running,
                    "configured_model_loaded": snapshot.configured_model_loaded,
                    "loaded_model_count": len(snapshot.loaded_identifiers),
                },
                error=None if ready else "configured LM Studio model is not ready",
            )
        except Exception as exc:
            return self._failure("status", exc)

    def stop(self) -> InferenceResult:
        """Idempotently unload only the exact configured identifier."""
        try:
            before = self.backend.snapshot()
            unrelated_before = {
                identifier
                for identifier in before.loaded_identifiers
                if identifier != self.config.identifier
            }
            if not before.configured_model_loaded:
                return self._result(
                    "stop",
                    status="pass",
                    retryable=False,
                    checks={
                        "configured_model": InferenceCheck(
                            status="ready", detail="already absent"
                        )
                    },
                    metrics={"unloaded": False},
                )

            self.backend.stop()
            after = self._poll_for_identifier(present=False)
            unrelated_after = {
                identifier
                for identifier in after.loaded_identifiers
                if identifier != self.config.identifier
            }
            if unrelated_after != unrelated_before:
                raise LmStudioError("LM Studio stop changed unrelated loaded identifiers")
            return self._result(
                "stop",
                status="pass",
                retryable=False,
                checks={
                    "configured_model": InferenceCheck(status="ready", detail="unloaded")
                },
                metrics={"unloaded": True},
            )
        except Exception as exc:
            return self._failure("stop", exc)
