"""Read-only policy checks for the Hermes parent and local delegation route."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from ai_video_factory.hermes_backend import (
    HERMES_EXECUTABLE_PATH,
    HermesBackend,
    HermesError,
    HermesFileOps,
)
from ai_video_factory.hermes_config import HermesConfig
from ai_video_factory.hermes_models import HermesCheck, HermesResult, HermesSnapshot
from ai_video_factory.inference_benchmark import current_text_capability
from ai_video_factory.inference_service import InferenceService
from ai_video_factory.run_store import RunStore
from ai_video_factory.sanitization import sanitize_diagnostic


TextCapability = Callable[[], bool]


class HermesService:
    """Evaluate fixed route invariants without mutating Hermes or LM Studio."""

    def __init__(
        self,
        config: HermesConfig,
        *,
        backend: HermesBackend | None = None,
        text_capability: TextCapability | None = None,
        inference_service: InferenceService | None = None,
        run_store: RunStore | None = None,
        data_root: Path | None = None,
        file_ops: HermesFileOps | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or HermesBackend(config, file_ops=file_ops)
        self.text_capability = text_capability
        self.inference_service = inference_service
        self.run_store = run_store
        self.data_root = None if data_root is None else Path(data_root)
        self.file_ops = file_ops

    @staticmethod
    def _check(ready: bool, detail: str | None = None) -> HermesCheck:
        return HermesCheck(status="ready" if ready else "not_ready", detail=detail)

    def _installation_check(self, snapshot: HermesSnapshot) -> HermesCheck:
        ready = (
            snapshot.config_valid
            and snapshot.version == self.config.required_version
            and snapshot.commit == self.config.required_commit
            and snapshot.profile == self.config.profile
            and snapshot.hermes_path == HERMES_EXECUTABLE_PATH
            and bool(snapshot.config_path)
        )
        return self._check(ready)

    def _parent_route_check(self, snapshot: HermesSnapshot) -> HermesCheck:
        return self._check(
            snapshot.parent_provider == self.config.parent_provider
            and snapshot.parent_model == self.config.parent_model
            and bool(snapshot.parent_base_url)
        )

    def _delegation_state(self, snapshot: HermesSnapshot) -> tuple[HermesCheck, bool]:
        values = (
            snapshot.delegation_model,
            snapshot.delegation_base_url,
            snapshot.delegation_api_mode,
            snapshot.delegation_max_iterations,
            snapshot.delegation_max_concurrent_children,
            snapshot.delegation_max_spawn_depth,
            snapshot.delegation_orchestrator_enabled,
            snapshot.delegation_subagent_auto_approve,
            snapshot.delegation_inherit_mcp_toolsets,
        )
        route_values = values[:3]
        safety_values = values[3:]
        expected_safety = (
            self.config.max_iterations,
            self.config.max_concurrent_children,
            1,
            False,
            False,
            False,
        )
        if all(value is None for value in route_values) and safety_values == expected_safety:
            return self._check(False, "delegation is not configured"), True
        if all(value is None for value in values):
            return self._check(False, "delegation is not configured"), True
        if any(value is None for value in values):
            return self._check(False), False
        ready = (
            snapshot.delegation_model == self.config.delegation_model
            and snapshot.delegation_base_url == self.config.delegation_base_url
            and snapshot.delegation_api_mode == self.config.delegation_api_mode
            and snapshot.delegation_max_iterations == self.config.max_iterations
            and snapshot.delegation_max_concurrent_children == self.config.max_concurrent_children
            and snapshot.delegation_max_spawn_depth == 1
            and snapshot.delegation_orchestrator_enabled is False
            and snapshot.delegation_subagent_auto_approve is False
            and snapshot.delegation_inherit_mcp_toolsets is False
        )
        return self._check(ready), False

    def _delegation_safety_check(self, snapshot: HermesSnapshot, unconfigured: bool) -> HermesCheck:
        if unconfigured:
            return self._check(False, "delegation is not configured")
        return self._check(
            snapshot.delegation_max_concurrent_children == 1
            and snapshot.delegation_max_spawn_depth == 1
            and snapshot.delegation_orchestrator_enabled is False
            and snapshot.delegation_subagent_auto_approve is False
            and snapshot.delegation_inherit_mcp_toolsets is False
        )

    def _result(
        self,
        *,
        status: Literal["pass", "fail", "not_ready"],
        retryable: bool,
        checks: dict[str, HermesCheck],
        error: str | None = None,
    ) -> HermesResult:
        return HermesResult(
            command="doctor",
            status=status,
            retryable=retryable,
            parent_provider="nous",
            parent_model=self.config.parent_model,
            attempts=[],
            checks=checks,
            metrics={},
            artifacts={},
            error=error,
        )

    def _current_text_capability(self) -> bool:
        if self.text_capability is not None:
            return self.text_capability() is True
        if (
            self.inference_service is None
            or self.run_store is None
            or self.data_root is None
        ):
            return False
        return current_text_capability(
            self.inference_service, self.run_store, self.data_root
        ) is True

    def doctor(self) -> HermesResult:
        """Inspect Hermes identity/routes and require a current text capability proof."""
        try:
            snapshot = self.backend.snapshot()
            delegation_route, unconfigured = self._delegation_state(snapshot)
            checks = {
                "installation": self._installation_check(snapshot),
                "parent_route": self._parent_route_check(snapshot),
                "delegation_route": delegation_route,
                "delegation_safety": self._delegation_safety_check(snapshot, unconfigured),
                "text_capability": self._check(self._current_text_capability()),
            }
            if any(
                checks[name].status != "ready"
                for name in ("installation", "parent_route")
            ):
                return self._result(status="fail", retryable=False, checks=checks)
            if not unconfigured and any(
                checks[name].status != "ready"
                for name in ("delegation_route", "delegation_safety")
            ):
                return self._result(status="fail", retryable=False, checks=checks)
            if unconfigured or checks["text_capability"].status != "ready":
                return self._result(status="not_ready", retryable=True, checks=checks)
            return self._result(status="pass", retryable=False, checks=checks)
        except HermesError as exc:
            return self._result(
                status="not_ready",
                retryable=True,
                checks={"operation": self._check(False)},
                error=sanitize_diagnostic(exc),
            )
        except Exception as exc:
            return self._result(
                status="fail",
                retryable=False,
                checks={"operation": self._check(False)},
                error=sanitize_diagnostic(exc),
            )
