import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ai_video_factory import cli
from ai_video_factory.inference_models import InferenceResult
from ai_video_factory.pipeline import PipelineResult


class _InferenceService:
    def __init__(self, command: str, *, status: str = "pass") -> None:
        self.command = command
        self.result_status = status
        self.config = SimpleNamespace(identifier="avf-tiel-coder-executor")

    def _result(self, command: str) -> InferenceResult:
        assert command == self.command
        return InferenceResult(
            command=command,
            status=self.result_status,
            retryable=self.result_status != "pass",
            model_identifier=self.config.identifier,
            checks={},
            metrics={},
            artifacts={},
            error=None if self.result_status == "pass" else "not ready",
        )

    def doctor(self) -> InferenceResult:
        return self._result("doctor")

    def estimate(self) -> InferenceResult:
        return self._result("estimate")

    def start(self) -> InferenceResult:
        return self._result("start")

    def status(self) -> InferenceResult:
        return self._result("status")

    def stop(self) -> InferenceResult:
        return self._result("stop")


def _inference_result(command: str, *, status: str = "pass") -> InferenceResult:
    return InferenceResult(
        command=command,
        status=status,
        retryable=status != "pass",
        model_identifier="avf-tiel-coder-executor",
        checks={},
        metrics={},
        artifacts={},
        error=None if status == "pass" else "not ready",
    )


def test_cli_exposes_phase_one_commands() -> None:
    result = CliRunner().invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "benchmark", "test-pipeline"):
        assert command in result.stdout


def test_cli_exposes_inference_command_group() -> None:
    result = CliRunner().invoke(cli.app, ["inference", "--help"])

    assert result.exit_code == 0
    for command in ("doctor", "estimate", "start", "status", "benchmark", "stop"):
        assert command in result.stdout


@pytest.mark.parametrize(
    "command",
    ["doctor", "estimate", "start", "status", "benchmark", "stop"],
)
def test_inference_commands_emit_one_json_document(
    monkeypatch, command: str
) -> None:
    service = _InferenceService(command)
    monkeypatch.setattr(cli, "build_inference_service", lambda: service)
    monkeypatch.setattr(
        cli,
        "run_capability_benchmark",
        lambda built_service, _store, _data_root: (
            _inference_result("benchmark")
            if built_service is service
            else pytest.fail("benchmark used a different inference service")
        ),
    )

    result = CliRunner().invoke(cli.app, ["inference", command])

    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert payload["schema_version"] == 1
    assert payload["command"] == command
    assert result.stdout.count("\n") == 1
    assert "Traceback" not in result.stdout


def test_inference_non_pass_result_exits_two(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "build_inference_service",
        lambda: _InferenceService("status", status="not_ready"),
    )

    result = CliRunner().invoke(cli.app, ["inference", "status"])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "not_ready"


def test_inference_failure_is_sanitized_and_exits_two(monkeypatch) -> None:
    class BrokenService(_InferenceService):
        def start(self) -> InferenceResult:
            raise RuntimeError("token=secret")

    monkeypatch.setattr(
        cli, "build_inference_service", lambda: BrokenService("start")
    )

    result = CliRunner().invoke(cli.app, ["inference", "start"])

    payload = json.loads(result.stdout)
    assert result.exit_code == 2
    assert payload["status"] != "pass"
    assert payload["command"] == "start"
    assert "secret" not in result.stdout
    assert "Traceback" not in result.stdout


def test_inference_benchmark_uses_repository_root_project_data(
    monkeypatch, tmp_path: Path
) -> None:
    service = _InferenceService("benchmark")
    observed: dict[str, Path] = {}

    class RecordingStore:
        def __init__(self, root: Path, *, artifact_root: Path) -> None:
            observed["state"] = root
            observed["artifacts"] = artifact_root

    def benchmark(
        built_service: _InferenceService,
        _store: RecordingStore,
        data_root: Path,
    ) -> InferenceResult:
        assert built_service is service
        observed["data"] = data_root
        return _inference_result("benchmark")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_inference_service", lambda: service)
    monkeypatch.setattr(cli, "RunStore", RecordingStore)
    monkeypatch.setattr(cli, "run_capability_benchmark", benchmark)

    result = CliRunner().invoke(cli.app, ["inference", "benchmark"])

    project_root = Path(cli.__file__).resolve().parents[2]
    assert result.exit_code == 0
    assert observed == {
        "state": project_root / "data" / "projects" / "system" / "state",
        "artifacts": project_root / "data" / "projects" / "system" / "runs",
        "data": project_root / "data",
    }


def test_build_inference_service_uses_repository_root_config(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    config = object()
    service = object()

    def load(path: Path) -> object:
        observed["config_path"] = path
        return config

    def build(loaded_config: object) -> object:
        observed["config"] = loaded_config
        return service

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_inference_config", load)
    monkeypatch.setattr(cli, "InferenceService", build)

    assert cli.build_inference_service() is service
    assert observed == {
        "config_path": Path(cli.__file__).resolve().parents[2]
        / "config"
        / "inference.toml",
        "config": config,
    }


def test_pipeline_json_command_emits_only_pipeline_result(monkeypatch) -> None:
    """Changing root resolution to a non-Path value must not break JSON mode."""
    expected = PipelineResult(
        schema_version=1,
        command="test-pipeline",
        status="pass",
        run_id="pipeline-run",
        resumed=False,
        retryable=False,
        artifacts={"master": "/tmp/master.mp4"},
        error=None,
    )

    def pipeline(project_root: Path, data_root: Path) -> PipelineResult:
        assert project_root.is_dir()
        assert data_root == project_root / "data"
        return expected

    monkeypatch.setattr(cli, "run_synthetic_pipeline", pipeline)

    result = CliRunner().invoke(cli.app, ["test-pipeline", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == expected.to_dict()


def test_pipeline_json_command_converts_unexpected_error_to_json_failure(monkeypatch) -> None:
    """The public JSON command must not leak a traceback from an inner failure."""
    def broken_pipeline(_project_root: Path, _data_root: Path) -> PipelineResult:
        raise FileNotFoundError("synthetic fixture is missing")

    monkeypatch.setattr(cli, "run_synthetic_pipeline", broken_pipeline)

    result = CliRunner().invoke(cli.app, ["test-pipeline", "--json"])

    payload = json.loads(result.stdout)
    assert result.exit_code == 2
    assert payload["status"] == "fail"
    assert payload["run_id"] is None
    assert payload["artifacts"] == {}
    assert payload["error"] == "synthetic fixture is missing"
    assert "Traceback" not in result.stdout
