import json
from pathlib import Path

from typer.testing import CliRunner

from ai_video_factory import cli
from ai_video_factory.pipeline import PipelineResult


def test_cli_exposes_phase_one_commands() -> None:
    result = CliRunner().invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "benchmark", "test-pipeline"):
        assert command in result.stdout


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
