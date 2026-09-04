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
