from typer.testing import CliRunner

from ai_video_factory.cli import app


def test_cli_exposes_phase_one_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "benchmark", "test-pipeline"):
        assert command in result.stdout
