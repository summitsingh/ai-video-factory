from pathlib import Path

import pytest

from ai_video_factory import pipeline
from ai_video_factory.pipeline import PipelineCommandError, run_synthetic_pipeline


def test_pipeline_resumes_completed_render(tmp_path: Path) -> None:
    """A completed render and QC run are reused when inputs have not changed."""
    calls: list[str] = []

    def render(_edit: Path, output: Path) -> None:
        calls.append("render")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"synthetic")

    def validate(_video: Path, output_dir: Path) -> dict[str, str]:
        calls.append("validate")
        return {"status": "pass", "report": str(output_dir / "qc_report.json")}

    first = run_synthetic_pipeline(Path.cwd(), tmp_path, render=render, validate=validate)
    second = run_synthetic_pipeline(Path.cwd(), tmp_path, render=render, validate=validate)

    assert first.status == second.status == "pass"
    assert calls == ["render", "validate"]
    assert Path(first.artifacts["master"]).parent.name == first.run_id
    assert second.resumed is True


@pytest.mark.parametrize("missing", ["fixture", "lockfile"])
def test_missing_pipeline_input_returns_stable_failure(tmp_path: Path, missing: str) -> None:
    """Missing setup inputs must become a result, not escape the JSON boundary."""
    project = tmp_path / "project"
    fixtures = project / "fixtures"
    remotion = project / "remotion"
    fixtures.mkdir(parents=True)
    remotion.mkdir()
    if missing != "fixture":
        (fixtures / "synthetic-edit.json").write_bytes(
            Path("fixtures/synthetic-edit.json").read_bytes()
        )
    if missing != "lockfile":
        (remotion / "package-lock.json").write_text("{}", encoding="utf-8")

    result = run_synthetic_pipeline(project, tmp_path / "data")

    assert result.status == "fail"
    assert result.run_id is None
    assert result.artifacts == {}
    assert result.error is not None
    assert set(result.to_dict()) == {
        "schema_version", "command", "status", "run_id", "resumed", "retryable", "artifacts", "error"
    }


def test_pipeline_qc_failure_preserves_completed_render_artifact(tmp_path: Path) -> None:
    """A QC rejection must retain the render output for diagnosis and resumption."""
    def render(_edit: Path, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"render-output")

    def validate(_video: Path, output_dir: Path) -> dict[str, str]:
        return {"status": "fail", "report": str(output_dir / "qc_report.json")}

    result = run_synthetic_pipeline(Path.cwd(), tmp_path, render=render, validate=validate)

    assert result.status == "fail"
    assert result.retryable is False
    assert Path(result.artifacts["master"]).read_bytes() == b"render-output"


def test_render_wires_verified_browser_to_remotion_wrapper(monkeypatch, tmp_path: Path) -> None:
    """The local browser path is an explicit wrapper argument, not a download fallback."""
    project = tmp_path / "project"
    remotion = project / "remotion"
    fixture = project / "fixtures" / "synthetic-edit.json"
    output = project / "data" / "render.tmp.mp4"
    remotion.mkdir(parents=True)
    fixture.parent.mkdir()
    fixture.write_text("{}", encoding="utf-8")
    calls: list[tuple[tuple[str, ...], Path, str]] = []
    browser = Path("/opt/local-browser")

    monkeypatch.setattr(pipeline, "_find_local_browser", lambda: browser, raising=False)
    monkeypatch.setattr(
        pipeline,
        "_run_command",
        lambda argv, *, cwd, name: calls.append((tuple(argv), cwd, name)),
    )

    pipeline._render_with_remotion(project, fixture, output)

    assert calls == [
        (
            (
                "npm", "run", "render", "--", "--props", "../fixtures/synthetic-edit.json",
                "../data/render.tmp.mp4", "--browser-executable", "/opt/local-browser",
            ),
            remotion,
            "Remotion render",
        )
    ]


def test_render_refuses_to_invoke_remotion_without_local_browser(monkeypatch, tmp_path: Path) -> None:
    """Browser absence fails before the command that could trigger a download."""
    calls: list[tuple[str, ...]] = []

    def missing_browser() -> Path:
        raise PipelineCommandError("no verified local browser executable")

    monkeypatch.setattr(pipeline, "_find_local_browser", missing_browser, raising=False)
    monkeypatch.setattr(
        pipeline,
        "_run_command",
        lambda argv, **_kwargs: calls.append(tuple(argv)),
    )

    with pytest.raises(PipelineCommandError, match="no verified local browser"):
        pipeline._render_with_remotion(tmp_path, tmp_path / "fixture.json", tmp_path / "output.mp4")

    assert calls == []


def test_pipeline_marks_missing_browser_retryable_before_remotion(monkeypatch, tmp_path: Path) -> None:
    """The public pipeline returns a retryable failure without starting npm."""
    commands: list[tuple[str, ...]] = []

    def missing_browser() -> Path:
        raise PipelineCommandError("no verified local browser executable")

    monkeypatch.setattr(pipeline, "_find_local_browser", missing_browser, raising=False)
    monkeypatch.setattr(
        pipeline,
        "_run_command",
        lambda argv, **_kwargs: commands.append(tuple(argv)),
    )

    result = run_synthetic_pipeline(Path.cwd(), tmp_path)

    assert result.status == "fail"
    assert result.retryable is True
    assert result.run_id is not None
    assert commands == []
