import json
import shutil
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
        return _write_fake_qc(output_dir)

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
        return _write_fake_qc(output_dir, status="fail")

    result = run_synthetic_pipeline(Path.cwd(), tmp_path, render=render, validate=validate)

    assert result.status == "fail"
    assert result.retryable is False
    assert Path(result.artifacts["master"]).read_bytes() == b"render-output"


@pytest.mark.parametrize("mutation", ["delete", "tamper"])
def test_pipeline_reexecutes_render_when_completed_master_is_not_intact(
    tmp_path: Path, mutation: str
) -> None:
    """A stale render manifest cannot bless a deleted or modified master."""
    calls: list[str] = []

    def render(_edit: Path, output: Path) -> None:
        calls.append("render")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"render-{calls.count('render')}".encode())

    def validate(_video: Path, output_dir: Path) -> dict[str, str]:
        calls.append("validate")
        return _write_fake_qc(output_dir)

    first = run_synthetic_pipeline(Path.cwd(), tmp_path, render=render, validate=validate)
    master = Path(first.artifacts["master"])
    if mutation == "delete":
        master.unlink()
    else:
        master.write_bytes(b"tampered-master")

    second = run_synthetic_pipeline(Path.cwd(), tmp_path, render=render, validate=validate)

    assert first.status == second.status == "pass"
    assert first.run_id != second.run_id
    assert calls == ["render", "validate", "render", "validate"]
    assert master != Path(second.artifacts["master"])


def test_pipeline_reexecutes_qc_when_completed_report_is_missing(tmp_path: Path) -> None:
    """QC cannot be resumed as passing when a claimed report disappeared."""
    calls: list[str] = []

    def render(_edit: Path, output: Path) -> None:
        calls.append("render")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"render")

    def validate(_video: Path, output_dir: Path) -> dict[str, str]:
        calls.append("validate")
        return _write_fake_qc(output_dir)

    first = run_synthetic_pipeline(Path.cwd(), tmp_path, render=render, validate=validate)
    Path(first.artifacts["report"]).unlink()

    second = run_synthetic_pipeline(Path.cwd(), tmp_path, render=render, validate=validate)

    assert second.status == "pass"
    assert second.resumed is False
    assert calls == ["render", "validate", "validate"]
    assert Path(second.artifacts["report"]).is_file()


def test_pipeline_fails_only_active_qc_stage_and_redacts_persisted_error(
    tmp_path: Path,
) -> None:
    """A validator exception must not demote its completed render dependency."""
    secret = "pipeline-secret"

    def render(_edit: Path, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"render")

    def validate(_video: Path, _output_dir: Path) -> dict[str, str]:
        raise PipelineCommandError(
            f"Authorization: Bearer {secret}\n" + ("x" * 5_000)
        )

    result = run_synthetic_pipeline(Path.cwd(), tmp_path, render=render, validate=validate)

    assert result.status == "fail"
    assert result.run_id is not None
    assert secret not in (result.error or "")
    assert len(result.error or "") <= 2_048
    state_root = tmp_path / "projects" / "synthetic" / "state"
    render_manifest = json.loads(
        (state_root / "synthetic-render" / result.run_id / "manifest.json").read_text()
    )
    qc_manifests = list((state_root / "synthetic-qc").glob("*/manifest.json"))
    assert render_manifest["status"] == "completed"
    assert len(qc_manifests) == 1
    qc_manifest = json.loads(qc_manifests[0].read_text())
    assert qc_manifest["status"] == "failed"
    assert secret not in qc_manifests[0].read_text()
    assert secret not in qc_manifests[0].with_name("events.jsonl").read_text()


def test_renderer_source_change_prevents_stale_render_reuse(tmp_path: Path) -> None:
    """Changing render-determining source must change the render fingerprint."""
    project = _copy_pipeline_project(tmp_path)
    calls: list[str] = []

    def render(_edit: Path, output: Path) -> None:
        calls.append("render")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"render-{calls.count('render')}".encode())

    def validate(_video: Path, output_dir: Path) -> dict[str, str]:
        calls.append("validate")
        return _write_fake_qc(output_dir)

    first = run_synthetic_pipeline(project, tmp_path / "data", render=render, validate=validate)
    renderer = project / "remotion" / "src" / "SyntheticVideo.tsx"
    renderer.write_text(renderer.read_text() + "\n// render identity change\n")

    second = run_synthetic_pipeline(project, tmp_path / "data", render=render, validate=validate)

    assert first.status == second.status == "pass"
    assert first.run_id != second.run_id
    assert calls == ["render", "validate", "render", "validate"]


def test_production_tool_versions_and_identities_are_recorded_in_render_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    """A production render manifest must identify the exact browser and media tools."""
    fake_tools = {
        name: {
            "status": "ready",
            "path": f"/tools/{name}",
            "sha256": name * 8,
            "version": f"{name} version 1",
        }
        for name in ("browser", "ffmpeg", "ffprobe", "node", "npm")
    }

    monkeypatch.setattr(pipeline, "_collect_pipeline_toolchain", lambda: fake_tools)

    def fake_remotion(_root: Path, _fixture: Path, output: Path, **_kwargs) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"temporary")

    def fake_mux(_source: Path, destination: Path, **_kwargs) -> None:
        destination.write_bytes(b"master")

    monkeypatch.setattr(pipeline, "_render_with_remotion", fake_remotion)
    monkeypatch.setattr(pipeline, "_mux_silent_audio", fake_mux)

    result = run_synthetic_pipeline(
        Path.cwd(),
        tmp_path,
        validate=lambda _video, output_dir: _write_fake_qc(output_dir),
    )

    manifest_path = (
        tmp_path
        / "projects"
        / "synthetic"
        / "state"
        / "synthetic-render"
        / str(result.run_id)
        / "manifest.json"
    )
    persisted = json.loads(manifest_path.read_text())
    assert persisted["inputs"]["provenance"]["tools"] == fake_tools
    assert "remotion/src/SyntheticVideo.tsx" in persisted["inputs"]["provenance"]["files"]


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


def _write_fake_qc(output_dir: Path, *, status: str = "pass") -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "qc_report.json"
    markdown_path = output_dir / "qc_report.md"
    json_path.write_text(json.dumps({"status": status}), encoding="utf-8")
    markdown_path.write_text(f"# QC: {status}\n", encoding="utf-8")
    return {
        "status": status,
        "report": str(json_path),
        "report_markdown": str(markdown_path),
    }


def _copy_pipeline_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "fixtures").mkdir(parents=True)
    shutil.copy2(Path("fixtures/synthetic-edit.json"), project / "fixtures")
    shutil.copytree(
        Path("remotion"),
        project / "remotion",
        ignore=shutil.ignore_patterns("node_modules", "dist"),
    )
    return project
