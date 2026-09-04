from pathlib import Path

from ai_video_factory.pipeline import run_synthetic_pipeline


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
