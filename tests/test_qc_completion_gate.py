"""Bug 2 regression: a failing QC must mark the run failed, not completed.

Fermi v2 logged run_completed on the video-qc run even though the QC
artifacts recorded status "fail" (91.9% near-black frames, 22:31 vs the
25-minute target). The QC result now gates completion: ``_settle_qc_run``
fails the run loudly on any QC failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_video_factory.qc_final import FinalQcCheck, FinalQcReport
from ai_video_factory.run_store import RunStore
from ai_video_factory.video_pipeline import _settle_qc_run


def _run_store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "state", artifact_root=tmp_path / "runs")


def _write_reports(tmp_path: Path, failed: list[str]) -> tuple[Path, Path]:
    # Artifacts must live under the run store's artifact root.
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    report = runs / "qc_report.json"
    markdown = runs / "qc_report.md"
    checks = [{"name": "dimensions", "passed": True, "detail": "ok"}]
    checks.extend({"name": name, "passed": False, "detail": "bad"} for name in failed)
    report.write_text(
        json.dumps({"status": "fail" if failed else "pass", "checks": checks}),
        encoding="utf-8",
    )
    markdown.write_text("# qc report", encoding="utf-8")
    return report, markdown


def _final_qc(failed: list[str]) -> FinalQcReport:
    checks = [FinalQcCheck(name="duration", passed=True, detail="ok")]
    checks.extend(FinalQcCheck(name=name, passed=False, detail="bad") for name in failed)
    return FinalQcReport(master="master.mp4", target={}, checks=checks)


def _manifest(tmp_path: Path, run_id: str) -> dict:
    return json.loads(
        (tmp_path / "state" / "video-qc" / run_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )


def _events(tmp_path: Path, run_id: str) -> list[str]:
    events_path = tmp_path / "state" / "video-qc" / run_id / "events.jsonl"
    return [
        json.loads(line)["event"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_qc_failure_marks_run_failed_not_completed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Simulate the Fermi v2 QC outcome: content QC failed (dark frames,
    short runtime) and final QC failed (black frames). The run must end
    failed, never completed."""
    store = _run_store(tmp_path)
    run = store.start("video-qc", {"edit_sha256": "abc"})
    report, markdown = _write_reports(tmp_path, ["content-brightness", "target-duration"])
    final_qc = _final_qc(["black_frames"])

    settled = _settle_qc_run(
        store,
        run,
        status="fail",
        result={
            "status": "fail",
            "report": str(report),
            "report_markdown": str(markdown),
        },
        final_qc=final_qc,
        report_path=report,
    )

    manifest = _manifest(tmp_path, run.run_id)
    assert manifest["status"] == "failed"
    assert settled.run_id == run.run_id
    events = _events(tmp_path, run.run_id)
    assert "run_failed" in events
    assert "run_completed" not in events
    # The failure names the failed checks so the run log is actionable.
    assert "content-brightness" in manifest["error"]
    assert "black_frames" in manifest["error"]
    # And it is loud on stdout, not just in the log file.
    assert "QC FAILED" in capsys.readouterr().out


def test_qc_pass_marks_run_completed(tmp_path: Path) -> None:
    store = _run_store(tmp_path)
    run = store.start("video-qc", {"edit_sha256": "abc"})
    report, markdown = _write_reports(tmp_path, [])
    final_qc = _final_qc([])

    _settle_qc_run(
        store,
        run,
        status="pass",
        result={
            "status": "pass",
            "report": str(report),
            "report_markdown": str(markdown),
        },
        final_qc=final_qc,
        report_path=report,
    )

    manifest = _manifest(tmp_path, run.run_id)
    assert manifest["status"] == "completed"
    events = _events(tmp_path, run.run_id)
    assert "run_completed" in events
    assert "run_failed" not in events
