"""Tests for pipeline reliability upgrades: command retries and mux timeouts."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_video_factory.video_pipeline import PipelineCommandError, _run_command


def _completed_ok(argv, **kwargs):
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def _completed_fail(argv, **kwargs):
    return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")


def test_run_command_succeeds_first_try(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=_completed_ok) as mock_run:
        _run_command(("echo", "hi"), cwd=tmp_path, name="echo")
    assert mock_run.call_count == 1


def test_run_command_retries_transient_timeout(tmp_path: Path) -> None:
    calls = []

    def flaky(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=1)
        return _completed_ok(argv)

    with patch("subprocess.run", side_effect=flaky):
        with patch("ai_video_factory.video_pipeline.time.sleep"):
            _run_command(("ffmpeg",), cwd=tmp_path, name="mux", retries=1)
    assert len(calls) == 2


def test_run_command_retries_failed_exit_code(tmp_path: Path) -> None:
    calls = []

    def flaky(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            return _completed_fail(argv)
        return _completed_ok(argv)

    with patch("subprocess.run", side_effect=flaky):
        with patch("ai_video_factory.video_pipeline.time.sleep"):
            _run_command(("ffmpeg",), cwd=tmp_path, name="mux", retries=1)
    assert len(calls) == 2


def test_run_command_raises_after_retries_exhausted(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=_completed_fail) as mock_run:
        with patch("ai_video_factory.video_pipeline.time.sleep"):
            with pytest.raises(PipelineCommandError):
                _run_command(("ffmpeg",), cwd=tmp_path, name="mux", retries=2)
    assert mock_run.call_count == 3


def test_run_command_default_no_retry_preserves_old_behavior(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=_completed_fail) as mock_run:
        with pytest.raises(PipelineCommandError):
            _run_command(("ffmpeg",), cwd=tmp_path, name="mux")
    assert mock_run.call_count == 1
