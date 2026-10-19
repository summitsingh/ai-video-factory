"""Tests for the resumable downloader in scripts/download_ltx2.py (Q6).

Exercises Range-resume, truncate-on-200-to-Range, retry-after-drop, and
size-mismatch behaviour against a local HTTP server (no network).
"""

from __future__ import annotations

import http.server
import importlib.util
import socket
import threading
import time
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_ltx2.py"
_spec = importlib.util.spec_from_file_location("download_ltx2", _SCRIPT)
assert _spec and _spec.loader
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)

PAYLOAD = bytes((i % 251 for i in range(2_500_000)))  # ~2.5 MB, multi-chunk


class _Handler(http.server.BaseHTTPRequestHandler):
    """Tiny test server. Class attrs are reset per test via ``configure``."""

    server_version = "TestDl/1.0"
    payload: bytes = PAYLOAD
    mode: str = "range"  # range | ignore-range | drop-once | drop-always
    dropped: bool = False
    seen: list = []

    @classmethod
    def configure(cls, mode: str):
        cls.mode = mode
        cls.dropped = False
        cls.seen = []

    def log_message(self, *args):
        pass

    def _parse_range(self):
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            return None
        try:
            return int(rng[len("bytes="):].rstrip("-"))
        except ValueError:
            return None

    def do_HEAD(self):
        type(self).seen.append(("HEAD", self.path, self.headers.get("Range")))
        body = type(self).payload
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

    def do_GET(self):
        cls = type(self)
        cls.seen.append(("GET", self.path, self.headers.get("Range")))
        body = cls.payload
        start = self._parse_range()
        if cls.mode == "drop-always" or (
            cls.mode == "drop-once" and not cls.dropped and not start
        ):
            cls.dropped = True
            tail = body[start or 0 :]
            if start:
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(body)-1}/{len(body)}"
                )
                self.send_header("Content-Length", str(len(tail)))
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(tail[: len(tail) // 2])
            self.wfile.flush()
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return

        if cls.mode == "ignore-range" or not start:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command == "GET":
                self.wfile.write(body)
            return

        # honour the Range request
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{len(body)-1}/{len(body)}")
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])


@pytest.fixture()
def server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


def _url(server, name="f.bin"):
    return f"{server}/{name}"


def test_fresh_download(server, tmp_path):
    _Handler.configure("range")
    dest = tmp_path / "f.bin"
    n = dl._download_file(_url(server), dest, expected_size=len(PAYLOAD), quiet=True)
    assert n == len(PAYLOAD)
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "f.bin.part").exists()


def test_resume_partial_file(server, tmp_path):
    """A pre-existing .part is resumed via Range, not re-downloaded."""
    _Handler.configure("range")
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(PAYLOAD[:1_000_000])
    dl._download_file(_url(server), dest, expected_size=len(PAYLOAD), quiet=True)
    assert dest.read_bytes() == PAYLOAD
    ranges = [r for m, _, r in _Handler.seen if m == "GET"]
    assert any(r == "bytes=1000000-" for r in ranges), ranges


def test_truncate_on_200_to_range(server, tmp_path):
    """Server ignoring Range (200) -> partial file truncated, fresh start."""
    _Handler.configure("ignore-range")
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(b"garbage-prefix")
    dl._download_file(_url(server), dest, expected_size=len(PAYLOAD), quiet=True)
    assert dest.read_bytes() == PAYLOAD


def test_retry_after_dropped_connection(server, tmp_path):
    _Handler.configure("drop-once")
    dest = tmp_path / "f.bin"
    dl._download_file(_url(server), dest, expected_size=len(PAYLOAD),
                      retries=3, quiet=True)
    assert dest.read_bytes() == PAYLOAD


def test_size_mismatch_raises(server, tmp_path):
    _Handler.configure("range")
    dest = tmp_path / "f.bin"
    with pytest.raises(dl.DownloadError, match="size mismatch"):
        dl._download_file(_url(server), dest, expected_size=len(PAYLOAD) + 10,
                          quiet=True)


def test_retries_exhausted_raises(server, tmp_path):
    _Handler.configure("drop-always")
    dest = tmp_path / "f.bin"
    with pytest.raises(dl.DownloadError, match="gave up after"):
        dl._download_file(_url(server), dest, expected_size=len(PAYLOAD),
                          retries=1, quiet=True)


def test_head_status(server):
    _Handler.configure("range")
    status, length = dl.head_status(_url(server))
    assert status == 200
    assert length == len(PAYLOAD)


def test_download_all_skips_complete(server, tmp_path, monkeypatch):
    _Handler.configure("range")
    monkeypatch.setattr(dl, "file_url", lambda rel: _url(server, rel))
    monkeypatch.setattr(dl, "REQUIRED_FILES", ("a.bin", "b.bin"))
    (tmp_path / "a.bin").write_bytes(PAYLOAD)  # complete already
    results = dl.download_all(tmp_path, quiet=True)
    assert results == {"a.bin": "skipped", "b.bin": "downloaded"}
    assert (tmp_path / "b.bin").read_bytes() == PAYLOAD
    gets = [m for m, p, _ in _Handler.seen if m == "GET"]
    assert gets == ["GET"]  # only b.bin fetched


def test_verify_mirror_reports_bad_status(monkeypatch):
    monkeypatch.setattr(dl, "head_status", lambda url: (404, None))
    problems = dl.verify_mirror(("nope.bin",))
    assert problems == [("nope.bin", 404)]


def test_verify_mirror_ok(monkeypatch):
    monkeypatch.setattr(dl, "head_status", lambda url: (200, 10))
    assert dl.verify_mirror(("ok.bin",)) == []
