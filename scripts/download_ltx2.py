#!/usr/bin/env python3
"""Download the supported LTX-2.5 weight set from the DeepBeepMeep/LTX-2 mirror.

Supported config: LTX-2.5 22B Distilled, int8_convrot transformer + int8
Gemma4 text encoder + int8 connectors + bf16 VAEs/vocoder. This is the exact
set WanGP benchmarked on summit-amd 2026-10-18 (768x512, 33f @24fps, 8
steps, ~10-12 min/clip, no OOM).

Mirror: ``DeepBeepMeep/LTX-2`` (re-host; sidesteps the gated Lightricks
repos that 401 without a token). Verified ungated 2026-10-18 from
summit-amd: HTTP 200 (after the CDN redirect) on a HEAD request for every
file below, no Hugging Face token needed.

Downloads are resumable: each file streams into ``<name>.part``; on a
dropped connection the next attempt sends ``Range: bytes=<done>-`` and
appends; a 200 response to a Range request (server ignored the range)
truncates the partial file and restarts from zero; up to 5 retries per file
with backoff. Existing complete files (size matches the mirror) are skipped,
so re-runs only fetch what is missing.

On-disk layout (flat; mirrors /home/summit/wangp/ckpts/)::

    <out>/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors
    <out>/gemma4-12b-ltx-v1/gemma4-12b-ltx-v1_int8_convrot.safetensors
    <out>/gemma4-12b-ltx-v1/{config,tokenizer,tokenizer_config}.json
    <out>/gemma4-12b-ltx-v1/chat_template.jinja
    <out>/ltx-2.5-22b_*_{bf16,int8_convrot}.safetensors  (VAEs, vocoder, connectors)

Usage:
    python scripts/download_ltx2.py [--verify] [--list] [--out-dir DIR]
                                     [--include-optional]

Env:
    LTX25_MODEL_DIR  overrides the default target dir (models/LTX-2.5 under
                     the repo root).

The download recipe (Range resume, temp-file append, retry budget,
truncate-on-200) follows the approach documented in WanGP's
shared/utils/download_progress.py, reimplemented here with the standard
library only.
"""

from __future__ import annotations

import argparse
import http.client
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MIRROR_REPO = "DeepBeepMeep/LTX-2"
MIRROR_BASE = f"https://huggingface.co/{MIRROR_REPO}/resolve/main/"

TEXT_ENCODER_DIR = "gemma4-12b-ltx-v1"

# Exact file set for the supported int8_convrot distilled config (WanGP
# defaults ltx2_25_22B_distilled.json + models/ltx2/ltx2_handler.py
# _ARCH_SPECS["ltx2_25_22B"]).
REQUIRED_FILES: tuple[str, ...] = (
    "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors",
    f"{TEXT_ENCODER_DIR}/gemma4-12b-ltx-v1_int8_convrot.safetensors",
    f"{TEXT_ENCODER_DIR}/config.json",
    f"{TEXT_ENCODER_DIR}/tokenizer.json",
    f"{TEXT_ENCODER_DIR}/tokenizer_config.json",
    f"{TEXT_ENCODER_DIR}/chat_template.jinja",
    "ltx-2.5-22b_video_embeddings_connector_int8_convrot.safetensors",
    "ltx-2.5-22b_audio_embeddings_connector_int8_convrot.safetensors",
    "ltx-2.5-22b_video_vae_bf16.safetensors",
    "ltx-2.5-22b_diffusion_video_vae_bf16.safetensors",
    "ltx-2.5-22b_audio_vae_bf16.safetensors",
    "ltx-2.5-22b_vocoder_bf16.safetensors",
    "ltx-2.5-22b_text_embedding_projection_bf16.safetensors",
)

# On the mirror and in WanGP's arch spec, but not needed for the distilled
# 8-step path the benchmark used (the distilled transformer already carries
# the distillation).
OPTIONAL_FILES: tuple[str, ...] = (
    "ltx-2.5-22b-distilled-lora-450_bf16.safetensors",
)

MAX_RETRIES = 5
CHUNK_BYTES = 1024 * 1024
HEAD_TIMEOUT = 30
GET_TIMEOUT = 60
USER_AGENT = "ai-video-factory-download/1.0"

# Network-level failures worth retrying with a Range resume.
_RETRYABLE = (
    urllib.error.URLError,
    http.client.HTTPException,
    socket.timeout,
    ConnectionError,
    TimeoutError,
)


class DownloadError(RuntimeError):
    """A file could not be downloaded (mirror issue, size mismatch, ...)."""


def _response_total(resp, status: int, done: int) -> int | None:
    """Total bytes this response should deliver, or None if unknowable."""
    if status == 206:
        content_range = resp.headers.get("Content-Range")  # "bytes 100-199/1000"
        if content_range and "/" in content_range:
            total = content_range.rsplit("/", 1)[1]
            if total.isdigit():
                return int(total)
        length = resp.headers.get("Content-Length")
        if length and length.isdigit():
            return done + int(length)
        return None
    length = resp.headers.get("Content-Length")
    if length and length.isdigit():
        return int(length)
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _request(url: str, *, method: str, headers: dict | None, timeout: float):
    req = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def head_status(url: str) -> tuple[int, int | None]:
    """HEAD a mirror URL. Returns (http_status, content_length or None)."""
    with _request(url, method="HEAD", headers=None, timeout=HEAD_TIMEOUT) as resp:
        length = resp.headers.get("Content-Length")
        return resp.status, (int(length) if length and length.isdigit() else None)


def _download_file(
    url: str,
    dest: Path,
    *,
    expected_size: int | None = None,
    retries: int = MAX_RETRIES,
    quiet: bool = False,
) -> int:
    """Download ``url`` to ``dest`` with Range-resume. Returns total bytes.

    Streams into ``dest.name + ".part"`` and atomically renames on success.
    On a dropped connection, retries send ``Range: bytes=<done>-`` and
    append; a 200 response to a Range request truncates and restarts.
    """
    part = dest.parent / (dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    attempt = 0
    last_report = 0.0
    while True:
        headers = {"Range": f"bytes={done}-"} if done else None
        try:
            with _request(url, method="GET", headers=headers, timeout=GET_TIMEOUT) as resp:
                status = resp.status
                if done and status == 200:
                    # Server ignored the Range header: restart from zero.
                    part.unlink(missing_ok=True)
                    done = 0
                elif done and status == 416:
                    # Our partial file is longer than the remote object.
                    part.unlink(missing_ok=True)
                    done = 0
                    continue  # re-issue without Range below via loop retry
                elif status not in (200, 206):
                    raise DownloadError(f"{url}: unexpected HTTP {status}")
                want = _response_total(resp, status, done)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(part, "ab") as fh:
                    while True:
                        chunk = resp.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if not quiet and now - last_report > 2.0:
                            last_report = now
                            if expected_size:
                                print(
                                    f"  {dest.name}: {done/1e9:.2f}/{expected_size/1e9:.2f} GB "
                                    f"({100*done/expected_size:.1f}%)",
                                    flush=True,
                                )
                            else:
                                print(f"  {dest.name}: {done/1e9:.2f} GB", flush=True)
                if want is not None and done < want:
                    # Truncated stream (server died mid-body): retry with Range.
                    raise http.client.IncompleteRead(b"", want - done)
            break
        except _RETRYABLE as exc:
            attempt += 1
            if attempt > retries:
                raise DownloadError(
                    f"{url}: gave up after {retries} retries ({exc})"
                ) from exc
            wait = min(2 ** attempt, 30)
            if not quiet:
                print(
                    f"  {dest.name}: connection interrupted at {done} bytes, "
                    f"retry {attempt}/{retries} in {wait}s",
                    flush=True,
                )
            time.sleep(wait)
            done = part.stat().st_size if part.exists() else 0
    if expected_size is not None and done != expected_size:
        raise DownloadError(
            f"{url}: size mismatch: expected {expected_size} bytes, got {done}"
        )
    os.replace(part, dest)
    return done


def file_url(relative_path: str) -> str:
    return MIRROR_BASE + relative_path


def download_all(
    out_dir: Path,
    *,
    include_optional: bool = False,
    quiet: bool = False,
) -> dict[str, str]:
    """Download the weight set into ``out_dir``. Returns {file: status}.

    Status is one of "downloaded", "skipped" (already complete), "optional".
    Raises DownloadError on the first hard failure.
    """
    files = REQUIRED_FILES + (OPTIONAL_FILES if include_optional else ())
    results: dict[str, str] = {}
    for rel in files:
        url = file_url(rel)
        dest = out_dir / rel
        try:
            status, expected = head_status(url)
        except _RETRYABLE as exc:
            raise DownloadError(f"{url}: HEAD failed ({exc})") from exc
        if status != 200:
            raise DownloadError(f"{url}: mirror returned HTTP {status}")
        if dest.exists() and expected is not None and dest.stat().st_size == expected:
            if not quiet:
                print(f"  {rel}: present ({expected/1e9:.2f} GB), skipping")
            results[rel] = "skipped"
            continue
        if dest.exists():
            # Stale or truncated copy: start clean.
            dest.unlink()
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        stale_part = dest.parent / (dest.name + ".part")
        if not quiet:
            resumed = stale_part.stat().st_size if stale_part.exists() else 0
            print(f"  {rel}: downloading ({expected/1e9:.2f} GB" +
                  (f", resuming at {resumed/1e9:.2f} GB" if resumed else "") + ")")
        _download_file(url, dest, expected_size=expected, quiet=quiet)
        results[rel] = "downloaded"
    return results


def verify_mirror(files: tuple[str, ...] = REQUIRED_FILES) -> list[tuple[str, int | str]]:
    """HEAD every file; return [(file, status)] for anything not HTTP 200."""
    problems: list[tuple[str, int | str]] = []
    for rel in files:
        try:
            status, _ = head_status(file_url(rel))
        except _RETRYABLE as exc:
            problems.append((rel, f"HEAD failed: {exc}"))
            continue
        if status != 200:
            problems.append((rel, status))
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None,
                    help="target dir (default: $LTX25_MODEL_DIR or models/LTX-2.5)")
    ap.add_argument("--verify", action="store_true",
                    help="HEAD every mirror file and report availability; download nothing")
    ap.add_argument("--list", action="store_true",
                    help="print the file list and exit")
    ap.add_argument("--include-optional", action="store_true",
                    help="also fetch OPTIONAL_FILES (distilled system LoRA)")
    args = ap.parse_args(argv)

    if args.list:
        print("required:")
        for rel in REQUIRED_FILES:
            print(f"  {rel}")
        print("optional:")
        for rel in OPTIONAL_FILES:
            print(f"  {rel}")
        return 0

    if args.verify:
        problems = verify_mirror(REQUIRED_FILES + OPTIONAL_FILES)
        for rel in REQUIRED_FILES + OPTIONAL_FILES:
            mark = "OK " if all(p[0] != rel for p in problems) else "FAIL"
            print(f"  [{mark}] {rel}")
        if problems:
            print(f"\n{len(problems)} file(s) not reachable on {MIRROR_REPO}:")
            for rel, status in problems:
                print(f"  {rel}: {status}")
            return 1
        print(f"\nmirror {MIRROR_REPO} OK: all files reachable, no token needed")
        return 0

    default_dir = os.environ.get("LTX25_MODEL_DIR")
    out_dir = Path(args.out_dir or default_dir or (_repo_root() / "models" / "LTX-2.5"))
    print(f"target: {out_dir}")
    try:
        results = download_all(out_dir, include_optional=args.include_optional)
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    n_dl = sum(1 for s in results.values() if s == "downloaded")
    n_skip = sum(1 for s in results.values() if s == "skipped")
    print(f"DONE: {n_dl} downloaded, {n_skip} already present -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
