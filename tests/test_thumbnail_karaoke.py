"""Tests for ITEM 6 (canonical thumbnail.jpg) and ITEM 9 (karaoke captions)."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from ai_video_factory.edit_schema import EditDocument, EditScene
from ai_video_factory.subtitle_export import (
    build_ass_karaoke,
    burn_karaoke_captions,
    distribute_word_timings,
    write_ass_karaoke,
)
from ai_video_factory.thumbnail import _truncate_text, build_thumbnails


def _doc() -> EditDocument:
    return EditDocument(
        schema_version=1,
        width=1280,
        height=720,
        fps=30,
        duration_frames=300,
        title="The Fermi Paradox Explained",
        scenes=[
            EditScene(
                id="s1",
                from_frame=0,
                duration_frames=150,
                title="Silent Stars",
                caption="intro card",
                kind="intro",
            ),
            EditScene(
                id="s2",
                from_frame=150,
                duration_frames=150,
                title="Deep Field",
                caption="deep field",
                kind="normal",
                narration=(
                    "The quick brown fox jumps over the lazy dog near the river bank today"
                ),
            ),
        ],
    )


def _short_doc() -> EditDocument:
    """5-second doc with caption cues inside the clip for the burn test."""
    return EditDocument(
        schema_version=1,
        width=1280,
        height=720,
        fps=30,
        duration_frames=150,
        title="Burn Test",
        scenes=[
            EditScene(
                id="s1",
                from_frame=0,
                duration_frames=150,
                title="Only Scene",
                caption="only",
                kind="normal",
                narration="one two three four five six seven eight nine ten",
            ),
        ],
    )


def test_truncate_text_word_cap_no_spurious_ellipsis():
    # Regression: the old char-length check appended "…" to short hooks.
    assert _truncate_text("BEYOND EARTH") == "BEYOND EARTH"
    assert _truncate_text("DEEP FIELD") == "DEEP FIELD"
    assert (
        _truncate_text("one two three four five six seven")
        == "ONE TWO THREE FOUR FIVE SIX…"
    )
    assert _truncate_text("  spaced   out  ") == "SPACED OUT"


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


needs_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(), reason="ffmpeg not available"
)


@needs_ffmpeg
def test_build_thumbnails_writes_primary_jpg(tmp_path: Path):
    master = tmp_path / "master.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=6:size=1280x720:rate=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(master),
        ],
        check=True,
        capture_output=True,
    )
    out = build_thumbnails(
        _doc(),
        master,
        tmp_path / "thumbs",
        count=1,
        primary_copy=tmp_path / "thumbnail.jpg",
    )
    primary = Path(out["thumbnail"])
    assert primary.is_file()
    with Image.open(primary) as img:
        assert img.size == (1280, 720)
        assert img.format == "JPEG"


def test_distribute_word_timings_tiles_segment():
    words = "The quick brown fox jumps over the lazy dog now".split()
    cues = distribute_word_timings(" ".join(words), 2.0, 12.0)
    assert [w for _, _, w in cues] == words
    assert cues[0][0] == pytest.approx(2.0)
    assert cues[-1][1] == pytest.approx(12.0)
    for (_, end_a, _), (start_b, _, _) in zip(cues, cues[1:]):
        assert start_b == pytest.approx(end_a)  # contiguous, no gaps/overlaps
    assert distribute_word_timings("", 0, 1) == []
    assert distribute_word_timings("hi", 5, 5) == []


def test_build_ass_karaoke_structure():
    ass = build_ass_karaoke(_doc())
    assert "[Script Info]" in ass
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass
    assert "Style: Karaoke" in ass
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    # Intro scene is skipped; the normal scene's narration yields >= 1 cue.
    assert len(dialogues) >= 1
    for line in dialogues:
        assert "{\\k" in line
        match = re.search(
            r"Dialogue: 0,(\d+):(\d\d):(\d\d)\.(\d\d),"
            r"(\d+):(\d\d):(\d\d)\.(\d\d),",
            line,
        )
        assert match, line
        start = (
            int(match.group(1)) * 3600
            + int(match.group(2)) * 60
            + int(match.group(3))
            + int(match.group(4)) / 100
        )
        end = (
            int(match.group(5)) * 3600
            + int(match.group(6)) * 60
            + int(match.group(7))
            + int(match.group(8)) / 100
        )
        # Karaoke sweep durations (centiseconds) tile the cue duration.
        total_k = sum(int(x) for x in re.findall(r"\{\\k(\d+)\}", line)) / 100
        assert abs(total_k - (end - start)) < 0.05


def test_write_ass_karaoke_file(tmp_path: Path):
    out = write_ass_karaoke(_doc(), tmp_path)
    assert out["ass"].is_file()
    assert out["ass"].read_text(encoding="utf-8").startswith("[Script Info]")


@needs_ffmpeg
def test_burn_karaoke_captions_changes_pixels(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=1280x720:rate=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )
    ass_path = write_ass_karaoke(_short_doc(), tmp_path)["ass"]
    burned = burn_karaoke_captions(clip, ass_path, tmp_path / "burned.mp4")
    assert burned.is_file()

    def frame_pixels(video: Path) -> bytes:
        frame = tmp_path / f"f-{video.stem}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "2", "-i", str(video), "-frames:v", "1", str(frame)],
            check=True,
            capture_output=True,
        )
        with Image.open(frame) as img:
            return img.convert("RGB").tobytes()

    assert frame_pixels(burned) != frame_pixels(clip)
