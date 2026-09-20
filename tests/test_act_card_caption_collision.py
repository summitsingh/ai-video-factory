"""Bug 1 regression: act-card scenes must not stack caption layers on the card.

Fermi v2's chapter cards (~10:30 "A New Dawn of Inquiry", ~18:30 "The Horizon
Beckons") rendered the act label, a full-paragraph lower third, and a
full-paragraph subtitle all at once: the act-card window and the
transient-text envelope both ran frames 0..83 of the scene. The renderer now
delays the transient-text envelope until the act card has fully faded.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from ai_video_factory.edit_schema import EditDocument, EditScene, save_edit
from ai_video_factory.longform import (
    BeatSpec,
    LongformBeat,
    LongformScene,
    LongformScript,
    longform_to_edit_document,
)
from ai_video_factory.video_pipeline import _render_with_remotion

ROOT = Path(__file__).resolve().parents[1]
REMOTION_ROOT = ROOT / "remotion"
TSX = REMOTION_ROOT / "src" / "SyntheticVideo.tsx"


def _make_script() -> LongformScript:
    narration = " ".join(["word"] * 60)
    beats = []
    for key, label in (
        ("cold_open", "COLD OPEN"),
        ("discovery", "A NEW DAWN OF INQUIRY"),
    ):
        spec = BeatSpec(key=key, label=label, fraction=0.5, purpose="", retention="")
        beats.append(
            LongformBeat(
                spec=spec,
                scenes=[
                    LongformScene(
                        title=f"{label} scene",
                        narration=narration,
                        visual_direction="starfield",
                    )
                ],
            )
        )
    return LongformScript(
        title="T", description="D", topic="T", target_minutes=25.0, beats=beats
    )


def test_beat_opening_scenes_carry_act_cards() -> None:
    """The act-card surface the renderer fix targets: beat-opening scenes."""
    edit = longform_to_edit_document(_make_script())
    assert len(edit.scenes) == 2
    assert edit.scenes[0].act == "COLD OPEN"
    assert edit.scenes[1].act == "A NEW DAWN OF INQUIRY"


def test_renderer_defers_transient_text_during_act_card() -> None:
    """Cross-language contract: the TSX renderer must delay the caption
    envelope on act-card scenes past the full act-card window."""
    src = TSX.read_text(encoding="utf-8")
    assert "const ACT_CARD_TOTAL_FRAMES" in src
    assert "scene.act ? frame - ACT_CARD_TOTAL_FRAMES : frame" in src


def _chrome_available() -> bool:
    exe = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    return bool(exe) and (REMOTION_ROOT / "node_modules/.bin/remotion").is_file()


needs_chrome = pytest.mark.skipif(
    not _chrome_available(),
    reason="act-card pixel test requires Chrome + the remotion build",
)


def _cyan_fraction(frame: Path) -> float:
    """Fraction of pixels that look like the LowerThird cyan box."""
    img = Image.open(frame).convert("RGB").resize((64, 36))
    px = img.load()
    count = sum(
        1
        for y in range(img.height)
        for x in range(img.width)
        if (lambda p: p[1] > 150 and p[2] > 150 and p[0] < 120)(px[x, y])
    )
    return count / (img.width * img.height)


def _bright_center_fraction(frame: Path) -> float:
    """Fraction of bright pixels in the frame's vertical center band at full
    resolution (the act label lives here)."""
    img = Image.open(frame).convert("RGB")
    px = img.load()
    count = 0
    total = 0
    for y in range(int(img.height * 0.3), int(img.height * 0.6), 2):
        for x in range(0, img.width, 2):
            r, g, b = px[x, y]
            total += 1
            if r > 180 and g > 180 and b > 180:
                count += 1
    return count / total


def _extract_frame(video: Path, at_seconds: float, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(at_seconds), "-i", str(video),
         "-frames:v", "1", str(dest)],
        check=True, capture_output=True, text=True, timeout=60,
    )


@needs_chrome
def test_act_card_timing_has_no_caption_collision(tmp_path: Path) -> None:
    """A real headless render of an act-card scene via the production render
    path: during the card window no caption layers may be visible, the act
    label must still render, and after the card fades the caption layers must
    return (not be deleted)."""
    exe = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    caption = (
        "For centuries humanity stared at the night sky and wondered whether "
        "anyone was staring back, a question that has only grown sharper as "
        "our telescopes have grown larger and our silence has grown longer."
    )
    scene = EditScene(
        id="discovery-0",
        from_frame=0,
        duration_frames=240,
        title="A New Dawn of Inquiry",
        caption=caption,
        kind="normal",
        narration=caption,
        act="A NEW DAWN OF INQUIRY",
    )
    edit = EditDocument(
        schema_version=1,
        width=640,
        height=360,
        fps=30,
        duration_frames=240,
        scenes=[scene],
    )
    fixture = tmp_path / "act-edit.json"
    save_edit(edit, fixture)
    dest = tmp_path / "act.mp4"
    _render_with_remotion(
        ROOT,
        fixture,
        dest,
        browser=Path(exe),
        npm="npm",
        timeout=600,
    )
    assert dest.is_file() and dest.stat().st_size > 0, "act-card master not produced"

    card_frame = tmp_path / "card.png"
    _extract_frame(dest, 1.5, card_frame)  # frame 45: act card fully visible
    assert _cyan_fraction(card_frame) < 0.001, (
        "caption layers visible during the act-card window "
        f"(cyan fraction {_cyan_fraction(card_frame):.4f})"
    )
    assert _bright_center_fraction(card_frame) > 0.01, (
        "act label itself did not render; the test would be vacuous"
    )

    post_frame = tmp_path / "post.png"
    _extract_frame(dest, 4.5, post_frame)  # frame 135: card gone, captions due
    assert _cyan_fraction(post_frame) > 0.005, (
        "caption layers never returned after the act card faded "
        f"(cyan fraction {_cyan_fraction(post_frame):.4f})"
    )
