"""Real-Remotion integration test: a staged still is visibly composited into the master.

The fast unit tests in ``test_production.py`` stub ``_run_remotion`` (they prove wiring,
fail-closed behavior, and ffmpeg muxing). This one drives the ACTUAL headless render
(``@remotion/cli`` + Chrome) and proves the approved asset *changed* the rendered pixels --
i.e. the master visibly contains it.

Design notes:
- SyntheticVideo paints an ``rgba(0,0,0,0.55)`` readability overlay over each scene, so a
  full-frame magenta still (255,0,255) renders at ~ (115,0,115). Detection therefore uses a
  ratio mask (R>70, B>70, G < 0.4*min(R,B)) rather than an exact color.
- A no-image control frame is rendered alongside the asset frame and compared: this proves
  the magenta comes from the ASSET, not any scene background. Empirically the control carries
  ~0% magenta while the asset frame carries ~89%, and the mean pixel difference is large.
- On assertion failure both frames are copied to ``output/pixel-test-failures/`` (gitignored)
  for inspection; cleanup removes only run-created children under ``public/assets/``.

Skipped when Chrome or the remotion build is unavailable so the fast suite never depends on
external binaries.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from ai_video_factory.edit_schema import EditDocument, EditScene
from ai_video_factory.production import RemotionKokoroRenderEngine, _write_tone_wav

ROOT = Path(__file__).resolve().parents[1] / "remotion"


def _chrome_available() -> bool:
    exe = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    return bool(exe) and (ROOT / "node_modules/.bin/remotion").is_file()


pytestmark = pytest.mark.skipif(
    not _chrome_available(),
    reason="real-Remotion pixel test requires Chrome + the remotion build",
)

MAGENTA = (255, 0, 255)  # full-frame still; dims to ~ (115, 0, 115) under the 0.55 black overlay


def _magenta_fraction(frame: Path) -> float:
    """Fraction of pixels whose color is magenta-ish (R~B high, G low)."""
    small = Image.open(frame).convert("RGB").resize((60, 34))
    total = small.width * small.height
    count = 0
    for r, g, b in small.getdata():
        if r > 70 and b > 70 and g < 0.4 * min(r, b):
            count += 1
    return count / total


def _mean_abs_diff(a: Path, b: Path) -> float:
    """Mean absolute per-channel pixel difference between two frames (0-255)."""
    ia = Image.open(a).convert("RGB").resize((60, 34))
    ib = Image.open(b).convert("RGB").resize((60, 34))
    pa = list(ia.getdata())
    pb = list(ib.getdata())
    total = sum(
        abs(x - y)
        for (ar, ag, ab), (cr, cg, cb) in zip(pa, pb)
        for x, y in ((ar, cr), (ag, cg), (ab, cb))
    )
    return total / (len(pa) * 3)


def _extract_frame(dest: Path, frame: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "1.0", "-i", str(dest), "-frames:v", "1", str(frame)],
        check=True, capture_output=True, text=True, timeout=60,
    )


def test_staged_still_is_composited_into_rendered_master(tmp_path: Path) -> None:
    """A real headless render of a scene whose only visual is a distinctive still must show
    that still's color dominating an in-scene frame AND differ from a no-image control -- so
    the master visibly contains the asset; a failed load renders like the black control."""
    public = ROOT / "public"
    assets_dir = public / "assets"
    existing = set(assets_dir.glob("*")) if assets_dir.exists() else set()
    exe = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    fail_dir = Path("output") / "pixel-test-failures"  # gitignored; only written on failure
    frame_with: Path | None = None
    frame_without: Path | None = None
    try:
        still = tmp_path / "distinctive.jpg"
        Image.new("RGB", (1920, 1080), MAGENTA).save(still, "JPEG", quality=95)

        scene = EditScene(id="scene-1", from_frame=0, duration_frames=60, title="Test Scene", caption="Neutral label")
        edit = EditDocument(schema_version=1, width=1920, height=1080, fps=30, duration_frames=60, scenes=[scene])

        engine = RemotionKokoroRenderEngine(remotion_root=ROOT, public_dir=public, browser_executable=Path(exe))

        narr = tmp_path / "n.wav"
        _write_tone_wav(narr, seconds=2.0, hertz=300)

        # Asset render: scene-1's only visual is the distinctive still.
        dest_with = tmp_path / "with.mp4"
        engine.render_master(
            edit,
            narration_segments=[(narr, 0.0)],
            assets_by_scene_id={"scene-1": still},
            destination=dest_with,
        )
        assert dest_with.is_file() and dest_with.stat().st_size > 0, "asset master not produced"

        # No-image control: same scene/timing, no asset -> proves background isn't the source.
        edit_none = EditDocument(
            schema_version=1, width=1920, height=1080, fps=30, duration_frames=60,
            scenes=[EditScene(id="scene-1", from_frame=0, duration_frames=60, title="Test Scene", caption="Neutral label")],
        )
        dest_none = tmp_path / "none.mp4"
        engine.render_master(
            edit_none,
            narration_segments=[(narr, 0.0)],
            assets_by_scene_id={},
            destination=dest_none,
        )

        frame_with = tmp_path / "with.png"
        frame_without = tmp_path / "none.png"
        _extract_frame(dest_with, frame_with)
        _extract_frame(dest_none, frame_without)

        frac = _magenta_fraction(frame_with)
        diff = _mean_abs_diff(frame_with, frame_without)
        assert frac > 0.3, f"staged still not visibly composited (magenta fraction {frac:.3f})"
        assert diff > 10, f"asset did not change rendered pixels vs control (mean diff {diff:.2f})"
    except AssertionError:
        fail_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        if frame_with is not None:
            shutil.copy2(frame_with, fail_dir / f"asset-{ts}.png")
        if frame_without is not None:
            shutil.copy2(frame_without, fail_dir / f"control-{ts}.png")
        raise
    finally:
        # Remove only children created by this run; never touch pre-existing/user assets.
        for entry in (assets_dir.glob("*") if assets_dir.exists() else []):
            if entry not in existing and entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
        shutil.rmtree(tmp_path, ignore_errors=True)
