"""Bug 3 regression: the final master must reach the promised runtime.

Fermi v2 came out at 22:31 vs the 25-minute target (9.92% short). Two holes
combined to let that through:

1. The script was generated before duration enforcement existed, and the
   resumed-script path never verified duration before rendering. A
   fail-closed gate (``_verify_longform_duration``) now refuses to render an
   underweight script.
2. Final QC compared the master against the edit document's own duration
   (self-consistent by construction), never against the promised target.
   The pipeline now passes the target runtime to final QC, whose
   ``check_duration`` enforces it within 5%.
"""

from __future__ import annotations

import pytest

from ai_video_factory.longform import (
    BeatSpec,
    LongformBeat,
    LongformScene,
    LongformScript,
)
from ai_video_factory.qc_final import check_duration
from ai_video_factory.video_pipeline import (
    PipelineCommandError,
    _verify_longform_duration,
)


def _script_with_words(total_words: int) -> LongformScript:
    spec = BeatSpec(
        key="discovery", label="DISCOVERY", fraction=1.0, purpose="", retention=""
    )
    scenes = []
    remaining = total_words
    while remaining > 0:
        take = min(300, remaining)
        scenes.append(
            LongformScene(
                title="S",
                narration=" ".join(["word"] * take),
                visual_direction="",
            )
        )
        remaining -= take
    return LongformScript(
        title="T",
        description="D",
        topic="T",
        target_minutes=25.0,
        beats=[LongformBeat(spec=spec, scenes=scenes)],
    )


def test_rejects_fermi_v2_shortfall() -> None:
    """3378 words vs the 25-minute (3750-word) target is 9.92% short: the
    pipeline must refuse to render instead of producing a short master."""
    script = _script_with_words(3378)
    assert script.total_words == 3378
    with pytest.raises(PipelineCommandError, match="9\\.9% short"):
        _verify_longform_duration(script, 25.0)


def test_accepts_script_within_tolerance() -> None:
    """1.3% short is inside the 5% tolerance; at/over target is fine."""
    _verify_longform_duration(_script_with_words(3700), 25.0)
    _verify_longform_duration(_script_with_words(3750), 25.0)
    _verify_longform_duration(_script_with_words(3900), 25.0)


def _duration_payload(seconds: float) -> dict:
    return {
        "streams": [{"codec_type": "video", "duration": str(seconds)}],
        "format": {"duration": str(seconds)},
    }


def test_final_qc_duration_fails_against_promised_target() -> None:
    """A 22:31 master must fail a duration check against the 25:00 target
    (9.92% drift, limit 5%). This is the check the pipeline now runs."""
    check = check_duration(_duration_payload(1351.2), 1500.0)
    assert not check.passed
    assert "drift" in check.detail


def test_final_qc_duration_passes_within_tolerance() -> None:
    """A 24:30 master (2% drift) passes against the 25:00 target."""
    check = check_duration(_duration_payload(1470.0), 1500.0)
    assert check.passed
