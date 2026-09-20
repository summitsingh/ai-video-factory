"""Tests for YouTube metadata generation (item 7)."""

from ai_video_factory.longform import (
    BeatSpec,
    LongformBeat,
    LongformScene,
    LongformScript,
)
from ai_video_factory.yt_metadata import (
    chapters_for_script,
    generate_metadata,
    render_markdown,
    save_metadata,
)


def make_stub_longform() -> LongformScript:
    specs = [
        ("cold_open", "COLD OPEN", 0.10),
        ("act1_setup", "ACT I", 0.30),
        ("act2a_evidence", "ACT II", 0.30),
        ("act3_climax", "ACT III", 0.20),
        ("outro", "OUTRO", 0.10),
    ]
    beats = []
    for key, label, fraction in specs:
        spec = BeatSpec(key=key, label=label, fraction=fraction,
                        purpose="p", retention="r")
        words = int(20.0 * 150 * fraction)
        beats.append(
            LongformBeat(
                spec=spec,
                scenes=[
                    LongformScene(
                        title=f"{label.title()} begins",
                        narration=(
                            "Where is everybody? The galaxy should be loud "
                            "with alien civilizations, yet we hear nothing. "
                            + " ".join(["word"] * words)
                        ),
                        visual_direction="starfield",
                    )
                ],
            )
        )
    return LongformScript(
        title="The Fermi Paradox",
        description="Why the universe is silent",
        topic="The Fermi Paradox",
        target_minutes=20.0,
        beats=beats,
        sources=[
            "https://en.wikipedia.org/wiki/Fermi_paradox",
            "https://www.seti.org/research/seti-101/fermi-paradox/",
        ],
    )


def test_chapters_derive_from_beat_boundaries():
    script = make_stub_longform()
    chapters = chapters_for_script(script)
    assert len(chapters) == 5
    assert chapters[0]["start_seconds"] == 0
    # Timestamps strictly increase along beat boundaries.
    starts = [c["start_seconds"] for c in chapters]
    assert starts == sorted(starts)
    assert starts[1] > 0
    # Total span matches the narration length at 150 wpm.
    expected = script.total_words / 150 * 60
    assert starts[-1] < expected


def test_generate_metadata_structure():
    script = make_stub_longform()
    metadata = generate_metadata(
        script, topic="The Fermi Paradox", summary="A look at the silence."
    )
    # 3 title options, each under 60 chars, curiosity-driven.
    assert len(metadata["title_options"]) == 3
    assert all(len(t) <= 60 for t in metadata["title_options"])
    assert all(t.strip() for t in metadata["title_options"])
    assert len(set(metadata["title_options"])) == 3
    # 10-15 tags.
    assert 10 <= len(metadata["tags"]) <= 15
    # Description: hook + summary + chapters + sources.
    description = metadata["description"]
    assert "Where is everybody?" in description  # hook from cold open
    assert "A look at the silence." in description  # summary
    assert "CHAPTERS" in description
    assert "SOURCES" in description
    assert "https://en.wikipedia.org/wiki/Fermi_paradox" in description
    assert "00:00" in description
    # Chapter records carry formatted timestamps.
    assert metadata["chapters"][0]["timestamp"] == "00:00"
    assert metadata["sources"] == script.sources


def test_generate_metadata_short_script_dict():
    script = {
        "title": "Quick Take",
        "scenes": [
            {"title": "Intro", "narration": "Hello world.",
             "duration_frames": 900},
            {"title": "Main", "narration": "The story.",
             "duration_frames": 1800},
            {"title": "End", "narration": "Goodbye.",
             "duration_frames": 900},
        ],
        "sources": ["https://www.nasa.gov"],
    }
    metadata = generate_metadata(script, topic="Quick Take")
    assert len(metadata["title_options"]) == 3
    assert len(metadata["chapters"]) == 3
    assert metadata["chapters"][1]["start_seconds"] == 30.0


def test_save_metadata_writes_json_and_md(tmp_path):
    script = make_stub_longform()
    metadata = generate_metadata(script, topic="The Fermi Paradox")
    paths = save_metadata(tmp_path, metadata)
    assert paths["json"].name == "metadata.json"
    assert paths["md"].name == "metadata.md"
    assert paths["json"].is_file()
    assert paths["md"].is_file()
    md = render_markdown(metadata)
    assert "# YouTube Metadata" in md
    assert "## Title options" in md
    assert "## Tags" in md
    for title in metadata["title_options"]:
        assert title in md
