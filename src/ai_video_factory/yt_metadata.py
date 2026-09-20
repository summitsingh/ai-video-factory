"""YouTube metadata generation: titles, description, tags, chapters.

Deterministic and LLM-free. After script finalization the pipeline calls
``generate_metadata`` and saves ``metadata.json`` + ``metadata.md`` to the
run dir. Chapter timestamps derive from the script's beat boundaries
(long-form) or scene durations (short path).

Reuses ``youtube.format_timestamp`` / ``youtube.build_chapter_lines`` for
chapter rendering and ``packaging.generate_title_variants`` /
``packaging.score_title`` for title options instead of duplicating them.

Standing rule: no em dashes anywhere in generated strings.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_video_factory import youtube as _youtube
from ai_video_factory.packaging import generate_title_variants, score_title

WORDS_PER_MINUTE = 150
TITLE_MAX_CHARS = 60
TAG_COUNT = 12

_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "with", "into",
    "from", "by", "at", "is", "are", "was", "were", "how", "that", "this",
    "our", "their", "its", "your", "as", "to", "be", "it", "we", "you",
    "what", "why", "when", "where", "which", "who", "will", "would",
    "could", "should", "about", "over", "under", "between", "through",
})


def _keywords(*texts: str, limit: int = 8) -> list[str]:
    """Meaningful keywords mined from the given texts, in priority order."""
    seen: list[str] = []
    for text in texts:
        for word in re.findall(r"[A-Za-z0-9']+", (text or "").lower()):
            cleaned = word.strip("'")
            if len(cleaned) >= 3 and cleaned not in _STOP_WORDS and cleaned not in seen:
                seen.append(cleaned)
            if len(seen) >= limit:
                return seen
    return seen


def _beat_chapters(script: Any) -> list[dict[str, Any]]:
    """Chapter per beat boundary; timestamps from cumulative word counts."""
    chapters: list[dict[str, Any]] = []
    elapsed = 0.0
    for beat in script.beats:
        first_title = beat.scenes[0].title if beat.scenes else beat.spec.key
        # Keep the beat label verbatim ("ACT II", not "Act Ii").
        label = beat.spec.label.strip()
        title = f"{label}: {first_title}".strip()
        chapters.append({"title": title, "start_seconds": elapsed})
        elapsed += beat.words / WORDS_PER_MINUTE * 60.0
    return chapters


def _scene_chapters(scenes: list[dict[str, Any]], fps: int = 30) -> list[dict[str, Any]]:
    """Chapter per scene; timestamps from duration_frames when available."""
    chapters: list[dict[str, Any]] = []
    elapsed = 0.0
    for scene in scenes:
        title = str(scene.get("title") or "Untitled").strip() or "Untitled"
        chapters.append({"title": title, "start_seconds": elapsed})
        try:
            elapsed += int(scene.get("duration_frames", 0) or 0) / fps
        except (TypeError, ValueError):
            pass
    return chapters


def chapters_for_script(script: Any) -> list[dict[str, Any]]:
    """Derive chapter boundaries from a LongformScript or ScriptOutput/dict."""
    beats = getattr(script, "beats", None)
    if beats:
        return _beat_chapters(script)
    scenes: list[dict[str, Any]] = []
    if isinstance(script, dict):
        raw = script.get("scenes", [])
        scenes = [s for s in raw if isinstance(s, dict)]
    else:
        raw = getattr(script, "scenes", []) or []
        scenes = [s for s in raw if isinstance(s, dict)]
    return _scene_chapters(scenes)


def chapter_lines(chapters: list[dict[str, Any]]) -> list[str]:
    """Render chapter lines, reusing youtube.py's recognized format."""
    lines = _youtube.build_chapter_lines(chapters)
    if lines:
        return lines
    # Fallback for scripts with fewer than 3 chapters: still timestamped.
    cleaned: list[tuple[float, str]] = []
    for chapter in chapters:
        title = str(chapter.get("title", "")).strip()
        try:
            start = float(chapter.get("start_seconds", 0))
        except (TypeError, ValueError):
            continue
        if title:
            cleaned.append((start, title))
    cleaned.sort(key=lambda item: item[0])
    return [f"{_youtube.format_timestamp(start)} {title}" for start, title in cleaned]


def _short_topic(topic: str, script_title: str) -> str:
    words = _keywords(topic, script_title, limit=2)
    return " ".join(w.capitalize() for w in words[:2]) or "This Story"


def _fallback_titles(topic: str, script_title: str) -> list[str]:
    """Curiosity-driven, defensible title options (no clickbait lies)."""
    short = _short_topic(topic, script_title)
    kws = _keywords(topic, script_title)
    kw = kws[0].capitalize() if kws else "New"
    return [
        f"The {short} Mystery Nobody Talks About",
        f"{short}: The Question Science Cannot Answer",
        f"What Nobody Tells You About {short}",
        f"Why {short} Could Change Everything",
        f"{kw} Discoveries That Rewrite {short}",
    ]


def generate_title_options(
    topic: str, script_title: str, count: int = 3, max_chars: int = TITLE_MAX_CHARS
) -> list[str]:
    """Pick ``count`` title options under ``max_chars``, best first.

    Candidates come from packaging's formula variants (built only from
    words in the topic material, so every option stays defensible from
    the script); packaging's own score ranks them. Guaranteed to return
    ``count`` options.
    """
    candidates: list[str] = []
    for variant in generate_title_variants(topic, script_title, "", count=8):
        clean = " ".join(variant.split())
        # No bare numbers: "The 7 Strangest Facts" would assert a count the
        # script may not contain (no clickbait lies).
        if re.search(r"\b\d+\b", clean):
            continue
        if clean and len(clean) <= max_chars and clean not in candidates:
            candidates.append(clean)
    for fallback in _fallback_titles(topic, script_title):
        if len(candidates) >= count:
            break
        if fallback not in candidates and len(fallback) <= max_chars:
            candidates.append(fallback)
    ranked = sorted(candidates, key=score_title, reverse=True)
    options = ranked[: max(1, count)]
    assert all(len(t) <= max_chars for t in options), "title exceeded char limit"
    assert all("\u2014" not in t for t in options), "em dash in title"
    return options


def generate_tags(topic: str, script_title: str, count: int = TAG_COUNT) -> list[str]:
    """10-15 YouTube tags: topic keywords first, then documentary staples."""
    kws = _keywords(topic, script_title, limit=8)
    tags: list[str] = []
    for kw in kws:
        tag = kw if len(kw.split()) > 1 else kw
        if tag not in tags:
            tags.append(tag)
    # Multi-word topic phrase as one tag when it reads naturally.
    short = _short_topic(topic, script_title).lower()
    if short not in tags:
        tags.insert(0, short)
    staples = [
        "documentary", "science documentary", "space documentary",
        "astronomy", "astrophysics", "seti", "exoplanets",
        "universe", "cosmology", "science", "space exploration",
        "mystery", "explained",
    ]
    for staple in staples:
        if len(tags) >= count:
            break
        if staple not in tags:
            tags.append(staple)
    tags = tags[:count]
    assert 10 <= len(tags) <= 15, f"tag count out of range: {len(tags)}"
    return tags


def _hook_sentence(script: Any, script_title: str, topic: str) -> str:
    """First sentence of the cold open: the description's hook paragraph."""
    narration = ""
    beats = getattr(script, "beats", None)
    if beats and beats[0].scenes:
        narration = beats[0].scenes[0].narration or ""
    elif isinstance(script, dict):
        scenes = script.get("scenes", [])
        if scenes:
            narration = str(scenes[0].get("narration", "") or "")
    else:
        scenes = getattr(script, "scenes", []) or []
        if scenes and isinstance(scenes[0], dict):
            narration = str(scenes[0].get("narration", "") or "")
    narration = " ".join(narration.split())
    if narration:
        first = re.split(r"(?<=[.!?])\s", narration, maxsplit=1)[0]
        return first[:280]
    title = script_title or topic
    return f"{title} - a documentary on {topic}."


def _source_label(url: str) -> str:
    lower = url.lower()
    if "wikipedia.org" in lower:
        return "Wikipedia"
    if "seti.org" in lower:
        return "SETI Institute"
    if "nasa.gov" in lower:
        return "NASA"
    if "arxiv.org" in lower:
        return "arXiv"
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else url


def generate_description(
    *,
    topic: str,
    script_title: str,
    summary: str,
    chapters: list[dict[str, Any]],
    sources: list[str],
    script: Any = None,
) -> str:
    """Build the description: hook paragraph + summary + chapters + sources."""
    hook = _hook_sentence(script, script_title, topic)
    summary_text = " ".join((summary or "").split())
    if not summary_text:
        summary_text = (
            f"We dig into the evidence, the leading theories, and the "
            f"questions about {topic} that nobody has answered yet."
        )
    lines = [hook, "", summary_text, "", "CHAPTERS"]
    lines.extend(chapter_lines(chapters) or ["00:00 Introduction"])
    lines += ["", "SOURCES"]
    seen: set[str] = set()
    for url in sources or []:
        label = _source_label(url)
        suffix = f" ({url})" if label in seen else ""
        seen.add(label)
        lines.append(f"- {label}{suffix}: {url}")
    if not sources:
        lines.append("- Sources available on request.")
    description = "\n".join(lines)
    assert "\u2014" not in description, "em dash leaked into description"
    return description


def generate_metadata(
    script: Any,
    *,
    topic: str,
    summary: str = "",
) -> dict[str, Any]:
    """Generate the full YouTube metadata package for a finished script."""
    if isinstance(script, dict):
        script_title = str(script.get("title", "") or "")
        sources = list(script.get("sources", []) or [])
    else:
        script_title = str(getattr(script, "title", "") or "")
        sources = list(getattr(script, "sources", []) or [])
    chapters = chapters_for_script(script)
    metadata = {
        "topic": topic,
        "script_title": script_title,
        "title_options": generate_title_options(topic, script_title),
        "description": generate_description(
            topic=topic,
            script_title=script_title,
            summary=summary,
            chapters=chapters,
            sources=sources,
            script=script,
        ),
        "tags": generate_tags(topic, script_title),
        "chapters": [
            {
                "title": c["title"],
                "start_seconds": round(float(c["start_seconds"]), 1),
                "timestamp": _youtube.format_timestamp(float(c["start_seconds"])),
            }
            for c in chapters
        ],
        "sources": sources,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return metadata


def render_markdown(metadata: dict[str, Any]) -> str:
    """Human-readable metadata.md from the metadata dict."""
    lines = [
        "# YouTube Metadata",
        "",
        f"Topic: {metadata.get('topic', '')}",
        f"Generated: {metadata.get('generated_at', '')}",
        "",
        "## Title options (pick one)",
        "",
    ]
    for index, title in enumerate(metadata.get("title_options", []), start=1):
        lines.append(f"{index}. {title}")
    lines += ["", "## Description", "", metadata.get("description", "")]
    lines += ["", "## Tags", ""]
    lines.append(", ".join(metadata.get("tags", [])))
    lines += ["", "## Chapters", ""]
    for chapter in metadata.get("chapters", []):
        lines.append(f"- {chapter['timestamp']} {chapter['title']}")
    lines += ["", "## Sources", ""]
    for url in metadata.get("sources", []):
        lines.append(f"- {url}")
    lines.append("")
    return "\n".join(lines)


def save_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> dict[str, Path]:
    """Write metadata.json + metadata.md to the run dir; return the paths."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "metadata.json"
    md_path = run_dir / "metadata.md"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(metadata), encoding="utf-8")
    return {"json": json_path, "md": md_path}
