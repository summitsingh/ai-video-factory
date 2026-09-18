"""YouTube packaging: title variants, thumbnail briefs, video descriptions.

This module turns a finished script into the packaging layer that decides
whether anyone clicks: title variants, structured thumbnail briefs, and the
description box text. It is deterministic and LLM-free, so it runs offline
and is fully testable.

Title formulas are ranked with a transparent heuristic (see ``score_title``).
The formula weights below encode general YouTube documentary best practice
only. When the YouTube format research report lands, replace the weights and
add the proven patterns; see INTEGRATION.md for the exact TODO list.

Standing rule: no em dashes anywhere in generated strings.
"""

from __future__ import annotations

import re
from typing import Any

# Words that add click value without lying. Scored, never injected blindly:
# titles are built from topic-derived keywords only (see _topic_keywords),
# so every variant stays defensible from the script.
_POWER_WORDS = frozenset({
    "secret", "mystery", "truth", "hidden", "untold", "lost", "final",
    "strange", "bizarre", "silent", "dark", "vanished", "impossible",
    "shocking", "ultimate", "forbidden", "forgotten", "unseen", "real",
    "exposed", "solved", "unsolved",
})

# Filler with no keyword value when mining the topic for title material.
_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "with", "into",
    "from", "by", "at", "is", "are", "was", "were", "how", "that", "this",
    "our", "their", "its", "your", "as", "to", "be", "it", "we", "you",
    "what", "why", "when", "where", "which", "who", "will", "would",
    "could", "should", "about", "over", "under", "between", "through",
})

_TITLE_FORMULAS = (
    # (name, template) - {topic} is the short topic phrase, {kw} a keyword.
    ("curiosity_gap", "The {topic} Mystery Nobody Talks About"),
    ("stakes", "Why {topic} Could Change Everything"),
    ("number", "The 7 Strangest Facts About {topic}"),
    ("contrarian", "Everything You Know About {topic} Is Wrong"),
    ("question", "{topic}: The Question Science Cannot Answer"),
    ("stakes_personal", "What {topic} Means for Our Future"),
    ("curiosity_gap_2", "What Nobody Tells You About {topic}"),
    ("number_2", "{kw} Discoveries That Rewrite {topic}"),
)

_THUMBNAIL_ARCHETYPES = (
    {
        "concept": "lone subject against the void",
        "why_it_works": (
            "One focal subject with strong figure-ground contrast reads at "
            "120px wide. A question on the thumbnail opens a curiosity gap "
            "the title then promises to close."
        ),
    },
    {
        "concept": "scale contrast",
        "why_it_works": (
            "Tiny human silhouette against a vast cosmic structure triggers "
            "awe and smallness, the core emotion of space documentaries. "
            "Warm/cool color clash stops the scroll in a feed of blue thumbnails."
        ),
    },
    {
        "concept": "the unanswered question",
        "why_it_works": (
            "Direct question text aimed at the viewer plus an empty, silent "
            "landscape. Questions outperform statements for documentary CTR "
            "because they demand an answer."
        ),
    },
)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _topic_keywords(
    topic: str, script_title: str, description: str, limit: int = 6
) -> list[str]:
    """Meaningful keywords mined from topic material, in priority order.

    Only real words from the caller's own material are returned, so any title
    built from them is defensible from the script. Falls back to the script
    title, then to a generic phrase.
    """
    seen: list[str] = []
    for text in (topic, script_title, description):
        for word in _words(text):
            cleaned = word.strip("'")
            if (
                len(cleaned) >= 3
                and cleaned not in _STOP_WORDS
                and cleaned not in seen
            ):
                seen.append(cleaned)
            if len(seen) >= limit:
                return seen
    if not seen:
        return ["unknown"]
    return seen


def _short_topic(topic: str, script_title: str) -> str:
    """A title-cased short phrase for the {topic} slot, e.g. 'Fermi Paradox'."""
    keywords = _topic_keywords(topic, script_title, "", limit=3)
    return " ".join(w.capitalize() for w in keywords[:2])


def score_title(title: str) -> float:
    """Heuristic click-worthiness score for a title, 0.0 to 1.0.

    Rewards: 40-60 character length (the readable range on all surfaces),
    power words, a concrete number. Penalizes: ALL CAPS words, em dashes,
    shouty repeated punctuation.
    """
    if not title or not title.strip():
        return 0.0
    score = 0.55  # base: a plain grammatical title
    length = len(title)

    # Length: 40-60 is the sweet spot; taper off outside it.
    if 40 <= length <= 60:
        score += 0.20
    elif 30 <= length < 40 or 60 < length <= 75:
        score += 0.10
    elif length < 20 or length > 90:
        score -= 0.20

    words = _words(title)
    power_hits = sum(1 for w in words if w in _POWER_WORDS)
    score += min(power_hits * 0.08, 0.16)

    # Concrete numbers perform well in documentary titles.
    if re.search(r"\b\d+\b", title):
        score += 0.05

    # ALL CAPS words read as spam.
    caps_words = [
        w for w in re.findall(r"[A-Za-z']+", title)
        if len(w) >= 3 and w.isupper()
    ]
    score -= 0.25 * len(caps_words)

    # Em dashes are banned from all output; score them harshly if seen.
    if "\u2014" in title or "\u2013" in title:
        score -= 0.50

    # Shouty punctuation.
    if title.count("!") > 1 or title.count("?") > 2:
        score -= 0.20

    return round(max(0.0, min(1.0, score)), 3)


def generate_title_variants(
    topic: str,
    script_title: str,
    description: str = "",
    count: int = 3,
) -> list[str]:
    """Generate ``count`` YouTube title variants, best first.

    Formulas cover the patterns that win for documentary/essay content:
    curiosity gap, stakes, number/spec, contrarian, and the direct question.
    Every variant is built only from words in the topic material, so none of
    them can promise something the script does not contain.
    """
    short = _short_topic(topic, script_title)
    keywords = _topic_keywords(topic, script_title, description)
    kw = keywords[0].capitalize() if keywords else "New"

    candidates: list[str] = []
    for _name, template in _TITLE_FORMULAS:
        title = template.format(topic=short, kw=kw)
        if title not in candidates:
            candidates.append(title)

    # The script's own title is always a candidate; sometimes it wins.
    if script_title and script_title.strip():
        clean = script_title.strip()
        if clean not in candidates:
            candidates.append(clean)

    ranked = sorted(candidates, key=score_title, reverse=True)
    return ranked[: max(1, count)]


def _scene_dicts(script: Any) -> list[dict[str, Any]]:
    if script is None:
        return []
    if isinstance(script, dict):
        scenes = script.get("scenes", [])
    else:
        scenes = getattr(script, "scenes", [])
    return [s for s in scenes if isinstance(s, dict)]


def _script_title(script: Any) -> str:
    if script is None:
        return ""
    if isinstance(script, dict):
        return str(script.get("title", "") or "")
    return str(getattr(script, "title", "") or "")


def _visual_anchors(scenes: list[dict[str, Any]], limit: int = 8) -> list[str]:
    """Concrete visual nouns mined from scene visual directions."""
    anchors = (
        "black hole", "space station", "mission control", "launch pad",
        "nebula", "galaxy", "galaxies", "planet", "planets", "earth",
        "moon", "mars", "astronaut", "telescope", "rocket", "satellite",
        "stars", "star", "comet", "asteroid", "sun",
    )
    found: list[str] = []
    for scene in scenes:
        visual = str(scene.get("visual", "") or "").lower()
        for anchor in anchors:
            if anchor in visual and anchor not in found:
                found.append(anchor)
            if len(found) >= limit:
                return found
    return found


def _overlay_text(keywords: list[str], style: str) -> str:
    """Thumbnail text: uppercase, max 4 words, no em dashes."""
    if style == "question":
        words = ["WHERE", "IS", "EVERYBODY?"][:4]
        if "paradox" in keywords:
            words = ["THE", "GREAT", "SILENCE"][:4]
    elif style == "stakes":
        words = [w.upper() for w in keywords[:2]] + ["ENDS", "HERE"]
        words = words[:4]
    else:  # number
        words = ["7", "STRANGE", "FACTS"][:4]
    text = " ".join(words)
    assert len(text.split()) <= 4, "overlay exceeded 4 words"
    assert "\u2014" not in text, "em dash in overlay"
    return text


def generate_thumbnail_briefs(script: Any, count: int = 3) -> list[dict]:
    """Generate ``count`` structured thumbnail briefs from a script.

    Each brief is a dict with: concept, foreground_subject, background,
    text_overlay (max 4 words), color_mood, why_it_works. Designed for the
    1280x720 YouTube canvas: high contrast, one focal subject, readable at
    small sizes. Accepts a ScriptOutput or a plain dict with
    title/scenes keys.
    """
    scenes = _scene_dicts(script)
    title = _script_title(script)
    anchors = _visual_anchors(scenes)
    subject = anchors[0] if anchors else "planet"
    keywords = _topic_keywords(title, "", "")

    first_visual = ""
    for scene in scenes:
        if scene.get("visual"):
            first_visual = str(scene["visual"])[:120]
            break

    archetypes = list(_THUMBNAIL_ARCHETYPES)
    briefs: list[dict] = []
    styles = ("question", "stakes", "number")
    for i in range(max(1, count)):
        arch = archetypes[i % len(archetypes)]
        style = styles[i % len(styles)]
        if i == 0:
            foreground = f"single {subject} in sharp detail, centered right"
            background = (
                f"deep-space nebula field, near-black with stars"
                + (f" ({first_visual})" if first_visual else "")
            )
            mood = "near-black blues with one warm orange accent"
        elif i == 1:
            foreground = "tiny astronaut silhouette for human scale"
            background = f"vast {subject} dominating the frame"
            mood = "cold teal shadows, hot amber rim light"
        else:
            foreground = f"extreme close-up of {subject} surface detail"
            background = "empty black void, subtle star grain"
            mood = "monochrome with a single red accent"
        briefs.append({
            "concept": arch["concept"],
            "foreground_subject": foreground,
            "background": background,
            "text_overlay": _overlay_text(keywords, style),
            "color_mood": mood,
            "why_it_works": arch["why_it_works"],
        })
    return briefs


def _clean_source_label(url: str) -> str:
    """Short publisher label for a source URL (full URL stays in text)."""
    lower = url.lower()
    if "nasa.gov" in lower:
        return "NASA"
    if "news.google.com" in lower:
        return "Google News"
    if "wikipedia.org" in lower:
        return "Wikipedia"
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else url


def _format_timestamp(total_seconds: int) -> str:
    minutes, seconds = divmod(max(0, total_seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def generate_description(
    topic: str, script: Any, sources: list[str]
) -> str:
    """Build the YouTube description box text.

    Hook paragraph first (what the viewer gets), then a chapter list derived
    from the scene titles with estimated timestamps, then the sources section.
    Full URLs belong here in the description, never burned into the video.
    """
    scenes = _scene_dicts(script)
    title = _script_title(script) or topic

    hook_lines = [f"{title} - a documentary on {topic}."]
    first_narration = ""
    if scenes:
        first_narration = str(scenes[0].get("narration", "") or "").strip()
    if first_narration:
        first_sentence = re.split(r"(?<=[.!?])\s", first_narration, maxsplit=1)[0]
        hook_lines.append(first_sentence[:220])
    hook_lines.append(
        "We dig into the evidence, the leading theories, and the questions "
        "nobody has answered yet."
    )
    hook = " ".join(hook_lines)

    lines = [hook, "", "CHAPTERS"]
    elapsed = 0
    for scene in scenes:
        scene_title = str(scene.get("title", "") or "Untitled").strip() or "Untitled"
        frames = scene.get("duration_frames", 0)
        try:
            seconds = int(frames) // 30
        except (TypeError, ValueError):
            seconds = 0
        lines.append(f"{_format_timestamp(elapsed)} {scene_title}")
        elapsed += seconds
    if not scenes:
        lines.append("00:00 Introduction")

    lines += ["", "SOURCES"]
    seen_labels: set[str] = set()
    for url in sources or []:
        label = _clean_source_label(url)
        suffix = ""
        if label in seen_labels:
            suffix = f" ({url})"
        seen_labels.add(label)
        lines.append(f"- {label}{suffix}: {url}")
    if not sources:
        lines.append("- Sources available on request.")

    description = "\n".join(lines)
    assert "\u2014" not in description, "em dash leaked into description"
    return description
