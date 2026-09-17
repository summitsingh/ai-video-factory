"""Per-topic theming for the video factory.

A theme bundles every piece of copy that used to be hardcoded for the
original space documentary: the outro caption, the thumbnail power words,
and the NASA search stop words. Pipelines take an optional ``ThemeConfig``
(defaulting to the built-in "space" theme so existing behavior is unchanged)
and can load custom themes from JSON for new channels or topics.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ThemeConfig(BaseModel):
    """Copy/branding bundle for one topic or channel."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(min_length=1)
    outro_caption: str = Field(min_length=1)
    thumbnail_power_words: list[str] = Field(default_factory=list)
    nasa_stop_words: list[str] = Field(default_factory=list)
    channel_name: str | None = None


# Exact copy carried over from the original space documentary pipeline.
SPACE_THEME = ThemeConfig(
    name="space",
    outro_caption="The Eyes That See Everything",
    thumbnail_power_words=[
        "eyes", "golden eye", "golden", "deepest", "oldest", "invisible",
        "beyond", "frontier", "origins", "cosmic", "universe", "life",
        "secrets", "hidden", "first light", "telescope", "jwst", "webb",
    ],
    nasa_stop_words=[
        "the", "a", "an", "of", "and", "or", "in", "on", "for", "with", "into",
        "from", "by", "at", "is", "are", "was", "were", "how", "that", "this",
        "our", "their", "its", "your", "as", "to", "be", "see", "everything",
        "rewrote", "cosmic", "history", "james", "webb", "telescope", "telescopes",
    ],
)

# Neutral copy for non-space topics. No documentary-specific wording.
GENERIC_THEME = ThemeConfig(
    name="generic",
    outro_caption="Thanks for watching",
    thumbnail_power_words=[
        "amazing", "incredible", "new", "future", "world", "life",
        "secret", "secrets", "hidden", "discover", "discovery",
        "breakthrough", "inside", "story", "untold", "first",
    ],
    nasa_stop_words=[
        "the", "a", "an", "of", "and", "or", "in", "on", "for", "with",
        "into", "from", "by", "at", "is", "are", "was", "were", "how",
        "that", "this", "our", "their", "its", "your", "as", "to", "be",
    ],
)

BUILTIN_THEMES: dict[str, ThemeConfig] = {
    "space": SPACE_THEME,
    "generic": GENERIC_THEME,
}

DEFAULT_THEME_NAME = "space"


def load_theme_config(path: Path) -> ThemeConfig:
    """Load a custom theme from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ThemeConfig.model_validate(data)


def resolve_theme(theme: str = DEFAULT_THEME_NAME, theme_json: Path | None = None) -> ThemeConfig:
    """Resolve the theme to use for a pipeline run.

    A JSON file wins over the builtin name when both are given. The builtin
    "space" theme preserves the original pipeline behavior exactly.
    """
    if theme_json is not None:
        custom = load_theme_config(theme_json)
        if theme != DEFAULT_THEME_NAME and custom.name != theme:
            raise ValueError(
                f"theme JSON name {custom.name!r} does not match --theme {theme!r}"
            )
        return custom
    try:
        return BUILTIN_THEMES[theme]
    except KeyError:
        known = ", ".join(sorted(BUILTIN_THEMES))
        raise ValueError(f"unknown theme {theme!r}; expected one of: {known}") from None
