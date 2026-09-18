"""YouTube format presets: the winning 20-40 minute long-form formats.

Derived from the 2026-09-18 format research (7 format blueprints, title
formulas, hook structures, chapter cadence, pacing cheat-sheet). Each preset
drives script generation (beats, hook, narration register), packaging
(titles, thumbnails), and visual style.

Ranked by fit for a fully-AI pipeline (no filming, TTS voice, stock/AI
visuals):
  TIER 1: business_autopsy, systems_explainer
  TIER 2: mystery_deep_dive, history_reconstruction
  TIER 3: armchair_true_crime (policy gates), horror_anthology
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HookSpec:
    """The universal 3-beat 60-second hook, per-format flavored."""

    beat1_0_10: str  # 0:00-0:10 pattern interrupt
    beat2_10_30: str  # 0:10-0:30 promise + credibility
    beat3_30_60: str  # 0:30-1:00 the map (retention contract)


@dataclass(frozen=True)
class FormatPreset:
    """Everything script generation needs to produce one winning format."""

    key: str
    label: str
    tier: int
    target_minutes: tuple[float, float]
    description: str
    title_formulas: tuple[str, ...]
    hook: HookSpec
    chapter_template: tuple[str, ...]
    narration_person: str
    narration_tense: str
    narration_tone: str
    wpm: int
    sentence_guidance: str
    visual_style: str
    beats: tuple[dict, ...]
    needs_human_review: bool = False


PRESETS: dict[str, FormatPreset] = {}


def _register(preset: FormatPreset) -> None:
    PRESETS[preset.key] = preset


_register(
    FormatPreset(
        key="business_autopsy",
        label="Business Autopsy / Rise-and-Fall",
        tier=1,
        target_minutes=(25.0, 40.0),
        description=(
            "The peak-to-collapse story of a company or empire. Cold open "
            "with peak numbers, fall numbers, and irony; 8-12 chapters "
            "around 2 minutes each; animated charts and AI recreations; "
            "a direct-address lesson ending that drives comments."
        ),
        title_formulas=(
            "The Rise and Fall of {X}",
            "The Rise and Fall of {X} - What Really Went Wrong",
            "How a {N} {X} Empire Died in {T}",
            "{X} Owned the World - Then It Laughed at {Rival}",
            "Everyone Knew. Nobody Stopped Him. - The {N} {X} Fraud",
            "The Collapse of {X}: Insiders Tell All",
        ),
        hook=HookSpec(
            beat1_0_10="The peak stated with numbers: at its height {X} was worth {N}.",
            beat2_10_30="The fall: {T} later it was worth nothing. Credibility: the complete timeline.",
            beat3_30_60="The irony hook: the one detail that makes THIS telling worth watching.",
        ),
        chapter_template=(
            "Cold open (peak + fall in 60s)",
            "Part 1 - The Rise (founding story, humanize founders)",
            "Part 2 - The Throne (dominance, numbers, cultural moment)",
            "The Turn (the disruption arrives; the fateful decision)",
            "Part 3 - The Fall (cascade of mistakes, each a mini-chapter)",
            "The Verdict (final numbers)",
            "The Lesson (direct address, comment bait)",
            "Epilogue + next-video tease",
        ),
        narration_person="third person, second person in the lesson section",
        narration_tense="past (present in lesson)",
        narration_tone="ironic, judgmental, numbers-forward",
        wpm=155,
        sentence_guidance="Staccato for numbers and shock beats, flowing for narrative.",
        visual_style=(
            "Archival photos, product shots, animated collapse charts (the "
            "money visual), AI cinematic recreations. On-screen text is "
            "always the numbers."
        ),
        beats=(
            {"key": "cold_open", "fraction": 0.04,
             "purpose": "Peak numbers, fall numbers, irony hook. State the "
                        "highest value {X} ever reached and the value at the end."},
            {"key": "rise", "fraction": 0.16,
             "purpose": "The founding story. Humanize the founders. What made {X} "
                        "special and who believed first."},
            {"key": "throne", "fraction": 0.16,
             "purpose": "Dominance: market share, revenue, cultural moment. "
                        "The empire at its most invincible."},
            {"key": "turn", "fraction": 0.12,
             "purpose": "The disruption arrives. The fateful decision, stated "
                        "so hindsight makes it sting."},
            {"key": "fall", "fraction": 0.24,
             "purpose": "The cascade of mistakes, each as its own mini-chapter "
                        "with numbers. Schadenfreude with receipts."},
            {"key": "verdict", "fraction": 0.10,
             "purpose": "Final accounting: market cap, market share, jobs, "
                        "the bottom line."},
            {"key": "lesson", "fraction": 0.10,
             "purpose": "The one lesson that applies to the viewer's life right "
                        "now. Direct address. End with a comment-bait question."},
            {"key": "outro", "fraction": 0.08,
             "purpose": "Epilogue and next-video tease."},
        ),
    )
)

_register(
    FormatPreset(
        key="systems_explainer",
        label="Systems Explainer",
        tier=1,
        target_minutes=(15.0, 25.0),
        description=(
            "A hidden system behind an everyday puzzle, explained layer by "
            "layer. 100% motion graphics and maps. Calm present-tense "
            "narration. The thesis is the closing line."
        ),
        title_formulas=(
            "Why {EverydayPuzzle}",
            "How {X} {Verb}ed {Y}",
            "The Economics of {X}",
        ),
        hook=HookSpec(
            beat1_0_10="The everyday observation the viewer has had.",
            beat2_10_30="Promise: there is a hidden system behind it, named in three nouns.",
            beat3_30_60="The first surprising fact proving the rabbit hole is deep.",
        ),
        chapter_template=(
            "The everyday puzzle",
            "The history of the system",
            "The economics (incentives of each player)",
            "The geography/physics constraint",
            "The edge cases and exceptions",
            "The future (what changes next)",
            "The thesis restated as the closing line",
        ),
        narration_person="third person",
        narration_tense="present ('here is how')",
        narration_tone="calm, even, slightly dry; 'I did the reading so you do not have to'",
        wpm=150,
        sentence_guidance="Medium-length, precise vocabulary, steady cadence.",
        visual_style=(
            "100% motion graphics: animated maps, flow diagrams, data charts. "
            "On-screen labels, figures, country names. No host, no camera footage."
        ),
        beats=(
            {"key": "cold_open", "fraction": 0.05,
             "purpose": "The everyday observation, the promise of the hidden "
                        "system, the first surprising fact."},
            {"key": "history", "fraction": 0.20,
             "purpose": "How this system came to be. The origins that explain "
                        "today's shape."},
            {"key": "economics", "fraction": 0.22,
             "purpose": "The incentives of each player. Who gets paid, who pays, "
                        "and why it stays that way."},
            {"key": "constraint", "fraction": 0.20,
             "purpose": "The geography, physics, or law that constrains the "
                        "system. The variable nobody can cheat."},
            {"key": "edge_cases", "fraction": 0.12,
             "purpose": "The exceptions that prove the rule. What happens at "
                        "the margins."},
            {"key": "future", "fraction": 0.12,
             "purpose": "What changes next. The trend lines and what breaks them."},
            {"key": "thesis", "fraction": 0.09,
             "purpose": "Restate the thesis as the closing line. Quotable, "
                        "shareable, one sentence."},
        ),
    )
)

_register(
    FormatPreset(
        key="history_reconstruction",
        label="AI History Reconstruction",
        tier=2,
        target_minutes=(15.0, 25.0),
        description=(
            "'24 Hours in...' and 'What Did X Look Like?' series formats. "
            "AI reconstruction imagery is the native visual identity."
        ),
        title_formulas=(
            "24 Hours in {Place}, {Year}",
            "What Did {X} Really Look Like?",
            "A Day in the Life of {Person}, {Year}",
        ),
        hook=HookSpec(
            beat1_0_10="The single most alien detail of this time and place, stated cold.",
            beat2_10_30="Promise: the full day reconstructed, hour by hour, from the sources.",
            beat3_30_60="The map: dawn, midday, dusk, night, and what each one reveals.",
        ),
        chapter_template=(
            "Cold open (the alien detail)",
            "Dawn: waking up in this world",
            "Midday: the work and the streets",
            "Dusk: the meal, the markets, the danger",
            "Night: what the dark was like",
            "The meaning (what this day explains about us)",
        ),
        narration_person="second/third person immersive",
        narration_tense="present immersive",
        narration_tone="cinematic, sensory, grounded in cited detail",
        wpm=145,
        sentence_guidance="Sensory detail, short beats mixed with flowing description.",
        visual_style=(
            "AI-generated period reconstructions in a consistent art style "
            "across the series. Maps and timeline inserts for orientation."
        ),
        beats=(
            {"key": "cold_open", "fraction": 0.05,
             "purpose": "The most alien detail of this time and place. Immerse "
                        "instantly."},
            {"key": "dawn", "fraction": 0.18,
             "purpose": "Waking up: homes, rituals, the first hours."},
            {"key": "midday", "fraction": 0.22,
             "purpose": "Work, streets, markets, power. The world in motion."},
            {"key": "dusk", "fraction": 0.22,
             "purpose": "Food, danger, religion, entertainment. The texture of life."},
            {"key": "night", "fraction": 0.18,
             "purpose": "Darkness before electric light. What people feared and "
                        "believed."},
            {"key": "meaning", "fraction": 0.15,
             "purpose": "What this day explains about us. The thesis closing."},
        ),
    )
)

_register(
    FormatPreset(
        key="mystery_deep_dive",
        label="Mystery Deep Dive",
        tier=2,
        target_minutes=(20.0, 30.0),
        description=(
            "Restrained, clinical narration over custom maps, timelines, and "
            "documents. Long chapters (5-6 min); pacing from within-chapter "
            "reveals. Honest unresolved endings."
        ),
        title_formulas=(
            "{Topic}: An {Adjective} Mystery",
            "The {Verb}ing of {X}",
            "The Enduring Mystery of {X}",
        ),
        hook=HookSpec(
            beat1_0_10="Cold open ON the mystery's most cinematic moment, narrated flatly.",
            beat2_10_30="One-sentence premise.",
            beat3_30_60="The first concrete detail proving this is real: a date, a coordinate.",
        ),
        chapter_template=(
            "The Inciting Artifact",
            "The Investigation (following the trail)",
            "The Complication (dead ends, what investigators got wrong)",
            "The Deepest Layer (the unsolved core)",
            "The Unresolved End (no fake resolution)",
        ),
        narration_person="third person",
        narration_tense="past, flat and calm",
        narration_tone="clinical, detached, understated; tension through understatement",
        wpm=140,
        sentence_guidance="Long complex sentences mixed with short fragments.",
        visual_style=(
            "Custom maps, timelines, document recreations, terminal-style text "
            "animations. No stock, no host. Dates, coordinates, document quotes "
            "as on-screen text."
        ),
        beats=(
            {"key": "cold_open", "fraction": 0.04,
             "purpose": "The inciting artifact: the image, the disappearance, "
                        "stated flatly."},
            {"key": "investigation", "fraction": 0.25,
             "purpose": "Follow the trail. What is known, in order, with "
                        "primary-source detail."},
            {"key": "complication", "fraction": 0.25,
             "purpose": "The dead ends. What investigators got wrong and why "
                        "it matters."},
            {"key": "deepest_layer", "fraction": 0.28,
             "purpose": "The unsolved core. Competing theories weighed "
                        "honestly, no favorite picked without evidence."},
            {"key": "unresolved_end", "fraction": 0.18,
             "purpose": "The honest admission of what is unknowable. Trust "
                        "over closure."},
        ),
    )
)

_register(
    FormatPreset(
        key="armchair_true_crime",
        label="Armchair True Crime",
        tier=3,
        target_minutes=(20.0, 35.0),
        description=(
            "Narrator-led crime documentary. Archival photos, news clippings, "
            "animated maps. Strict human review gates: no invented cases, "
            "no graphic detail, no defamation risk."
        ),
        title_formulas=(
            "The Hunt for {Superlative} {Archetype}",
            "The {Adjective} {Case}",
            "{Name}: {IronicSubtitle}",
        ),
        hook=HookSpec(
            beat1_0_10="The crime's most shocking beat as cold fact.",
            beat2_10_30="The scope: why this case matters.",
            beat3_30_60="The angle: the part of the story most coverage leaves out.",
        ),
        chapter_template=(
            "Cold open (the crime)",
            "The victims (human stakes)",
            "The investigation begins",
            "The suspect emerges (first open loop)",
            "The evidence (twist/complication)",
            "The mistake / the near-miss",
            "The aftermath and the unanswered question",
            "Epilogue",
        ),
        narration_person="third person omniscient",
        narration_tense="past",
        narration_tone="journalistic, serious, no jokes; respectful of victims",
        wpm=150,
        sentence_guidance="Short punchy sentences for shock beats, longer for context.",
        visual_style=(
            "Archival photos, news clippings, maps with animated routes, "
            "AI-generated atmospheric recreations. Names, dates, locations, "
            "primary-source quotes as on-screen text. Nothing graphic."
        ),
        beats=(
            {"key": "cold_open", "fraction": 0.05,
             "purpose": "The crime as cold fact. The scope. The angle."},
            {"key": "victims", "fraction": 0.14,
             "purpose": "Human stakes. Who was affected, told with dignity."},
            {"key": "investigation", "fraction": 0.18,
             "purpose": "The investigation begins. Evidence in order."},
            {"key": "suspect", "fraction": 0.16,
             "purpose": "The suspect emerges. First open loop."},
            {"key": "twist", "fraction": 0.16,
             "purpose": "The evidence twist or complication."},
            {"key": "near_miss", "fraction": 0.13,
             "purpose": "The mistake or the near-miss."},
            {"key": "aftermath", "fraction": 0.12,
             "purpose": "Resolution or lack of it. The unanswered question."},
            {"key": "epilogue", "fraction": 0.06,
             "purpose": "What remains. Respectful close."},
        ),
        needs_human_review=True,
    )
)

_register(
    FormatPreset(
        key="horror_anthology",
        label="AI Horror / Story Anthology",
        tier=3,
        target_minutes=(20.0, 60.0),
        description=(
            "Escalating fictional stories with a payoff, twist, or reveal. "
            "Clearly labeled fiction. Escalation ladder: each story stranger "
            "than the last."
        ),
        title_formulas=(
            "{N} {Adjective} Stories That {Verb} {X}",
            "Why You Should NEVER {Verb}...",
            "This {Noun} Is {Adjective}",
        ),
        hook=HookSpec(
            beat1_0_10="The count and the promise: N stories, each stranger than the last.",
            beat2_10_30="One-sentence teaser of each story.",
            beat3_30_60="The escalation ladder stated: the last one is the reason to stay.",
        ),
        chapter_template=(
            "Intro + all teasers",
            "Story 1 (the weakest, still strong)",
            "Story 2 (escalation)",
            "Story 3 (the best)",
            "Outro",
        ),
        narration_person="first person storyteller",
        narration_tense="present ('he walks into the tunnel...')",
        narration_tone="conversational, campfire rhythm, deliberate pauses",
        wpm=140,
        sentence_guidance="Short sentences. Pauses before the twist. Occasional asides.",
        visual_style=(
            "Dark cinematic AI imagery, slow zooms. Location and date stamps. "
            "Clearly marked FICTION."
        ),
        beats=(
            {"key": "intro", "fraction": 0.05,
             "purpose": "N teasers and the escalation ladder."},
            {"key": "story1", "fraction": 0.25,
             "purpose": "Story one: strong but the weakest of the three. "
                        "Payoff and twist."},
            {"key": "story2", "fraction": 0.30,
             "purpose": "Story two: escalation. Stranger, darker, better twist."},
            {"key": "story3", "fraction": 0.33,
             "purpose": "Story three: the best. The full payoff the hook promised."},
            {"key": "outro", "fraction": 0.07,
             "purpose": "Close. FICTION label reinforced."},
        ),
    )
)

_register(
    FormatPreset(
        key="science_doc",
        label="Cinematic Science Documentary (upgraded lane)",
        tier=2,
        target_minutes=(20.0, 30.0),
        description=(
            "The existing space/science lane, upgraded with hook engineering, "
            "question-debt chaptering, pacing discipline, and better assets."
        ),
        title_formulas=(
            "The {Superlative} {X} We Have Ever {Verb}",
            "Why {X} {CounterintuitiveVerb}",
            "{N} {X} That {Verb} {Y}",
        ),
        hook=HookSpec(
            beat1_0_10="The single most mind-bending fact about this topic, stated cold.",
            beat2_10_30="The promise: the complete story, from the evidence to the implication.",
            beat3_30_60="The map: the 3-4 revelations the video will deliver.",
        ),
        chapter_template=(
            "Cold open (the mind-bending fact)",
            "The discovery",
            "How it works (the mechanism)",
            "The evidence",
            "What it means for us",
            "The open question",
        ),
        narration_person="third person",
        narration_tense="past for history, present for mechanism",
        narration_tone="awed, precise, cinematic",
        wpm=150,
        sentence_guidance="Varied: staccato for scale, flowing for wonder.",
        visual_style=(
            "Cinematic B-roll, NASA/public-domain imagery, AI reconstructions "
            "for what no camera can show. Motion graphics for scale and mechanism."
        ),
        beats=(
            {"key": "cold_open", "fraction": 0.04,
             "purpose": "The mind-bending fact. The promise. The map."},
            {"key": "discovery", "fraction": 0.18,
             "purpose": "How we found this out. The human story of the discovery."},
            {"key": "mechanism", "fraction": 0.24,
             "purpose": "How it works. The mechanism, built layer by layer."},
            {"key": "evidence", "fraction": 0.20,
             "purpose": "The evidence. What convinced the skeptics."},
            {"key": "meaning", "fraction": 0.20,
             "purpose": "What it means for us. The implication, stated plainly."},
            {"key": "open_question", "fraction": 0.14,
             "purpose": "The open question. End on wonder, not summary."},
        ),
    )
)


def get_preset(key: str) -> FormatPreset:
    """Return the preset for key, defaulting to business_autopsy."""
    return PRESETS.get(key, PRESETS["business_autopsy"])


def list_presets() -> list[FormatPreset]:
    """All presets, Tier 1 first."""
    return sorted(PRESETS.values(), key=lambda p: (p.tier, p.key))


def format_names() -> list[str]:
    """CLI-friendly format keys."""
    return [p.key for p in list_presets()]
