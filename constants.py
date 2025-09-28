"""Global constants for the MIBot analysis scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Final

BASE_PATH: Final[Path] = Path(__file__).resolve().parent
TRANSCRIPTS_DIR: Final[Path] = BASE_PATH / "transcripts"
CONVERSATIONS_PATH: Final[Path] = BASE_PATH / "conversations.csv"
SURVEY_PATH: Final[Path] = BASE_PATH / "data.csv"
ANALYSIS_DIR: Final[Path] = BASE_PATH / "analysis"
THEMATIC_ANALYSIS_OUTPUT_PATH: Final[Path] = ANALYSIS_DIR / "thematic_analysis.md"
THEMATIC_ANALYSIS_LOG_PATH: Final[Path] = ANALYSIS_DIR / "thematic_analysis_run.md"

DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o"
OPENAI_TEMPERATURE: Final[float] = 0.2
OPENAI_MAX_OUTPUT_TOKENS: Final[int] = 800

MAX_UTTERANCES_PER_TRANSCRIPT: Final[int] = 400
QUOTE_WORD_LIMIT: Final[int] = 60
THEME_QUOTE_LIMIT: Final[int] = 3

THEME_TITLES: Final[dict[str, str]] = {
    "stress_and_coping": "Stress and Coping Narratives",
    "social_ritual": "Social and Ritualistic Aspects",
    "ambivalence": "Ambivalence Themes",
}

THEME_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "stress_and_coping": (
        "stress",
        "stressed",
        "anxious",
        "cope",
        "coping",
        "overwhelmed",
        "relax",
        "calm",
        "anxiety",
        "tension",
        "pressure",
        "break",
        "stressful",
    ),
    "social_ritual": (
        "friends",
        "coworker",
        "coworkers",
        "social",
        "party",
        "break",
        "ritual",
        "together",
        "group",
        "hang",
        "pub",
        "bar",
        "coffee",
        "family",
    ),
    "ambivalence": (
        "part of me",
        "want to quit",
        "should quit",
        "need to quit",
        "don't want",
        "do not want",
        "can't imagine",
        "mixed",
        "conflicted",
        "torn",
        "ambivalent",
    ),
}

THEME_BASE_DESCRIPTIONS: Final[dict[str, str]] = {
    "stress_and_coping": (
        "Participants described cigarettes as tools for emotion regulation and short-term relief. "
        "Discussions often focused on stress at work, family pressures, or using smoke breaks as a momentary escape."
    ),
    "social_ritual": (
        "Smoking was entwined with social identity and daily rituals. Clients highlighted smoke breaks with colleagues, "
        "shared routines with partners, and the fear of losing social contact if they quit."
    ),
    "ambivalence": (
        "Clients expressed motivational ambivalence, simultaneously acknowledging reasons to quit while defending the role of smoking "
        "in their lives. Conversations frequently normalized these mixed feelings before exploring change talk."
    ),
}

NEGATIVE_CASE_SUBCATEGORIES: Final[tuple[str, ...]] = (
    "Mandated Participation",
    "Enjoyment-Focused Smokers",
    "Technical Therapeutic Mismatches",
)

SHORT_CONVERSATION_THRESHOLD: Final[int] = 50
MINIMAL_RESPONSE_WORD_THRESHOLD: Final[int] = 12
ENJOYMENT_KEYWORDS: Final[tuple[str, ...]] = (
    "enjoy",
    "love",
    "like smoking",
    "happy smoker",
    "fun",
)
MISMATCH_KEYWORDS: Final[tuple[str, ...]] = (
    "push",
    "pressure",
    "pushed",
    "rushed",
    "off",
    "misread",
    "misunderstand",
)

SIGNIFICANT_CONFIDENCE_GAIN: Final[int] = 2

SUCCESS_KEYWORD_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "Reflecting on past attempts": ("attempt", "tried", "past", "before"),
    "Reframing setbacks": ("failure", "learn", "lesson", "setback"),
    "Planning next steps": ("plan", "strategy", "prepare", "ready", "goal"),
    "Coping strategies": ("trigger", "craving", "coping", "replace", "support"),
}

OPENAI_SYSTEM_PROMPT: Final[str] = (
    "You are a qualitative research assistant who specialises in motivational interviewing (MI) for smoking cessation. "
    "Use the quantitative summary, keyword frequencies, and representative quotes to craft rich, human-readable summaries "
    "of each theme. Cite numerical values directly and maintain the narrative voice of an academic report."
)

OPENAI_THEME_USER_TEMPLATE: Final[str] = (
    "Dataset summary:\n"
    "Total transcripts: {total_transcripts}\n\n"
    "Theme data:\n{theme_rows}\n\n"
    "Quotes by theme:\n{quote_rows}\n\n"
    "Write a markdown section that mirrors the structure of the report excerpt provided in the repository's README."
)
