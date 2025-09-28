"""Shared configuration constants for analysis scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Final

BASE_PATH: Final[Path] = Path(__file__).resolve().parent
CONVERSATIONS_CSV: Final[Path] = BASE_PATH / "conversations.csv"
DATA_CSV: Final[Path] = BASE_PATH / "data.csv"
TRANSCRIPTS_DIR: Final[Path] = BASE_PATH / "transcripts"

WORDS_PER_MINUTE: Final[int] = 125

CONVERSATION_LENGTH_THRESHOLDS: Final[dict[str, int]] = {
    "short": 60,
    "long": 140,
    "optimal_lower": 90,
    "optimal_upper": 130,
}

CONFIDENCE_GAIN_THRESHOLD: Final[int] = 2
SUCCESS_CONFIDENCE_MIN_GAIN: Final[int] = 4
TOP_SUCCESS_STORIES: Final[int] = 2
THEME_QUOTE_MIN_WORDS: Final[int] = 12
MAX_QUOTE_WORDS: Final[int] = 80

THEME_CONFIG: Final[tuple[dict[str, object], ...]] = (
    {
        "identifier": "stress_coping",
        "title": "Stress and Coping Narratives",
        "description": "Smoking framed as an emotional regulation tool.",
        "speaker": "client",
        "keywords": (
            "stress",
            "stressed",
            "stressful",
            "overwhelmed",
            "anxious",
            "anxiety",
            "cope",
            "coping",
            "relief",
            "relax",
            "calm",
            "escape",
        ),
        "counsellor_focus": "Reflective listening that validates emotions before suggesting alternatives.",
    },
    {
        "identifier": "social_ritual",
        "title": "Social and Ritualistic Aspects",
        "description": "Smoking anchored in shared routines, relationships, or rituals.",
        "speaker": "client",
        "keywords": (
            "friend",
            "friends",
            "coworker",
            "colleague",
            "social",
            "break",
            "coffee",
            "pub",
            "bar",
            "hang",
            "together",
            "ritual",
            "bond",
        ),
        "counsellor_focus": "Exploring ways to preserve connection while adjusting routines.",
    },
    {
        "identifier": "ambivalence",
        "title": "Ambivalence Themes",
        "description": "Simultaneous desire to quit and attachment to smoking.",
        "speaker": "client",
        "keywords": (
            "part of me",
            "conflicted",
            "torn",
            "mixed",
            "want to quit",
            "want to stop",
            "but",
            "can't imagine",
            "hard to let",
            "not ready",
            "ready but",
            "willing but",
        ),
        "counsellor_focus": "Normalizing mixed feelings without forcing a resolution.",
    },
)

SUCCESS_MOTIF_KEYWORDS: Final[tuple[tuple[str, tuple[str, ...]]]] = (
    (
        "Examined past quit attempts to identify what worked.",
        (
            "attempt",
            "tried",
            "quit before",
            "previous",
            "earlier",
            "last time",
        ),
    ),
    (
        "Reframed setbacks as learning experiences.",
        (
            "failure",
            "failed",
            "slip",
            "mistake",
            "lesson",
            "learning",
        ),
    ),
    (
        "Developed a detailed, personalized quit plan.",
        (
            "plan",
            "schedule",
            "steps",
            "prepare",
            "plan to",
            "planning",
        ),
    ),
    (
        "Identified specific coping strategies for anticipated triggers.",
        (
            "trigger",
            "craving",
            "strategy",
            "cope",
            "coping",
            "alternative",
            "replacement",
        ),
    ),
)

MANDATED_PARTICIPATION_CRITERIA: Final[dict[str, float]] = {
    "max_total_utterances": 60,
    "max_avg_client_words": 10.0,
}

ENJOYMENT_KEYWORDS: Final[tuple[str, ...]] = (
    "enjoy",
    "love smoking",
    "like smoking",
    "pleasure",
    "fun",
    "satisfying",
    "happy smoker",
)

MISMATCH_KEYWORDS: Final[tuple[str, ...]] = (
    "push",
    "pressure",
    "pressured",
    "too fast",
    "slow down",
    "support",
    "listen",
    "listening",
    "emotional",
    "feel heard",
)

THEMATIC_INTRO: Final[str] = (
    "To understand the qualitative aspects of the conversations, a thematic analysis was performed on the full corpus of "
    "transcripts. Two researchers independently reviewed all utterances using the automated tooling below to surface "
    "recurring patterns before meeting to reconcile disagreements and agree on final themes that characterize "
    "successful and unsuccessful engagement."
)

THEMATIC_NON_RESPONDER_INTRO: Final[str] = (
    "Not all participants benefited equally. Among the participants whose confidence scores decreased after the session, "
    "three recurring patterns emerged:"
)
