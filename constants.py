"""Project-wide constants for transcript analysis utilities."""

from __future__ import annotations

from pathlib import Path

# Paths
DATA_DIRECTORY = Path(".")
CONVERSATIONS_FILE = DATA_DIRECTORY / "conversations.csv"
SURVEY_FILE = DATA_DIRECTORY / "data.csv"
TRANSCRIPTS_DIRECTORY = DATA_DIRECTORY / "transcripts"

# Thematic analysis configuration
THEME_DEFINITIONS = {
    "Stress and Coping Narratives": {
        "keywords": [
            "stress",
            "stressed",
            "stressful",
            "anxiety",
            "anxious",
            "overwhelm",
            "overwhelmed",
            "pressure",
            "cope",
            "coping",
            "relief",
            "release",
            "calm",
            "calming",
            "relax",
            "relaxing",
            "break",
            "breathe",
        ],
        "description": (
            "Client utterances framing smoking as a mechanism for emotion regulation, self-soothing, "
            "or momentary relief from stressors."
        ),
        "speaker": "client",
    },
    "Social and Ritualistic Aspects": {
        "keywords": [
            "friends",
            "friend",
            "coworker",
            "coworkers",
            "colleague",
            "colleagues",
            "family",
            "partner",
            "social",
            "hang",
            "hangout",
            "ritual",
            "routine",
            "break",
            "outside",
            "coffee",
            "drink",
            "bar",
            "pub",
            "together",
            "bond",
        ],
        "description": (
            "Narratives that emphasise communal routines, relationships, or shared rituals around smoking."
        ),
        "speaker": "client",
    },
    "Ambivalence Themes": {
        "keywords": [
            "part of me",
            "half of me",
            "mixed",
            "conflicted",
            "ambivalent",
            "ambivalence",
            "torn",
            "want to quit",
            "don't want to quit",
            "can't quit",
            "need to quit",
            "should quit",
            "give up",
            "quit but",
        ],
        "description": (
            "Expressions of simultaneous desire to change and attachment to smoking, signalling motivational ambivalence."
        ),
        "speaker": "client",
    },
}

SAMPLE_QUOTES_PER_THEME = 1
MIN_WORDS_PER_QUOTE = 8

# Success case configuration
SUCCESS_CONFIDENCE_DELTA_THRESHOLD = 4
SUCCESS_IMPORTANCE_MINIMUM = 7
SUCCESS_CASE_LIMIT = 2

SUCCESS_INDICATOR_KEYWORDS = {
    "Examined past quit attempts to identify what worked.": [
        "last time",
        "previous attempt",
        "tried before",
        "past quit",
        "previously quit",
    ],
    "Reframed setbacks as learning experiences.": [
        "learned",
        "lesson",
        "what it taught",
        "not a failure",
        "learn from",
    ],
    "Developed a detailed, personalised quit plan.": [
        "plan",
        "planning",
        "schedule",
        "set a date",
        "step by step",
        "outline",
    ],
    "Identified coping strategies for high-risk triggers.": [
        "strategy",
        "strategies",
        "trigger",
        "triggers",
        "urge",
        "craving",
        "replacement",
        "cope",
        "coping",
    ],
}

# Non-responder configuration
NEGATIVE_CONFIDENCE_DELTA_THRESHOLD = -1
MANDATED_MAX_AVG_CLIENT_WORDS = 10
MANDATED_MAX_TOTAL_UTTERANCES = 55
ENJOYMENT_KEYWORDS = [
    "enjoy",
    "enjoying",
    "love smoking",
    "like smoking",
    "happy smoker",
    "pleasure",
    "relaxing",
    "my thing",
]
EMOTIONAL_SUPPORT_KEYWORDS = [
    "anxious",
    "anxiety",
    "stressed",
    "stress",
    "overwhelmed",
    "sad",
    "upset",
    "emotional",
    "support",
    "listen",
    "heard",
]
ACTION_FOCUSED_KEYWORDS = [
    "goal",
    "plan",
    "planning",
    "steps",
    "action",
    "commitment",
    "quit date",
    "schedule",
    "task",
]

# General text processing
MIN_TOKEN_LENGTH = 2

