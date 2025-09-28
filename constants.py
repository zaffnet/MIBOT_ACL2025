"""Shared configuration values for analysis scripts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

CONVERSATIONS_FILENAME = "conversations.csv"
SURVEY_FILENAME = "data.csv"

THEMATIC_ANALYSIS_OUTPUT = Path("analysis/thematic_analysis.md")

THEME_CONFIG = {
    "stress_and_coping": {
        "label": "Stress and Coping Narratives",
        "keywords": [
            "stress",
            "stressed",
            "stressful",
            "anxious",
            "anxiety",
            "cope",
            "coping",
            "relax",
            "calm",
            "overwhelmed",
            "overwhelming",
            "breathe",
            "break",
            "tension",
        ],
        "phrases": [
            "safety blanket",
            "keeps me sane",
            "only thing",
            "only way",
            "get through",
        ],
        "minimum_matches": 2,
        "counsellor_note": (
            "In these exchanges counsellors most often leaned on reflective listening before offering "
            "alternative coping ideas, mirroring double-sided reflections in MI."
        ),
    },
    "social_and_ritual": {
        "label": "Social and Ritualistic Aspects",
        "keywords": [
            "friend",
            "friends",
            "coworker",
            "coworkers",
            "colleague",
            "colleagues",
            "social",
            "group",
            "breaks",
            "break",
            "coffee",
            "pub",
            "bar",
            "outside",
            "hang",
            "ritual",
            "routine",
        ],
        "phrases": [
            "smoke break",
            "go outside",
            "with coffee",
            "after dinner",
        ],
        "minimum_matches": 2,
        "counsellor_note": (
            "Counsellor turns commonly acknowledged the relational value of cigarettes before exploring "
            "ways to preserve connection without tobacco."
        ),
    },
    "ambivalence": {
        "label": "Ambivalence Themes",
        "keywords": [
            "want to quit",
            "want to stop",
            "need to quit",
            "should quit",
            "part of me",
            "can't imagine",
            "hard to quit",
            "mixed",
            "torn",
        ],
        "phrases": [
            "but I",
            "but it's",
            "but at the same time",
        ],
        "minimum_matches": 1,
        "counsellor_note": (
            "Sessions typically normalised ambivalence rather than forcing resolution, which aligns with "
            "MI best practices for strengthening change talk."
        ),
    },
}

THEME_REPRESENTATIVE_MAX_WORDS = 120

SUCCESS_CONFIDENCE_GAIN_THRESHOLD = 4
SUCCESS_LOW_CONFIDENCE_THRESHOLD = 3
SUCCESS_HIGH_IMPORTANCE_THRESHOLD = 8
SUCCESS_INDICATOR_KEYWORDS = {
    "Examined past quit attempts": ["attempt", "tried", "before", "previous"],
    "Reframed setbacks as learning": ["learn", "lesson", "progress", "fail"],
    "Built a concrete quit plan": ["plan", "planning", "schedule", "set a date", "game plan"],
    "Identified coping strategies": ["trigger", "craving", "urge", "strategy", "replace"],
}

NON_RESPONDER_MANDATED_MAX_VOLLEYS = 60
NON_RESPONDER_MAX_CLIENT_WORDS_PER_UTTERANCE = 12
NON_RESPONDER_ENJOYMENT_KEYWORDS = [
    "enjoy",
    "love",
    "like smoking",
    "pleasure",
    "savor",
    "relaxing",
]
NON_RESPONDER_CATEGORY_MIN_SAMPLE = 1

