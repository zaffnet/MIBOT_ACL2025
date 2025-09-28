"""Central configuration for analysis scripts."""

from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = BASE_PATH / "transcripts"
CONVERSATIONS_PATH = BASE_PATH / "conversations.csv"
DATA_PATH = BASE_PATH / "data.csv"
CLIENT_SPEAKER = "client"
COUNSELLOR_SPEAKER = "counsellor"
SMOKING_KEYWORD = "smok"

THEME_KEYWORDS = {
    "stress_coping": [
        "stress",
        "stressed",
        "stressful",
        "anxious",
        "anxiety",
        "cope",
        "coping",
        "relax",
        "overwhelm",
        "tension",
        "calm",
        "escape",
    ],
    "social_ritual": [
        "friend",
        "coworker",
        "colleague",
        "social",
        "hang",
        "break",
        "coffee",
        "drink",
        "bar",
        "pub",
        "ritual",
        "routine",
        "bond",
        "together",
    ],
    "enjoyment": [
        "enjoy",
        "enjoying",
        "enjoyment",
        "love",
        "pleasure",
        "like",
        "savor",
        "cherish",
        "favorite",
        "fun",
    ],
}

AMBIVALENCE_POSITIVE_KEYWORDS = [
    "quit",
    "stop",
    "give up",
    "cut back",
    "less",
    "health",
    "kids",
    "family",
    "doctor",
    "should",
    "need",
    "want",
    "plan",
]
AMBIVALENCE_DEPENDENCE_KEYWORDS = [
    "crave",
    "need",
    "can't",
    "habit",
    "help",
    "relax",
    "calm",
    "cope",
    "enjoy",
    "like",
    "addict",
    "long",
    "decades",
    "ritual",
]
AMBIVALENCE_BRIDGE_WORDS = [
    "but",
    "though",
    "however",
    "yet",
    "still",
    "even though",
]

SUCCESS_KEYWORDS = {
    "past_attempts": ["last time", "previous", "before", "attempt", "tried"],
    "learning_reframe": ["learn", "lesson", "progress", "strength", "proud", "improve"],
    "plan": ["plan", "schedule", "goal", "prepare", "strategy"],
    "coping": ["trigger", "craving", "urge", "stress", "replacement", "tool", "technique"],
}

NEGATIVE_OUTCOME_THRESHOLDS = {
    "mandated_max_utterances": 60,
    "mandated_max_avg_client_words": 9.5,
    "question_ratio_threshold": 0.55,
}

SUPPORT_KEYWORDS = [
    "support",
    "listen",
    "understand",
    "emotional",
    "feel",
    "feelings",
    "overwhelm",
    "stressed",
    "anxious",
    "lonely",
    "frustrated",
]

PERCENTAGE_PRECISION = 1
