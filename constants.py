"""Global constants for thematic analysis scripts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = REPO_ROOT
CONVERSATIONS_FILE_NAME = "conversations.csv"
SURVEY_FILE_NAME = "data.csv"
PARTICIPANT_ID_COLUMN = "ParticipantID"
SPEAKER_COLUMN = "Speaker"
UTTERANCE_COLUMN = "Utterance"
UTTERANCE_NUMBER_COLUMN = "Utterance#"
COUNSELLOR_LABEL = "counsellor"
CLIENT_LABEL = "client"
DEFAULT_OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.2
TRANSCRIPT_CHUNK_CHARACTER_LIMIT = 6000
THEME_KEYS = (
    "stress_and_coping",
    "social_and_ritual",
    "ambivalence",
    "success_profile",
    "non_responder_profile",
)
THEME_DEFINITIONS = {
    "stress_and_coping": "Smoking discussed as a coping mechanism for stress or emotional regulation.",
    "social_and_ritual": "Smoking framed as part of social rituals, routines, or relationship maintenance.",
    "ambivalence": "Statements revealing conflicting motivations about quitting versus continuing to smoke.",
    "success_profile": "Evidence of strong engagement, confidence gains, or detailed quit planning aligned with success stories.",
    "non_responder_profile": "Indicators of disengagement, mandated participation, enjoyment-focused smokers, or therapeutic mismatches.",
}
SURVEY_RULER_COLUMNS = {
    "pre_importance": "PreRulerImportance",
    "pre_confidence": "PreRulerConfidence",
    "pre_readiness": "PreRulerReadiness",
    "post_importance": "PostRulerImportance",
    "post_confidence": "PostRulerConfidence",
    "post_readiness": "PostRulerReadiness",
    "week_importance": "WeekLaterRulerImportance",
    "week_confidence": "WeekLaterRulerConfidence",
    "week_readiness": "WeekLaterRulerReadiness",
}
THEME_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert qualitative researcher specializing in motivational interviewing for smoking cessation. "
    "Analyse the provided transcript for the following candidate themes and return a compact JSON object. "
    "Themes to consider: {theme_definitions}. For each theme, return an object with keys 'present' (boolean), "
    "'evidence' (one-sentence rationale), and 'quotes' (list of up to two short verbatim quotes). Also include 'notable_outcomes' "
    "containing keys 'confidence_shift', 'importance_shift', and 'readiness_shift' summarizing any meaningful change described "
    "in the metadata. Focus on fidelity to the transcript and avoid speculation."
)
THEME_EXTRACTION_USER_TEMPLATE = (
    "Conversation metadata:\n"
    "Participant ID: {participant_id}\n"
    "Pre-conversation rulers: importance={pre_importance}, confidence={pre_confidence}, readiness={pre_readiness}\n"
    "Post-conversation rulers: importance={post_importance}, confidence={post_confidence}, readiness={post_readiness}\n"
    "Week-later rulers: importance={week_importance}, confidence={week_confidence}, readiness={week_readiness}\n\n"
    "Transcript:\n{transcript}\n"
    "Respond with JSON only."
)
CORPUS_SYNTHESIS_SYSTEM_PROMPT = (
    "You are writing the thematic analysis section of a qualitative results report. "
    "Use the aggregated theme frequencies, exemplar quotes, and outcome notes to craft a narrative similar in tone and structure "
    "to peer-reviewed reporting on motivational interviewing studies. Highlight prevalence percentages, describe patterns, and "
    "reference supporting quotes. Conclude with observations about notable successes and non-responder patterns."
)
CORPUS_SYNTHESIS_USER_TEMPLATE = (
    "Theme summary statistics:\n{theme_table}\n\n"
    "Representative evidence:\n{evidence_block}\n\n"
    "Outcome highlights:\n{outcome_block}\n\n"
    "Draft a thematic analysis narrative."
)
REPORT_HEADER = "# Thematic Analysis of MIBot Conversations"
MAX_QUOTES_PER_THEME = 2
TOP_CONVERSATION_COUNT = 3
SCRIPT_DESCRIPTION = "Perform GPT-assisted thematic analysis on MIBot transcripts."
DEFAULT_OUTPUT_FILENAME = "thematic_analysis.md"
TRANSCRIPT_LINE_TEMPLATE = "{speaker}: {utterance}"
MISSING_VALUE_PLACEHOLDER = "N/A"
TRANSCRIPT_SEGMENT_TEMPLATE = "[Segment {index}]"
BASE_PATH_HELP = "Directory containing dataset CSV files."
MODEL_HELP = "OpenAI model name for analysis."
OUTPUT_HELP = "Relative path for saving the markdown report (use '-' for stdout)."
