"""Global constants for the MIBot analysis toolkit.

This module centralizes configuration values so that other modules do not
have to rely on hard-coded literals. Update the values here to customize
paths, model selections, or prompt templates.
"""

from pathlib import Path

# Dataset paths
CONVERSATIONS_CSV_PATH: Path = Path("conversations.csv")
TRANSCRIPTS_DIRECTORY: Path = Path("transcripts")

# Column names used throughout the corpus
PARTICIPANT_ID_COLUMN: str = "ParticipantID"
SPEAKER_COLUMN: str = "Speaker"
UTTERANCE_COLUMN: str = "Utterance"
VOLLEY_COLUMN: str = "Volley#"
UTTERANCE_INDEX_COLUMN: str = "Utterance#"

# Speaker labels
COUNSELLOR_LABEL: str = "counsellor"
CLIENT_LABEL: str = "client"

# OpenAI configuration
THEMATIC_ANALYSIS_MODEL: str = "gpt-4o"
THEMATIC_ANALYSIS_TEMPERATURE: float = 0.2
THEMATIC_SYNTHESIS_MODEL: str = "gpt-4o"
THEMATIC_SYNTHESIS_TEMPERATURE: float = 0.3

# Prompt templates
THEME_EXTRACTION_SYSTEM_PROMPT: str = (
    "You are a qualitative research assistant helping with motivational "
    "interviewing (MI) smoking cessation transcripts. You identify themes, "
    "patterns, and evidence grounded in the text. Always produce well-structured "
    "JSON so that analysts can audit your reasoning."
)

THEME_EXTRACTION_USER_PROMPT_TEMPLATE: str = (
    "You will receive a conversation transcript between the MI chatbot MIBot "
    "and a smoker seeking support. Perform a rigorous thematic coding of this "
    "conversation. Use motivational interviewing concepts where appropriate.\n\n"
    "Return a JSON object with the following keys:\n"
    "- participant_id: the identifier of the participant.\n"
    "- dominant_themes: list of objects with `name`, `description`, and `evidence` "
    "fields summarizing how the theme appears (cite speaker turns by number).\n"
    "- change_talk_indicators: bullet list describing any markers of behavior change.\n"
    "- sustain_talk_indicators: bullet list describing resistance or barriers.\n"
    "- notable_quotes: list of short quotes (<= 3 sentences) that exemplify the themes.\n"
    "- recommended_follow_ups: suggestions for future conversations based on MI best practices.\n\n"
    "Conversation transcript:\n{transcript}"
)

THEME_SYNTHESIS_SYSTEM_PROMPT: str = (
    "You are synthesizing qualitative findings from motivational interviewing "
    "sessions. Consolidate coded themes into a narrative mirroring academic "
    "reporting standards for thematic analysis. Reference theme prevalence and "
    "quote identifiers when available."
)

THEME_SYNTHESIS_USER_PROMPT_TEMPLATE: str = (
    "You are given a set of conversation-level analyses produced by researchers. "
    "Each entry contains dominant themes, change and sustain talk indicators, "
    "and notable quotes with speaker references. Produce a synthesized thematic "
    "analysis across the corpus.\n\n"
    "Your response must include:\n"
    "1. An overview paragraph of the analytic process.\n"
    "2. A numbered list of 3-5 major cross-cutting themes.\n"
    "3. A section highlighting success stories with participant identifiers.\n"
    "4. A section discussing non-responders or challenging cases.\n"
    "5. Methodological notes about limitations or future analytic steps.\n\n"
    "Conversation analyses:\n{analyses}"
)

# Output defaults
DEFAULT_ANALYSIS_OUTPUT_PATH: Path = Path("analysis/thematic_analysis_output.json")
DEFAULT_SYNTHESIS_OUTPUT_PATH: Path = Path("analysis/thematic_synthesis_report.md")

# Chunking configuration
MAX_TURNS_PER_CHUNK: int = 40
TURN_DELIMITER: str = "\n---\n"
