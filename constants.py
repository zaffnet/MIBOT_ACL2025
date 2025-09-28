"""Centralized constants for the MIBot analysis utilities."""

from pathlib import Path

TRANSCRIPTS_DIRECTORY = Path("transcripts")
OPENAI_MODEL_GPT4O = "gpt-4o"
OPENAI_TEMPERATURE_DEFAULT = 0.2
TRANSCRIPT_SUMMARY_MAX_OUTPUT_TOKENS = 1200
THEME_SYNTHESIS_MAX_OUTPUT_TOKENS = 1800
TRANSCRIPT_SUMMARY_UTTERANCE_LIMIT = 120
DIALOGUE_JOINER = "\n"
JSON_INDENT = 2
TEXT_ENCODING = "utf-8"
PARTICIPANT_ID_PREFIX = "Participant ID: "
CONVERSATION_SECTION_HEADER = "Conversation:\n"
PIPELINE_DESCRIPTION = "Run thematic analysis over MIBot transcripts using GPT-4o."
TRANSCRIPTS_DIR_HELP = "Directory containing transcript JSON files."
OUTPUT_PATH_HELP = "Where to save the thematic analysis JSON report."
MODEL_HELP = "OpenAI model name to use for analysis."
TEMPERATURE_HELP = "Sampling temperature for the model."
MAX_TRANSCRIPTS_HELP = "Optional limit on the number of transcripts to analyze."
JSON_PARSE_ERROR_TEMPLATE = "Model returned invalid JSON for {context}: {content}"
SUMMARY_CONTEXT_LABEL = "transcript summary"
THEME_CONTEXT_LABEL = "theme synthesis"
TRANSCRIPT_SUMMARY_PROMPT = (
    "You are a qualitative research assistant helping motivational interviewing "
    "researchers analyze smoking cessation conversations. Review the provided "
    "dialogue carefully and return a JSON object with the following keys: "
    "`conversation_overview` (2-3 sentence summary), `client_state` (insights "
    "about motivations, emotions, and change talk), `therapeutic_moves` "
    "(effective counselling strategies that appeared), `notable_quotes` (up to "
    "two short client quotes), and `preliminary_themes` (list of 1-3 short "
    "labels capturing the main ideas in the conversation)."
)
TRANSCRIPT_SUMMARY_USER_PROMPT_PREFIX = (
    "Summarize the following motivational interviewing conversation and respond with valid JSON."
)
THEME_SYNTHESIS_PROMPT = (
    "You are a senior qualitative researcher finalizing a thematic analysis of "
    "motivational interviewing transcripts. You will receive structured "
    "conversation summaries. Synthesize them into a JSON document with the "
    "following keys: `themes` (list of objects with `name`, `prevalence`, "
    "`description`, and `representative_quotes`), `success_cases` (list of "
    "objects each with `participant_id`, `summary`, and `key_actions`), and "
    "`negative_cases` (list of objects each with `participant_id`, `summary`, and "
    "`lessons`). Prevalence should be a percentage estimate derived from the "
    "proportion of summaries mentioning a theme. Quote selections should come "
    "from the supplied conversation summaries."
)
THEME_SYNTHESIS_USER_PROMPT_PREFIX = (
    "Use the following transcript summaries to produce the requested thematic analysis."
)
DEFAULT_OUTPUT_PATH = Path("analysis/thematic_analysis_report.json")
