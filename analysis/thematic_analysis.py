#!/usr/bin/env python3
"""GPT-assisted thematic analysis workflow for MIBot transcripts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from openai import OpenAI

from constants import (
    BASE_PATH_HELP,
    CLIENT_LABEL,
    CONVERSATIONS_FILE_NAME,
    CORPUS_SYNTHESIS_SYSTEM_PROMPT,
    CORPUS_SYNTHESIS_USER_TEMPLATE,
    COUNSELLOR_LABEL,
    DATA_DIRECTORY,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OUTPUT_FILENAME,
    MAX_QUOTES_PER_THEME,
    MODEL_HELP,
    MISSING_VALUE_PLACEHOLDER,
    OPENAI_TEMPERATURE,
    OUTPUT_HELP,
    PARTICIPANT_ID_COLUMN,
    REPORT_HEADER,
    SCRIPT_DESCRIPTION,
    SPEAKER_COLUMN,
    SURVEY_FILE_NAME,
    SURVEY_RULER_COLUMNS,
    THEME_DEFINITIONS,
    THEME_EXTRACTION_SYSTEM_PROMPT,
    THEME_EXTRACTION_USER_TEMPLATE,
    THEME_KEYS,
    TOP_CONVERSATION_COUNT,
    TRANSCRIPT_CHUNK_CHARACTER_LIMIT,
    TRANSCRIPT_LINE_TEMPLATE,
    TRANSCRIPT_SEGMENT_TEMPLATE,
    UTTERANCE_COLUMN,
    UTTERANCE_NUMBER_COLUMN,
)


@dataclass(slots=True)
class ConversationRecord:
    """Container for a participant transcript and associated survey data."""

    participant_id: str
    transcript: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ThemeEvidence:
    """Structured representation of a single theme classification."""

    present: bool
    evidence: str
    quotes: list[str]


@dataclass(slots=True)
class ConversationThemeAnalysis:
    """LLM-derived analysis artefacts for a conversation."""

    participant_id: str
    themes: dict[str, ThemeEvidence]
    notable_outcomes: dict[str, str]


def load_datasets(base_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read conversation and survey data from disk."""

    conversations_path = base_path / CONVERSATIONS_FILE_NAME
    survey_path = base_path / SURVEY_FILE_NAME
    conversations = pd.read_csv(conversations_path)
    survey = pd.read_csv(survey_path)
    return conversations, survey


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
        if pd.isna(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _format_ruler_value(value: float | None) -> str:
    return MISSING_VALUE_PLACEHOLDER if value is None else f"{value:.1f}"


def build_transcript(conversation: pd.DataFrame) -> str:
    """Return a human-readable transcript for a participant."""

    ordered = conversation.sort_values(UTTERANCE_NUMBER_COLUMN)
    lines: list[str] = []
    for _, row in ordered.iterrows():
        speaker = str(row[SPEAKER_COLUMN]).strip()
        utterance = str(row[UTTERANCE_COLUMN]).strip()
        if not utterance:
            continue
        label = speaker.title() if speaker in {COUNSELLOR_LABEL, CLIENT_LABEL} else speaker
        lines.append(
            TRANSCRIPT_LINE_TEMPLATE.format(speaker=label, utterance=utterance)
        )
    transcript = "\n".join(lines)
    if len(transcript) <= TRANSCRIPT_CHUNK_CHARACTER_LIMIT:
        return transcript

    segments: list[str] = []
    current_segment: list[str] = []
    current_length = 0
    for line in lines:
        prospective_length = current_length + len(line) + 1
        if prospective_length > TRANSCRIPT_CHUNK_CHARACTER_LIMIT and current_segment:
            segment_index = len(segments) + 1
            segments.append(
                "\n".join(
                    [TRANSCRIPT_SEGMENT_TEMPLATE.format(index=segment_index)] + current_segment
                )
            )
            current_segment = [line]
            current_length = len(line) + 1
        else:
            current_segment.append(line)
            current_length = prospective_length
    if current_segment:
        segment_index = len(segments) + 1
        segments.append(
            "\n".join(
                [TRANSCRIPT_SEGMENT_TEMPLATE.format(index=segment_index)] + current_segment
            )
        )
    return "\n\n".join(segments)


def build_conversation_records(
    conversations: pd.DataFrame, survey: pd.DataFrame
) -> list[ConversationRecord]:
    """Construct typed conversation records with survey metadata attached."""

    survey_indexed = survey.set_index(PARTICIPANT_ID_COLUMN)
    records: list[ConversationRecord] = []

    for participant_id, participant_rows in conversations.groupby(PARTICIPANT_ID_COLUMN):
        metadata: dict[str, Any] = {}
        survey_row = survey_indexed.loc[participant_id] if participant_id in survey_indexed.index else None
        for key, column in SURVEY_RULER_COLUMNS.items():
            value = None
            if survey_row is not None and column in survey_row:
                value = _safe_float(survey_row[column])
            metadata[key] = value

        metadata["post_minus_pre_confidence"] = (
            metadata["post_confidence"] - metadata["pre_confidence"]
            if metadata["post_confidence"] is not None and metadata["pre_confidence"] is not None
            else None
        )
        metadata["week_minus_pre_confidence"] = (
            metadata["week_confidence"] - metadata["pre_confidence"]
            if metadata["week_confidence"] is not None and metadata["pre_confidence"] is not None
            else None
        )

        transcript = build_transcript(participant_rows)
        records.append(
            ConversationRecord(
                participant_id=str(participant_id),
                transcript=transcript,
                metadata=metadata,
            )
        )

    return records


def _format_theme_definitions() -> str:
    items = [f"{key}: {definition}" for key, definition in THEME_DEFINITIONS.items()]
    return "; ".join(items)


def render_theme_prompt(record: ConversationRecord) -> tuple[str, str]:
    """Return system and user prompts for the theme extraction request."""

    system_prompt = THEME_EXTRACTION_SYSTEM_PROMPT.format(
        theme_definitions=_format_theme_definitions()
    )
    metadata = record.metadata
    user_prompt = THEME_EXTRACTION_USER_TEMPLATE.format(
        participant_id=record.participant_id,
        pre_importance=_format_ruler_value(metadata.get("pre_importance")),
        pre_confidence=_format_ruler_value(metadata.get("pre_confidence")),
        pre_readiness=_format_ruler_value(metadata.get("pre_readiness")),
        post_importance=_format_ruler_value(metadata.get("post_importance")),
        post_confidence=_format_ruler_value(metadata.get("post_confidence")),
        post_readiness=_format_ruler_value(metadata.get("post_readiness")),
        week_importance=_format_ruler_value(metadata.get("week_importance")),
        week_confidence=_format_ruler_value(metadata.get("week_confidence")),
        week_readiness=_format_ruler_value(metadata.get("week_readiness")),
        transcript=record.transcript,
    )
    return system_prompt, user_prompt


def parse_theme_response(content: str, participant_id: str) -> ConversationThemeAnalysis:
    """Convert the JSON LLM response into typed structures."""

    parsed = json.loads(content)
    themes: dict[str, ThemeEvidence] = {}
    for theme_key in THEME_KEYS:
        theme_payload = parsed.get("themes", {}).get(theme_key, {})
        themes[theme_key] = ThemeEvidence(
            present=bool(theme_payload.get("present", False)),
            evidence=str(theme_payload.get("evidence", "")).strip(),
            quotes=[str(q).strip() for q in theme_payload.get("quotes", []) if str(q).strip()],
        )
    notable_outcomes = {
        key: str(value).strip()
        for key, value in parsed.get("notable_outcomes", {}).items()
        if str(value).strip()
    }
    return ConversationThemeAnalysis(
        participant_id=participant_id,
        themes=themes,
        notable_outcomes=notable_outcomes,
    )


def analyse_conversation(
    client: OpenAI, record: ConversationRecord, model: str
) -> ConversationThemeAnalysis:
    """Call the OpenAI Responses API to classify a transcript."""

    system_prompt, user_prompt = render_theme_prompt(record)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=OPENAI_TEMPERATURE,
    )
    content = response.output_text
    return parse_theme_response(content, record.participant_id)


def aggregate_theme_statistics(
    analyses: Iterable[ConversationThemeAnalysis],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Compute prevalence percentages and gather exemplar quotes per theme."""

    analyses_list = list(analyses)
    total = len(analyses_list)
    prevalence: dict[str, float] = {key: 0.0 for key in THEME_KEYS}
    quotes: dict[str, list[str]] = {key: [] for key in THEME_KEYS}
    for analysis in analyses_list:
        for theme_key, evidence in analysis.themes.items():
            if evidence.present:
                prevalence[theme_key] += 1
                available_slots = MAX_QUOTES_PER_THEME - len(quotes[theme_key])
                if available_slots > 0:
                    quotes[theme_key].extend(evidence.quotes[:available_slots])
    if total:
        for theme_key in prevalence:
            prevalence[theme_key] = (prevalence[theme_key] / total) * 100
    return prevalence, quotes


def summarise_outcomes(
    records: list[ConversationRecord], analyses: list[ConversationThemeAnalysis]
) -> dict[str, list[str]]:
    """Extract notable success and challenge narratives for synthesis."""

    record_lookup = {record.participant_id: record for record in records}
    positive_changes: list[tuple[str, float]] = []
    negative_changes: list[tuple[str, float]] = []
    for record in records:
        delta = record.metadata.get("post_minus_pre_confidence")
        if delta is None:
            continue
        if delta >= 0:
            positive_changes.append((record.participant_id, delta))
        else:
            negative_changes.append((record.participant_id, delta))
    positive_changes.sort(key=lambda item: item[1], reverse=True)
    negative_changes.sort(key=lambda item: item[1])

    def _format_change(participant_id: str, delta: float) -> str:
        metadata = record_lookup[participant_id].metadata
        pre_value = metadata.get("pre_confidence")
        post_value = metadata.get("post_confidence")
        return (
            f"Participant {participant_id}: confidence {pre_value:.1f} → {post_value:.1f}"
            if pre_value is not None and post_value is not None
            else f"Participant {participant_id}: Δconfidence={delta:+.1f}"
        )

    success_notes = [
        _format_change(participant_id, delta)
        for participant_id, delta in positive_changes[:TOP_CONVERSATION_COUNT]
    ]
    challenge_notes = [
        _format_change(participant_id, delta)
        for participant_id, delta in negative_changes[:TOP_CONVERSATION_COUNT]
    ]

    qualitative_notes: list[str] = []
    for analysis in analyses:
        if analysis.themes["success_profile"].present:
            qualitative_notes.append(
                f"Participant {analysis.participant_id}: {analysis.themes['success_profile'].evidence}"
            )
        if analysis.themes["non_responder_profile"].present:
            qualitative_notes.append(
                f"Participant {analysis.participant_id}: {analysis.themes['non_responder_profile'].evidence}"
            )

    return {
        "success_highlights": success_notes,
        "challenge_highlights": challenge_notes,
        "qualitative_notes": qualitative_notes,
    }


def format_theme_table(prevalence: dict[str, float]) -> str:
    """Render a markdown table summarising theme prevalence."""

    rows = ["| Theme | Prevalence (%) |", "|-------|----------------|"]
    for key, percentage in prevalence.items():
        label = key.replace("_", " ").title()
        rows.append(f"| {label} | {percentage:.1f} |")
    return "\n".join(rows)


def format_evidence_block(quotes: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for key, theme_quotes in quotes.items():
        if not theme_quotes:
            continue
        label = key.replace("_", " ").title()
        lines.append(f"### {label}")
        for quote in theme_quotes[:MAX_QUOTES_PER_THEME]:
            lines.append(f"> {quote}")
    return "\n".join(lines) if lines else "No exemplar quotes available."


def format_outcome_block(outcomes: dict[str, list[str]]) -> str:
    segments: list[str] = []
    if outcomes["success_highlights"]:
        segments.append("**Positive confidence shifts**")
        segments.extend(f"- {item}" for item in outcomes["success_highlights"])
    if outcomes["challenge_highlights"]:
        segments.append("**Confidence decreases**")
        segments.extend(f"- {item}" for item in outcomes["challenge_highlights"])
    if outcomes["qualitative_notes"]:
        segments.append("**Qualitative notes**")
        segments.extend(f"- {note}" for note in outcomes["qualitative_notes"])
    return "\n".join(segments) if segments else "No outcome highlights available."


def synthesise_corpus_narrative(
    client: OpenAI,
    prevalence: dict[str, float],
    quotes: dict[str, list[str]],
    outcomes: dict[str, list[str]],
    model: str,
) -> str:
    """Request a final narrative from the LLM using aggregated evidence."""

    theme_table = format_theme_table(prevalence)
    evidence_block = format_evidence_block(quotes)
    outcome_block = format_outcome_block(outcomes)
    user_prompt = CORPUS_SYNTHESIS_USER_TEMPLATE.format(
        theme_table=theme_table,
        evidence_block=evidence_block,
        outcome_block=outcome_block,
    )
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": CORPUS_SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=OPENAI_TEMPERATURE,
    )
    return response.output_text


def generate_report_text(narrative: str) -> str:
    return f"{REPORT_HEADER}\n\n{narrative.strip()}\n"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
    parser.add_argument(
        "--base-path",
        type=Path,
        default=DATA_DIRECTORY,
        help=BASE_PATH_HELP,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OPENAI_MODEL,
        help=MODEL_HELP,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help=OUTPUT_HELP,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    conversations, survey = load_datasets(args.base_path)
    records = build_conversation_records(conversations, survey)
    client = OpenAI()
    analyses = [analyse_conversation(client, record, args.model) for record in records]
    prevalence, quotes = aggregate_theme_statistics(analyses)
    outcomes = summarise_outcomes(records, analyses)
    narrative = synthesise_corpus_narrative(client, prevalence, quotes, outcomes, args.model)
    report = generate_report_text(narrative)
    if args.output == "-":
        print(report)
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = args.base_path / output_path
        output_path.write_text(report, encoding="utf-8")
        print(f"Saved thematic analysis to {output_path}")


if __name__ == "__main__":
    main()
