#!/usr/bin/env python3
"""Automated thematic analysis of MIBot counselling transcripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from textwrap import indent
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from constants import (
    ACTION_FOCUSED_KEYWORDS,
    CONVERSATIONS_FILE,
    DATA_DIRECTORY,
    EMOTIONAL_SUPPORT_KEYWORDS,
    ENJOYMENT_KEYWORDS,
    MANDATED_MAX_AVG_CLIENT_WORDS,
    MANDATED_MAX_TOTAL_UTTERANCES,
    MIN_WORDS_PER_QUOTE,
    NEGATIVE_CONFIDENCE_DELTA_THRESHOLD,
    SAMPLE_QUOTES_PER_THEME,
    SUCCESS_CASE_LIMIT,
    SUCCESS_CONFIDENCE_DELTA_THRESHOLD,
    SUCCESS_IMPORTANCE_MINIMUM,
    SUCCESS_INDICATOR_KEYWORDS,
    SURVEY_FILE,
    THEME_DEFINITIONS,
)


@dataclass
class ThemeSummary:
    name: str
    description: str
    participant_count: int
    percentage: float
    quotes: list[str]


@dataclass
class SuccessCase:
    participant_id: str
    pre_confidence: float
    post_confidence: float
    week_later_confidence: float
    importance: float
    utterance_count: int
    client_utterances: int
    indicators: list[str]
    representative_quote: str | None


@dataclass
class NonResponderSummary:
    mandated_ids: list[str]
    enjoyment_ids: list[str]
    mismatch_ids: list[str]
    mandated_average_length: float


def load_conversations(conversation_path: Path) -> pd.DataFrame:
    conversations = pd.read_csv(conversation_path)
    conversations["Utterance"] = conversations["Utterance"].fillna("")
    conversations["normalised"] = conversations["Utterance"].str.lower()
    conversations["speaker_lower"] = conversations["Speaker"].str.lower()
    conversations["word_count"] = conversations["Utterance"].str.split().map(len)
    conversations["is_client"] = conversations["speaker_lower"].eq("client")
    conversations["is_counsellor"] = conversations["speaker_lower"].eq("counsellor")
    conversations["client_word_count"] = np.where(
        conversations["is_client"], conversations["word_count"], 0
    )
    conversations["counsellor_word_count"] = np.where(
        conversations["is_counsellor"], conversations["word_count"], 0
    )
    return conversations


def load_survey(survey_path: Path) -> pd.DataFrame:
    survey = pd.read_csv(survey_path)
    numeric_columns = [
        "PreRulerImportance",
        "PreRulerConfidence",
        "PostRulerConfidence",
        "WeekLaterRulerConfidence",
    ]
    for column in numeric_columns:
        survey[column] = pd.to_numeric(survey[column], errors="coerce")
    return survey


def contains_keywords(text: str, keywords: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def score_text(text: str, keywords: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(keyword) for keyword in keywords)


def select_representative_quotes(
    utterances: pd.Series, keywords: Sequence[str], limit: int
) -> list[str]:
    candidates: list[tuple[int, int, str]] = []
    for text in utterances:
        if not isinstance(text, str):
            continue
        if len(text.split()) < MIN_WORDS_PER_QUOTE:
            continue
        score = score_text(text, keywords)
        if score == 0:
            continue
        candidates.append((score, len(text), text))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [text for _, _, text in candidates[:limit]]


def compute_theme_summaries(conversations: pd.DataFrame) -> list[ThemeSummary]:
    summaries: list[ThemeSummary] = []
    participant_groups = conversations.groupby("ParticipantID")
    total_participants = participant_groups.ngroups

    for theme_name, config in THEME_DEFINITIONS.items():
        speaker = config["speaker"].lower()
        keywords = config["keywords"]
        description = config["description"]

        participant_hits: list[str] = []
        quote_candidates: list[str] = []

        for participant_id, group in participant_groups:
            speaker_utterances = group[group["speaker_lower"].eq(speaker)]
            matches = speaker_utterances[speaker_utterances["normalised"].apply(
                lambda text: contains_keywords(text, keywords)
            )]
            if not matches.empty:
                participant_hits.append(participant_id)
                quote_candidates.extend(matches["Utterance"].tolist())

        percentage = (len(participant_hits) / total_participants) * 100 if total_participants else 0.0
        quotes = select_representative_quotes(pd.Series(quote_candidates), keywords, SAMPLE_QUOTES_PER_THEME)
        summaries.append(
            ThemeSummary(
                name=theme_name,
                description=description,
                participant_count=len(participant_hits),
                percentage=percentage,
                quotes=quotes,
            )
        )

    return summaries


def compute_conversation_metrics(conversations: pd.DataFrame) -> pd.DataFrame:
    aggregated = conversations.groupby("ParticipantID").agg(
        total_utterances=("Utterance", "size"),
        client_utterances=("is_client", "sum"),
        counsellor_utterances=("is_counsellor", "sum"),
        client_words=("client_word_count", "sum"),
        counsellor_words=("counsellor_word_count", "sum"),
    )
    aggregated["avg_client_words"] = (
        aggregated["client_words"]
        / aggregated["client_utterances"].replace({0: np.nan})
    )
    return aggregated


def gather_success_cases(
    survey: pd.DataFrame, metrics: pd.DataFrame, conversations: pd.DataFrame
) -> list[SuccessCase]:
    survey = survey.copy()
    survey["confidence_delta"] = (
        survey["WeekLaterRulerConfidence"] - survey["PreRulerConfidence"]
    )
    eligible = survey[
        (survey["confidence_delta"] >= SUCCESS_CONFIDENCE_DELTA_THRESHOLD)
        & (survey["PreRulerImportance"] >= SUCCESS_IMPORTANCE_MINIMUM)
    ]
    eligible = eligible.sort_values(
        by=["confidence_delta", "WeekLaterRulerConfidence"], ascending=False
    ).head(SUCCESS_CASE_LIMIT)

    success_cases: list[SuccessCase] = []
    for _, row in eligible.iterrows():
        participant_id = row["ParticipantID"]
        if participant_id not in metrics.index:
            continue
        convo_metrics = metrics.loc[participant_id]
        conversation_view = conversations[conversations["ParticipantID"] == participant_id]
        client_texts = conversation_view[conversation_view["is_client"]]["Utterance"]

        indicators = [
            label
            for label, keywords in SUCCESS_INDICATOR_KEYWORDS.items()
            if client_texts.apply(lambda text: contains_keywords(text, keywords)).any()
        ]

        quote = None
        if not client_texts.empty:
            quotes = select_representative_quotes(client_texts, SUCCESS_INDICATOR_KEYWORDS[
                "Identified coping strategies for high-risk triggers."
            ], SAMPLE_QUOTES_PER_THEME)
            quote = quotes[0] if quotes else None

        success_cases.append(
            SuccessCase(
                participant_id=str(participant_id),
                pre_confidence=float(row["PreRulerConfidence"]),
                post_confidence=float(row["PostRulerConfidence"]),
                week_later_confidence=float(row["WeekLaterRulerConfidence"]),
                importance=float(row["PreRulerImportance"]),
                utterance_count=int(convo_metrics["total_utterances"]),
                client_utterances=int(convo_metrics["client_utterances"]),
                indicators=indicators,
                representative_quote=quote,
            )
        )

    return success_cases


def summarise_non_responders(
    survey: pd.DataFrame, metrics: pd.DataFrame, conversations: pd.DataFrame
) -> NonResponderSummary:
    survey = survey.copy()
    survey["confidence_change"] = survey["PostRulerConfidence"] - survey["PreRulerConfidence"]
    negative = survey[survey["confidence_change"] <= NEGATIVE_CONFIDENCE_DELTA_THRESHOLD]

    mandated_ids: list[str] = []
    enjoyment_ids: list[str] = []
    mismatch_ids: list[str] = []

    mandated_lengths: list[int] = []

    for _, row in negative.iterrows():
        participant_id = row["ParticipantID"]
        metrics_row = metrics.loc[participant_id]
        conversation_view = conversations[conversations["ParticipantID"] == participant_id]
        client_texts = conversation_view[conversation_view["is_client"]]
        counsellor_texts = conversation_view[conversation_view["is_counsellor"]]

        avg_client_words = metrics_row["avg_client_words"]
        total_utterances = metrics_row["total_utterances"]

        if (
            np.isfinite(avg_client_words)
            and avg_client_words <= MANDATED_MAX_AVG_CLIENT_WORDS
            and total_utterances <= MANDATED_MAX_TOTAL_UTTERANCES
        ):
            mandated_ids.append(participant_id)
            mandated_lengths.append(total_utterances)

        if client_texts["normalised"].apply(lambda text: contains_keywords(text, ENJOYMENT_KEYWORDS)).any():
            enjoyment_ids.append(participant_id)

        if (
            client_texts["normalised"].apply(lambda text: contains_keywords(text, EMOTIONAL_SUPPORT_KEYWORDS)).any()
            and counsellor_texts["normalised"].apply(lambda text: contains_keywords(text, ACTION_FOCUSED_KEYWORDS)).any()
        ):
            mismatch_ids.append(participant_id)

    mandated_average_length = float(np.mean(mandated_lengths)) if mandated_lengths else float("nan")

    return NonResponderSummary(
        mandated_ids=[str(pid) for pid in mandated_ids],
        enjoyment_ids=[str(pid) for pid in enjoyment_ids],
        mismatch_ids=[str(pid) for pid in mismatch_ids],
        mandated_average_length=mandated_average_length,
    )


def format_success_case(case: SuccessCase) -> str:
    header = f"Participant `{case.participant_id}`"
    details = [
        header + ":",
        f"Starting confidence {case.pre_confidence:.0f}/10, post-session {case.post_confidence:.0f}/10, week-later {case.week_later_confidence:.0f}/10.",
        f"Importance entering the chat: {case.importance:.0f}/10.",
        f"Conversation volume: {case.utterance_count} total utterances ({case.client_utterances} from the client).",
    ]
    if case.indicators:
        details.append("Observed conversation moves:")
        for index, indicator in enumerate(case.indicators, start=1):
            details.append(f"   {index}. {indicator}")
    if case.representative_quote:
        details.append("Representative coping quote:")
        details.append(indent(f"\"{case.representative_quote}\"", "   "))
    return "\n".join(details)


def format_non_responder_summary(summary: NonResponderSummary) -> str:
    lines = []
    if summary.mandated_ids:
        lines.append(
            "Mandated participation signals (brief replies, short conversations): "
            f"{len(summary.mandated_ids)} participants, average {summary.mandated_average_length:.0f} utterances."
        )
    if summary.enjoyment_ids:
        lines.append(
            "Enjoyment-focused smokers whose dialogue reaffirmed satisfaction with smoking: "
            f"{len(summary.enjoyment_ids)} participants."
        )
    if summary.mismatch_ids:
        lines.append(
            "Therapeutic misalignments (emotional bids met with action planning): "
            f"{len(summary.mismatch_ids)} participants."
        )
    return "\n".join(lines)


def build_report(
    theme_summaries: Sequence[ThemeSummary],
    success_cases: Sequence[SuccessCase],
    non_responder_summary: NonResponderSummary,
) -> str:
    lines = []
    lines.append(
        "To understand the qualitative contours of the MIBot conversations, this script performs an automated thematic "
        "analysis over the full transcript corpus. Patterns are derived by scanning client and counsellor utterances "
        "for linguistic markers associated with key motivational interviewing constructs."
    )

    for summary in theme_summaries:
        lines.append(f"\n### {summary.name}\n")
        lines.append(
            f"{summary.description} The pattern appears in {summary.participant_count} participants "
            f"({summary.percentage:.0f}% of conversations)."
        )
        for quote in summary.quotes:
            lines.append("\n> \"" + quote.strip() + "\"")

    if success_cases:
        lines.append("\n### Success Stories\n")
        for case in success_cases:
            lines.append(format_success_case(case) + "\n")

    non_responder_text = format_non_responder_summary(non_responder_summary)
    if non_responder_text:
        lines.append("\n### Non-Responders and Negative Cases\n")
        lines.append(non_responder_text)

    return "\n".join(lines).strip() + "\n"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thematic analysis on MIBot transcripts.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIRECTORY,
        help="Base directory containing conversations.csv and data.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    base_dir = args.data_dir

    conversations = load_conversations(base_dir / CONVERSATIONS_FILE.name)
    survey = load_survey(base_dir / SURVEY_FILE.name)
    metrics = compute_conversation_metrics(conversations)

    theme_summaries = compute_theme_summaries(conversations)
    success_cases = gather_success_cases(survey, metrics, conversations)
    non_responder_summary = summarise_non_responders(survey, metrics, conversations)

    report = build_report(theme_summaries, success_cases, non_responder_summary)
    print(report)


if __name__ == "__main__":
    main()
