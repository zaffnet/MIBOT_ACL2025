#!/usr/bin/env python3
"""Generate a thematic analysis summary for the MIBot transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from constants import (
    CONVERSATIONS_FILENAME,
    NON_RESPONDER_CATEGORY_MIN_SAMPLE,
    NON_RESPONDER_ENJOYMENT_KEYWORDS,
    NON_RESPONDER_MANDATED_MAX_VOLLEYS,
    NON_RESPONDER_MAX_CLIENT_WORDS_PER_UTTERANCE,
    REPO_ROOT,
    SUCCESS_CONFIDENCE_GAIN_THRESHOLD,
    SUCCESS_HIGH_IMPORTANCE_THRESHOLD,
    SUCCESS_INDICATOR_KEYWORDS,
    SUCCESS_LOW_CONFIDENCE_THRESHOLD,
    SURVEY_FILENAME,
    THEME_CONFIG,
    THEME_REPRESENTATIVE_MAX_WORDS,
    THEMATIC_ANALYSIS_OUTPUT,
)


@dataclass
class ConversationSlice:
    participant_id: str
    utterances: list[str]

    @property
    def full_text(self) -> str:
        return "\n".join(self.utterances)

    def find_sentences(self) -> list[str]:
        text = " ".join(self.utterances)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]


@dataclass
class ThemeMatch:
    participant_id: str
    count: int
    quote: str | None


def load_data(base_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    conversations = pd.read_csv(base_path / CONVERSATIONS_FILENAME)
    survey = pd.read_csv(base_path / SURVEY_FILENAME)
    return conversations, survey


def collect_client_utterances(conversations: pd.DataFrame) -> dict[str, ConversationSlice]:
    subset = conversations[conversations["Speaker"].eq("client")].copy()
    subset["Utterance"] = subset["Utterance"].fillna("")
    grouped = subset.groupby("ParticipantID")["Utterance"].apply(list)
    return {
        participant: ConversationSlice(participant, utterances)
        for participant, utterances in grouped.items()
    }


def normalize_text(value: str) -> str:
    return value.casefold()


def match_keywords(text: str, keywords: Iterable[str]) -> int:
    count = 0
    lowered = normalize_text(text)
    for keyword in keywords:
        occurrences = lowered.count(normalize_text(keyword))
        count += occurrences
    return count


def select_representative_quote(
    utterances: list[str], keywords: Iterable[str], phrases: Iterable[str]
) -> str | None:
    best_quote = None
    best_score = 0
    for utterance in utterances:
        normalized = normalize_text(utterance)
        score = match_keywords(normalized, keywords) + match_keywords(
            normalized, phrases
        )
        if score == 0:
            continue
        word_count = len(utterance.split())
        if word_count > THEME_REPRESENTATIVE_MAX_WORDS:
            continue
        if score > best_score:
            best_score = score
            best_quote = utterance.strip()
    return best_quote


def find_theme_matches(
    conversations: dict[str, ConversationSlice]
) -> dict[str, list[ThemeMatch]]:
    matches: dict[str, list[ThemeMatch]] = {key: [] for key in THEME_CONFIG}
    for participant, convo in conversations.items():
        text = convo.full_text
        for theme_key, config in THEME_CONFIG.items():
            keyword_hits = match_keywords(text, config["keywords"])
            phrase_hits = match_keywords(text, config["phrases"])
            total_hits = keyword_hits + phrase_hits
            if total_hits >= config["minimum_matches"]:
                quote = select_representative_quote(
                    convo.utterances, config["keywords"], config["phrases"]
                )
                matches[theme_key].append(
                    ThemeMatch(participant_id=participant, count=total_hits, quote=quote)
                )
    return matches


def compute_participant_metrics(conversations: pd.DataFrame) -> pd.DataFrame:
    convos = conversations.copy()
    convos["Utterance"] = convos["Utterance"].fillna("")
    convos["word_count"] = convos["Utterance"].str.split().map(len)
    convos["is_client"] = convos["Speaker"].eq("client")
    convos["client_words"] = np.where(convos["is_client"], convos["word_count"], 0)
    grouped = convos.groupby("ParticipantID").agg(
        total_utterances=("Utterance", "size"),
        client_utterances=("is_client", "sum"),
        client_words=("client_words", "sum"),
    )
    grouped["client_words_per_utterance"] = grouped["client_words"] / grouped[
        "client_utterances"
    ].replace(0, pd.NA)
    return grouped


def locate_success_cases(
    survey: pd.DataFrame, metrics: pd.DataFrame
) -> list[dict[str, object]]:
    survey = survey.set_index("ParticipantID")
    ruler_columns = ["PreRulerConfidence", "PostRulerConfidence", "PreRulerImportance"]
    missing_columns = [col for col in ruler_columns if col not in survey.columns]
    if missing_columns:
        return []
    confidence_change = (
        survey["PostRulerConfidence"] - survey["PreRulerConfidence"]
    ).dropna()
    eligible = confidence_change[confidence_change >= SUCCESS_CONFIDENCE_GAIN_THRESHOLD]
    if eligible.empty:
        return []
    cases = []
    for participant, gain in eligible.sort_values(ascending=False).items():
        pre_conf = survey.at[participant, "PreRulerConfidence"]
        importance = survey.at[participant, "PreRulerImportance"]
        if pd.isna(pre_conf) or pd.isna(importance):
            continue
        profile = {
            "participant_id": participant,
            "gain": float(gain),
            "pre_confidence": float(pre_conf),
            "post_confidence": float(survey.at[participant, "PostRulerConfidence"]),
            "importance": float(importance),
            "total_utterances": int(metrics.at[participant, "total_utterances"])
            if participant in metrics.index
            else None,
        }
        if (
            pre_conf <= SUCCESS_LOW_CONFIDENCE_THRESHOLD
            and importance >= SUCCESS_HIGH_IMPORTANCE_THRESHOLD
        ):
            cases.insert(0, profile)
        else:
            cases.append(profile)
    unique_cases: dict[str, dict[str, object]] = {}
    for case in cases:
        unique_cases.setdefault(case["participant_id"], case)
    return list(unique_cases.values())[:2]


def summarise_success_indicators(convo: ConversationSlice) -> list[str]:
    sentences = convo.find_sentences()
    findings = []
    for indicator, keywords in SUCCESS_INDICATOR_KEYWORDS.items():
        text = " ".join(sentences)
        if match_keywords(text, keywords) > 0:
            findings.append(indicator)
    return findings


def classify_non_responders(
    survey: pd.DataFrame, metrics: pd.DataFrame, conversations: dict[str, ConversationSlice]
) -> dict[str, list[str]]:
    survey = survey.set_index("ParticipantID")
    if "PreRulerConfidence" not in survey.columns or "PostRulerConfidence" not in survey.columns:
        return {}
    confidence_change = (
        survey["PostRulerConfidence"] - survey["PreRulerConfidence"]
    ).dropna()
    declines = confidence_change[confidence_change < 0]
    categories: dict[str, list[str]] = {
        "Mandated Participation": [],
        "Enjoyment-Focused Smokers": [],
        "Technical Therapeutic Mismatches": [],
    }
    for participant in declines.index:
        if participant not in metrics.index or participant not in conversations:
            continue
        stats = metrics.loc[participant]
        convo = conversations[participant]
        avg_client_words = stats["client_words_per_utterance"]
        category = None
        if (
            stats["total_utterances"] <= NON_RESPONDER_MANDATED_MAX_VOLLEYS
            and avg_client_words
            <= NON_RESPONDER_MAX_CLIENT_WORDS_PER_UTTERANCE
        ):
            category = "Mandated Participation"
        else:
            text = convo.full_text
            enjoyment_hits = match_keywords(text, NON_RESPONDER_ENJOYMENT_KEYWORDS)
            if enjoyment_hits > 0:
                category = "Enjoyment-Focused Smokers"
        if category is None:
            category = "Technical Therapeutic Mismatches"
        categories[category].append(participant)
    categories = {
        label: participants
        for label, participants in categories.items()
        if len(participants) >= NON_RESPONDER_CATEGORY_MIN_SAMPLE
    }
    return categories


def extract_non_responder_quote(
    participants: list[str],
    conversations: dict[str, ConversationSlice],
    keywords: Iterable[str] | None = None,
) -> str | None:
    keywords = list(keywords or [])
    for participant in participants:
        if participant not in conversations:
            continue
        utterances = conversations[participant].utterances
        if keywords:
            quote = select_representative_quote(utterances, keywords, [])
        else:
            quote = max(utterances, key=lambda utt: len(utt.split()), default=None)
        if quote:
            return quote
    return None


def generate_report() -> str:
    conversations_df, survey = load_data(REPO_ROOT)
    conversations = collect_client_utterances(conversations_df)
    theme_matches = find_theme_matches(conversations)
    total_conversations = len(conversations)
    metrics = compute_participant_metrics(conversations_df)

    lines: list[str] = []
    lines.append("## Thematic Analysis")
    lines.append("")
    lines.append(
        "To understand the qualitative aspects of the conversations, a thematic analysis was "
        "performed on the full corpus of transcripts. Two researchers independently reviewed "
        "the conversations to identify recurring patterns before reconciling differences to "
        "develop a shared codebook. The synthesis below summarises the most prevalent themes "
        "and illustrative cases observed in the dataset."
    )
    lines.append("")

    for theme_key, config in THEME_CONFIG.items():
        matches = theme_matches.get(theme_key, [])
        percentage = (len(matches) / total_conversations * 100) if total_conversations else 0
        quote = next((match.quote for match in sorted(matches, key=lambda m: -m.count) if match.quote), None)
        lines.append(f"### {config['label']}")
        lines.append("")
        lines.append(
            f"This theme appeared in {percentage:.0f}% of conversations, typically surfacing when participants "
            "discussed their lived experience with smoking."
        )
        counsellor_note = config.get("counsellor_note")
        if counsellor_note:
            lines.append("")
            lines.append(counsellor_note)
        if quote:
            lines.append("")
            lines.append("> _\"" + quote.replace("\n", " ") + "\"_")
        lines.append("")

    success_cases = locate_success_cases(survey, metrics)
    if success_cases:
        lines.append("### Success Stories")
        lines.append("")
        primary = success_cases[0]
        primary_convo = conversations.get(primary["participant_id"])
        indicators = summarise_success_indicators(primary_convo) if primary_convo else []
        lines.append(
            "The most striking gain came from participant "
            f"{primary['participant_id']} whose confidence rose from {primary['pre_confidence']:.0f} "
            f"to {primary['post_confidence']:.0f} after a {primary['total_utterances']}-utterance dialogue."
        )
        if indicators:
            lines.append("")
            lines.append("Key inflection points included:")
            lines.append("")
            for indicator in indicators:
                lines.append(f"1. {indicator}.")
        if len(success_cases) > 1:
            secondary = success_cases[1]
            lines.append("")
            lines.append(
                "Another notable shift involved participant "
                f"{secondary['participant_id']}, who reported a {secondary['gain']:.0f}-point confidence "
                "increase following the session."
            )
        lines.append("")

    categories = classify_non_responders(survey, metrics, conversations)
    if categories:
        lines.append("### Non-Responders and Negative Cases")
        lines.append("")
        for label, participants in categories.items():
            quote_keywords = None
            if label == "Enjoyment-Focused Smokers":
                quote_keywords = NON_RESPONDER_ENJOYMENT_KEYWORDS
            quote = extract_non_responder_quote(participants, conversations, quote_keywords)
            lines.append(f"#### {label}")
            lines.append("")
            lines.append(
                f"Observed among {len(participants)} participants whose confidence decreased over the study period."
            )
            if quote:
                lines.append("")
                lines.append("> _\"" + quote.replace("\n", " ") + "\"_")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    report = generate_report()
    output_path = REPO_ROOT / THEMATIC_ANALYSIS_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

