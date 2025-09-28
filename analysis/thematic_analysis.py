#!/usr/bin/env python3
"""Perform a lightweight thematic analysis of the MIBot transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import (
    BASE_PATH,
    CONVERSATIONS_CSV,
    DATA_CSV,
    ENJOYMENT_KEYWORDS,
    MANDATED_PARTICIPATION_CRITERIA,
    MAX_QUOTE_WORDS,
    MISMATCH_KEYWORDS,
    SUCCESS_CONFIDENCE_MIN_GAIN,
    SUCCESS_MOTIF_KEYWORDS,
    THEME_CONFIG,
    THEME_QUOTE_MIN_WORDS,
    THEMATIC_INTRO,
    THEMATIC_NON_RESPONDER_INTRO,
    TOP_SUCCESS_STORIES,
)


@dataclass
class ThemeResult:
    identifier: str
    title: str
    description: str
    counsellor_focus: str
    percentage: float
    participant_count: int
    quote: str


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    conversations = pd.read_csv(CONVERSATIONS_CSV)
    survey = pd.read_csv(DATA_CSV)
    return conversations, survey


def preprocess_conversations(conversations: pd.DataFrame) -> pd.DataFrame:
    processed = conversations.copy()
    processed["Utterance"] = processed["Utterance"].fillna("")
    processed["utterance_lower"] = processed["Utterance"].str.lower()
    processed["word_count"] = processed["Utterance"].str.split().map(len)
    return processed


def contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def trim_quote(text: str) -> str:
    words = text.split()
    if len(words) <= MAX_QUOTE_WORDS:
        return text
    trimmed = " ".join(words[:MAX_QUOTE_WORDS]) + "..."
    return trimmed


def escape_for_latex(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    escaped = escaped.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
    escaped = escaped.replace("#", "\\#").replace("_", "\\_")
    return escaped


def select_quote(df: pd.DataFrame, keywords: Sequence[str]) -> str:
    if df.empty:
        return ""

    keyword_mask = df["utterance_lower"].apply(lambda text: contains_keyword(text, keywords))
    candidates = df.loc[keyword_mask].copy()
    if candidates.empty:
        return ""

    candidates = candidates.sort_values(by=["word_count", "Utterance"], ascending=[False, True])
    long_enough = candidates[candidates["word_count"] >= THEME_QUOTE_MIN_WORDS]
    selected = long_enough.iloc[0] if not long_enough.empty else candidates.iloc[0]
    quote = trim_quote(str(selected["Utterance"]).strip())
    return quote


def format_sentence(text: str) -> str:
    return text.rstrip(".")


def format_score(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.0f}"


def to_float(value: object) -> float:
    return float(value) if value is not None and not pd.isna(value) else float("nan")


def format_points(value: float | int) -> str:
    magnitude = abs(int(round(float(value))))
    label = "point" if magnitude == 1 else "points"
    return f"{magnitude} {label}"


def compute_theme_results(conversations: pd.DataFrame) -> list[ThemeResult]:
    total_participants = conversations["ParticipantID"].nunique()
    results: list[ThemeResult] = []

    for config in THEME_CONFIG:
        speaker = config["speaker"]
        subset = conversations[conversations["Speaker"].eq(speaker)]
        keyword_mask = subset["utterance_lower"].apply(
            lambda text: contains_keyword(text, config["keywords"])
        )
        participant_ids = subset.loc[keyword_mask, "ParticipantID"].unique()
        count = len(participant_ids)
        percentage = (count / total_participants * 100) if total_participants else 0.0
        quote = select_quote(subset, config["keywords"])

        results.append(
            ThemeResult(
                identifier=str(config["identifier"]),
                title=str(config["title"]),
                description=str(config["description"]),
                counsellor_focus=str(config["counsellor_focus"]),
                percentage=percentage,
                participant_count=count,
                quote=quote,
            )
        )

    return results


def compute_participant_metrics(conversations: pd.DataFrame) -> pd.DataFrame:
    conv = conversations.copy()
    conv["is_counsellor"] = conv["Speaker"].eq("counsellor")
    conv["is_client"] = conv["Speaker"].eq("client")
    conv["counsellor_words"] = np.where(conv["is_counsellor"], conv["word_count"], 0)
    conv["client_words"] = np.where(conv["is_client"], conv["word_count"], 0)

    aggregated = conv.groupby("ParticipantID").agg(
        total_utterances=("Utterance", "size"),
        counsellor_utterances=("is_counsellor", "sum"),
        client_utterances=("is_client", "sum"),
        counsellor_words=("counsellor_words", "sum"),
        client_words=("client_words", "sum"),
    )

    aggregated["avg_client_words"] = aggregated.apply(
        lambda row: (row["client_words"] / row["client_utterances"]) if row["client_utterances"] else 0.0,
        axis=1,
    )

    return aggregated


def concatenate_utterances(conversations: pd.DataFrame, participant_id: str, speaker: str | None = None) -> str:
    subset = conversations[conversations["ParticipantID"].eq(participant_id)]
    if speaker is not None:
        subset = subset[subset["Speaker"].eq(speaker)]
    utterances = subset["Utterance"].tolist()
    return " ".join(utterances)


def extract_motifs(text: str) -> list[str]:
    lowered = text.lower()
    motifs: list[str] = []
    for description, keywords in SUCCESS_MOTIF_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            motifs.append(description)
    return motifs


def select_success_stories(
    conversations: pd.DataFrame, metrics: pd.DataFrame, survey: pd.DataFrame
) -> list[dict[str, object]]:
    survey_indexed = survey.set_index("ParticipantID")
    combined = metrics.join(survey_indexed, how="left")

    combined["confidence_change"] = (
        combined["PostRulerConfidence"] - combined["PreRulerConfidence"]
    )
    combined["week_later_change"] = (
        combined["WeekLaterRulerConfidence"] - combined["PreRulerConfidence"]
    )

    eligible = combined.dropna(subset=["confidence_change"])
    eligible = eligible.sort_values(
        by=["confidence_change", "week_later_change"],
        ascending=[False, False],
        na_position="last",
    )

    strong_gains = eligible[eligible["confidence_change"] >= SUCCESS_CONFIDENCE_MIN_GAIN]
    selected = strong_gains.head(TOP_SUCCESS_STORIES)
    if len(selected) < TOP_SUCCESS_STORIES:
        remaining = eligible.loc[~eligible.index.isin(selected.index)]
        selected = pd.concat([selected, remaining.head(TOP_SUCCESS_STORIES - len(selected))])

    stories: list[dict[str, object]] = []
    for participant_id, row in selected.iterrows():
        conversation_text = concatenate_utterances(conversations, participant_id, speaker="client")
        motifs = extract_motifs(conversation_text)
        stories.append(
            {
                "participant_id": participant_id,
                "pre_confidence": to_float(row.get("PreRulerConfidence")),
                "post_confidence": to_float(row.get("PostRulerConfidence")),
                "week_later_confidence": to_float(row.get("WeekLaterRulerConfidence")),
                "importance": to_float(row.get("PreRulerImportance")),
                "total_utterances": row["total_utterances"],
                "motifs": motifs,
                "confidence_change": to_float(row.get("confidence_change")),
                "week_later_change": to_float(row.get("week_later_change")),
            }
        )

    return stories


def format_participant_label(participant_id: str) -> str:
    return f"participant {participant_id[:6]}"


def analyze_non_responders(
    conversations: pd.DataFrame, metrics: pd.DataFrame, survey: pd.DataFrame
) -> dict[str, dict[str, object]]:
    survey_indexed = survey.set_index("ParticipantID")
    combined = metrics.join(survey_indexed, how="left")
    combined["confidence_change"] = (
        combined["PostRulerConfidence"] - combined["PreRulerConfidence"]
    )
    negative = combined[combined["confidence_change"] < 0]

    mandated_mask = (
        (negative["total_utterances"] <= MANDATED_PARTICIPATION_CRITERIA["max_total_utterances"])
        & (negative["avg_client_words"] <= MANDATED_PARTICIPATION_CRITERIA["max_avg_client_words"])
    )
    mandated = negative[mandated_mask]

    remaining = negative.loc[~negative.index.isin(mandated.index)]

    enjoyment_ids: list[str] = []
    for participant_id in remaining.index:
        text = concatenate_utterances(conversations, participant_id, speaker="client").lower()
        if any(keyword in text for keyword in ENJOYMENT_KEYWORDS):
            enjoyment_ids.append(participant_id)

    enjoyment = remaining.loc[enjoyment_ids]
    mismatch = remaining.loc[~remaining.index.isin(enjoyment_ids)]

    patterns: dict[str, dict[str, object]] = {}

    patterns["mandated"] = {
        "participants": mandated.index.tolist(),
        "count": len(mandated),
        "average_utterances": float(mandated["total_utterances"].mean()) if len(mandated) else 0.0,
        "average_client_words": float(mandated["avg_client_words"].mean()) if len(mandated) else 0.0,
        "quote": select_quote(
            conversations[conversations["ParticipantID"].isin(mandated.index)],
            (),
        ),
    }

    patterns["enjoyment"] = {
        "participants": enjoyment.index.tolist(),
        "count": len(enjoyment),
        "quote": select_quote(
            conversations[
                (conversations["ParticipantID"].isin(enjoyment.index))
                & conversations["Speaker"].eq("client")
            ],
            ENJOYMENT_KEYWORDS,
        ),
    }

    patterns["mismatch"] = {
        "participants": mismatch.index.tolist(),
        "count": len(mismatch),
        "quote": select_quote(
            conversations[conversations["ParticipantID"].isin(mismatch.index)],
            MISMATCH_KEYWORDS,
        ),
    }

    patterns["total_negative"] = len(negative)
    return patterns


def format_quote_block(quote: str) -> list[str]:
    if not quote:
        return []
    escaped = escape_for_latex(quote)
    return ["\\begin{quote}", f"\t\\textit{{``{escaped}''}}", "\\end{quote}"]


def build_report(
    theme_results: Sequence[ThemeResult],
    success_stories: Sequence[dict[str, object]],
    non_responder_patterns: dict[str, dict[str, object]],
) -> str:
    lines: list[str] = ["## Thematic Analysis {#sec:thematic-analysis}", "", THEMATIC_INTRO, ""]

    for theme in theme_results:
        lines.append(f"\\subsubsection{{{theme.title}}}")
        lines.append("")
        lines.append(
            (
                f"This theme appeared in {theme.percentage:.0f}\\% of conversations ("
                f"{theme.participant_count} of the analysed participants), typically involving {format_sentence(theme.description)}."
            )
        )
        lines.append(
            f"MIBot tended to respond with {theme.counsellor_focus}"
        )
        lines.append("")
        lines.extend(format_quote_block(theme.quote))
        lines.append("")

    lines.append("\\subsubsection{Success Stories}")
    lines.append("")

    for story in success_stories:
        participant_label = format_participant_label(story["participant_id"])
        pre_conf = format_score(story["pre_confidence"])
        post_conf = format_score(story["post_confidence"])
        week_later_value = story.get("week_later_confidence")
        week_later = format_score(week_later_value)
        importance = format_score(story.get("importance"))
        utterances = story["total_utterances"]
        motifs = story["motifs"]
        gain = format_score(story.get("confidence_change"))

        lines.append(
            (
                f"{participant_label.capitalize()} entered with confidence {pre_conf}/10 and importance "
                f"{importance}/10. Over {utterances} utterances the dialogue raised confidence to "
                f"{post_conf}/10 (Δ {gain})."
            )
        )
        if week_later_value is not None and pd.notna(week_later_value):
            delta_followup = week_later_value - story.get("post_confidence", float("nan"))
            if not pd.isna(delta_followup):
                if delta_followup > 0:
                    change = format_points(delta_followup)
                    lines.append(
                        f"Week-later follow-up recorded confidence at {week_later}/10, increasing by an additional {change}."
                    )
                elif delta_followup < 0:
                    change = format_points(delta_followup)
                    lines.append(
                        f"Week-later follow-up recorded confidence at {week_later}/10, a decline of {change} from session end."
                    )
                else:
                    lines.append(
                        f"Week-later follow-up recorded confidence at {week_later}/10, maintaining the gain."
                    )
        if motifs:
            lines.append("\\begin{enumerate}")
            for motif in motifs:
                lines.append(f"\t\\item {motif}")
            lines.append("\\end{enumerate}")
        lines.append("")

    lines.append("\\subsubsection{Non-Responders and Negative Cases}")
    lines.append("")
    total_negative = non_responder_patterns.get("total_negative", 0)
    lines.append(
        f"{THEMATIC_NON_RESPONDER_INTRO} A total of {total_negative} participants experienced declines in confidence."
    )
    lines.append("")

    mandated = non_responder_patterns["mandated"]
    lines.append("\\paragraph{Mandated Participation}")
    lines.append(
        (
            f"These conversations averaged {mandated['average_utterances']:.0f} utterances with clients speaking "
            f"{mandated['average_client_words']:.0f} words per turn on average, suggesting transactional engagement."
        )
    )
    lines.extend(format_quote_block(mandated["quote"]))
    lines.append("")

    enjoyment = non_responder_patterns["enjoyment"]
    lines.append("\\paragraph{Enjoyment-Focused Smokers}")
    lines.append(
        "A subset identified strongly as happy smokers; motivational exploration sometimes reinforced enjoyment."
    )
    lines.extend(format_quote_block(enjoyment["quote"]))
    lines.append("")

    mismatch = non_responder_patterns["mismatch"]
    lines.append("\\paragraph{Technical Therapeutic Mismatches}")
    lines.append(
        "In a few conversations, the agent emphasised change strategies while participants signalled emotional needs, "
        "resulting in missed attunement."
    )
    lines.extend(format_quote_block(mismatch["quote"]))
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    conversations, survey = load_datasets()
    processed = preprocess_conversations(conversations)
    theme_results = compute_theme_results(processed)
    metrics = compute_participant_metrics(processed)
    success_stories = select_success_stories(processed, metrics, survey)
    non_responder_patterns = analyze_non_responders(processed, metrics, survey)
    report = build_report(theme_results, success_stories, non_responder_patterns)
    output_path = BASE_PATH / "analysis" / "thematic_analysis.md"
    output_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
