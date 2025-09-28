#!/usr/bin/env python3
"""Automated thematic analysis of MIBot counselling transcripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from constants import (
    AMBIVALENCE_BRIDGE_WORDS,
    AMBIVALENCE_DEPENDENCE_KEYWORDS,
    AMBIVALENCE_POSITIVE_KEYWORDS,
    CLIENT_SPEAKER,
    COUNSELLOR_SPEAKER,
    DATA_PATH,
    NEGATIVE_OUTCOME_THRESHOLDS,
    PERCENTAGE_PRECISION,
    SMOKING_KEYWORD,
    SUCCESS_KEYWORDS,
    SUPPORT_KEYWORDS,
    THEME_KEYWORDS,
    TRANSCRIPTS_DIR,
)


@dataclass
class ParticipantTranscript:
    participant_id: str
    client_text: list[str]
    counsellor_text: list[str]
    client_word_counts: list[int]
    counsellor_labels: list[str]
    total_utterances: int

    @property
    def average_client_words(self) -> float:
        if not self.client_word_counts:
            return 0.0
        return sum(self.client_word_counts) / len(self.client_word_counts)


@dataclass
class ThemeResult:
    label: str
    percentage: float
    quote: str


@dataclass
class SuccessCase:
    participant_id: str
    pre_confidence: float
    post_confidence: float
    week_confidence: float
    pre_importance: float
    week_importance: float
    utterance_count: int
    counsellor_reflections: int
    metrics: dict[str, int]
    exemplar_quote: str


@dataclass
class NegativeCaseCategory:
    label: str
    count: int
    quote: str


def load_transcripts(transcripts_dir: Path) -> dict[str, ParticipantTranscript]:
    transcripts: dict[str, ParticipantTranscript] = {}
    for path in transcripts_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        participant_id = data["participant_id"]
        client_text: list[str] = []
        counsellor_text: list[str] = []
        client_word_counts: list[int] = []
        counsellor_labels: list[str] = []
        utterances = data.get("utterances", [])
        for item in utterances:
            speaker = item.get("speaker", "").strip().lower()
            text = item.get("text", "")
            label = item.get("auto_misc_label", "")
            if speaker == CLIENT_SPEAKER:
                client_text.append(text)
                client_word_counts.append(int(item.get("word_count", 0)))
            elif speaker == COUNSELLOR_SPEAKER:
                counsellor_text.append(text)
                counsellor_labels.append(label)
        transcripts[participant_id] = ParticipantTranscript(
            participant_id=participant_id,
            client_text=client_text,
            counsellor_text=counsellor_text,
            client_word_counts=client_word_counts,
            counsellor_labels=counsellor_labels,
            total_utterances=len(utterances),
        )
    return transcripts


def normalize(text: str) -> str:
    return text.lower()


def contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    normalized = normalize(text)
    return any(keyword in normalized for keyword in keywords)


def theme_matches(utterances: Iterable[str], keywords: Sequence[str]) -> list[str]:
    matches: list[str] = []
    for utterance in utterances:
        normalized = normalize(utterance)
        if SMOKING_KEYWORD not in normalized:
            continue
        if contains_keyword(normalized, keywords):
            matches.append(utterance)
    return matches


def select_quote(candidates: Sequence[str]) -> str:
    if not candidates:
        return ""
    return max(candidates, key=lambda text: len(text))


def sanitize_for_latex(text: str) -> str:
    sanitized = text.replace("\\", "\\\\")
    sanitized = sanitized.replace("\"", "''")
    sanitized = sanitized.replace("{", "\\{")
    sanitized = sanitized.replace("}", "\\}")
    return sanitized


def compute_theme_results(transcripts: dict[str, ParticipantTranscript]) -> dict[str, ThemeResult]:
    results: dict[str, ThemeResult] = {}
    total_participants = len(transcripts)

    stress_matches = {
        pid: theme_matches(record.client_text, THEME_KEYWORDS["stress_coping"])
        for pid, record in transcripts.items()
    }
    social_matches = {
        pid: theme_matches(record.client_text, THEME_KEYWORDS["social_ritual"])
        for pid, record in transcripts.items()
    }

    ambivalence_matches: dict[str, list[str]] = {}
    for pid, record in transcripts.items():
        combined: list[str] = []
        positives: list[str] = []
        dependence: list[str] = []
        for utterance in record.client_text:
            normalized = normalize(utterance)
            if SMOKING_KEYWORD not in normalized:
                continue
            has_positive = contains_keyword(normalized, AMBIVALENCE_POSITIVE_KEYWORDS)
            has_dependence = contains_keyword(normalized, AMBIVALENCE_DEPENDENCE_KEYWORDS)
            has_bridge = contains_keyword(normalized, AMBIVALENCE_BRIDGE_WORDS)
            if has_positive:
                positives.append(utterance)
            if has_dependence:
                dependence.append(utterance)
            if has_positive and has_dependence:
                combined.append(utterance)
            elif has_positive and has_bridge:
                combined.append(utterance)
            elif has_dependence and has_bridge:
                combined.append(utterance)
        if positives and dependence:
            ambivalence_matches[pid] = combined or positives + dependence

    def build_result(label: str, match_map: dict[str, list[str]]) -> ThemeResult:
        count = sum(1 for items in match_map.values() if items)
        percentage = round(count / total_participants * 100, PERCENTAGE_PRECISION)
        exemplar_source = max(
            match_map.items(),
            key=lambda item: len(select_quote(item[1])),
            default=("", []),
        )
        exemplar_quote = sanitize_for_latex(select_quote(exemplar_source[1]).strip())
        return ThemeResult(label=label, percentage=percentage, quote=exemplar_quote)

    results["stress"] = build_result("Stress and Coping Narratives", stress_matches)
    results["social"] = build_result("Social and Ritualistic Aspects", social_matches)
    results["ambivalence"] = build_result("Ambivalence Themes", ambivalence_matches)
    return results


def load_survey_data(data_path: Path) -> pd.DataFrame:
    survey = pd.read_csv(data_path)
    return survey.set_index("ParticipantID")


def compute_success_cases(
    transcripts: dict[str, ParticipantTranscript], survey: pd.DataFrame
) -> list[SuccessCase]:
    available = survey.dropna(subset=["WeekLaterRulerConfidence", "PreRulerConfidence"])
    confidence_gain = available["WeekLaterRulerConfidence"] - available["PreRulerConfidence"]
    top_ids = confidence_gain.sort_values(ascending=False).head(5).index

    cases: list[SuccessCase] = []
    for participant_id in top_ids:
        if participant_id not in transcripts:
            continue
        record = transcripts[participant_id]
        participant_row = survey.loc[participant_id]
        metrics = count_success_indicators(record)
        reflections = sum(label == "R" for label in record.counsellor_labels)
        exemplar_candidates = theme_matches(
            record.client_text, SUCCESS_KEYWORDS["coping"]
        )
        exemplar = select_quote(exemplar_candidates)
        if not exemplar:
            exemplar = select_quote(record.client_text)
        cases.append(
            SuccessCase(
                participant_id=participant_id,
                pre_confidence=float(participant_row.get("PreRulerConfidence", 0.0)),
                post_confidence=float(participant_row.get("PostRulerConfidence", 0.0)),
                week_confidence=float(participant_row.get("WeekLaterRulerConfidence", 0.0)),
                pre_importance=float(participant_row.get("PreRulerImportance", 0.0)),
                week_importance=float(participant_row.get("WeekLaterRulerImportance", 0.0)),
                utterance_count=record.total_utterances,
                counsellor_reflections=reflections,
                metrics=metrics,
                exemplar_quote=exemplar,
            )
        )
    cases.sort(key=lambda case: case.week_confidence - case.pre_confidence, reverse=True)
    return cases[:2]


def count_success_indicators(record: ParticipantTranscript) -> dict[str, int]:
    metrics = {key: 0 for key in SUCCESS_KEYWORDS}
    for utterance in record.counsellor_text + record.client_text:
        normalized = normalize(utterance)
        for key, keywords in SUCCESS_KEYWORDS.items():
            if contains_keyword(normalized, keywords) and SMOKING_KEYWORD in normalized:
                metrics[key] += 1
    return metrics


def categorize_negative_cases(
    transcripts: dict[str, ParticipantTranscript], survey: pd.DataFrame
) -> list[NegativeCaseCategory]:
    mask = (
        survey["PostRulerConfidence"].notna()
        & survey["PreRulerConfidence"].notna()
        & (survey["PostRulerConfidence"] < survey["PreRulerConfidence"])
    )
    negative_participants = survey[mask].index

    mandated: list[str] = []
    enjoyment: list[str] = []
    mismatches: list[str] = []

    for participant_id in negative_participants:
        if participant_id not in transcripts:
            continue
        record = transcripts[participant_id]
        if (
            record.total_utterances <= NEGATIVE_OUTCOME_THRESHOLDS["mandated_max_utterances"]
            and record.average_client_words
            <= NEGATIVE_OUTCOME_THRESHOLDS["mandated_max_avg_client_words"]
        ):
            mandated.append(participant_id)
            continue
        client_matches = theme_matches(record.client_text, THEME_KEYWORDS["enjoyment"])
        if client_matches:
            enjoyment.append(participant_id)
            continue
        mismatches.append(participant_id)

    categories: list[NegativeCaseCategory] = []
    categories.append(
        NegativeCaseCategory(
            label="Mandated Participation",
            count=len(mandated),
            quote=select_category_quote(transcripts, mandated),
        )
    )
    categories.append(
        NegativeCaseCategory(
            label="Enjoyment-Focused Smokers",
            count=len(enjoyment),
            quote=select_category_quote(transcripts, enjoyment),
        )
    )
    categories.append(
        NegativeCaseCategory(
            label="Technical Therapeutic Mismatches",
            count=len(mismatches),
            quote=select_category_quote(transcripts, mismatches, require_support=True),
        )
    )
    return categories


def select_category_quote(
    transcripts: dict[str, ParticipantTranscript],
    participant_ids: Sequence[str],
    require_support: bool = False,
) -> str:
    candidate_quotes: list[str] = []
    for participant_id in participant_ids:
        record = transcripts.get(participant_id)
        if record is None:
            continue
        for utterance in record.client_text:
            normalized = normalize(utterance)
            if SMOKING_KEYWORD not in normalized:
                continue
            if require_support and not contains_keyword(normalized, SUPPORT_KEYWORDS):
                continue
            candidate_quotes.append(utterance)
    return sanitize_for_latex(select_quote(candidate_quotes))


def format_success_case(case: SuccessCase) -> str:
    gain = case.week_confidence - case.pre_confidence
    importance_gain = case.week_importance - case.pre_importance
    lines = [
        (
            f"The participant ({case.participant_id[:8]}...) entered with confidence "
            f"{case.pre_confidence:.1f}/10 and reached {case.week_confidence:.1f}/10 at one week "
            f"(Δ {gain:.1f}). Their importance rating moved from {case.pre_importance:.1f} to "
            f"{case.week_importance:.1f} (Δ {importance_gain:.1f}), across {case.utterance_count} utterances "
            f"with {case.counsellor_reflections} reflective responses."
        ),
        "",
        "\\begin{enumerate}",
    ]
    steps = [
        (
            "Examined past quit attempts", case.metrics.get("past_attempts", 0)
        ),
        (
            "Reframed setbacks as learning moments",
            case.metrics.get("learning_reframe", 0),
        ),
        ("Outlined a concrete quit plan", case.metrics.get("plan", 0)),
        (
            "Identified coping tools for anticipated triggers",
            case.metrics.get("coping", 0),
        ),
    ]
    for description, count in steps:
        lines.append(f"\\item {description} ({count} mentions)")
    lines.append("\\end{enumerate}")
    if case.exemplar_quote:
        formatted_quote = sanitize_for_latex(case.exemplar_quote)
        lines.append("")
        lines.append("\\begin{quote}")
        lines.append(f"\\textit{{\"{formatted_quote}\"}}")
        lines.append("\\end{quote}")
    return "\n".join(lines)


def build_negative_case_section(categories: Sequence[NegativeCaseCategory], total: int) -> str:
    lines = ["\\subsubsection{Non-Responders and Negative Cases}"]
    lines.append(
        f"Analysis of the {total} participants whose confidence decreased revealed three patterns:"
    )
    for category in categories:
        lines.append(f"\\paragraph{{{category.label}}}")
        lines.append(
            (
                f"{category.count} participants met this pattern."
                if category.count
                else "No participants exhibited this pattern in the current dataset."
            )
        )
        if category.quote:
            lines.append("\\begin{quote}")
            lines.append(f"\\textit{{\"{category.quote}\"}}")
            lines.append("\\end{quote}")
    return "\n".join(lines)


def generate_report(base_path: Path) -> str:
    transcripts = load_transcripts(TRANSCRIPTS_DIR)
    survey = load_survey_data(DATA_PATH)

    theme_results = compute_theme_results(transcripts)
    success_cases = compute_success_cases(transcripts, survey)
    negative_categories = categorize_negative_cases(transcripts, survey)
    negative_total = sum(category.count for category in negative_categories)

    lines: list[str] = []
    lines.append("## Thematic Analysis {#sec:thematic-analysis}")
    lines.append("")
    lines.append(
        (
            "To surface conversational patterns linked with therapeutic engagement, we "
            "combined lexical heuristics over all transcripts with readiness-ruler outcomes. "
            "Client-side utterances were scanned for smoking-related references aligned with "
            "stress, social context, and ambivalence lexicons, while survey deltas informed "
            "success and non-responder profiles."
        )
    )
    lines.append("")
    lines.append("This process revealed three recurring patterns:")
    lines.append("")

    for key in ("stress", "social", "ambivalence"):
        result = theme_results[key]
        section_header = f"\\subsubsection{{{result.label}}}"
        lines.append(section_header)
        lines.append("")
        lines.append(
            (
                f"The theme appeared in {result.percentage:.{PERCENTAGE_PRECISION}f}% of conversations, "
                "highlighting clients' narratives."
            )
        )
        if result.quote:
            lines.append("")
            lines.append("\\begin{quote}")
            lines.append(f"\t\\textit{{\"{result.quote}\"}}")
            lines.append("\\end{quote}")
        lines.append("")

    lines.append("\\subsubsection*{Success Stories}")
    lines.append("")
    for case in success_cases:
        lines.append(format_success_case(case))
        lines.append("")

    lines.append(
        build_negative_case_section(negative_categories, negative_total)
    )

    report = "\n".join(lines).strip() + "\n"
    return report


def main() -> None:
    base_path = Path(__file__).resolve().parents[1]
    report = generate_report(base_path)
    output_path = base_path / "analysis" / "thematic_analysis.md"
    output_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
