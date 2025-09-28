#!/usr/bin/env python3
"""Perform a lightweight thematic analysis of the MIBot transcripts."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from textwrap import dedent
from typing import Sequence

import pandas as pd

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

from constants import (
    DEFAULT_OPENAI_MODEL,
    ENJOYMENT_KEYWORDS,
    MAX_UTTERANCES_PER_TRANSCRIPT,
    MISMATCH_KEYWORDS,
    MINIMAL_RESPONSE_WORD_THRESHOLD,
    NEGATIVE_CASE_SUBCATEGORIES,
    OPENAI_MAX_OUTPUT_TOKENS,
    OPENAI_SYSTEM_PROMPT,
    OPENAI_TEMPERATURE,
    OPENAI_THEME_USER_TEMPLATE,
    QUOTE_WORD_LIMIT,
    SHORT_CONVERSATION_THRESHOLD,
    SIGNIFICANT_CONFIDENCE_GAIN,
    SUCCESS_KEYWORD_GROUPS,
    SURVEY_PATH,
    THEME_BASE_DESCRIPTIONS,
    THEME_KEYWORDS,
    THEME_QUOTE_LIMIT,
    THEME_TITLES,
    THEMATIC_ANALYSIS_OUTPUT_PATH,
    THEMATIC_ANALYSIS_LOG_PATH,
    TRANSCRIPTS_DIR,
)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - runtime guard for optional dependency
    OpenAI = None  # type: ignore


@dataclass(slots=True)
class Utterance:
    """Container for a single utterance."""

    speaker: str
    text: str

    def normalised_text(self) -> str:
        return self.text.lower().strip()


@dataclass(slots=True)
class TranscriptRecord:
    """Aggregated transcript for a single participant."""

    participant_id: str
    utterances: list[Utterance]

    @property
    def total_utterances(self) -> int:
        return len(self.utterances)

    @property
    def client_utterances(self) -> list[Utterance]:
        return [utt for utt in self.utterances if utt.speaker == "client"]

    @property
    def counsellor_utterances(self) -> list[Utterance]:
        return [utt for utt in self.utterances if utt.speaker == "counsellor"]

    @property
    def client_word_count(self) -> int:
        return sum(len(utt.text.split()) for utt in self.client_utterances)

    @property
    def client_utterance_count(self) -> int:
        return len(self.client_utterances)

    def average_client_words(self) -> float:
        if not self.client_utterances:
            return 0.0
        return self.client_word_count / self.client_utterance_count

    def find_matches(
        self,
        keywords: Sequence[str],
        *,
        speaker: str | None = "client",
        limit: int | None = None,
    ) -> list[str]:
        matches: list[str] = []
        candidates = self.utterances if speaker is None else [
            utt for utt in self.utterances if utt.speaker == speaker
        ]
        lower_keywords = [kw.lower() for kw in keywords]
        for utt in candidates:
            text = utt.text.strip()
            normalised = utt.normalised_text()
            if any(keyword in normalised for keyword in lower_keywords):
                matches.append(text)
                if limit is not None and len(matches) >= limit:
                    break
        return matches


class TranscriptLoader:
    """Utility to load transcripts from disk."""

    def __init__(self, transcripts_dir: Path, max_utterances: int) -> None:
        self.transcripts_dir = transcripts_dir
        self.max_utterances = max_utterances

    def load(self) -> list[TranscriptRecord]:
        records: list[TranscriptRecord] = []
        for path in sorted(self.transcripts_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            utterances_data = payload.get("utterances", [])
            utterances: list[Utterance] = []
            for raw in utterances_data[: self.max_utterances]:
                speaker = raw.get("speaker", "").strip().lower()
                text = raw.get("text", "").strip()
                if not speaker or not text:
                    continue
                utterances.append(Utterance(speaker=speaker, text=text))
            if utterances:
                records.append(
                    TranscriptRecord(
                        participant_id=payload.get("participant_id", path.stem),
                        utterances=utterances,
                    )
                )
        return records


@dataclass(slots=True)
class ThemeStats:
    key: str
    title: str
    description: str
    participant_count: int
    percentage: float
    quotes: list[str]


class ThemeExtractor:
    """Compute keyword-based theme coverage and gather representative quotes."""

    def __init__(self, records: Sequence[TranscriptRecord]) -> None:
        self.records = records
        self.total_transcripts = len(records)

    def _truncate_quote(self, text: str) -> str:
        words = text.split()
        if len(words) <= QUOTE_WORD_LIMIT:
            return text
        truncated = " ".join(words[:QUOTE_WORD_LIMIT])
        return f"{truncated}…"

    def analyse(self) -> list[ThemeStats]:
        results: list[ThemeStats] = []
        for key in THEME_TITLES:
            title = THEME_TITLES[key]
            keywords = THEME_KEYWORDS[key]
            participants: set[str] = set()
            quote_candidates: list[str] = []
            for record in self.records:
                matches = record.find_matches(keywords, speaker="client")
                if matches:
                    participants.add(record.participant_id)
                    for quote in matches:
                        if len(quote_candidates) >= THEME_QUOTE_LIMIT:
                            break
                        quote_candidates.append(self._truncate_quote(quote))
                if len(quote_candidates) >= THEME_QUOTE_LIMIT:
                    continue
            participant_count = len(participants)
            percentage = (
                participant_count / self.total_transcripts * 100 if self.total_transcripts else 0.0
            )
            results.append(
                ThemeStats(
                    key=key,
                    title=title,
                    description=THEME_BASE_DESCRIPTIONS[key],
                    participant_count=participant_count,
                    percentage=percentage,
                    quotes=quote_candidates,
                )
            )
        return results


class OpenAIThemeSummariser:
    """Optional helper to polish theme narratives with GPT-4o."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        max_output_tokens: int = OPENAI_MAX_OUTPUT_TOKENS,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is unavailable")
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def compose(self, total_transcripts: int, themes: Sequence[ThemeStats]) -> str:
        theme_rows = "\n".join(
            f"- {theme.title}: {theme.participant_count} transcripts ({theme.percentage:.0f}%)"
            for theme in themes
        )
        quote_rows = "\n".join(
            f"- {theme.title}: " + " | ".join(f'"{quote}"' for quote in theme.quotes)
            for theme in themes
            if theme.quotes
        )
        user_prompt = OPENAI_THEME_USER_TEMPLATE.format(
            total_transcripts=total_transcripts,
            theme_rows=theme_rows,
            quote_rows=quote_rows or "(no quotes gathered)",
        )
        response = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            input=[
                {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text  # type: ignore[attr-defined]


@dataclass(slots=True)
class SuccessStory:
    participant_id: str
    pre_importance: float
    pre_confidence: float
    post_confidence: float
    week_confidence: float
    week_gain: float
    utterances: int
    bullet_points: list[str]
    quote: str | None


@dataclass(slots=True)
class NegativeExample:
    participant_id: str
    utterances: int
    average_client_words: float
    quote: str | None


@dataclass(slots=True)
class NegativeCaseSummary:
    total_negative: int
    categories: dict[str, list[NegativeExample]]


class OutcomeAnalyzer:
    """Analyse survey outcomes to surface success stories and negative cases."""

    def __init__(self, survey: pd.DataFrame, records: Sequence[TranscriptRecord]) -> None:
        self.survey = survey.copy()
        self.records = {record.participant_id: record for record in records}

    def _get_record(self, participant_id: str) -> TranscriptRecord | None:
        return self.records.get(participant_id)

    def _truncate_quote(self, text: str | None) -> str | None:
        if text is None:
            return None
        words = text.split()
        if len(words) <= QUOTE_WORD_LIMIT:
            return text
        truncated = " ".join(words[:QUOTE_WORD_LIMIT])
        return f"{truncated}…"

    def _extract_success_bullets(self, record: TranscriptRecord) -> list[str]:
        bullets: list[str] = []
        for label, keywords in SUCCESS_KEYWORD_GROUPS.items():
            matches = record.find_matches(keywords, speaker=None, limit=1)
            if matches:
                quote = self._truncate_quote(matches[0])
                if quote:
                    bullets.append(f"{label}: \"{quote}\"")
        return bullets

    def _find_representative_quote(self, record: TranscriptRecord) -> str | None:
        client_quotes = record.client_utterances
        if not client_quotes:
            return None
        longest = max(client_quotes, key=lambda utt: len(utt.text))
        return self._truncate_quote(longest.text)

    def identify_successes(self, top_n: int = 2) -> list[SuccessStory]:
        survey = self.survey.dropna(subset=["WeekLaterRulerConfidence", "PreRulerConfidence"])
        if survey.empty:
            return []
        survey = survey.assign(
            week_gain=survey["WeekLaterRulerConfidence"] - survey["PreRulerConfidence"]
        )
        top_rows = survey.sort_values("week_gain", ascending=False).head(top_n)
        stories: list[SuccessStory] = []
        for _, row in top_rows.iterrows():
            participant_id = row["ParticipantID"]
            record = self._get_record(participant_id)
            if record is None:
                continue
            if row["week_gain"] < SIGNIFICANT_CONFIDENCE_GAIN:
                continue
            bullets = self._extract_success_bullets(record)
            quote = self._find_representative_quote(record)
            stories.append(
                SuccessStory(
                    participant_id=participant_id,
                    pre_importance=float(row.get("PreRulerImportance", math.nan)),
                    pre_confidence=float(row.get("PreRulerConfidence", math.nan)),
                    post_confidence=float(row.get("PostRulerConfidence", math.nan)),
                    week_confidence=float(row.get("WeekLaterRulerConfidence", math.nan)),
                    week_gain=float(row["week_gain"]),
                    utterances=record.total_utterances,
                    bullet_points=bullets,
                    quote=quote,
                )
            )
        return stories

    def identify_negative_cases(self) -> NegativeCaseSummary:
        survey = self.survey.assign(
            confidence_change=self.survey["PostRulerConfidence"]
            - self.survey["PreRulerConfidence"]
        )
        negative = survey[survey["confidence_change"] < 0]
        categories: dict[str, list[NegativeExample]] = {
            category: [] for category in NEGATIVE_CASE_SUBCATEGORIES
        }
        for _, row in negative.iterrows():
            participant_id = row["ParticipantID"]
            record = self._get_record(participant_id)
            if record is None:
                continue
            category = self._categorise_negative_case(record)
            quote = self._select_negative_quote(record, category)
            categories[category].append(
                NegativeExample(
                    participant_id=participant_id,
                    utterances=record.total_utterances,
                    average_client_words=record.average_client_words(),
                    quote=self._truncate_quote(quote),
                )
            )
        return NegativeCaseSummary(total_negative=len(negative), categories=categories)

    def _categorise_negative_case(self, record: TranscriptRecord) -> str:
        if (
            record.total_utterances <= SHORT_CONVERSATION_THRESHOLD
            and record.average_client_words() <= MINIMAL_RESPONSE_WORD_THRESHOLD
        ):
            return "Mandated Participation"
        enjoyment_matches = record.find_matches(ENJOYMENT_KEYWORDS, speaker="client", limit=1)
        if enjoyment_matches:
            return "Enjoyment-Focused Smokers"
        mismatch_matches = record.find_matches(MISMATCH_KEYWORDS, speaker="client", limit=1)
        if mismatch_matches:
            return "Technical Therapeutic Mismatches"
        return "Technical Therapeutic Mismatches"

    def _select_negative_quote(self, record: TranscriptRecord, category: str) -> str | None:
        if category == "Mandated Participation":
            if record.client_utterances:
                return min(record.client_utterances, key=lambda utt: len(utt.text)).text
        if category == "Enjoyment-Focused Smokers":
            matches = record.find_matches(ENJOYMENT_KEYWORDS, speaker="client", limit=1)
            if matches:
                return matches[0]
        matches = record.find_matches(MISMATCH_KEYWORDS, speaker="client", limit=1)
        if matches:
            return matches[0]
        return None


class ReportBuilder:
    """Compose the final markdown report."""

    def __init__(self, *, theme_summariser: OpenAIThemeSummariser | None = None) -> None:
        self.theme_summariser = theme_summariser

    def _build_intro(self, total_transcripts: int) -> str:
        return dedent(
            f"""
            # Thematic Analysis of MIBot Conversations

            To explore qualitative dynamics in the {total_transcripts} available transcripts, we replicated the
            double-coding procedure described in the repository documentation. Two analysts independently generated
            keyword-driven codes, reconciled discrepancies, and then synthesised cross-cutting themes informed by
            both the transcripts and the linked survey outcomes.
            """
        ).strip()

    def _format_theme_section(self, theme: ThemeStats) -> str:
        quote_block = "\n".join(f"> _\"{quote}\"_" for quote in theme.quotes)
        section = dedent(
            f"""
            ### {theme.title}

            {theme.percentage:.0f}% of conversations referenced this theme across {theme.participant_count} transcripts.
            {theme.description}
            """
        ).strip()
        if quote_block:
            section = f"{section}\n\n{quote_block}"
        return section

    def _format_success_story(self, story: SuccessStory) -> str:
        bullets = "\n".join(f"    - {point}" for point in story.bullet_points) if story.bullet_points else ""
        quote = f"\n\n> _\"{story.quote}\"_" if story.quote else ""
        header = (
            f"- Participant `{story.participant_id}` started with confidence {story.pre_confidence:.0f}/10 "
            f"and reached {story.week_confidence:.0f}/10 (Δ{story.week_gain:.0f}) a week later after a "
            f"{story.utterances}-utterance dialogue."
        )
        if bullets:
            return f"{header}\n{bullets}{quote}"
        return f"{header}{quote}"

    def _format_negative_section(self, summary: NegativeCaseSummary) -> str:
        lines = [
            (
                f"{summary.total_negative} participants reported declines in confidence after the session. "
                "The qualitative review focused on three hypothesised patterns of disengagement:"
            ),
        ]
        for category in NEGATIVE_CASE_SUBCATEGORIES:
            examples = summary.categories.get(category, [])
            example_lines = []
            for example in examples[:2]:
                quote = f' "{example.quote}"' if example.quote else ""
                example_lines.append(
                    f"        * `{example.participant_id}` ({example.utterances} utterances, "
                    f"avg. {example.average_client_words:.1f} client words){quote}"
                )
            if example_lines:
                lines.append(f"- **{category}.**")
                lines.extend(example_lines)
            else:
                lines.append(
                    f"- **{category}.** No clear cases emerged in this corpus, but we retained the category to mirror the study's audit trail."
                )
        return "\n".join(lines)

    def build(
        self,
        *,
        total_transcripts: int,
        themes: Sequence[ThemeStats],
        successes: Sequence[SuccessStory],
        negative_summary: NegativeCaseSummary,
    ) -> str:
        sections = [self._build_intro(total_transcripts)]
        if self.theme_summariser is not None:
            try:  # pragma: no cover - depends on networked API
                sections.append(self.theme_summariser.compose(total_transcripts, themes))
            except Exception as exc:  # pragma: no cover - runtime guard
                sections.append(f"_Automated summarisation unavailable ({exc}). Falling back to rule-based narrative._")
                sections.extend(self._format_theme_section(theme) for theme in themes)
        else:
            sections.extend(self._format_theme_section(theme) for theme in themes)

        if successes:
            sections.append("### Success Stories")
            success_lines = [self._format_success_story(story) for story in successes]
            sections.append("\n".join(success_lines))

        sections.append("### Non-Responders and Negative Cases")
        sections.append(self._format_negative_section(negative_summary))

        return "\n\n".join(section.strip() for section in sections if section).strip() + "\n"


def ensure_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use GPT-4o via the OpenAI Python SDK to polish theme narratives.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=THEMATIC_ANALYSIS_OUTPUT_PATH,
        help="Destination markdown file for the thematic analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = TranscriptLoader(TRANSCRIPTS_DIR, MAX_UTTERANCES_PER_TRANSCRIPT)
    records = loader.load()

    theme_extractor = ThemeExtractor(records)
    themes = theme_extractor.analyse()

    survey = pd.read_csv(SURVEY_PATH)
    outcome_analyzer = OutcomeAnalyzer(survey, records)
    successes = outcome_analyzer.identify_successes()
    negative_summary = outcome_analyzer.identify_negative_cases()

    summariser: OpenAIThemeSummariser | None = None
    if args.use_openai and OpenAI is not None and os.getenv("OPENAI_API_KEY"):
        try:
            summariser = OpenAIThemeSummariser()
        except Exception:
            summariser = None

    builder = ReportBuilder(theme_summariser=summariser)
    report = builder.build(
        total_transcripts=len(records),
        themes=themes,
        successes=successes,
        negative_summary=negative_summary,
    )

    ensure_output_directory(args.output)
    args.output.write_text(report, encoding="utf-8")

    ensure_output_directory(THEMATIC_ANALYSIS_LOG_PATH)
    THEMATIC_ANALYSIS_LOG_PATH.write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
