"""Tools for performing thematic analysis on MIBot transcripts.

This module orchestrates transcript loading, chunking, and qualitative coding
via the OpenAI Responses API. It provides two public entry points:

1. ``run_conversation_analyses`` for generating conversation-level thematic
   codings.
2. ``synthesize_corpus_themes`` for summarizing the conversation findings into
   a corpus-level report.

Example usage::

    conda activate py311
    export OPENAI_API_KEY=...
    python -m analysis.thematic_analysis --max-conversations 5

The script writes JSON and Markdown artefacts that mirror the manual thematic
analysis described in the accompanying manuscript.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from openai import OpenAI

from constants import (
    CLIENT_LABEL,
    CONVERSATIONS_CSV_PATH,
    DEFAULT_ANALYSIS_OUTPUT_PATH,
    DEFAULT_SYNTHESIS_OUTPUT_PATH,
    MAX_TURNS_PER_CHUNK,
    PARTICIPANT_ID_COLUMN,
    SPEAKER_COLUMN,
    THEME_EXTRACTION_SYSTEM_PROMPT,
    THEME_EXTRACTION_USER_PROMPT_TEMPLATE,
    THEME_SYNTHESIS_SYSTEM_PROMPT,
    THEME_SYNTHESIS_USER_PROMPT_TEMPLATE,
    THEMATIC_ANALYSIS_MODEL,
    THEMATIC_ANALYSIS_TEMPERATURE,
    THEMATIC_SYNTHESIS_MODEL,
    THEMATIC_SYNTHESIS_TEMPERATURE,
    TURN_DELIMITER,
    UTTERANCE_COLUMN,
    UTTERANCE_INDEX_COLUMN,
    VOLLEY_COLUMN,
)


@dataclass
class ConversationTurn:
    """Represents a single utterance in the transcript."""

    speaker: str
    text: str
    volley: Optional[int]
    utterance_index: Optional[int]

    def formatted(self) -> str:
        """Return a formatted string with identifiers for qualitative coding."""

        speaker_label = "Client" if self.speaker == CLIENT_LABEL else "MIBot"
        volley_text = f"V{self.volley}" if self.volley is not None else "V?"
        utterance_text = (
            f"U{self.utterance_index}" if self.utterance_index is not None else "U?"
        )
        return f"{speaker_label} ({volley_text}-{utterance_text}): {self.text.strip()}"


@dataclass
class ConversationTranscript:
    """Container for an ordered list of conversation turns."""

    participant_id: str
    turns: List[ConversationTurn]

    def chunked_text(self, max_turns: int = MAX_TURNS_PER_CHUNK) -> str:
        """Combine turns into manageable blocks to control prompt length."""

        chunks: List[str] = []
        for start in range(0, len(self.turns), max_turns):
            subset = self.turns[start : start + max_turns]
            chunk_body = "\n".join(turn.formatted() for turn in subset)
            chunk_header = f"Participant {self.participant_id} — Turns {start + 1}-{start + len(subset)}"
            chunks.append(f"{chunk_header}\n{chunk_body}")
        return TURN_DELIMITER.join(chunks)


class ThematicAnalyzer:
    """Encapsulates OpenAI calls for conversation-level thematic analysis."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = THEMATIC_ANALYSIS_MODEL,
        temperature: float = THEMATIC_ANALYSIS_TEMPERATURE,
    ) -> None:
        self.client = client or OpenAI()
        self.model = model
        self.temperature = temperature

    def analyze_conversation(self, transcript: ConversationTranscript) -> Dict[str, Any]:
        """Run the thematic extraction prompt on a single conversation."""

        user_prompt = THEME_EXTRACTION_USER_PROMPT_TEMPLATE.format(
            transcript=transcript.chunked_text()
        )
        response = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=[
                {"role": "system", "content": THEME_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return self._parse_json_response(response)

    @staticmethod
    def _parse_json_response(response: Any) -> Dict[str, Any]:
        """Extract JSON content from a Responses API payload."""

        try:
            if hasattr(response, "output_text") and response.output_text:
                return json.loads(response.output_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError("Failed to decode JSON output_text") from exc

        try:
            first_chunk = response.output[0].content[0]
            text_payload = getattr(first_chunk, "text")
            if isinstance(text_payload, str):
                return json.loads(text_payload)
            if isinstance(text_payload, list):
                combined = "".join(segment.get("text", "") for segment in text_payload)
                return json.loads(combined)
        except (AttributeError, IndexError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("OpenAI response did not contain valid JSON") from exc

        raise ValueError("OpenAI response did not include recognizable content")


class ThematicSynthesizer:
    """Aggregates conversation analyses into a corpus-level report."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = THEMATIC_SYNTHESIS_MODEL,
        temperature: float = THEMATIC_SYNTHESIS_TEMPERATURE,
    ) -> None:
        self.client = client or OpenAI()
        self.model = model
        self.temperature = temperature

    def synthesize(self, analyses: Iterable[Dict[str, Any]]) -> str:
        """Produce a narrative thematic synthesis report."""

        serialized = json.dumps(list(analyses), ensure_ascii=False, indent=2)
        user_prompt = THEME_SYNTHESIS_USER_PROMPT_TEMPLATE.format(analyses=serialized)
        response = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=[
                {"role": "system", "content": THEME_SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        try:
            return response.output[0].content[0].text
        except (AttributeError, IndexError):  # pragma: no cover - defensive
            raise ValueError("OpenAI response did not contain text content")


def load_transcripts(csv_path: str = str(CONVERSATIONS_CSV_PATH)) -> List[ConversationTranscript]:
    """Load transcripts from the canonical conversations CSV file."""

    dataframe = pd.read_csv(csv_path)
    transcripts: List[ConversationTranscript] = []

    for participant_id, group in dataframe.groupby(PARTICIPANT_ID_COLUMN):
        sorted_group = group.sort_values(by=UTTERANCE_INDEX_COLUMN, kind="mergesort")
        turns = [
            ConversationTurn(
                speaker=row[SPEAKER_COLUMN],
                text=str(row[UTTERANCE_COLUMN]),
                volley=_safe_int(row.get(VOLLEY_COLUMN)),
                utterance_index=_safe_int(row.get(UTTERANCE_INDEX_COLUMN)),
            )
            for _, row in sorted_group.iterrows()
        ]
        transcripts.append(ConversationTranscript(participant_id=str(participant_id), turns=turns))

    return transcripts


def _safe_int(value: Any) -> Optional[int]:
    """Convert values from pandas into optional integers."""

    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def run_conversation_analyses(
    transcripts: Iterable[ConversationTranscript],
    analyzer: Optional[ThematicAnalyzer] = None,
) -> List[Dict[str, Any]]:
    """Iterate through transcripts and generate thematic codings."""

    analyzer = analyzer or ThematicAnalyzer()
    analyses: List[Dict[str, Any]] = []
    for transcript in transcripts:
        analysis = analyzer.analyze_conversation(transcript)
        analyses.append(analysis)
    return analyses


def synthesize_corpus_themes(
    analyses: Iterable[Dict[str, Any]],
    synthesizer: Optional[ThematicSynthesizer] = None,
) -> str:
    """Summarize conversation analyses into a single thematic report."""

    synthesizer = synthesizer or ThematicSynthesizer()
    return synthesizer.synthesize(list(analyses))


def save_json(data: Any, destination: str) -> None:
    """Persist JSON data to disk with UTF-8 encoding."""

    with open(destination, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)


def save_text(report: str, destination: str) -> None:
    """Save textual reports (e.g., Markdown) to disk."""

    with open(destination, "w", encoding="utf-8") as text_file:
        text_file.write(report)


def parse_arguments() -> argparse.Namespace:
    """Create a CLI for end-to-end thematic analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversations",
        type=str,
        default=str(CONVERSATIONS_CSV_PATH),
        help="Path to the conversations CSV file.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Optional limit on the number of conversations to process.",
    )
    parser.add_argument(
        "--analysis-output",
        type=str,
        default=str(DEFAULT_ANALYSIS_OUTPUT_PATH),
        help="Destination JSON file for conversation-level analyses.",
    )
    parser.add_argument(
        "--synthesis-output",
        type=str,
        default=str(DEFAULT_SYNTHESIS_OUTPUT_PATH),
        help="Destination Markdown file for the corpus-level synthesis.",
    )
    parser.add_argument(
        "--skip-synthesis",
        action="store_true",
        help="Only compute conversation-level analyses without corpus synthesis.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for command-line execution."""

    args = parse_arguments()
    transcripts = load_transcripts(args.conversations)
    if args.max_conversations is not None:
        transcripts = transcripts[: args.max_conversations]

    analyzer = ThematicAnalyzer()
    analyses = run_conversation_analyses(transcripts, analyzer)
    save_json(analyses, args.analysis_output)

    if args.skip_synthesis:
        return

    synthesizer = ThematicSynthesizer(client=analyzer.client)
    report = synthesize_corpus_themes(analyses, synthesizer)
    save_text(report, args.synthesis_output)


if __name__ == "__main__":
    main()
