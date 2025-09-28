"""End-to-end thematic analysis pipeline for MIBot transcripts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from openai import OpenAI

from constants import (
    CONVERSATION_SECTION_HEADER,
    DEFAULT_OUTPUT_PATH,
    DIALOGUE_JOINER,
    JSON_INDENT,
    JSON_PARSE_ERROR_TEMPLATE,
    MAX_TRANSCRIPTS_HELP,
    MODEL_HELP,
    OPENAI_MODEL_GPT4O,
    OPENAI_TEMPERATURE_DEFAULT,
    OUTPUT_PATH_HELP,
    PARTICIPANT_ID_PREFIX,
    PIPELINE_DESCRIPTION,
    SUMMARY_CONTEXT_LABEL,
    TEMPERATURE_HELP,
    TEXT_ENCODING,
    THEME_CONTEXT_LABEL,
    THEME_SYNTHESIS_MAX_OUTPUT_TOKENS,
    THEME_SYNTHESIS_PROMPT,
    THEME_SYNTHESIS_USER_PROMPT_PREFIX,
    TRANSCRIPT_SUMMARY_MAX_OUTPUT_TOKENS,
    TRANSCRIPT_SUMMARY_PROMPT,
    TRANSCRIPT_SUMMARY_UTTERANCE_LIMIT,
    TRANSCRIPT_SUMMARY_USER_PROMPT_PREFIX,
    TRANSCRIPTS_DIRECTORY,
    TRANSCRIPTS_DIR_HELP,
)


def parse_model_json(raw_text: str, context: str) -> dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        message = JSON_PARSE_ERROR_TEMPLATE.format(context=context, content=raw_text)
        raise ValueError(message) from exc


@dataclass
class Utterance:
    """A single conversational turn."""

    speaker: str
    text: str

    def to_prompt_line(self) -> str:
        return f"{self.speaker.title()}: {self.text.strip()}".strip()


@dataclass
class Transcript:
    """Full transcript for a participant."""

    participant_id: str
    utterances: Sequence[Utterance]

    def limited_utterances(self, limit: Optional[int]) -> Sequence[Utterance]:
        if limit is None or limit >= len(self.utterances):
            return self.utterances
        return self.utterances[:limit]

    def to_prompt(self, utterance_limit: Optional[int], joiner: str) -> str:
        snippet = [utt.to_prompt_line() for utt in self.limited_utterances(utterance_limit)]
        return joiner.join(snippet)


class TranscriptLoader:
    """Load transcript JSON files into structured objects."""

    def __init__(self, transcripts_dir: Path):
        self.transcripts_dir = transcripts_dir

    def load_transcripts(self, limit: Optional[int] = None) -> List[Transcript]:
        transcripts: List[Transcript] = []
        files = sorted(self.transcripts_dir.glob("*.json"))
        for index, path in enumerate(files):
            if limit is not None and index >= limit:
                break
            with path.open("r", encoding=TEXT_ENCODING) as fp:
                payload = json.load(fp)
            utterances = [
                Utterance(speaker=item["speaker"], text=item["text"]) for item in payload["utterances"]
            ]
            transcripts.append(Transcript(participant_id=payload["participant_id"], utterances=utterances))
        return transcripts


@dataclass
class TranscriptSummary:
    """Structured summary returned by the LLM for a single transcript."""

    participant_id: str
    conversation_overview: str
    client_state: str
    therapeutic_moves: str
    notable_quotes: Sequence[str]
    preliminary_themes: Sequence[str]

    @classmethod
    def from_model_payload(cls, participant_id: str, payload: dict) -> "TranscriptSummary":
        return cls(
            participant_id=participant_id,
            conversation_overview=payload.get("conversation_overview", ""),
            client_state=payload.get("client_state", ""),
            therapeutic_moves=payload.get("therapeutic_moves", ""),
            notable_quotes=payload.get("notable_quotes", []),
            preliminary_themes=payload.get("preliminary_themes", []),
        )

    def to_serializable(self) -> dict:
        return {
            "participant_id": self.participant_id,
            "conversation_overview": self.conversation_overview,
            "client_state": self.client_state,
            "therapeutic_moves": self.therapeutic_moves,
            "notable_quotes": list(self.notable_quotes),
            "preliminary_themes": list(self.preliminary_themes),
        }


class TranscriptSummarizer:
    """Use GPT-4o to summarize each transcript in a structured way."""

    def __init__(
        self,
        client: OpenAI,
        model: str = OPENAI_MODEL_GPT4O,
        temperature: float = OPENAI_TEMPERATURE_DEFAULT,
        utterance_limit: Optional[int] = TRANSCRIPT_SUMMARY_UTTERANCE_LIMIT,
        max_output_tokens: int = TRANSCRIPT_SUMMARY_MAX_OUTPUT_TOKENS,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.utterance_limit = utterance_limit
        self.max_output_tokens = max_output_tokens

    def build_user_prompt(self, transcript: Transcript) -> str:
        conversation_text = transcript.to_prompt(self.utterance_limit, DIALOGUE_JOINER)
        return (
            f"{TRANSCRIPT_SUMMARY_USER_PROMPT_PREFIX}\n"
            f"{PARTICIPANT_ID_PREFIX}{transcript.participant_id}\n"
            f"{CONVERSATION_SECTION_HEADER}" + conversation_text
        )

    def summarize(self, transcript: Transcript) -> TranscriptSummary:
        user_prompt = self.build_user_prompt(transcript)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": [{"type": "text", "text": TRANSCRIPT_SUMMARY_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        content = response.output_text
        payload = parse_model_json(content, SUMMARY_CONTEXT_LABEL)
        return TranscriptSummary.from_model_payload(transcript.participant_id, payload)


@dataclass
class ThematicAnalysisReport:
    """Final structured thematic analysis."""

    themes: Sequence[dict]
    success_cases: Sequence[dict]
    negative_cases: Sequence[dict]

    @classmethod
    def from_payload(cls, payload: dict) -> "ThematicAnalysisReport":
        return cls(
            themes=payload.get("themes", []),
            success_cases=payload.get("success_cases", []),
            negative_cases=payload.get("negative_cases", []),
        )

    def to_serializable(self) -> dict:
        return {
            "themes": list(self.themes),
            "success_cases": list(self.success_cases),
            "negative_cases": list(self.negative_cases),
        }


class ThemeSynthesizer:
    """Aggregate transcript summaries into overarching themes."""

    def __init__(
        self,
        client: OpenAI,
        model: str = OPENAI_MODEL_GPT4O,
        temperature: float = OPENAI_TEMPERATURE_DEFAULT,
        max_output_tokens: int = THEME_SYNTHESIS_MAX_OUTPUT_TOKENS,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def build_user_prompt(self, summaries: Sequence[TranscriptSummary]) -> str:
        payload = [summary.to_serializable() for summary in summaries]
        data = json.dumps(payload, ensure_ascii=False, indent=JSON_INDENT)
        return f"{THEME_SYNTHESIS_USER_PROMPT_PREFIX}\n" + data

    def synthesize(self, summaries: Sequence[TranscriptSummary]) -> ThematicAnalysisReport:
        user_prompt = self.build_user_prompt(summaries)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": [{"type": "text", "text": THEME_SYNTHESIS_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        payload = parse_model_json(response.output_text, THEME_CONTEXT_LABEL)
        return ThematicAnalysisReport.from_payload(payload)


def save_report(report: ThematicAnalysisReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding=TEXT_ENCODING) as fp:
        json.dump(report.to_serializable(), fp, ensure_ascii=False, indent=JSON_INDENT)


def run_pipeline(
    transcripts_dir: Path = TRANSCRIPTS_DIRECTORY,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model: str = OPENAI_MODEL_GPT4O,
    temperature: float = OPENAI_TEMPERATURE_DEFAULT,
    max_transcripts: Optional[int] = None,
) -> ThematicAnalysisReport:
    client = OpenAI()
    loader = TranscriptLoader(transcripts_dir)
    transcripts = loader.load_transcripts(limit=max_transcripts)
    summarizer = TranscriptSummarizer(client, model=model, temperature=temperature)
    summaries = [summarizer.summarize(transcript) for transcript in transcripts]
    synthesizer = ThemeSynthesizer(client, model=model, temperature=temperature)
    report = synthesizer.synthesize(summaries)
    save_report(report, output_path)
    return report


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PIPELINE_DESCRIPTION)
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=TRANSCRIPTS_DIRECTORY,
        help=TRANSCRIPTS_DIR_HELP,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=OUTPUT_PATH_HELP,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OPENAI_MODEL_GPT4O,
        help=MODEL_HELP,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=OPENAI_TEMPERATURE_DEFAULT,
        help=TEMPERATURE_HELP,
    )
    parser.add_argument(
        "--max-transcripts",
        type=int,
        default=None,
        help=MAX_TRANSCRIPTS_HELP,
    )
    return parser.parse_args(args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = parse_args(argv)
    report = run_pipeline(
        transcripts_dir=arguments.transcripts_dir,
        output_path=arguments.output_path,
        model=arguments.model,
        temperature=arguments.temperature,
        max_transcripts=arguments.max_transcripts,
    )
    print(json.dumps(report.to_serializable(), ensure_ascii=False, indent=JSON_INDENT))


if __name__ == "__main__":
    main()
