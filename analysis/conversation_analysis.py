"""Compute summary statistics for conversation dynamics and outcomes."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERSATIONS_PATH = REPO_ROOT / "conversations.csv"
SURVEY_PATH = REPO_ROOT / "data.csv"
OUTPUT_PATH = Path(__file__).with_name("conversation_metrics.json")

# Regex to capture self-reported durations in minutes from open-text feedback.
MINUTES_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:minutes|min|minute)", re.IGNORECASE)


@dataclass
class MetricSummary:
    mean: float
    median: float
    std: float | None
    minimum: int | float
    maximum: int | float

    @classmethod
    def from_series(cls, values: Iterable[float]) -> "MetricSummary":
        data = list(values)
        if not data:
            return cls(
                mean=float("nan"),
                median=float("nan"),
                std=None,
                minimum=float("nan"),
                maximum=float("nan"),
            )
        return cls(
            mean=float(mean(data)),
            median=float(pd.Series(data).median()),
            std=float(pstdev(data)) if len(data) > 1 else None,
            minimum=float(min(data)),
            maximum=float(max(data)),
        )

    def rounded(self, digits: int = 2) -> "MetricSummary":
        def _round(value: float | None) -> float | None:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return value
            return round(value, digits)

        return MetricSummary(
            mean=_round(self.mean),
            median=_round(self.median),
            std=_round(self.std) if self.std is not None else None,
            minimum=_round(self.minimum),
            maximum=_round(self.maximum),
        )


def load_conversation_metrics() -> pd.DataFrame:
    conversations = pd.read_csv(CONVERSATIONS_PATH)
    conversations["word_count"] = (
        conversations["Utterance"].fillna("").astype(str).str.split().map(len)
    )

    def _mean_word_count(df: pd.DataFrame, speaker: str) -> float:
        subset = df[df["Speaker"] == speaker]["word_count"]
        return subset.mean() if not subset.empty else float("nan")

    metrics: List[Dict[str, float]] = []
    for participant_id, group in conversations.groupby("ParticipantID"):
        counsellor_count = int((group["Speaker"] == "counsellor").sum())
        client_count = int((group["Speaker"] == "client").sum())
        metrics.append(
            {
                "ParticipantID": participant_id,
                "total_utterances": int(len(group)),
                "counsellor_utterances": counsellor_count,
                "client_utterances": client_count,
                "words_per_counsellor_utterance": float(_mean_word_count(group, "counsellor")),
                "words_per_client_utterance": float(_mean_word_count(group, "client")),
            }
        )

    return pd.DataFrame(metrics)


def parse_self_reported_durations(surveys: pd.DataFrame) -> List[Tuple[str, float]]:
    durations: List[Tuple[str, float]] = []
    for _, row in surveys.iterrows():
        for col in ("FeedbackQ1", "FeedbackQ2", "FeedbackQ3"):
            text = row.get(col)
            if not isinstance(text, str):
                continue
            for match in MINUTES_PATTERN.findall(text):
                try:
                    durations.append((row["ParticipantID"], float(match)))
                except ValueError:
                    continue
    return durations


def compute_confidence_correlation(metrics: pd.DataFrame, surveys: pd.DataFrame) -> float:
    merged = metrics.merge(
        surveys[["ParticipantID", "PreRulerConfidence", "PostRulerConfidence"]],
        on="ParticipantID",
        how="left",
    )
    merged["confidence_change"] = (
        merged["PostRulerConfidence"] - merged["PreRulerConfidence"]
    )
    return float(merged[["total_utterances", "confidence_change"]].corr().iloc[0, 1])


def compute_gain_thresholds(metrics: pd.DataFrame, surveys: pd.DataFrame) -> Dict[str, object]:
    merged = metrics.merge(
        surveys[["ParticipantID", "PreRulerConfidence", "PostRulerConfidence"]],
        on="ParticipantID",
        how="left",
    )
    merged["confidence_change"] = (
        merged["PostRulerConfidence"] - merged["PreRulerConfidence"]
    )

    substantial_threshold = 2
    short_threshold = 60
    long_threshold = 130

    share_substantial_short = float(
        (merged[merged["total_utterances"] < short_threshold]["confidence_change"] >= substantial_threshold)
        .mean()
    )
    share_substantial_long = float(
        (merged[merged["total_utterances"] >= long_threshold]["confidence_change"] >= substantial_threshold)
        .mean()
    )

    # Summarise confidence change by 10-utterance bins (minimum 5 conversations per bin).
    bin_edges = list(range(30, int(metrics["total_utterances"].max()) + 10, 10))
    merged["utterance_bin"] = pd.cut(
        merged["total_utterances"], bins=bin_edges, right=False
    )
    bin_summary = (
        merged.groupby("utterance_bin", observed=False)
        .agg(
            mean_change=("confidence_change", "mean"),
            median_change=("confidence_change", "median"),
            count=("ParticipantID", "count"),
        )
        .dropna(subset=["count"])
    )

    bin_summary = bin_summary[bin_summary["count"] > 0]

    eligible_bins = bin_summary[bin_summary["count"] >= 5]
    if not eligible_bins.empty:
        peak_mean = float(eligible_bins["mean_change"].max())
        close_to_peak = eligible_bins[
            eligible_bins["mean_change"] >= (peak_mean - 0.5)
        ]
        left_edges = [int(interval.left) for interval in close_to_peak.index]
        right_edges = [int(interval.right) for interval in close_to_peak.index]
        optimal_window = (
            min(left_edges),
            max(right_edges),
        )
    else:
        optimal_window = (None, None)

    return {
        "substantial_threshold": substantial_threshold,
        "short_threshold": short_threshold,
        "long_threshold": long_threshold,
        "share_substantial_short": share_substantial_short,
        "share_substantial_long": share_substantial_long,
        "bin_summary": {
            f"[{int(interval.left)}, {int(interval.right)})": {
                "mean_change": float(row["mean_change"]) if not math.isnan(row["mean_change"]) else float("nan"),
                "median_change": float(row["median_change"])
                if not math.isnan(row["median_change"])
                else float("nan"),
                "count": int(row["count"]),
            }
            for interval, row in bin_summary.iterrows()
        },
        "optimal_window": optimal_window,
    }


def main() -> None:
    metrics = load_conversation_metrics()
    surveys = pd.read_csv(SURVEY_PATH)

    summaries = {
        name: MetricSummary.from_series(metrics[name]).rounded(2)
        for name in (
            "total_utterances",
            "counsellor_utterances",
            "client_utterances",
            "words_per_counsellor_utterance",
            "words_per_client_utterance",
        )
    }

    counsellor_share = float(
        metrics["counsellor_utterances"].sum()
        / metrics["total_utterances"].sum()
    )

    durations = parse_self_reported_durations(surveys)
    duration_summary = MetricSummary.from_series([value for _, value in durations]).rounded(2)
    duration_summary_dict = asdict(duration_summary)
    duration_summary_dict["count"] = len(durations)

    correlation = compute_confidence_correlation(metrics, surveys)
    thresholds = compute_gain_thresholds(metrics, surveys)

    result = {
        "metric_summaries": {name: asdict(summary) for name, summary in summaries.items()},
        "counsellor_utterance_share": round(counsellor_share, 4),
        "duration_summary": duration_summary_dict,
        "confidence_change_correlation": round(correlation, 4),
        "gain_thresholds": thresholds,
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
