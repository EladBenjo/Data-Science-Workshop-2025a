"""Data loading and aggregation utilities for speech and inflation datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


def load_speeches(path: str, date_column: str = "speech_date") -> pd.DataFrame:
    """Load the scraped speeches CSV.

    Parameters
    ----------
    path:
        Path to the CSV file produced by the scraping notebook.
    date_column:
        Column name containing the speech date.
    """
    df = pd.read_csv(path)
    df[date_column] = pd.to_datetime(df[date_column])
    return df


def load_fred_expectations(path: str, date_column: str = "DATE") -> pd.DataFrame:
    """Load FRED inflation expectations data.

    The notebook originally used the EXPINF1YR series. This helper simply
    standardizes the date column to ``datetime`` and leaves the remaining columns
    untouched.
    """
    df = pd.read_csv(path)
    df[date_column] = pd.to_datetime(df[date_column])
    return df


@dataclass
class MonthlyAggregates:
    """Container for monthly speech aggregates.

    Attributes
    ----------
    aggregated_month:
        Timestamp for the month that speeches are aligned to.
    role_encoded:
        1 for chairman, -1 for vice chairman, ``None`` otherwise.
    speech_length_avg:
        Mean speech length for the month.
    target_sentence_ratio_avg:
        Average ratio of target sentences for the month.
    sentiment_measure_avg:
        Average hawkish-dovish score for the month.
    dominant_sentiment_encoded:
        1 for hawkish, -1 for dovish, ``None`` otherwise.
    """

    aggregated_month: pd.Timestamp
    role_encoded: Optional[int]
    speech_length_avg: float
    target_sentence_ratio_avg: float
    sentiment_measure_avg: float
    dominant_sentiment_encoded: Optional[int]


ROLE_PRIORITIES = ("Chairman", "Vice Chairman")
SENTIMENT_PRIORITIES = ("Hawkish", "Dovish")


def _mode_with_priority(values: Iterable[str], priority: tuple[str, ...]) -> Optional[str]:
    filtered = [value for value in values if value in priority]
    if not filtered:
        return None
    counts = pd.Series(filtered).value_counts()
    return counts.idxmax()


def encode_role(role: Optional[str]) -> Optional[int]:
    if role is None:
        return None
    return 1 if role == "Chairman" else -1 if role == "Vice Chairman" else None


def encode_sentiment_label(label: Optional[str]) -> Optional[int]:
    if label is None:
        return None
    return 1 if label == "Hawkish" else -1 if label == "Dovish" else None


def aggregate_monthly_speeches(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate speech-level features to the following month.

    The notebook aligns each speech with the inflation expectations of the
    *following* month. This helper mirrors that behavior and produces the same
    encoded features used in the modeling section.
    """
    if "speech_date" not in df.columns:
        raise KeyError("DataFrame must include a 'speech_date' column before aggregation.")

    working = df.copy()
    working["speech_date"] = pd.to_datetime(working["speech_date"])
    working["aggregated_month"] = (working["speech_date"] + pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp()

    grouped = working.groupby("aggregated_month")

    def _most_frequent(series: pd.Series, priority: tuple[str, ...]) -> Optional[str]:
        return _mode_with_priority(series.dropna().tolist(), priority)

    aggregated = grouped.agg(
        role_encoded=("role", lambda roles: encode_role(_most_frequent(roles, ROLE_PRIORITIES))),
        speech_length_avg=("speech_length", "mean"),
        target_sentence_ratio_avg=("target_sentence_ratio", "mean"),
        sentiment_measure_avg=("sentiment_measure", "mean"),
        dominant_sentiment_encoded=(
            "dominant_sentiment",
            lambda sentiments: encode_sentiment_label(_most_frequent(sentiments, SENTIMENT_PRIORITIES)),
        ),
    ).reset_index()

    aggregated["aggregated_month"] = aggregated["aggregated_month"].dt.to_period("M").dt.to_timestamp()
    return aggregated

