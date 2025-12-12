"""Sentiment helpers derived from the notebook's HuggingFace workflow."""
from __future__ import annotations

import ast
from typing import Iterable, Sequence

import pandas as pd

LABEL_MAP = {"LABEL_0": "Dovish", "LABEL_1": "Hawkish", "LABEL_2": "Neutral"}


def classify_sentences(sentences: Sequence[str], classifier) -> list[dict]:
    """Run a Transformers pipeline on a list of sentences."""
    if not sentences:
        return []
    results = classifier(sentences, batch_size=128, truncation=False)
    return [{"label": res["label"], "score": res["score"]} for res in results]


def _ensure_result_list(sentiment_results: Iterable | str | None) -> list[dict]:
    if sentiment_results is None:
        return []
    if isinstance(sentiment_results, str):
        try:
            parsed = ast.literal_eval(sentiment_results)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            return []
    if isinstance(sentiment_results, list):
        return sentiment_results
    return []


def compute_sentiment_measure(sentiment_results: Iterable | str | None) -> float | None:
    """Compute the hawkish-dovish ratio used in the notebook."""
    parsed = _ensure_result_list(sentiment_results)
    label_counts = {"LABEL_0": 0, "LABEL_1": 0, "LABEL_2": 0}
    for res in parsed:
        if isinstance(res, dict) and "label" in res:
            label_counts[res["label"]] += 1
    total = sum(label_counts.values())
    if total == 0:
        return None
    return (label_counts["LABEL_1"] - label_counts["LABEL_0"]) / total


def determine_dominant_sentiment(sentiment_results: Iterable | str | None) -> str | None:
    """Replicate the notebook's dominant sentiment logic."""
    parsed = _ensure_result_list(sentiment_results)
    label_counts = {"LABEL_0": 0, "LABEL_1": 0, "LABEL_2": 0}
    for res in parsed:
        if isinstance(res, dict) and "label" in res:
            label_counts[res["label"]] += 1
    total_sentences = sum(label_counts.values())
    if total_sentences == 0:
        return None
    neutral_ratio = label_counts["LABEL_2"] / total_sentences
    if neutral_ratio >= 0.75:
        return "Neutral"
    dominant_label = max(["LABEL_0", "LABEL_1"], key=lambda label: label_counts[label])
    return LABEL_MAP[dominant_label]


def add_sentiment_columns(df: pd.DataFrame, results_column: str) -> pd.DataFrame:
    """Add the sentiment measure and dominant sentiment columns to a DataFrame."""
    df = df.copy()
    df["sentiment_measure"] = df[results_column].apply(compute_sentiment_measure)
    df["dominant_sentiment"] = df[results_column].apply(determine_dominant_sentiment)
    return df


__all__ = [
    "LABEL_MAP",
    "classify_sentences",
    "compute_sentiment_measure",
    "determine_dominant_sentiment",
    "add_sentiment_columns",
]
