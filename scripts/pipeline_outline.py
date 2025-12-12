"""High-level outline for reproducing the notebook workflow in code.

This script does not download data automatically. It demonstrates how to glue
``src`` components together once the scraped speeches CSV and FRED CSV are
available locally.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

from src.data.loading import aggregate_monthly_speeches, load_fred_expectations, load_speeches
from src.features.sentiment_features import add_sentiment_columns
from src.features.text_features import add_text_features
from src.models.architectures import build_gru_regressor
from src.models.evaluation import evaluate_model
from src.models.training import set_global_seed, train_with_callbacks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the speech + inflation pipeline.")
    parser.add_argument("speeches_csv", type=Path, help="Path to the scraped speeches CSV.")
    parser.add_argument("fred_csv", type=Path, help="Path to the FRED expectations CSV (e.g., EXPINF1YR).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(42)

    nlp = spacy.load("en_core_web_sm")
    speeches = load_speeches(str(args.speeches_csv))
    speeches = add_text_features(speeches, text_column="speech_text", nlp=nlp)

    if "sentiment_results" in speeches.columns:
        speeches = add_sentiment_columns(speeches, "sentiment_results")

    speeches = speeches.dropna(subset=["sentiment_measure"])
    monthly = aggregate_monthly_speeches(speeches)

    fred = load_fred_expectations(str(args.fred_csv))
    fred["DATE"] = fred["DATE"].dt.to_period("M").dt.to_timestamp()

    dataset = monthly.merge(fred, left_on="aggregated_month", right_on="DATE", how="inner")
    feature_cols = [
        "target_sentence_ratio_avg",
        "sentiment_measure_avg",
        "speech_length_avg",
        "role_encoded",
        "dominant_sentiment_encoded",
    ]
    speech_features = dataset[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    speech_features = np.expand_dims(speech_features, axis=1)

    macro_cols = [col for col in dataset.columns if col not in feature_cols + ["aggregated_month", "DATE", "EXPINF1YR"]]
    macro_features = dataset[macro_cols].fillna(0).to_numpy(dtype=np.float32)
    targets = dataset[["EXPINF1YR"]].to_numpy(dtype=np.float32)

    # Simple chronological split: last 20% of months for validation
    split_index = int(len(dataset) * 0.8)
    train_speech, val_speech = speech_features[:split_index], speech_features[split_index:]
    train_macro, val_macro = macro_features[:split_index], macro_features[split_index:]
    train_target, val_target = targets[:split_index], targets[split_index:]

    model = build_gru_regressor(
        speech_feature_count=speech_features.shape[2],
        macro_feature_count=macro_features.shape[1],
    )
    history = train_with_callbacks(
        model,
        train_inputs=[train_speech, train_macro],
        train_target=train_target,
        val_inputs=[val_speech, val_macro],
        val_target=val_target,
        epochs=5,
    )

    evaluate_model(model, train_speech, train_macro, train_target, dataset_name="Train")
    evaluate_model(model, val_speech, val_macro, val_target, dataset_name="Validation")


if __name__ == "__main__":
    main()
