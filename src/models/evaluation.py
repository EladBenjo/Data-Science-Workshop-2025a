"""Evaluation utilities to mirror the notebook's metrics and plots."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model


def plot_training_history(history) -> None:
    """Plot training vs. validation loss."""
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, "o-", label="Training Loss")
    plt.plot(epochs, val_loss, "s--", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)


def evaluate_model(
    model: Model,
    speech_data: np.ndarray,
    inflation_data: np.ndarray,
    target_data: np.ndarray,
    dataset_name: str = "Dataset",
) -> dict[str, float]:
    """Compute MSE, MAE, and R² for a model and dataset."""
    predictions = model.predict([speech_data, inflation_data])
    mse = mean_squared_error(target_data, predictions)
    mae = mean_absolute_error(target_data, predictions)
    r2 = r2_score(target_data, predictions)
    print(f"\n{dataset_name} Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    return {"MSE": mse, "MAE": mae, "R²": r2}


def evaluate_mse_over_time(
    predictions: np.ndarray,
    targets: pd.Series,
    dates: pd.Series,
    label: str = "Model",
) -> pd.DataFrame:
    """Return the squared error for each date, matching the notebook helper."""
    predictions_flat = predictions.flatten()
    targets_flat = targets.values.flatten()
    squared_errors = (predictions_flat - targets_flat) ** 2
    return pd.DataFrame(
        {
            "Date": dates,
            "Squared_Error": squared_errors,
            "Target": targets_flat,
            "Prediction": predictions_flat,
            "Model": label,
        }
    )


__all__ = ["plot_training_history", "evaluate_model", "evaluate_mse_over_time"]
