"""Training helpers for time-series speech models."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model


def set_global_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_with_callbacks(
    model: Model,
    train_inputs: Sequence[np.ndarray],
    train_target: np.ndarray,
    val_inputs: Sequence[np.ndarray],
    val_target: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 8,
    early_stopping_patience: int = 5,
    reduce_lr_patience: int = 3,
    min_lr: float = 1e-6,
):
    """Fit a model with early stopping and learning-rate scheduling."""
    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=reduce_lr_patience, min_lr=min_lr)

    history = model.fit(
        train_inputs,
        train_target,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_inputs, val_target),
        callbacks=[early_stopping, reduce_lr],
    )
    return history


__all__ = ["set_global_seed", "train_with_callbacks"]
