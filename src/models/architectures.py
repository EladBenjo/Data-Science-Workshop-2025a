"""Neural network architectures lifted from the notebook's modeling section."""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, GRU, Input, LSTM, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_lstm_regressor(
    speech_feature_count: int = 9,
    macro_feature_count: int = 7,
    lstm_units: Tuple[int, int] = (32, 16),
    dropout: float = 0.2,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001,
) -> Model:
    """Construct the regularized LSTM architecture used in the notebook."""
    speech_input = Input(shape=(None, speech_feature_count), name="speech_input")
    masked = Masking(mask_value=0.0)(speech_input)
    lstm_out = LSTM(lstm_units[0], return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(masked)
    lstm_out = LSTM(lstm_units[1], dropout=dropout, recurrent_dropout=dropout)(lstm_out)

    macro_input = Input(shape=(macro_feature_count,), name="macro_input")
    macro_dense = Dense(16, activation="tanh", kernel_regularizer=l2(l2_reg))(macro_input)

    merged = Concatenate()([lstm_out, macro_dense])
    dense1 = Dense(32, activation="tanh", kernel_regularizer=l2(l2_reg))(merged)
    dense1 = Dropout(dropout)(dense1)
    dense2 = Dense(16, activation="tanh", kernel_regularizer=l2(l2_reg))(dense1)
    dense2 = Dropout(dropout)(dense2)
    output = Dense(1, name="output")(dense2)

    model = Model(inputs=[speech_input, macro_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_gru_regressor(
    speech_feature_count: int = 9,
    macro_feature_count: int = 7,
    gru_units: Tuple[int, int] = (32, 32),
    dropout: float = 0.2,
    l2_reg: float = 0.001,
    learning_rate: float = 0.008,
) -> Model:
    """Construct the GRU architecture that performed best in the notebook."""
    speech_input = Input(shape=(None, speech_feature_count), name="speech_input")
    masked = Masking(mask_value=0.0)(speech_input)
    gru_out = GRU(gru_units[0], return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(masked)
    gru_out = GRU(gru_units[1], dropout=dropout, recurrent_dropout=dropout)(gru_out)

    macro_input = Input(shape=(macro_feature_count,), name="macro_input")
    macro_dense = Dense(16, activation="tanh", kernel_regularizer=l2(l2_reg))(macro_input)

    merged = Concatenate()([gru_out, macro_dense])
    dense1 = Dense(32, activation="tanh", kernel_regularizer=l2(l2_reg))(merged)
    dense1 = Dropout(dropout)(dense1)
    dense2 = Dense(32, activation="tanh", kernel_regularizer=l2(l2_reg))(dense1)
    dense2 = Dropout(dropout)(dense2)
    output = Dense(1, name="output")(dense2)

    model = Model(inputs=[speech_input, macro_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model

