"""Feedforward MLP model definition and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from aerqualitas.modeling.baseline import RegressionMetrics, regression_metrics
from aerqualitas.modeling.split import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True, slots=True)
class MLPTrainingConfig:
    """Training hyperparameters for the AerQualitas MLP."""

    hidden_1: int = 64
    hidden_2: int = 32
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 200
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 6
    random_seed: int = 42

    def validate(self) -> None:
        """Validate hyperparameter ranges."""
        if self.hidden_1 <= 0 or self.hidden_2 <= 0:
            raise ValueError("Las capas ocultas deben tener al menos una neurona.")
        if self.learning_rate <= 0:
            raise ValueError("La tasa de aprendizaje debe ser positiva.")
        if self.batch_size <= 0:
            raise ValueError("El tamaño de mini-batch debe ser positivo.")
        if self.max_epochs <= 0:
            raise ValueError("El número máximo de épocas debe ser positivo.")
        if self.early_stopping_patience <= 0:
            raise ValueError("La paciencia de early stopping debe ser positiva.")
        if self.reduce_lr_patience <= 0:
            raise ValueError("La paciencia de reducción de LR debe ser positiva.")


def _import_tensorflow() -> Any:
    """Import TensorFlow lazily so data-only modules remain lightweight."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow no está instalado. Ejecute `pip install -r requirements-dev.txt`."
        ) from exc
    return tf


def configure_tensorflow(seed: int) -> Any:
    """Set TensorFlow and NumPy seeds and request deterministic operations."""
    tf = _import_tensorflow()
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    return tf


def build_mlp(input_dim: int, config: MLPTrainingConfig) -> Any:
    """Build and compile the feedforward regression network."""
    if input_dim <= 0:
        raise ValueError("input_dim debe ser un entero positivo.")
    config.validate()
    tf = configure_tensorflow(config.random_seed)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="entrada_preprocesada"),
            tf.keras.layers.Dense(
                config.hidden_1,
                activation="relu",
                name="dense_64_relu",
            ),
            tf.keras.layers.Dense(
                config.hidden_2,
                activation="relu",
                name="dense_32_relu",
            ),
            tf.keras.layers.Dense(1, activation="linear", name="pm25_estimado"),
        ],
        name="AerQualitas_MLP",
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_callbacks(
    model_path: Path,
    config: MLPTrainingConfig,
) -> list[Any]:
    """Create callbacks for stable training and best-model persistence."""
    tf = _import_tensorflow()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            min_delta=0.01,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.reduce_lr_patience,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def fit_mlp(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    model_path: Path,
    config: MLPTrainingConfig,
) -> Any:
    """Train the MLP using explicit chronological validation data."""
    callbacks = build_callbacks(model_path, config)
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )


def predict_array(model: Any, x: np.ndarray) -> np.ndarray:
    """Return flattened finite predictions from a fitted Keras model."""
    predicted = np.asarray(model.predict(x, verbose=0), dtype=float).reshape(-1)
    if not np.isfinite(predicted).all():
        raise ValueError("El modelo produjo predicciones no finitas.")
    return predicted


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """Evaluate regression predictions with the common project metrics."""
    return regression_metrics(y_true, y_pred)


def prediction_frame(
    validation: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Create an auditable validation prediction table."""
    if len(validation) != len(predictions):
        raise ValueError("El número de predicciones no coincide con validación.")

    output = validation.loc[:, ["datetime", TARGET_COLUMN]].copy()
    output = output.rename(columns={TARGET_COLUMN: "pm25_real"})
    output["pm25_predicho"] = np.asarray(predictions, dtype=float)
    output["residuo"] = output["pm25_real"] - output["pm25_predicho"]
    output["error_absoluto"] = output["residuo"].abs()
    return output


def compare_with_baseline(
    mlp: RegressionMetrics,
    baseline: RegressionMetrics,
) -> dict[str, Any]:
    """Compare validation metrics without touching the reserved test set."""
    improvements = {
        "mae": baseline.mae - mlp.mae,
        "rmse": baseline.rmse - mlp.rmse,
        "r2": mlp.r2 - baseline.r2,
    }
    improved_count = sum(value > 0 for value in improvements.values())
    return {
        "improvements": improvements,
        "metrics_improved": improved_count,
        "beats_baseline": improved_count >= 2,
    }


def model_summary_text(model: Any) -> str:
    """Capture Keras model.summary() as text for documentation."""
    lines: list[str] = []
    printer: Callable[[str], None] = lines.append
    model.summary(print_fn=printer)
    return "\n".join(lines)
