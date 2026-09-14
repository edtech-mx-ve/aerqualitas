"""Executable Sprint 3 feedforward MLP training pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aerqualitas.config import SETTINGS
from aerqualitas.data.io import save_dataframe_csv, save_json
from aerqualitas.logging_config import configure_logging
from aerqualitas.modeling.baseline import RegressionMetrics
from aerqualitas.modeling.mlp import (
    MLPTrainingConfig,
    build_mlp,
    compare_with_baseline,
    evaluate_predictions,
    fit_mlp,
    model_summary_text,
    predict_array,
    prediction_frame,
)
from aerqualitas.modeling.split import FEATURE_COLUMNS, TARGET_COLUMN, chronological_split
from aerqualitas.paths import MODEL_DIR, PROCESSED_DATA_DIR
from aerqualitas.validation import validate_required_file


def _save_text(content: str, path: Path, *, overwrite: bool) -> Path:
    """Persist UTF-8 text without silent overwrite."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"El artefacto ya existe y no se sobrescribirá: {path.name}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {path.name}.")
    return payload


def _training_config() -> MLPTrainingConfig:
    """Build the immutable MLP configuration from application settings."""
    return MLPTrainingConfig(
        hidden_1=SETTINGS.mlp_hidden_1,
        hidden_2=SETTINGS.mlp_hidden_2,
        learning_rate=SETTINGS.mlp_learning_rate,
        batch_size=SETTINGS.mlp_batch_size,
        max_epochs=SETTINGS.mlp_max_epochs,
        early_stopping_patience=SETTINGS.mlp_early_stopping_patience,
        reduce_lr_patience=SETTINGS.mlp_reduce_lr_patience,
        random_seed=SETTINGS.random_seed,
    )


def _metrics_from_payload(payload: dict[str, Any]) -> RegressionMetrics:
    """Convert a metrics mapping to the common immutable metric object."""
    return RegressionMetrics(
        mae=float(payload["mae"]),
        rmse=float(payload["rmse"]),
        r2=float(payload["r2"]),
    )


def run_mlp_pipeline(*, overwrite: bool = False) -> dict[str, Any]:
    """Train and validate the Sprint 3 MLP without evaluating the test partition."""
    logger = configure_logging()

    clean_path = PROCESSED_DATA_DIR / SETTINGS.processed_dataset_filename
    preprocessor_path = MODEL_DIR / SETTINGS.preprocessor_filename
    baseline_metrics_path = PROCESSED_DATA_DIR / SETTINGS.baseline_metrics_filename
    model_path = MODEL_DIR / SETTINGS.mlp_model_filename

    validate_required_file(clean_path, "dataset limpio del Sprint 1")
    validate_required_file(preprocessor_path, "preprocesador del Sprint 2")
    validate_required_file(baseline_metrics_path, "métricas baseline del Sprint 2")

    if model_path.exists() and not overwrite:
        raise FileExistsError(
            f"El modelo ya existe y no se sobrescribirá: {model_path.name}"
        )

    data = pd.read_csv(clean_path, parse_dates=["datetime"])
    split = chronological_split(
        data,
        train_fraction=SETTINGS.train_fraction,
        validation_fraction=SETTINGS.validation_fraction,
    )

    preprocessor = joblib.load(preprocessor_path)
    x_train = np.asarray(
        preprocessor.transform(split.train.loc[:, list(FEATURE_COLUMNS)]),
        dtype=np.float32,
    )
    x_validation = np.asarray(
        preprocessor.transform(split.validation.loc[:, list(FEATURE_COLUMNS)]),
        dtype=np.float32,
    )
    y_train = split.train[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_validation = split.validation[TARGET_COLUMN].to_numpy(dtype=np.float32)

    if x_train.ndim != 2 or x_validation.ndim != 2:
        raise ValueError("El preprocesamiento debe producir matrices bidimensionales.")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("Entrenamiento y validación tienen dimensiones incompatibles.")

    config = _training_config()
    model = build_mlp(input_dim=x_train.shape[1], config=config)
    history = fit_mlp(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
        model_path=model_path,
        config=config,
    )

    train_predictions = predict_array(model, x_train)
    validation_predictions = predict_array(model, x_validation)
    train_metrics = evaluate_predictions(y_train, train_predictions)
    validation_metrics = evaluate_predictions(y_validation, validation_predictions)

    baseline_payload = _load_json(baseline_metrics_path)
    baseline_validation = _metrics_from_payload(baseline_payload["validation"])
    comparison = compare_with_baseline(validation_metrics, baseline_validation)

    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    save_dataframe_csv(
        history_frame,
        PROCESSED_DATA_DIR / SETTINGS.mlp_history_filename,
        overwrite=overwrite,
    )

    predictions_frame = prediction_frame(split.validation, validation_predictions)
    save_dataframe_csv(
        predictions_frame,
        PROCESSED_DATA_DIR / SETTINGS.mlp_validation_predictions_filename,
        overwrite=overwrite,
    )

    metrics_payload = {
        "model": "MLP_feedforward",
        "framework": "TensorFlow/Keras",
        "purpose": "candidate_deep_learning_model",
        "train": train_metrics.to_dict(),
        "validation": validation_metrics.to_dict(),
        "baseline_validation": baseline_validation.to_dict(),
        "comparison": comparison,
        "test": None,
        "test_status": "reserved",
    }
    save_json(
        metrics_payload,
        PROCESSED_DATA_DIR / SETTINGS.mlp_metrics_filename,
        overwrite=overwrite,
    )

    metadata = {
        "project": SETTINGS.name,
        "version": SETTINGS.version,
        "model_type": "feedforward_multilayer_perceptron",
        "task": "supervised_regression",
        "original_features": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "preprocessed_input_dim": int(x_train.shape[1]),
        "architecture": [
            {"layer": "Input", "units": int(x_train.shape[1])},
            {
                "layer": "Dense",
                "units": config.hidden_1,
                "activation": "relu",
            },
            {
                "layer": "Dense",
                "units": config.hidden_2,
                "activation": "relu",
            },
            {"layer": "Dense", "units": 1, "activation": "linear"},
        ],
        "optimizer": {
            "name": "Adam",
            "learning_rate_initial": config.learning_rate,
        },
        "loss": "mean_squared_error",
        "metric_during_training": "mean_absolute_error",
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "epochs_executed": int(len(history_frame)),
        "early_stopping_patience": config.early_stopping_patience,
        "reduce_lr_patience": config.reduce_lr_patience,
        "random_seed": config.random_seed,
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows_reserved": int(len(split.test)),
        "test_evaluated": False,
        "preprocessor": SETTINGS.preprocessor_filename,
        "model_file": SETTINGS.mlp_model_filename,
    }
    save_json(
        metadata,
        MODEL_DIR / SETTINGS.mlp_metadata_filename,
        overwrite=overwrite,
    )
    _save_text(
        model_summary_text(model),
        MODEL_DIR / SETTINGS.mlp_summary_filename,
        overwrite=overwrite,
    )

    logger.info(
        "Sprint 3 completado: épocas=%s, MAE val=%.3f, RMSE val=%.3f, R² val=%.3f.",
        len(history_frame),
        validation_metrics.mae,
        validation_metrics.rmse,
        validation_metrics.r2,
    )

    return {
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows_reserved": int(len(split.test)),
        "input_dim": int(x_train.shape[1]),
        "epochs_executed": int(len(history_frame)),
        "validation_mae": validation_metrics.mae,
        "validation_rmse": validation_metrics.rmse,
        "validation_r2": validation_metrics.r2,
        "beats_baseline": bool(comparison["beats_baseline"]),
        "metrics_improved": int(comparison["metrics_improved"]),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de la red neuronal feedforward — Sprint 3."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite regenerar los artefactos del MLP existentes.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Sprint 3 from the command line."""
    args = parse_args()
    results = run_mlp_pipeline(overwrite=args.overwrite)

    print("AerQualitas — Sprint 3 completado")
    print(f"Entrenamiento: {results['train_rows']}")
    print(f"Validación: {results['validation_rows']}")
    print(f"Prueba reservada: {results['test_rows_reserved']}")
    print(f"Dimensión de entrada preprocesada: {results['input_dim']}")
    print(f"Épocas ejecutadas: {results['epochs_executed']}")
    print(f"MAE validación: {results['validation_mae']:.3f}")
    print(f"RMSE validación: {results['validation_rmse']:.3f}")
    print(f"R² validación: {results['validation_r2']:.3f}")
    print(
        "Comparación baseline: "
        f"{results['metrics_improved']}/3 métricas mejoradas — "
        f"{'SUPERA baseline' if results['beats_baseline'] else 'NO supera baseline'}"
    )


if __name__ == "__main__":
    main()
