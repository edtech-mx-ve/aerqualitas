"""One-time final evaluation pipeline for AerQualitas Sprint 4."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aerqualitas.config import SETTINGS
from aerqualitas.data.io import save_dataframe_csv, save_json
from aerqualitas.logging_config import configure_logging
from aerqualitas.modeling.evaluation import (
    compare_test_metrics,
    error_by_target_quartile,
    final_prediction_frame,
    metrics_from_arrays,
    residual_analysis,
    sha256_file,
)
from aerqualitas.modeling.split import FEATURE_COLUMNS, TARGET_COLUMN, chronological_split
from aerqualitas.paths import MODEL_DIR, PROCESSED_DATA_DIR
from aerqualitas.validation import validate_required_file


def _import_tensorflow() -> Any:
    """Import TensorFlow lazily."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow no está instalado. Ejecute `pip install -r requirements-dev.txt`."
        ) from exc
    return tf


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {path.name}.")
    return payload


def _predict_mlp(model: Any, x_test: np.ndarray) -> np.ndarray:
    """Return finite flattened predictions."""
    predicted = np.asarray(model.predict(x_test, verbose=0), dtype=float).reshape(-1)
    if not np.isfinite(predicted).all():
        raise ValueError("El MLP produjo predicciones no finitas.")
    return predicted


def run_final_evaluation(*, overwrite: bool = False) -> dict[str, Any]:
    """Open the reserved test set once and consolidate the selected MLP.

    The MLP was selected in Sprint 3 using validation data only. Sprint 4 does not
    retrain or retune the model. The test partition is used exclusively to estimate
    final generalization performance.
    """
    logger = configure_logging()

    clean_path = PROCESSED_DATA_DIR / SETTINGS.processed_dataset_filename
    preprocessor_path = MODEL_DIR / SETTINGS.preprocessor_filename
    mlp_model_path = MODEL_DIR / SETTINGS.mlp_model_filename
    baseline_model_path = MODEL_DIR / SETTINGS.baseline_model_filename
    mlp_metrics_path = PROCESSED_DATA_DIR / SETTINGS.mlp_metrics_filename

    for path, label in (
        (clean_path, "dataset limpio"),
        (preprocessor_path, "preprocesador"),
        (mlp_model_path, "MLP seleccionado"),
        (baseline_model_path, "baseline lineal"),
        (mlp_metrics_path, "métricas de validación del MLP"),
    ):
        validate_required_file(path, label)

    final_metrics_path = PROCESSED_DATA_DIR / SETTINGS.final_test_metrics_filename
    manifest_path = MODEL_DIR / SETTINGS.final_model_manifest_filename
    if (final_metrics_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            "La evaluación final ya existe. No se abrirá de nuevo el test sin "
            "usar explícitamente --overwrite."
        )

    data = pd.read_csv(clean_path, parse_dates=["datetime"])
    split = chronological_split(
        data,
        train_fraction=SETTINGS.train_fraction,
        validation_fraction=SETTINGS.validation_fraction,
    )
    test = split.test

    preprocessor = joblib.load(preprocessor_path)
    x_test = np.asarray(
        preprocessor.transform(test.loc[:, list(FEATURE_COLUMNS)]),
        dtype=np.float32,
    )
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

    tf = _import_tensorflow()
    mlp_model = tf.keras.models.load_model(mlp_model_path)
    mlp_predictions = _predict_mlp(mlp_model, x_test)

    baseline_model = joblib.load(baseline_model_path)
    baseline_predictions = np.asarray(
        baseline_model.predict(test.loc[:, list(FEATURE_COLUMNS)]),
        dtype=float,
    ).reshape(-1)
    if not np.isfinite(baseline_predictions).all():
        raise ValueError("El baseline produjo predicciones no finitas.")

    mlp_metrics = metrics_from_arrays(y_test, mlp_predictions)
    baseline_metrics = metrics_from_arrays(y_test, baseline_predictions)
    comparison = compare_test_metrics(mlp_metrics, baseline_metrics)

    predictions = final_prediction_frame(test, mlp_predictions, baseline_predictions)
    mlp_error = residual_analysis(predictions, prefix="mlp")
    baseline_error = residual_analysis(predictions, prefix="baseline")
    quartiles = error_by_target_quartile(predictions)

    save_dataframe_csv(
        predictions,
        PROCESSED_DATA_DIR / SETTINGS.final_test_predictions_filename,
        overwrite=overwrite,
    )
    save_dataframe_csv(
        quartiles,
        PROCESSED_DATA_DIR / SETTINGS.final_error_by_quartile_filename,
        overwrite=overwrite,
    )

    sprint3_metrics = _load_json(mlp_metrics_path)
    validation_metrics = sprint3_metrics["validation"]

    final_metrics = {
        "evaluation": "final_reserved_test",
        "selection_basis": "validation_only_sprint3",
        "no_retraining_or_tuning_with_test": True,
        "test_rows": int(len(test)),
        "test_period": {
            "start": test["datetime"].min().isoformat(),
            "end": test["datetime"].max().isoformat(),
        },
        "mlp": mlp_metrics.to_dict(),
        "baseline": baseline_metrics.to_dict(),
        "comparison": comparison,
        "validation_reference_mlp": validation_metrics,
    }
    save_json(final_metrics, final_metrics_path, overwrite=overwrite)

    error_payload = {
        "mlp": mlp_error.to_dict(),
        "baseline": baseline_error.to_dict(),
        "residual_definition": "observed_pm25_minus_predicted_pm25",
        "positive_mean_residual_interpretation": "average_underprediction",
        "negative_mean_residual_interpretation": "average_overprediction",
        "quartile_note": (
            "Q1-Q4 are descriptive quartiles of observed PM2.5 in the test set; "
            "they are not air-quality categories."
        ),
    }
    save_json(
        error_payload,
        PROCESSED_DATA_DIR / SETTINGS.final_error_analysis_filename,
        overwrite=overwrite,
    )

    manifest = {
        "project": SETTINGS.name,
        "version": SETTINGS.version,
        "status": "final_model_evaluated",
        "selected_model": "MLP_feedforward",
        "model_file": SETTINGS.mlp_model_filename,
        "preprocessor_file": SETTINGS.preprocessor_filename,
        "selection_decided_before_test": True,
        "selection_basis": "validation_results_sprint3",
        "test_opened_for_final_evaluation": True,
        "post_test_tuning_allowed": False,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_rows": int(len(test)),
        "test_metrics": mlp_metrics.to_dict(),
        "artifact_sha256": {
            "model": sha256_file(mlp_model_path),
            "preprocessor": sha256_file(preprocessor_path),
        },
    }
    save_json(manifest, manifest_path, overwrite=overwrite)

    logger.info(
        "Sprint 4 completado: test=%s, MLP MAE=%.3f, RMSE=%.3f, R²=%.3f.",
        len(test),
        mlp_metrics.mae,
        mlp_metrics.rmse,
        mlp_metrics.r2,
    )

    return {
        "test_rows": int(len(test)),
        "mlp_mae": mlp_metrics.mae,
        "mlp_rmse": mlp_metrics.rmse,
        "mlp_r2": mlp_metrics.r2,
        "baseline_mae": baseline_metrics.mae,
        "baseline_rmse": baseline_metrics.rmse,
        "baseline_r2": baseline_metrics.r2,
        "metrics_improved": int(comparison["metrics_improved"]),
        "mlp_better_on_test": bool(comparison["mlp_better_on_test"]),
        "mean_residual_mlp": mlp_error.mean_residual,
        "p90_absolute_error_mlp": mlp_error.p90_absolute_error,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluación final del MLP con el test reservado — Sprint 4."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenera artefactos de evaluación. No debe usarse para iterar sobre "
            "el test ni ajustar el modelo."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run Sprint 4 from the command line."""
    args = parse_args()
    results = run_final_evaluation(overwrite=args.overwrite)

    print("AerQualitas — Sprint 4 completado")
    print(f"Registros de prueba: {results['test_rows']}")
    print("MLP — evaluación final")
    print(f"  MAE: {results['mlp_mae']:.3f}")
    print(f"  RMSE: {results['mlp_rmse']:.3f}")
    print(f"  R²: {results['mlp_r2']:.3f}")
    print("Baseline lineal — prueba")
    print(f"  MAE: {results['baseline_mae']:.3f}")
    print(f"  RMSE: {results['baseline_rmse']:.3f}")
    print(f"  R²: {results['baseline_r2']:.3f}")
    print(
        "Comparación test: "
        f"{results['metrics_improved']}/3 métricas favorables al MLP — "
        f"{'MLP superior' if results['mlp_better_on_test'] else 'sin superioridad suficiente'}"
    )
    print(f"Sesgo residual medio MLP: {results['mean_residual_mlp']:.3f}")
    print(f"P90 error absoluto MLP: {results['p90_absolute_error_mlp']:.3f}")


if __name__ == "__main__":
    main()
