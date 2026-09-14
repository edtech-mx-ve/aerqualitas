"""Executable Sprint 2 preprocessing and baseline pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from aerqualitas.config import SETTINGS
from aerqualitas.data.io import save_dataframe_csv, save_json
from aerqualitas.logging_config import configure_logging
from aerqualitas.modeling.baseline import (
    evaluate_pipeline,
    fit_linear_baseline,
    validation_predictions,
)
from aerqualitas.modeling.preprocessing import build_input_metadata
from aerqualitas.modeling.split import FEATURE_COLUMNS, chronological_split
from aerqualitas.paths import MODEL_DIR, PROCESSED_DATA_DIR
from aerqualitas.validation import validate_required_file


def _save_joblib(obj: object, path: Path, *, overwrite: bool) -> Path:
    """Persist a joblib artifact without silent overwrite."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"El artefacto ya existe y no se sobrescribirá: {path.name}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def _period_payload(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Describe a chronological partition without serializing its rows."""
    return {
        "rows": int(len(dataframe)),
        "datetime_start": dataframe["datetime"].min().isoformat(),
        "datetime_end": dataframe["datetime"].max().isoformat(),
    }


def run_baseline_pipeline(*, overwrite: bool = False) -> dict[str, Any]:
    """Create chronological splits, preprocessing and the linear baseline.

    The test partition is created and documented but intentionally not evaluated
    in Sprint 2. It remains reserved for final model evaluation.
    """
    logger = configure_logging()
    clean_path = PROCESSED_DATA_DIR / SETTINGS.processed_dataset_filename
    validate_required_file(clean_path, "dataset limpio del Sprint 1")

    data = pd.read_csv(clean_path, parse_dates=["datetime"])
    split = chronological_split(
        data,
        train_fraction=SETTINGS.train_fraction,
        validation_fraction=SETTINGS.validation_fraction,
    )

    baseline = fit_linear_baseline(split.train)
    train_metrics = evaluate_pipeline(baseline, split.train)
    validation_metrics = evaluate_pipeline(baseline, split.validation)
    predictions = validation_predictions(baseline, split.validation)

    split_manifest = {
        "project": SETTINGS.name,
        "version": SETTINGS.version,
        "policy": "chronological_no_shuffle",
        "fractions": {
            "train": SETTINGS.train_fraction,
            "validation": SETTINGS.validation_fraction,
            "test": SETTINGS.test_fraction,
        },
        "features": list(FEATURE_COLUMNS),
        "target": "pm2.5",
        "train": _period_payload(split.train),
        "validation": _period_payload(split.validation),
        "test": _period_payload(split.test),
        "test_reserved": True,
        "note": (
            "El conjunto de prueba se reserva sin métricas en Sprint 2 para evitar "
            "utilizarlo durante decisiones de modelado."
        ),
    }
    save_json(
        split_manifest,
        PROCESSED_DATA_DIR / SETTINGS.split_manifest_filename,
        overwrite=overwrite,
    )

    metrics_payload = {
        "model": "LinearRegression",
        "purpose": "baseline_only",
        "train": train_metrics.to_dict(),
        "validation": validation_metrics.to_dict(),
        "test": None,
        "test_status": "reserved",
    }
    save_json(
        metrics_payload,
        PROCESSED_DATA_DIR / SETTINGS.baseline_metrics_filename,
        overwrite=overwrite,
    )

    save_dataframe_csv(
        predictions,
        PROCESSED_DATA_DIR / SETTINGS.validation_predictions_filename,
        overwrite=overwrite,
    )

    input_metadata = build_input_metadata(split.train)
    input_metadata["features_original"] = list(FEATURE_COLUMNS)
    input_metadata["target"] = "pm2.5"
    save_json(
        input_metadata,
        MODEL_DIR / SETTINGS.input_metadata_filename,
        overwrite=overwrite,
    )

    _save_joblib(
        baseline,
        MODEL_DIR / SETTINGS.baseline_model_filename,
        overwrite=overwrite,
    )
    fitted_preprocessor = baseline.named_steps["preprocessor"]
    _save_joblib(
        fitted_preprocessor,
        MODEL_DIR / SETTINGS.preprocessor_filename,
        overwrite=overwrite,
    )

    transformed_features = [
        str(name) for name in fitted_preprocessor.get_feature_names_out()
    ]

    logger.info(
        "Sprint 2 completado: train=%s, validación=%s, prueba=%s.",
        len(split.train),
        len(split.validation),
        len(split.test),
    )

    return {
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
        "validation_mae": validation_metrics.mae,
        "validation_rmse": validation_metrics.rmse,
        "validation_r2": validation_metrics.r2,
        "transformed_features": transformed_features,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pipeline de preprocesamiento y baseline — Sprint 2."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite regenerar artefactos existentes.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Sprint 2 from the command line."""
    args = parse_args()
    results = run_baseline_pipeline(overwrite=args.overwrite)

    print("AerQualitas — Sprint 2 completado")
    print(f"Entrenamiento: {results['train_rows']}")
    print(f"Validación: {results['validation_rows']}")
    print(f"Prueba reservada: {results['test_rows']}")
    print(f"MAE validación: {results['validation_mae']:.3f}")
    print(f"RMSE validación: {results['validation_rmse']:.3f}")
    print(f"R² validación: {results['validation_r2']:.3f}")


if __name__ == "__main__":
    main()
