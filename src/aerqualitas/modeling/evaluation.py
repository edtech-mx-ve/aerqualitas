"""Final evaluation utilities for AerQualitas Sprint 4."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aerqualitas.modeling.baseline import RegressionMetrics, regression_metrics


@dataclass(frozen=True, slots=True)
class ResidualAnalysis:
    """Summary of residual behavior on an untouched test partition."""

    mean_residual: float
    median_absolute_error: float
    p90_absolute_error: float
    max_absolute_error: float

    def to_dict(self) -> dict[str, float]:
        """Return JSON-serializable residual statistics."""
        return {
            "mean_residual": self.mean_residual,
            "median_absolute_error": self.median_absolute_error,
            "p90_absolute_error": self.p90_absolute_error,
            "max_absolute_error": self.max_absolute_error,
        }


def final_prediction_frame(
    test: pd.DataFrame,
    mlp_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
) -> pd.DataFrame:
    """Create an auditable table with test predictions from both models."""
    if len(test) != len(mlp_predictions) or len(test) != len(baseline_predictions):
        raise ValueError("Las predicciones deben coincidir con el tamaño de prueba.")

    output = test.loc[:, ["datetime", "pm2.5"]].copy()
    output = output.rename(columns={"pm2.5": "pm25_real"})
    output["pm25_mlp"] = np.asarray(mlp_predictions, dtype=float)
    output["pm25_baseline"] = np.asarray(baseline_predictions, dtype=float)
    output["residuo_mlp"] = output["pm25_real"] - output["pm25_mlp"]
    output["error_absoluto_mlp"] = output["residuo_mlp"].abs()
    output["residuo_baseline"] = output["pm25_real"] - output["pm25_baseline"]
    output["error_absoluto_baseline"] = output["residuo_baseline"].abs()

    numeric = output.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("La tabla de evaluación contiene valores no finitos.")
    return output


def residual_analysis(predictions: pd.DataFrame, *, prefix: str) -> ResidualAnalysis:
    """Summarize residual and absolute-error distributions."""
    residual_col = f"residuo_{prefix}"
    absolute_col = f"error_absoluto_{prefix}"
    missing = {residual_col, absolute_col}.difference(predictions.columns)
    if missing:
        raise ValueError(f"Faltan columnas para análisis residual: {sorted(missing)}")

    residual = predictions[residual_col].to_numpy(dtype=float)
    absolute = predictions[absolute_col].to_numpy(dtype=float)
    if residual.size == 0:
        raise ValueError("No hay predicciones para analizar.")

    return ResidualAnalysis(
        mean_residual=float(np.mean(residual)),
        median_absolute_error=float(np.median(absolute)),
        p90_absolute_error=float(np.quantile(absolute, 0.90)),
        max_absolute_error=float(np.max(absolute)),
    )


def error_by_target_quartile(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare model errors across quartiles of observed PM2.5.

    Quartiles are descriptive and must not be interpreted as air-quality classes.
    """
    required = {
        "pm25_real",
        "error_absoluto_mlp",
        "error_absoluto_baseline",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Faltan columnas para análisis por cuartiles: {sorted(missing)}")

    frame = predictions.copy()
    try:
        frame["cuartil_pm25"] = pd.qcut(
            frame["pm25_real"],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
    except ValueError as exc:
        raise ValueError("No fue posible construir los cuartiles de PM2.5.") from exc

    grouped = (
        frame.groupby("cuartil_pm25", observed=True)
        .agg(
            registros=("pm25_real", "size"),
            pm25_min=("pm25_real", "min"),
            pm25_max=("pm25_real", "max"),
            mae_mlp=("error_absoluto_mlp", "mean"),
            mae_baseline=("error_absoluto_baseline", "mean"),
        )
        .reset_index()
    )
    grouped["mejora_mae_mlp"] = grouped["mae_baseline"] - grouped["mae_mlp"]
    return grouped


def compare_test_metrics(
    mlp: RegressionMetrics,
    baseline: RegressionMetrics,
) -> dict[str, Any]:
    """Compare final test metrics without redefining the selected model."""
    differences = {
        "mae": baseline.mae - mlp.mae,
        "rmse": baseline.rmse - mlp.rmse,
        "r2": mlp.r2 - baseline.r2,
    }
    improved = sum(value > 0 for value in differences.values())

    relative = {
        "mae_percent": (
            100.0 * differences["mae"] / baseline.mae if baseline.mae != 0 else None
        ),
        "rmse_percent": (
            100.0 * differences["rmse"] / baseline.rmse if baseline.rmse != 0 else None
        ),
    }
    return {
        "differences": differences,
        "relative_improvement": relative,
        "metrics_improved": improved,
        "mlp_better_on_test": improved >= 2,
    }


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Calculate common regression metrics for final evaluation."""
    return regression_metrics(
        np.asarray(y_true, dtype=float).reshape(-1),
        np.asarray(y_pred, dtype=float).reshape(-1),
    )


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 for a persisted model artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el artefacto: {path}")
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
