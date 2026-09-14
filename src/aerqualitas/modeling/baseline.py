"""Linear-regression baseline for AerQualitas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from aerqualitas.modeling.preprocessing import build_preprocessor
from aerqualitas.modeling.split import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Regression metrics used across model comparisons."""

    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        """Return JSON-serializable metrics."""
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Calculate MAE, RMSE and R²."""
    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )


def build_linear_baseline() -> Pipeline:
    """Create an unfitted preprocessing + linear-regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )


def fit_linear_baseline(train: pd.DataFrame) -> Pipeline:
    """Fit the baseline using training data only."""
    pipeline = build_linear_baseline()
    pipeline.fit(train.loc[:, list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    return pipeline


def evaluate_pipeline(model: BaseEstimator, dataframe: pd.DataFrame) -> RegressionMetrics:
    """Evaluate a fitted regression pipeline on a labeled dataframe."""
    y_true = dataframe[TARGET_COLUMN].to_numpy(dtype=float)
    y_pred = np.asarray(
        model.predict(dataframe.loc[:, list(FEATURE_COLUMNS)]),
        dtype=float,
    )
    return regression_metrics(y_true, y_pred)


def validation_predictions(model: BaseEstimator, validation: pd.DataFrame) -> pd.DataFrame:
    """Return auditable validation predictions and residuals."""
    predicted = np.asarray(
        model.predict(validation.loc[:, list(FEATURE_COLUMNS)]),
        dtype=float,
    )
    output = validation.loc[:, ["datetime", TARGET_COLUMN]].copy()
    output = output.rename(columns={TARGET_COLUMN: "pm25_real"})
    output["pm25_predicho"] = predicted
    output["residuo"] = output["pm25_real"] - output["pm25_predicho"]
    output["error_absoluto"] = output["residuo"].abs()
    return output
