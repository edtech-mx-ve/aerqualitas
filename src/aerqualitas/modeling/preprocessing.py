"""Preprocessing definitions for AerQualitas models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aerqualitas.modeling.split import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Create an unfitted preprocessing transformer.

    Numeric variables are standardized and wind direction is one-hot encoded.
    The transformer must be fitted only on the training partition.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_input_metadata(train: pd.DataFrame) -> dict[str, Any]:
    """Build future UI input constraints using training data only."""
    numeric: dict[str, dict[str, float]] = {}
    for column in NUMERIC_FEATURES:
        numeric[column] = {
            "min": float(train[column].min()),
            "max": float(train[column].max()),
        }

    categorical = {
        column: sorted(str(value) for value in train[column].dropna().unique())
        for column in CATEGORICAL_FEATURES
    }

    return {
        "source": "training_partition_only",
        "numeric": numeric,
        "categorical": categorical,
    }
