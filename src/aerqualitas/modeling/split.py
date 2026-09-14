"""Chronological data splitting for AerQualitas."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "TEMP",
    "PRES",
    "DEWP",
    "cbwd",
    "Iws",
    "Is",
    "Ir",
)
NUMERIC_FEATURES: tuple[str, ...] = ("TEMP", "PRES", "DEWP", "Iws", "Is", "Ir")
CATEGORICAL_FEATURES: tuple[str, ...] = ("cbwd",)
TARGET_COLUMN = "pm2.5"


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Immutable train/validation/test partition."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def validate_modeling_dataset(dataframe: pd.DataFrame) -> None:
    """Validate the clean supervised dataframe before splitting."""
    required = {"datetime", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Faltan columnas para modelado: {sorted(missing)}")

    if dataframe.empty:
        raise ValueError("El dataset de modelado está vacío.")

    if dataframe[list(FEATURE_COLUMNS) + [TARGET_COLUMN]].isna().any().any():
        raise ValueError("El dataset de modelado contiene valores faltantes.")

    if not dataframe["datetime"].is_monotonic_increasing:
        raise ValueError("El dataset debe estar ordenado cronológicamente.")


def chronological_split(
    dataframe: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ChronologicalSplit:
    """Split an ordered dataframe without shuffling.

    The remaining fraction is reserved for the test partition.
    """
    validate_modeling_dataset(dataframe)

    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction debe estar entre 0 y 1.")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction debe estar entre 0 y 1.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Debe quedar una fracción positiva para prueba.")

    n_rows = len(dataframe)
    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))

    if train_end == 0 or validation_end <= train_end or validation_end >= n_rows:
        raise ValueError("El dataset es demasiado pequeño para la división solicitada.")

    train = dataframe.iloc[:train_end].copy()
    validation = dataframe.iloc[train_end:validation_end].copy()
    test = dataframe.iloc[validation_end:].copy()

    if not (
        train["datetime"].max() <= validation["datetime"].min()
        and validation["datetime"].max() <= test["datetime"].min()
    ):
        raise ValueError("Las particiones temporales se solapan.")

    return ChronologicalSplit(train=train, validation=validation, test=test)
