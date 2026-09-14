"""Pure cleaning transformations for the supervised PM2.5 dataset."""

from __future__ import annotations

import pandas as pd

from aerqualitas.data.quality import TARGET_COLUMN, build_datetime, validate_dataset_schema

CLEAN_COLUMNS: tuple[str, ...] = (
    "datetime",
    "year",
    "month",
    "day",
    "hour",
    "pm2.5",
    "DEWP",
    "TEMP",
    "PRES",
    "cbwd",
    "Iws",
    "Is",
    "Ir",
)


def clean_supervised_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepare the dataset for later supervised-learning stages.

    Policy
    ------
    * Validate the original schema.
    * Add a single hourly ``datetime`` field.
    * Remove rows whose target PM2.5 is missing; the target is never imputed.
    * Remove ``No`` because it is only a sequential identifier.
    * Sort chronologically and reset the index.
    * Preserve all meteorological and temporal information for later modeling.

    The input dataframe is never mutated.
    """
    validate_dataset_schema(dataframe)
    cleaned = dataframe.copy()
    cleaned.insert(0, "datetime", build_datetime(cleaned))
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned = cleaned.drop(columns=["No"])
    cleaned = cleaned.sort_values("datetime").reset_index(drop=True)

    if cleaned.isna().any().any():
        raise ValueError("El dataset limpio no debe contener valores faltantes.")

    if cleaned["datetime"].duplicated().any():
        raise ValueError("El dataset limpio contiene timestamps duplicados.")

    return cleaned.loc[:, list(CLEAN_COLUMNS)]
