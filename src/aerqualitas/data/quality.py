"""Structural and quality validation for the Beijing PM2.5 dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

EXPECTED_COLUMNS: tuple[str, ...] = (
    "No",
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

NUMERIC_COLUMNS: tuple[str, ...] = (
    "No",
    "year",
    "month",
    "day",
    "hour",
    "pm2.5",
    "DEWP",
    "TEMP",
    "PRES",
    "Iws",
    "Is",
    "Ir",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "DEWP",
    "TEMP",
    "PRES",
    "cbwd",
    "Iws",
    "Is",
    "Ir",
)

TARGET_COLUMN = "pm2.5"
WIND_DIRECTIONS: tuple[str, ...] = ("NE", "NW", "SE", "cv")


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Compact, serializable data-quality report."""

    rows: int
    columns: int
    missing_target: int
    missing_features: int
    duplicate_rows: int
    duplicate_timestamps: int
    non_hourly_gaps: int
    datetime_min: str
    datetime_max: str
    wind_categories: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation."""
        payload = asdict(self)
        payload["wind_categories"] = list(self.wind_categories)
        return payload


def build_datetime(dataframe: pd.DataFrame) -> pd.Series:
    """Build and validate the hourly timestamp from date component columns."""
    try:
        return pd.to_datetime(
            dataframe[["year", "month", "day", "hour"]],
            errors="raise",
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("Existen fechas u horas inválidas en el dataset.") from exc


def validate_dataset_schema(dataframe: pd.DataFrame) -> None:
    """Validate the expected schema and values required for Sprint 1."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            "Faltan columnas obligatorias: " + ", ".join(missing_columns)
        )

    unexpected = [column for column in dataframe.columns if column not in EXPECTED_COLUMNS]
    if unexpected:
        raise ValueError(
            "El dataset contiene columnas no esperadas: " + ", ".join(unexpected)
        )

    if dataframe.empty:
        raise ValueError("El dataset está vacío.")

    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise TypeError(f"La columna {column} debe ser numérica.")

    if dataframe[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Las variables predictoras contienen valores faltantes.")

    observed_target = dataframe[TARGET_COLUMN].dropna()
    if (observed_target < 0).any():
        raise ValueError("PM2.5 contiene valores negativos no válidos.")

    if not dataframe["month"].between(1, 12).all():
        raise ValueError("La columna month contiene valores fuera de 1..12.")

    if not dataframe["hour"].between(0, 23).all():
        raise ValueError("La columna hour contiene valores fuera de 0..23.")

    build_datetime(dataframe)

    wind_values = set(dataframe["cbwd"].astype(str).unique())
    unknown_wind = sorted(wind_values.difference(WIND_DIRECTIONS))
    if unknown_wind:
        raise ValueError(
            "Direcciones de viento no reconocidas: " + ", ".join(unknown_wind)
        )


def build_quality_report(dataframe: pd.DataFrame) -> DataQualityReport:
    """Create a deterministic quality report after schema validation."""
    validate_dataset_schema(dataframe)
    timestamps = build_datetime(dataframe)
    ordered = timestamps.sort_values()
    gaps = ordered.diff().dropna()

    return DataQualityReport(
        rows=int(len(dataframe)),
        columns=int(len(dataframe.columns)),
        missing_target=int(dataframe[TARGET_COLUMN].isna().sum()),
        missing_features=int(dataframe[list(FEATURE_COLUMNS)].isna().sum().sum()),
        duplicate_rows=int(dataframe.duplicated().sum()),
        duplicate_timestamps=int(timestamps.duplicated().sum()),
        non_hourly_gaps=int((gaps != pd.Timedelta(hours=1)).sum()),
        datetime_min=timestamps.min().isoformat(),
        datetime_max=timestamps.max().isoformat(),
        wind_categories=tuple(sorted(dataframe["cbwd"].astype(str).unique())),
    )


def numeric_ranges(dataframe: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return observed numeric ranges for descriptive Sprint 1 reporting."""
    columns = ("pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir")
    output: dict[str, dict[str, float]] = {}

    for column in columns:
        series = dataframe[column].dropna()
        output[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
        }

    return output
