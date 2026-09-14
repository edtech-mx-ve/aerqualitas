"""Safe input/output helpers for the AerQualitas data pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aerqualitas.validation import validate_required_file


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load the immutable raw Beijing PM2.5 CSV.

    Parameters
    ----------
    path:
        Path to the original CSV.

    Returns
    -------
    pandas.DataFrame
        Newly loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    ValueError
        If the path is not a regular file or the CSV cannot be parsed.
    """
    validate_required_file(path, "dataset Beijing PM2.5")
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"No fue posible leer el dataset: {path.name}") from exc


def save_dataframe_csv(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a dataframe without silently overwriting an existing artifact."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"El archivo de salida ya existe y no se sobrescribirá: {path.name}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path


def save_json(
    payload: dict[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Serialize JSON metadata with controlled overwrite behavior."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"El archivo de salida ya existe y no se sobrescribirá: {path.name}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path
