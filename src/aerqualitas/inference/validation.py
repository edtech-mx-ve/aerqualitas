"""Strict user-input validation for AerQualitas inference."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from aerqualitas.inference.metadata import validate_metadata_schema


class InputValidationError(ValueError):
    """Raised when a user-supplied inference value violates the model domain."""


def validate_user_input(
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, float | str]:
    """Validate and normalize one inference request.

    Values outside the training-derived domain are rejected, not clipped.
    """
    validate_metadata_schema(metadata)
    feature_order = list(metadata["feature_order"])

    missing = [feature for feature in feature_order if feature not in payload]
    extras = [key for key in payload if key not in feature_order]
    if missing:
        raise InputValidationError(
            "Faltan variables requeridas: " + ", ".join(missing)
        )
    if extras:
        raise InputValidationError(
            "Se recibieron variables no permitidas: " + ", ".join(extras)
        )

    validated: dict[str, float | str] = {}
    numeric_specs = metadata["numeric"]

    for feature in ("TEMP", "PRES", "DEWP", "Iws", "Is", "Ir"):
        raw = payload[feature]
        if isinstance(raw, bool):
            raise InputValidationError(f"{feature} debe ser numérico.")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"{feature} debe ser numérico.") from exc

        if not math.isfinite(value):
            raise InputValidationError(f"{feature} debe ser un valor finito.")

        spec = numeric_specs[feature]
        minimum = float(spec["min"])
        maximum = float(spec["max"])
        if value < minimum or value > maximum:
            raise InputValidationError(
                f"{feature}={value:g} está fuera del rango permitido "
                f"[{minimum:g}, {maximum:g}]."
            )
        validated[feature] = value

    category = str(payload["cbwd"])
    allowed = [str(value) for value in metadata["categorical"]["cbwd"]["allowed"]]
    if category not in allowed:
        raise InputValidationError(
            f"cbwd={category!r} no es una categoría permitida."
        )
    validated["cbwd"] = category

    return {feature: validated[feature] for feature in feature_order}


def input_frame(
    validated: dict[str, float | str],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Convert a validated request to a one-row frame in model feature order."""
    feature_order = list(metadata["feature_order"])
    return pd.DataFrame(
        [{feature: validated[feature] for feature in feature_order}],
        columns=feature_order,
    )
