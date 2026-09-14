"""Build and validate deployment metadata for AerQualitas inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aerqualitas.modeling.split import FEATURE_COLUMNS, chronological_split


_UI_SPEC: dict[str, dict[str, Any]] = {
    "TEMP": {
        "label": "Temperatura",
        "unit": "°C",
        "step": 1.0,
        "description": "Temperatura del aire.",
    },
    "PRES": {
        "label": "Presión atmosférica",
        "unit": "hPa",
        "step": 1.0,
        "description": "Presión atmosférica.",
    },
    "DEWP": {
        "label": "Punto de rocío",
        "unit": "°C",
        "step": 1.0,
        "description": "Temperatura de punto de rocío.",
    },
    "Iws": {
        "label": "Velocidad acumulada del viento",
        "unit": "m/s",
        "step": 0.01,
        "description": "Velocidad acumulada del viento registrada por la fuente.",
    },
    "Is": {
        "label": "Horas acumuladas de nieve",
        "unit": "h",
        "step": 1.0,
        "description": "Horas acumuladas con nieve.",
    },
    "Ir": {
        "label": "Horas acumuladas de lluvia",
        "unit": "h",
        "step": 1.0,
        "description": "Horas acumuladas con lluvia.",
    },
}


def build_inference_metadata(data: pd.DataFrame) -> dict[str, Any]:
    """Create hard input constraints using the training partition only."""
    split = chronological_split(data)
    train = split.train

    numeric: dict[str, dict[str, Any]] = {}
    for feature, spec in _UI_SPEC.items():
        series = pd.to_numeric(train[feature], errors="raise")
        minimum = float(series.min())
        maximum = float(series.max())
        default = float(series.median())
        if not np.isfinite([minimum, maximum, default]).all():
            raise ValueError(f"Metadatos no finitos para {feature}.")
        if minimum > maximum:
            raise ValueError(f"Rango inválido para {feature}.")
        numeric[feature] = {
            **spec,
            "min": minimum,
            "max": maximum,
            "default": min(max(default, minimum), maximum),
        }

    allowed_wind = sorted(str(value) for value in train["cbwd"].dropna().unique())
    if not allowed_wind:
        raise ValueError("No se encontraron categorías válidas para cbwd.")

    wind_mode = train["cbwd"].mode(dropna=True)
    default_wind = str(wind_mode.iloc[0]) if not wind_mode.empty else allowed_wind[0]

    return {
        "schema_version": 1,
        "source_partition": "training_only",
        "hard_constraints": True,
        "feature_order": list(FEATURE_COLUMNS),
        "numeric": numeric,
        "categorical": {
            "cbwd": {
                "label": "Dirección combinada del viento",
                "allowed": allowed_wind,
                "default": default_wind,
                "description": (
                    "Código de dirección del viento tal como aparece en el dataset UCI."
                ),
            }
        },
        "target": {
            "name": "pm2.5",
            "label": "PM2.5 estimado",
            "unit": "µg/m³",
        },
    }


def validate_metadata_schema(metadata: dict[str, Any]) -> None:
    """Validate the minimum schema required by the inference engine."""
    if metadata.get("source_partition") != "training_only":
        raise ValueError("Las restricciones deben derivarse solo de entrenamiento.")
    if metadata.get("hard_constraints") is not True:
        raise ValueError("AerQualitas requiere restricciones intrínsecas activas.")

    expected = list(FEATURE_COLUMNS)
    if metadata.get("feature_order") != expected:
        raise ValueError("El orden de variables no coincide con el modelo.")

    numeric = metadata.get("numeric")
    categorical = metadata.get("categorical")
    if not isinstance(numeric, dict) or not isinstance(categorical, dict):
        raise ValueError("Metadatos de entrada incompletos.")

    for feature in ("TEMP", "PRES", "DEWP", "Iws", "Is", "Ir"):
        spec = numeric.get(feature)
        if not isinstance(spec, dict):
            raise ValueError(f"Falta la especificación numérica de {feature}.")
        minimum = float(spec["min"])
        maximum = float(spec["max"])
        default = float(spec["default"])
        if minimum > maximum or not (minimum <= default <= maximum):
            raise ValueError(f"Rango o valor por defecto inválido para {feature}.")

    cbwd = categorical.get("cbwd")
    if not isinstance(cbwd, dict) or not cbwd.get("allowed"):
        raise ValueError("Faltan categorías permitidas para cbwd.")
