"""Prediction engine for the final AerQualitas model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from aerqualitas.inference.metadata import validate_metadata_schema
from aerqualitas.inference.validation import input_frame, validate_user_input
from aerqualitas.modeling.evaluation import sha256_file
from aerqualitas.validation import validate_required_file


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """One validated PM2.5 inference result."""

    pm25: float
    transformed_features: int
    physical_warning: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceAssets:
    """Loaded final-model artifacts."""

    model: Any
    preprocessor: Any
    metadata: dict[str, Any]


def _import_tensorflow() -> Any:
    """Import TensorFlow lazily."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow no está instalado. Ejecute `pip install -r requirements-dev.txt`."
        ) from exc
    return tf


def load_inference_metadata(path: Path) -> dict[str, Any]:
    """Load and validate inference metadata from disk."""
    validate_required_file(path, "metadatos de inferencia")
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("Los metadatos de inferencia deben ser un objeto JSON.")
    validate_metadata_schema(metadata)
    return metadata


def verify_final_artifacts(
    model_path: Path,
    preprocessor_path: Path,
    final_manifest_path: Path,
) -> None:
    """Verify that inference uses the exact artifacts consolidated in Sprint 4."""
    validate_required_file(model_path, "modelo MLP final")
    validate_required_file(preprocessor_path, "preprocesador")
    validate_required_file(final_manifest_path, "manifiesto del modelo final")

    with final_manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    expected = manifest.get("artifact_sha256", {})
    model_hash = expected.get("model")
    preprocessor_hash = expected.get("preprocessor")
    if not model_hash or not preprocessor_hash:
        raise ValueError("El manifiesto final no contiene hashes de los artefactos.")

    if sha256_file(model_path) != model_hash:
        raise ValueError("El modelo no coincide con el artefacto validado en Sprint 4.")
    if sha256_file(preprocessor_path) != preprocessor_hash:
        raise ValueError(
            "El preprocesador no coincide con el artefacto validado en Sprint 4."
        )


def load_inference_assets(
    *,
    model_path: Path,
    preprocessor_path: Path,
    metadata_path: Path,
    final_manifest_path: Path,
) -> InferenceAssets:
    """Load the exact validated model and its preprocessing artifacts."""
    verify_final_artifacts(model_path, preprocessor_path, final_manifest_path)
    metadata = load_inference_metadata(metadata_path)
    preprocessor = joblib.load(preprocessor_path)
    tf = _import_tensorflow()
    model = tf.keras.models.load_model(model_path)
    return InferenceAssets(
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
    )


def predict_pm25(
    payload: dict[str, Any],
    *,
    model: Any,
    preprocessor: Any,
    metadata: dict[str, Any],
) -> PredictionResult:
    """Validate, transform and predict PM2.5 for one user request."""
    validated = validate_user_input(payload, metadata)
    frame = input_frame(validated, metadata)

    transformed = np.asarray(preprocessor.transform(frame), dtype=np.float32)
    if transformed.ndim != 2 or transformed.shape[0] != 1:
        raise ValueError("El preprocesador produjo una forma de entrada inválida.")
    if not np.isfinite(transformed).all():
        raise ValueError("El preprocesamiento produjo valores no finitos.")

    predicted = np.asarray(model.predict(transformed, verbose=0), dtype=float).reshape(-1)
    if predicted.size != 1 or not np.isfinite(predicted[0]):
        raise ValueError("El modelo produjo una predicción inválida.")

    pm25 = float(predicted[0])
    warning = None
    if pm25 < 0:
        warning = (
            "El modelo produjo una concentración negativa. No se corrige "
            "silenciosamente porque ello cambiaría el comportamiento evaluado."
        )

    return PredictionResult(
        pm25=pm25,
        transformed_features=int(transformed.shape[1]),
        physical_warning=warning,
    )
