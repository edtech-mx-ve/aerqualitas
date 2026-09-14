"""Executable Sprint 1 data-preparation pipeline."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from aerqualitas.config import SETTINGS
from aerqualitas.data.cleaning import clean_supervised_dataset
from aerqualitas.data.io import load_raw_dataset, save_dataframe_csv, save_json
from aerqualitas.data.quality import build_quality_report, numeric_ranges
from aerqualitas.logging_config import configure_logging
from aerqualitas.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

CLEAN_FILENAME = "beijing_pm25_clean.csv"
REPORT_FILENAME = "data_quality_report.json"
MANIFEST_FILENAME = "cleaning_manifest.json"


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file using bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_data_pipeline(*, overwrite: bool = False) -> dict[str, Any]:
    """Validate, clean and persist reproducible Sprint 1 data artifacts."""
    logger = configure_logging()
    raw_path = RAW_DATA_DIR / SETTINGS.dataset_filename

    logger.info("Cargando dataset original: %s", raw_path.name)
    raw = load_raw_dataset(raw_path)
    quality = build_quality_report(raw)
    cleaned = clean_supervised_dataset(raw)

    clean_path = PROCESSED_DATA_DIR / CLEAN_FILENAME
    report_path = PROCESSED_DATA_DIR / REPORT_FILENAME
    manifest_path = PROCESSED_DATA_DIR / MANIFEST_FILENAME

    save_dataframe_csv(cleaned, clean_path, overwrite=overwrite)

    quality_payload = quality.to_dict()
    quality_payload["numeric_ranges_raw"] = numeric_ranges(raw)
    quality_payload["clean_rows"] = int(len(cleaned))
    quality_payload["rows_removed_missing_target"] = int(len(raw) - len(cleaned))
    quality_payload["clean_missing_values"] = int(cleaned.isna().sum().sum())
    save_json(quality_payload, report_path, overwrite=overwrite)

    manifest = {
        "project": SETTINGS.name,
        "version": SETTINGS.version,
        "source_file": raw_path.name,
        "source_sha256": sha256_file(raw_path),
        "clean_file": clean_path.name,
        "clean_sha256": sha256_file(clean_path),
        "source_rows": int(len(raw)),
        "clean_rows": int(len(cleaned)),
        "cleaning_policy": [
            "El dataset original no se modifica.",
            "Se crea datetime a partir de year, month, day y hour.",
            "Se eliminan filas con PM2.5 faltante; la variable objetivo no se imputa.",
            "Se elimina No por ser un identificador secuencial.",
            "Los registros se ordenan cronológicamente.",
        ],
    }
    save_json(manifest, manifest_path, overwrite=overwrite)

    logger.info(
        "Sprint 1 completado: %s filas originales -> %s filas limpias.",
        len(raw),
        len(cleaned),
    )
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(cleaned)),
        "removed_rows": int(len(raw) - len(cleaned)),
        "clean_path": str(clean_path),
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Pipeline de datos Sprint 1 de AerQualitas.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite regenerar los artefactos procesados existentes.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line data pipeline."""
    args = parse_args()
    results = run_data_pipeline(overwrite=args.overwrite)
    print("AerQualitas — Sprint 1 completado")
    print(f"Filas originales: {results['raw_rows']}")
    print(f"Filas limpias: {results['clean_rows']}")
    print(f"Filas eliminadas: {results['removed_rows']}")
    print(f"Dataset limpio: {results['clean_path']}")


if __name__ == "__main__":
    main()
