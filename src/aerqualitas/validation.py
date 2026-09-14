"""Pure validation helpers used across the project."""

from pathlib import Path


def validate_required_file(path: Path, label: str = "archivo") -> Path:
    """Validate that a required file exists and is a regular file.

    Parameters
    ----------
    path:
        File path to validate.
    label:
        Human-readable label used in error messages.

    Returns
    -------
    Path
        The validated path.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the path does not point to a regular file.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {label}: {path.name}")

    if not path.is_file():
        raise ValueError(f"La ruta de {label} no corresponde a un archivo.")

    return path
