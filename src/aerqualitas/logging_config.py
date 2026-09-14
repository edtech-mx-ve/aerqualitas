"""Logging configuration for AerQualitas."""

import logging
from pathlib import Path

from aerqualitas.paths import LOG_DIR


def configure_logging(
    level: int = logging.INFO,
    log_dir: Path = LOG_DIR,
) -> logging.Logger:
    """Configure and return the project logger.

    Parameters
    ----------
    level:
        Logging severity threshold.
    log_dir:
        Directory in which the application log is stored.

    Returns
    -------
    logging.Logger
        Configured project logger.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("aerqualitas")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(
        log_dir / "aerqualitas.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
