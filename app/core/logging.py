import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure application-wide logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module."""
    return logging.getLogger(name)
