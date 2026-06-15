"""Logging helpers for ptarmigan-flow."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_ANSI_YELLOW = "\x1b[33m"
_ANSI_RESET = "\x1b[0m"


def _supports_ansi(stream: object) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty) or not isatty():
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    term = str(os.environ.get("TERM", "")).strip().lower()
    if term in {"", "dumb"}:
        return False
    return True


class _WarningColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, *, color_warnings: bool) -> None:
        super().__init__(fmt)
        self._color_warnings = color_warnings

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if self._color_warnings and record.levelno == logging.WARNING:
            return f"{_ANSI_YELLOW}{rendered}{_ANSI_RESET}"
        return rendered


def configure_logging(level: str) -> None:
    """Configure project logging once."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)
    handler.setFormatter(
        _WarningColorFormatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            color_warnings=_supports_ansi(handler.stream),
        )
    )
    root.addHandler(handler)


def configure_app_file_logging(level: str = "DEBUG") -> Path:
    """Configure additive file logging for the macOS GUI app."""
    path = (
        Path.home()
        / "Library"
        / "Logs"
        / "ptarmigan-flow"
        / "app.log"
    ).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    for handler in root.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == path
        ):
            return path

    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    root.addHandler(handler)
    return path
