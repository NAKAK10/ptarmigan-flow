from __future__ import annotations

import logging
from pathlib import Path

import pytest

import ptarmigan_flow.logging_setup as logging_setup


@pytest.fixture(autouse=True)
def reset_root_logger_handlers():
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    yield

    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.handlers[:] = original_handlers
    root.setLevel(original_level)


def _configure_app_file_logging(level: str = "DEBUG") -> Path:
    assert hasattr(logging_setup, "configure_app_file_logging")
    return logging_setup.configure_app_file_logging(level)


def test_configure_app_file_logging_returns_app_log_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    log_path = _configure_app_file_logging()

    assert log_path.name == "app.log"
    assert log_path.parts[-4:] == ("Library", "Logs", "ptarmigan-flow", "app.log")


def test_configure_app_file_logging_creates_log_file_and_directory(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    log_path = _configure_app_file_logging()

    assert log_path.parent.is_dir()
    assert log_path.is_file()


def test_configure_app_file_logging_is_idempotent_for_same_path(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    log_path = _configure_app_file_logging()
    _configure_app_file_logging()

    matching_handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, logging.FileHandler)
        and str(handler.baseFilename) == str(log_path)
    ]
    assert len(matching_handlers) == 1


def test_configure_app_file_logging_writes_messages(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    log_path = _configure_app_file_logging()

    logging.getLogger("x").info("hello")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "hello" in log_path.read_text(encoding="utf-8")
