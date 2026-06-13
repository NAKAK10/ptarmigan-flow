"""In-process daemon lifecycle helpers for the macOS app."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ptarmigan_flow.config import AppConfig, ensure_config_exists, load_config
from ptarmigan_flow.text_processing.interfaces import TextPostProcessor

LOGGER = logging.getLogger(__name__)


class DaemonLike(Protocol):
    """Small protocol used by the app controller."""

    def run_forever(self) -> None:
        """Run the daemon until stopped."""

    def stop(self) -> None:
        """Request daemon shutdown."""


class DaemonController:
    """Start and stop a daemon on a background thread."""

    def __init__(
        self,
        daemon_factory: Callable[[], DaemonLike],
        *,
        join_timeout: float | None = 5.0,
    ) -> None:
        self._daemon_factory = daemon_factory
        self._join_timeout = join_timeout
        self._lock = threading.Lock()
        self._daemon: DaemonLike | None = None
        self._thread: threading.Thread | None = None
        self._is_running = False
        self._last_error: Exception | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    @property
    def last_error(self) -> Exception | None:
        with self._lock:
            return self._last_error

    def start(self) -> None:
        """Start the daemon worker unless one is already active."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._is_running = True
            self._last_error = None
            thread = threading.Thread(
                target=self._run_daemon,
                daemon=True,
                name="ptarmigan-flow-app-daemon",
            )
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        """Stop the daemon and wait briefly for the worker thread to exit."""
        with self._lock:
            daemon = self._daemon
            thread = self._thread

        if daemon is not None:
            try:
                daemon.stop()
            except Exception as exc:
                LOGGER.exception("Failed to stop app daemon")
                with self._lock:
                    self._last_error = exc

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._join_timeout)

        with self._lock:
            if thread is None or not thread.is_alive():
                if self._thread is thread:
                    self._thread = None
                self._daemon = None
                self._is_running = False

    def _run_daemon(self) -> None:
        try:
            daemon = self._daemon_factory()
            with self._lock:
                self._daemon = daemon
            daemon.run_forever()
        except Exception as exc:
            LOGGER.exception("App daemon exited with an error")
            with self._lock:
                self._last_error = exc
        finally:
            with self._lock:
                self._daemon = None
                self._is_running = False


def build_daemon(
    config: AppConfig,
    *,
    post_processor: TextPostProcessor | None = None,
    enable_streaming: bool = True,
) -> DaemonLike:
    """Build the runtime daemon shared by CLI and macOS app entry points."""
    from ptarmigan_flow.daemon import PtarmiganFlowDaemon

    return PtarmiganFlowDaemon(
        config,
        post_processor=post_processor,
        enable_streaming=enable_streaming,
    )


def build_daemon_from_config(config_path: Path | str) -> DaemonLike:
    """Ensure, load, and build a daemon from a config path."""
    resolved_path = Path(config_path).expanduser()
    ensure_config_exists(resolved_path)
    config = load_config(resolved_path)
    return build_daemon(config)
