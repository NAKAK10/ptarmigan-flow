"""Daemon lifecycle helpers for the macOS app."""

from __future__ import annotations

import logging
import subprocess
import sys
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


class ProcessLike(Protocol):
    """Small subprocess protocol used by the app controller."""

    returncode: int | None

    def poll(self) -> int | None:
        """Return the process exit code when it has exited."""

    def terminate(self) -> None:
        """Request graceful process termination."""

    def wait(self, timeout: float | None = None) -> int | None:
        """Wait for process exit."""

    def kill(self) -> None:
        """Force process termination."""


class DaemonController:
    """Start and stop the daemon as a signed child process."""

    def __init__(
        self,
        command_builder: Callable[[], list[str]],
        *,
        runner: Callable[[list[str]], ProcessLike] = subprocess.Popen,
        join_timeout: float | None = 5.0,
    ) -> None:
        self._command_builder = command_builder
        self._runner = runner
        self._join_timeout = join_timeout
        self._lock = threading.Lock()
        self._process: ProcessLike | None = None
        self._last_error: Exception | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            process = self._process
            if process is None:
                return False
            returncode = process.poll()
            self._record_exit_error_locked(returncode)
            return returncode is None

    @property
    def last_error(self) -> Exception | None:
        with self._lock:
            return self._last_error

    def start(self) -> None:
        """Start the daemon worker unless one is already active."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            self._last_error = None

        try:
            command = self._command_builder()
            process = self._runner(command)
        except Exception as exc:
            LOGGER.exception("Failed to start app daemon subprocess")
            with self._lock:
                self._process = None
                self._last_error = exc
            return

        with self._lock:
            self._process = process
            self._record_exit_error_locked(process.poll())

    def stop(self) -> None:
        """Stop the daemon subprocess and wait briefly for it to exit."""
        with self._lock:
            process = self._process
        if process is None:
            return

        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self._join_timeout)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=self._join_timeout)
                except subprocess.TimeoutExpired as exc:
                    LOGGER.exception("Timed out waiting for app daemon subprocess to exit")
                    with self._lock:
                        self._last_error = exc
        except Exception as exc:
            LOGGER.exception("Failed to stop app daemon subprocess")
            with self._lock:
                self._last_error = exc
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None

    def _record_exit_error_locked(self, returncode: int | None) -> None:
        if returncode is not None and returncode != 0:
            self._last_error = RuntimeError(
                f"App daemon subprocess exited with return code {returncode}"
            )


def daemon_run_command(config_path: Path | str) -> list[str]:
    """Build the child-process command that runs the daemon through the CLI."""
    return [
        sys.executable,
        "-m",
        "ptarmigan_flow.cli",
        "run",
        "--config",
        str(Path(config_path).expanduser()),
    ]


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
