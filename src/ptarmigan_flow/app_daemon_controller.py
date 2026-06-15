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
            if self._runner is subprocess.Popen:
                process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
                if process.stderr is not None:
                    threading.Thread(
                        target=self._drain_stderr,
                        args=(process,),
                        daemon=True,
                    ).start()
            else:
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

    @staticmethod
    def _drain_stderr(process: ProcessLike) -> None:
        try:
            for line in process.stderr:
                line = line.rstrip()
                if line:
                    LOGGER.warning("daemon stderr: %s", line)
        except Exception:
            pass


class InProcessDaemonController:
    """Start and stop the daemon as an in-process background thread."""

    def __init__(self, config_path: Path | str) -> None:
        self._config_path = Path(config_path)
        self._daemon: DaemonLike | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_error: Exception | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            t = self._thread
        return t is not None and t.is_alive()

    @property
    def last_error(self) -> Exception | None:
        with self._lock:
            return self._last_error

    def start(self) -> None:
        """Start the daemon on a background thread unless one is already active."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._last_error = None

        def _run() -> None:
            try:
                daemon = build_daemon_from_config(self._config_path)
            except Exception as exc:
                LOGGER.exception("Failed to build in-process daemon")
                with self._lock:
                    self._last_error = exc
                return
            with self._lock:
                self._daemon = daemon
            try:
                daemon.run_forever()
            except Exception as exc:
                LOGGER.exception("In-process daemon exited with error")
                with self._lock:
                    self._last_error = exc

        t = threading.Thread(target=_run, daemon=True, name="ptarmigan-daemon")
        with self._lock:
            self._thread = t
        t.start()

    def stop(self) -> None:
        """Stop the in-process daemon and wait briefly for its thread to exit."""
        with self._lock:
            daemon = self._daemon
            thread = self._thread
        if daemon is not None:
            try:
                daemon.stop()
            except Exception:
                LOGGER.exception("Failed to stop in-process daemon")
        if thread is not None:
            thread.join(timeout=5.0)
        with self._lock:
            self._daemon = None
            self._thread = None

    def notify_hotkey_press(self) -> None:
        with self._lock:
            daemon = self._daemon
        hotkey = getattr(daemon, "hotkey", None)
        notify_press = getattr(hotkey, "notify_press", None)
        if callable(notify_press):
            notify_press()

    def notify_hotkey_release(self) -> None:
        with self._lock:
            daemon = self._daemon
        hotkey = getattr(daemon, "hotkey", None)
        notify_release = getattr(hotkey, "notify_release", None)
        if callable(notify_release):
            notify_release()


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
