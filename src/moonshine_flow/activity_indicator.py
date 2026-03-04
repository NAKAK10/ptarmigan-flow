"""Runtime activity indicator control."""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import threading

from moonshine_flow.config import AppConfig
from moonshine_flow.ports.runtime import ActivityIndicatorPort

LOGGER = logging.getLogger(__name__)
_TERMINATE_TIMEOUT_SECONDS = 1.5
_OVERLAY_UNAVAILABLE_WARNED = False


class NullActivityIndicator(ActivityIndicatorPort):
    """No-op indicator used when overlay UI is disabled/unavailable."""

    def show_recording(self) -> None:
        return None

    def show_processing(self) -> None:
        return None

    def hide(self) -> None:
        return None

    def close(self) -> None:
        return None


class SubprocessActivityIndicator(ActivityIndicatorPort):
    """Control right-bottom overlay indicator via subprocess."""

    def __init__(
        self,
        *,
        enabled: bool,
        size: int,
        margin_right: int,
        margin_bottom: int,
        parent_pid: int | None = None,
    ) -> None:
        self._enabled = enabled
        self._size = max(16, int(size))
        self._margin_right = max(0, int(margin_right))
        self._margin_bottom = max(0, int(margin_bottom))
        self._parent_pid = int(parent_pid or os.getpid())
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def _build_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "moonshine_flow.activity_overlay",
            "--size",
            str(self._size),
            "--margin-right",
            str(self._margin_right),
            "--margin-bottom",
            str(self._margin_bottom),
            "--parent-pid",
            str(self._parent_pid),
        ]

    def _ensure_started_locked(self) -> bool:
        if not self._enabled:
            return False
        process = self._process
        if process is not None and process.poll() is None:
            return True

        self._process = None
        command = self._build_command()
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception as exc:
            LOGGER.warning("Failed to start activity overlay process: %s", exc)
            return False

        if process.stdin is None:
            try:
                process.terminate()
            except Exception:
                pass
            LOGGER.warning("Activity overlay process started without stdin; disabling overlay")
            return False

        self._process = process
        LOGGER.debug("Started activity overlay process")
        return True

    @staticmethod
    def _write_command_locked(process: subprocess.Popen[str], command: str) -> bool:
        try:
            if process.stdin is None:
                return False
            process.stdin.write(command + "\n")
            process.stdin.flush()
            return True
        except Exception:
            return False

    def _send_command(self, command: str, *, ensure_started: bool) -> None:
        with self._lock:
            if ensure_started and not self._ensure_started_locked():
                return

            process = self._process
            if process is None:
                return

            if process.poll() is not None:
                self._process = None
                if ensure_started and self._ensure_started_locked():
                    process = self._process
                else:
                    return

            if process is None:
                return

            if self._write_command_locked(process, command):
                return

            self._process = None
            if not ensure_started:
                return
            if not self._ensure_started_locked():
                return
            process = self._process
            if process is None:
                return
            if not self._write_command_locked(process, command):
                self._process = None

    def show_recording(self) -> None:
        self._send_command("SHOW_RECORDING", ensure_started=True)

    def show_processing(self) -> None:
        self._send_command("SHOW_PROCESSING", ensure_started=True)

    def hide(self) -> None:
        self._send_command("HIDE", ensure_started=False)

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None

        if process is None:
            return

        try:
            self._write_command_locked(process, "EXIT")
            process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                process.terminate()
                process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=0.5)
                except Exception:
                    pass
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass


def create_activity_indicator(config: AppConfig) -> ActivityIndicatorPort:
    global _OVERLAY_UNAVAILABLE_WARNED

    runtime_cfg = getattr(config, "runtime", None)
    enabled = bool(getattr(runtime_cfg, "activity_indicator_enabled", True))
    if not enabled:
        return NullActivityIndicator()
    if not _cocoa_overlay_available():
        if not _OVERLAY_UNAVAILABLE_WARNED:
            LOGGER.warning(
                "Activity indicator is disabled because Cocoa/AppKit runtime is unavailable"
            )
            _OVERLAY_UNAVAILABLE_WARNED = True
        return NullActivityIndicator()

    return SubprocessActivityIndicator(
        enabled=True,
        size=int(getattr(runtime_cfg, "activity_indicator_size", 42)),
        margin_right=int(getattr(runtime_cfg, "activity_indicator_margin_right", 24)),
        margin_bottom=int(getattr(runtime_cfg, "activity_indicator_margin_bottom", 24)),
    )


def _cocoa_overlay_available() -> bool:
    if sys.platform != "darwin":
        return False
    return importlib.util.find_spec("AppKit") is not None
