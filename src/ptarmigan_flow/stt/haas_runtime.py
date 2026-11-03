"""Discovery and cancellable, bounded speech.cpp invocation; no ML imports."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

SPEECH_CPP_PATH_ENV = "PTARMIGAN_FLOW_SPEECH_CPP_PATH"
MIN_SPEECH_CPP_VERSION = (0, 9, 4)
_VERSION_TIMEOUT_SECONDS = 5.0
_REAP_TIMEOUT_SECONDS = 2.0
_MAX_STDOUT_BYTES = 1024 * 1024
_MAX_DIAGNOSTIC_BYTES = 2000
_INSTALL_GUIDANCE = (
    "Install speech.cpp >=0.9.4 from https://github.com/nyosegawa/speech.cpp/releases "
    f"and set {SPEECH_CPP_PATH_ENV} to its speech executable if needed."
)


def find_speech_executable() -> Path | None:
    """Find speech even in a GUI's restricted PATH; an explicit override must work.

    Search order: PTARMIGAN_FLOW_SPEECH_CPP_PATH, PATH, ~/.local/bin,
    /opt/homebrew/bin, /usr/local/bin. The override is one path, never shell text.
    This only checks files; version verification belongs in background preflight.
    """
    override = os.environ.get(SPEECH_CPP_PATH_ENV)
    if override is not None:
        candidate = Path(override).expanduser()
        if not override.strip() or not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RuntimeError(f"Invalid {SPEECH_CPP_PATH_ENV}: {override!r}. {_INSTALL_GUIDANCE}")
        return candidate.absolute()
    on_path = shutil.which("speech")
    candidates = ([Path(on_path)] if on_path else []) + [
        Path.home() / ".local/bin/speech",
        Path("/opt/homebrew/bin/speech"),
        Path("/usr/local/bin/speech"),
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.absolute()
    return None


def parse_speech_version(output: str) -> tuple[int, int, int]:
    """Require speech.cpp's actual banner, rather than accepting an unrelated speech."""
    match = re.fullmatch(r"speech\.cpp (\d+)\.(\d+)\.(\d+), C API \d+\.\d+", output.strip())
    if match is None:
        raise RuntimeError(f"Unrecognized speech.cpp version banner. {_INSTALL_GUIDANCE}")
    version = tuple(int(part) for part in match.groups())
    if version < MIN_SPEECH_CPP_VERSION:
        raise RuntimeError(f"speech.cpp {'.'.join(match.groups())} is too old. {_INSTALL_GUIDANCE}")
    return version


class SpeechCppCancelledError(RuntimeError):
    """An explicit cancellation invalidated the current request."""


class SpeechCppRuntime:
    """Own at most one direct child; cancellation never waits on the request lock.

    A generation spans all steps of a backend request, so cancellation between
    version checking and inference cannot accidentally start another process.
    stdout/stderr use temporary files to avoid pipe deadlocks and unbounded RAM.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._executable: Path | None = None
        self._generation = 0
        self._closed = False

    @property
    def generation(self) -> int:
        with self._lock:
            self._check_generation(self._generation)
            return self._generation

    def _check_generation(self, generation: int) -> None:
        if self._closed or generation != self._generation:
            raise SpeechCppCancelledError("speech.cpp request cancelled or backend closed")

    @staticmethod
    def _kill_and_reap(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("speech.cpp did not exit after being killed") from exc

    def cancel(self) -> None:
        """Cancel active and intervening work; a later request may start afresh."""
        with self._lock:
            self._generation += 1
            process = self._process
        if process is not None:
            self._kill_and_reap(process)
            with self._lock:
                if self._process is process:
                    self._process = None

    def close(self) -> None:
        """Permanently close this runtime and reap any active child."""
        with self._lock:
            self._closed = True
        self.cancel()

    def check_version(self, *, generation: int) -> tuple[int, int, int]:
        return parse_speech_version(
            self.run(["--version"], timeout_seconds=_VERSION_TIMEOUT_SECONDS, generation=generation)
        )

    def run(self, args: list[str], *, timeout_seconds: float, generation: int) -> str:
        """Run an argv list, rejecting cancellation, failures, and invalid UTF-8."""
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("speech.cpp timeout must be finite and positive")
        with (
            self._request_lock,
            tempfile.TemporaryFile() as stdout,
            tempfile.TemporaryFile() as stderr,
        ):
            with self._lock:
                self._check_generation(generation)
                if self._process is not None:
                    raise RuntimeError(
                        "Previous speech.cpp process has not exited; close and retry"
                    )
                if self._executable is None:
                    self._executable = find_speech_executable()
                if self._executable is None:
                    raise RuntimeError(f"speech.cpp executable not found. {_INSTALL_GUIDANCE}")
                env = dict(os.environ, SPEECH_NO_UPDATE_CHECK="1", NO_COLOR="1")
                try:
                    process = subprocess.Popen(
                        [str(self._executable), *args],
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        env=env,
                        shell=False,
                    )
                except OSError as exc:
                    raise RuntimeError(
                        f"Cannot start speech.cpp at {self._executable}: {exc}. {_INSTALL_GUIDANCE}"
                    ) from exc
                self._process = process
            try:
                try:
                    process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f"speech.cpp timed out after {timeout_seconds:.1f}s"
                    ) from exc
                with self._lock:
                    self._check_generation(generation)
                if process.returncode != 0:
                    stderr.seek(0, os.SEEK_END)
                    stderr.seek(max(0, stderr.tell() - _MAX_DIAGNOSTIC_BYTES))
                    detail = stderr.read(_MAX_DIAGNOSTIC_BYTES).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"speech.cpp exited with status {process.returncode}: {detail.strip()}"
                    )
                stdout.seek(0)
                payload = stdout.read(_MAX_STDOUT_BYTES + 1)
                if len(payload) > _MAX_STDOUT_BYTES:
                    raise RuntimeError("speech.cpp output exceeds the response size limit")
                try:
                    return payload.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise RuntimeError("speech.cpp returned invalid UTF-8") from exc
            finally:
                # Retain ownership if even SIGKILL cannot be reaped within the
                # bound. Never overwrite the handle with a second child.
                self._kill_and_reap(process)
                with self._lock:
                    if self._process is process:
                        self._process = None
