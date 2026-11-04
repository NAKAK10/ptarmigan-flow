"""Helpers for restarting the macOS app and development invocations."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

Runner = Callable[[list[str]], object]


def _app_bundle_path_for(executable: str | Path) -> Path | None:
    executable_path = Path(executable)
    for candidate in (executable_path, *executable_path.parents):
        if candidate.suffix == ".app":
            return candidate
    return None


def current_app_bundle_path() -> Path | None:
    """Return the enclosing ``.app`` bundle for the current executable."""
    return _app_bundle_path_for(sys.executable)


def relaunch_app(
    *,
    runner: Runner = subprocess.Popen,
    executable: str | Path | None = None,
) -> bool:
    """Schedule a relaunch and report whether the relaunch command was started."""
    executable = sys.executable if executable is None else executable
    bundle_path = _app_bundle_path_for(executable)
    if bundle_path is not None:
        command = ["/usr/bin/open", str(bundle_path)]
    else:
        # orig_argv preserves -m, interpreter flags, script paths and CLI options.
        command = [str(executable), *sys.orig_argv[1:]]
        if len(command) == 1:
            return False
    try:
        # Cleanup can take longer than a fixed delay. Start exactly one successor
        # after this process exits, with paths/arguments passed without interpolation.
        runner(
            [
                "/bin/sh",
                "-c",
                'while /bin/kill -0 "$1" 2>/dev/null; do /bin/sleep 0.1; done; shift; exec "$@"',
                "ptarmigan-flow-restart",
                str(os.getpid()),
                *command,
            ]
        )
    except OSError:
        return False
    return True
