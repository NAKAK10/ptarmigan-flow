"""Helpers for restarting the packaged macOS app."""

from __future__ import annotations

import shlex
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
    executable: str | Path = sys.executable,
) -> bool:
    """Schedule a relaunch and report whether the relaunch command was started."""
    bundle_path = _app_bundle_path_for(executable)
    try:
        if bundle_path is not None:
            runner(["/bin/sh", "-c", f"sleep 1; open {shlex.quote(str(bundle_path))}"])
        else:
            runner([str(executable)])
    except OSError:
        return False
    return True
