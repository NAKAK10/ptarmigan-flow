"""Current resident memory for the macOS app and its subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys

_PS_TIMEOUT_SECONDS = 1.0


def sample_process_memory(pid: int | None = None) -> int | None:
    """Return current summed RSS bytes for a root and its live descendants.

    Defaults to this process, including children such as speech.cpp and the
    overlay. Shared pages can be counted more than once: this sum is neither
    total physical memory use nor total GPU allocation. The ps snapshot can
    race with process exits; unavailable or invalid samples return None.
    """
    if sys.platform != "darwin":
        return None
    root_pid = os.getpid() if pid is None else pid
    if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0:
        return None

    try:
        process = subprocess.Popen(
            ["/bin/ps", "-A", "-o", "pid=,ppid=,rss="],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
        )
    except OSError:
        return None

    try:
        output, _ = process.communicate(timeout=_PS_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.communicate(timeout=_PS_TIMEOUT_SECONDS)
        except (OSError, subprocess.SubprocessError, UnicodeError):
            pass
        finally:
            if process.stdout is not None:
                process.stdout.close()
        return None
    if process.returncode != 0:
        return None

    rows: dict[int, tuple[int, int]] = {}
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3 or not all(f.isascii() and f.isdecimal() for f in fields):
            return None
        try:
            row_pid, parent_pid, rss_kib = map(int, fields)
        except ValueError:
            return None
        row = (parent_pid, rss_kib)
        if row_pid in rows:
            if rows[row_pid] != row:
                return None
            continue
        rows[row_pid] = row
        children.setdefault(parent_pid, []).append(row_pid)

    if root_pid not in rows or root_pid == process.pid:
        return None
    total_kib = 0
    visited: set[int] = set()
    pending = [root_pid]
    while pending:
        current = pending.pop()
        if current == process.pid:
            continue
        if current in visited:
            return None
        visited.add(current)
        total_kib += rows[current][1]
        pending.extend(children.get(current, ()))
    return total_kib * 1024
