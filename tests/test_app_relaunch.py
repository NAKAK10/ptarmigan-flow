from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ptarmigan_flow import app_relaunch


def test_current_app_bundle_path_detects_enclosing_app_bundle(monkeypatch) -> None:
    executable = "/Applications/PtarmiganFlow.app/Contents/MacOS/PtarmiganFlow"
    monkeypatch.setattr(sys, "executable", executable)

    assert app_relaunch.current_app_bundle_path() == Path("/Applications/PtarmiganFlow.app")


def test_current_app_bundle_path_returns_none_outside_app_bundle(monkeypatch) -> None:
    monkeypatch.setattr(sys, "executable", "/usr/local/bin/python")

    assert app_relaunch.current_app_bundle_path() is None


@pytest.mark.parametrize("name", ["PtarmiganFlow", "Ptarmigan Flow", "Ptarmigan '$(exit 1)'"])
def test_relaunch_bundle_waits_for_exit_and_passes_path_literally(monkeypatch, name) -> None:
    calls = []
    monkeypatch.setattr(app_relaunch.os, "getpid", lambda: 12345)
    bundle = f"/Applications/{name}.app"

    assert (
        app_relaunch.relaunch_app(
            runner=calls.append, executable=f"{bundle}/Contents/MacOS/PtarmiganFlow"
        )
        is True
    )

    assert len(calls) == 1
    assert calls[0] == [
        "/bin/sh",
        "-c",
        'while /bin/kill -0 "$1" 2>/dev/null; do /bin/sleep 0.1; done; shift; exec "$@"',
        "ptarmigan-flow-restart",
        "12345",
        "/usr/bin/open",
        bundle,
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["-m", "ptarmigan_flow.macos_app"],
        ["-X", "dev", "-m", "ptarmigan_flow.cli", "app"],
        ["/project with spaces/.venv/bin/pflow", "app", "--help"],
    ],
)
def test_development_relaunch_preserves_invocation(monkeypatch, arguments) -> None:
    calls = []
    monkeypatch.setattr(sys, "executable", "/project with spaces/.venv/bin/python")
    monkeypatch.setattr(sys, "orig_argv", ["python", *arguments])

    assert app_relaunch.relaunch_app(runner=calls.append) is True

    assert len(calls) == 1
    assert calls[0][5:] == [sys.executable, *arguments]


def test_relaunch_without_development_entry_point_keeps_running(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(sys, "orig_argv", ["python"])

    assert app_relaunch.relaunch_app(runner=calls.append, executable="/tmp/python") is False
    assert calls == []


def test_relaunch_spawn_failure_keeps_running() -> None:
    def failing_runner(_args):
        raise OSError("cannot launch helper")

    assert (
        app_relaunch.relaunch_app(
            runner=failing_runner, executable="/Applications/PtarmiganFlow.app/Contents/MacOS/app"
        )
        is False
    )
