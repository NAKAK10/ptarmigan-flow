from __future__ import annotations

import sys
from pathlib import Path

from ptarmigan_flow import app_relaunch


def test_current_app_bundle_path_detects_enclosing_app_bundle(monkeypatch) -> None:
    executable = (
        "/Applications/PtarmiganFlow.app/Contents/MacOS/PtarmiganFlow"
    )
    monkeypatch.setattr(sys, "executable", executable)

    assert app_relaunch.current_app_bundle_path() == Path("/Applications/PtarmiganFlow.app")


def test_current_app_bundle_path_returns_none_outside_app_bundle(monkeypatch) -> None:
    monkeypatch.setattr(sys, "executable", "/usr/local/bin/python")

    assert app_relaunch.current_app_bundle_path() is None


def test_relaunch_app_opens_detected_bundle_with_delayed_shell_command() -> None:
    calls: list[list[str]] = []
    executable = "/Applications/PtarmiganFlow.app/Contents/MacOS/PtarmiganFlow"

    def fake_runner(args: list[str]) -> object:
        calls.append(args)
        return object()

    assert app_relaunch.relaunch_app(runner=fake_runner, executable=executable) is True

    assert calls == [
        [
            "/bin/sh",
            "-c",
            "sleep 1; open /Applications/PtarmiganFlow.app",
        ],
    ]


def test_relaunch_app_quotes_bundle_paths_with_spaces() -> None:
    calls: list[list[str]] = []
    executable = "/Applications/Ptarmigan Flow.app/Contents/MacOS/PtarmiganFlow"

    def fake_runner(args: list[str]) -> object:
        calls.append(args)
        return object()

    assert app_relaunch.relaunch_app(runner=fake_runner, executable=executable) is True

    assert calls[0] == [
        "/bin/sh",
        "-c",
        "sleep 1; open '/Applications/Ptarmigan Flow.app'",
    ]


def test_relaunch_app_falls_back_to_reexecuting_current_executable() -> None:
    calls: list[list[str]] = []

    def fake_runner(args: list[str]) -> object:
        calls.append(args)
        return object()

    assert app_relaunch.relaunch_app(runner=fake_runner, executable="/tmp/pflow") is True

    assert calls == [["/tmp/pflow"]]
