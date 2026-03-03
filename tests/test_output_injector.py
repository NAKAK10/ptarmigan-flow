import subprocess

import pyperclip
import pytest

from moonshine_flow.output_injector import OutputInjector


def test_parse_shortcut() -> None:
    key, modifiers = OutputInjector._parse_shortcut("cmd+shift+v")

    assert key == "v"
    assert modifiers == ["command down", "shift down"]


def test_inject_direct_typing_does_not_touch_clipboard(monkeypatch) -> None:
    calls: dict[str, object] = {"copy": None, "command": None}

    def _fake_copy(text: str) -> None:
        calls["copy"] = text

    def _fake_run(command, check: bool):
        calls["command"] = list(command)
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pyperclip, "copy", _fake_copy)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(OutputInjector, "_send_text_via_quartz", staticmethod(lambda _text: True))

    injector = OutputInjector(mode="direct_typing", paste_shortcut="cmd+v")
    assert injector.inject("hello") is True
    assert calls["copy"] is None
    assert calls["command"] is None


def test_inject_direct_typing_falls_back_to_applescript(monkeypatch) -> None:
    calls: dict[str, object] = {"copy": None, "command": None}

    def _fake_copy(text: str) -> None:
        calls["copy"] = text

    def _fake_run(command, check: bool):
        calls["command"] = list(command)
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pyperclip, "copy", _fake_copy)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(OutputInjector, "_send_text_via_quartz", staticmethod(lambda _text: False))

    injector = OutputInjector(mode="direct_typing", paste_shortcut="cmd+v")
    assert injector.inject("hello") is True
    assert calls["copy"] is None
    assert calls["command"] == [
        "osascript",
        "-e",
        'tell application "System Events" to keystroke "hello"',
    ]


def test_inject_clipboard_paste_uses_clipboard(monkeypatch) -> None:
    calls: dict[str, object] = {"copy": None, "command": None}

    def _fake_copy(text: str) -> None:
        calls["copy"] = text

    def _fake_run(command, check: bool):
        calls["command"] = list(command)
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pyperclip, "copy", _fake_copy)
    monkeypatch.setattr(subprocess, "run", _fake_run)

    injector = OutputInjector(mode="clipboard_paste", paste_shortcut="cmd+v")
    assert injector.inject("hello") is True
    assert calls["copy"] == "hello"
    assert calls["command"] == [
        "osascript",
        "-e",
        'tell application "System Events" to keystroke "v" using {command down}',
    ]


def test_inject_unsupported_mode_raises() -> None:
    injector = OutputInjector(mode="unknown", paste_shortcut="cmd+v")
    with pytest.raises(ValueError, match="Unsupported output mode"):
        injector.inject("hello")
