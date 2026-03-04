from __future__ import annotations

from moonshine_flow.activity_indicator import (
    NullActivityIndicator,
    SubprocessActivityIndicator,
    create_activity_indicator,
)
from moonshine_flow.config import AppConfig


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, text: str) -> int:
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        return None


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.running = True
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return None if self.running else 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        self.running = False
        return 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.running = False

    def kill(self) -> None:
        self.kill_calls += 1
        self.running = False


def test_show_recording_starts_overlay_and_sends_command(monkeypatch) -> None:
    started: list[_FakeProcess] = []

    def _fake_popen(*_args, **_kwargs):
        process = _FakeProcess()
        started.append(process)
        return process

    monkeypatch.setattr("moonshine_flow.activity_indicator.subprocess.Popen", _fake_popen)

    indicator = SubprocessActivityIndicator(
        enabled=True,
        size=42,
        margin_right=24,
        margin_bottom=24,
        parent_pid=123,
    )
    indicator.show_recording()

    assert len(started) == 1
    assert started[0].stdin.writes == ["SHOW_RECORDING\n"]


def test_show_processing_reuses_running_overlay(monkeypatch) -> None:
    started: list[_FakeProcess] = []

    def _fake_popen(*_args, **_kwargs):
        process = _FakeProcess()
        started.append(process)
        return process

    monkeypatch.setattr("moonshine_flow.activity_indicator.subprocess.Popen", _fake_popen)

    indicator = SubprocessActivityIndicator(
        enabled=True,
        size=42,
        margin_right=24,
        margin_bottom=24,
        parent_pid=123,
    )
    indicator.show_recording()
    indicator.show_processing()

    assert len(started) == 1
    assert started[0].stdin.writes == ["SHOW_RECORDING\n", "SHOW_PROCESSING\n"]


def test_hide_without_process_is_noop() -> None:
    indicator = SubprocessActivityIndicator(
        enabled=True,
        size=42,
        margin_right=24,
        margin_bottom=24,
        parent_pid=123,
    )

    indicator.hide()


def test_close_sends_exit_and_waits(monkeypatch) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(
        "moonshine_flow.activity_indicator.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )

    indicator = SubprocessActivityIndicator(
        enabled=True,
        size=42,
        margin_right=24,
        margin_bottom=24,
        parent_pid=123,
    )
    indicator.show_recording()

    indicator.close()

    assert process.stdin.writes == ["SHOW_RECORDING\n", "EXIT\n"]
    assert process.wait_calls >= 1


def test_create_activity_indicator_respects_runtime_toggle() -> None:
    config = AppConfig()
    config.runtime.activity_indicator_enabled = False

    indicator = create_activity_indicator(config)

    assert isinstance(indicator, NullActivityIndicator)


def test_create_activity_indicator_respects_ui_enabled_toggle() -> None:
    config = AppConfig()
    config.runtime.ui_enabled = False
    config.runtime.activity_indicator_enabled = True

    indicator = create_activity_indicator(config)

    assert isinstance(indicator, NullActivityIndicator)


def test_create_activity_indicator_returns_null_when_cocoa_unavailable(
    monkeypatch, caplog
) -> None:
    config = AppConfig()
    config.runtime.activity_indicator_enabled = True

    monkeypatch.setattr("moonshine_flow.activity_indicator._cocoa_overlay_available", lambda: False)
    monkeypatch.setattr("moonshine_flow.activity_indicator._OVERLAY_UNAVAILABLE_WARNED", False)
    with caplog.at_level("WARNING"):
        first = create_activity_indicator(config)
        second = create_activity_indicator(config)

    assert isinstance(first, NullActivityIndicator)
    assert isinstance(second, NullActivityIndicator)
    warnings = [
        entry.message
        for entry in caplog.records
        if (
            "Activity indicator is disabled because Cocoa/AppKit runtime is unavailable"
            in entry.message
        )
    ]
    assert len(warnings) == 1


def test_create_activity_indicator_uses_runtime_bounds(monkeypatch) -> None:
    config = AppConfig()
    config.runtime.activity_indicator_enabled = True
    config.runtime.activity_indicator_size = 10
    config.runtime.activity_indicator_margin_right = -3
    config.runtime.activity_indicator_margin_bottom = -4

    monkeypatch.setattr("moonshine_flow.activity_indicator._cocoa_overlay_available", lambda: True)
    indicator = create_activity_indicator(config)

    assert isinstance(indicator, SubprocessActivityIndicator)
    assert indicator._size == 16
    assert indicator._margin_right == 0
    assert indicator._margin_bottom == 0
