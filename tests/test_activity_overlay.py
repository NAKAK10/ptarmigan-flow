from __future__ import annotations

import moonshine_flow.activity_overlay as overlay_module


class _FakeWindow:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.pump_calls = 0

    def show_recording(self) -> None:
        self.calls.append("show_recording")

    def show_processing(self) -> None:
        self.calls.append("show_processing")

    def hide(self) -> None:
        self.calls.append("hide")

    def close(self) -> None:
        self.calls.append("close")

    def pump_events(self, timeout_seconds: float) -> None:
        del timeout_seconds
        self.pump_calls += 1


def test_runtime_processes_show_and_hide_commands() -> None:
    window = _FakeWindow()
    runtime = overlay_module.OverlayRuntime(window=window, parent_pid=123)

    runtime.enqueue_command("SHOW_RECORDING")
    runtime.enqueue_command("SHOW_PROCESSING")
    runtime.enqueue_command("HIDE")
    runtime.process_commands()

    assert window.calls == ["show_recording", "show_processing", "hide"]
    assert runtime.is_stopped() is False


def test_runtime_ignores_unknown_command() -> None:
    window = _FakeWindow()
    runtime = overlay_module.OverlayRuntime(window=window, parent_pid=123)

    runtime.enqueue_command("UNKNOWN")
    runtime.process_commands()

    assert window.calls == []


def test_runtime_parent_mismatch_requests_exit(monkeypatch) -> None:
    window = _FakeWindow()
    runtime = overlay_module.OverlayRuntime(window=window, parent_pid=999)

    monkeypatch.setattr(overlay_module.os, "getppid", lambda: 111)
    runtime.check_parent_alive()
    runtime.process_commands()

    assert runtime.is_stopped() is True
    assert "hide" in window.calls
    assert "close" in window.calls


def test_runtime_run_drains_exit_command_and_stops() -> None:
    window = _FakeWindow()
    runtime = overlay_module.OverlayRuntime(window=window, parent_pid=123)

    runtime.enqueue_command("SHOW_RECORDING")
    runtime.enqueue_command("EXIT")
    exit_code = runtime.run()

    assert exit_code == 0
    assert "show_recording" in window.calls
    assert "hide" in window.calls
    assert "close" in window.calls


def test_parse_args_clamps_bounds(monkeypatch) -> None:
    monkeypatch.setattr(overlay_module.os, "getppid", lambda: 555)

    parsed = overlay_module._parse_args(
        ["--size", "10", "--margin-right", "-7", "--margin-bottom", "-3"]
    )

    assert parsed.size == 16
    assert parsed.margin_right == 0
    assert parsed.margin_bottom == 0
    assert parsed.parent_pid == 555


def test_parse_args_accepts_explicit_parent_pid() -> None:
    parsed = overlay_module._parse_args(["--size", "56", "--parent-pid", "4321"])

    assert parsed.size == 56
    assert parsed.parent_pid == 4321
