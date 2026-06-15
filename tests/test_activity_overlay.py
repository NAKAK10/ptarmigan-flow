from __future__ import annotations

from pathlib import Path

import ptarmigan_flow.activity_overlay as overlay_module

ROOT = Path(__file__).resolve().parents[1]


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


class _FakeLayer:
    def __init__(self) -> None:
        self.hidden = None
        self.background_color = None
        self.border_color = None
        self.border_width = None
        self.shadow_color = None
        self.shadow_opacity = None
        self.shadow_radius = None
        self.shadow_offset = None
        self.opacity = None

    def setHidden_(self, value) -> None:
        self.hidden = value

    def setBackgroundColor_(self, value) -> None:
        self.background_color = value

    def setBorderColor_(self, value) -> None:
        self.border_color = value

    def setBorderWidth_(self, value) -> None:
        self.border_width = value

    def setShadowColor_(self, value) -> None:
        self.shadow_color = value

    def setShadowOpacity_(self, value) -> None:
        self.shadow_opacity = value

    def setShadowRadius_(self, value) -> None:
        self.shadow_radius = value

    def setShadowOffset_(self, value) -> None:
        self.shadow_offset = value

    def setOpacity_(self, value) -> None:
        self.opacity = value


class _FakeGlassWindow:
    def __init__(self) -> None:
        self._glass_layer = _FakeLayer()

    def _color(self, r: float, g: float, b: float, a: float):
        return (r, g, b, a)


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


def test_clear_backplate_visuals_hides_glass_layer() -> None:
    window = _FakeGlassWindow()

    overlay_module.AppKitOverlayWindow._clear_backplate_visuals(window)

    assert window._glass_layer.hidden is True
    assert window._glass_layer.background_color == (0.0, 0.0, 0.0, 0.0)
    assert window._glass_layer.border_color == (0.0, 0.0, 0.0, 0.0)
    assert window._glass_layer.border_width == 0.0
    assert window._glass_layer.shadow_color == (0.0, 0.0, 0.0, 0.0)
    assert window._glass_layer.shadow_opacity == 0.0
    assert window._glass_layer.shadow_radius == 0.0
    assert window._glass_layer.shadow_offset == (0.0, 0.0)
    assert window._glass_layer.opacity == 0.0


def test_recording_and_processing_overlay_icons_use_distinct_color_families() -> None:
    source = (ROOT / "src/ptarmigan_flow/activity_overlay.py").read_text(encoding="utf-8")
    recording_method = source.split("def _start_recording_animation", maxsplit=1)[1].split(
        "def _start_processing_animation",
        maxsplit=1,
    )[0]
    processing_method = source.split("def _start_processing_animation", maxsplit=1)[1].split(
        "def _position_bottom_right",
        maxsplit=1,
    )[0]

    assert 'self._set_mode("recording")' in recording_method
    assert "self._color(1.0, 0.28, 0.41, 0.95)" in recording_method
    assert "recording.core.scale" in recording_method
    assert 'self._set_mode("processing")' in processing_method
    assert "self._color(0.41, 0.96, 1.0, 0.92)" in processing_method
    assert '"processing.ring.a"' in processing_method
    assert 'f"{key_prefix}.rotation"' in processing_method
