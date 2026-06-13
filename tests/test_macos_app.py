from __future__ import annotations

import sys
from pathlib import Path

import ptarmigan_flow.cli as cli_module
from ptarmigan_flow import macos_app
from ptarmigan_flow.app_icon import APP_ICON_FILE


def _macos_app_source() -> str:
    return (Path(__file__).resolve().parents[1] / "src/ptarmigan_flow/macos_app.py").read_text(
        encoding="utf-8",
    )


def test_dispatch_cli_args_handles_launchd_python_module_form(monkeypatch) -> None:
    executable = "/Applications/PtarmiganFlow.app/Contents/MacOS/PtarmiganFlow"
    app_argv = [
        executable,
        "-m",
        "ptarmigan_flow.cli",
        "run",
        "--config",
        "/tmp/config.toml",
    ]
    observed: dict[str, list[str]] = {}

    def fake_cli_main() -> int:
        observed["argv"] = sys.argv[:]
        return 7

    monkeypatch.setattr(cli_module, "main", fake_cli_main)
    monkeypatch.setattr(sys, "argv", app_argv[:])

    result = macos_app._dispatch_cli_args(sys.argv[1:])

    assert result == 7
    assert observed["argv"] == [executable, "run", "--config", "/tmp/config.toml"]
    assert sys.argv == app_argv


def test_dispatch_cli_args_routes_activity_overlay_subprocess(monkeypatch) -> None:
    import ptarmigan_flow.activity_overlay as overlay_module

    observed: dict[str, list[str] | None] = {}

    def fake_overlay_main(argv: list[str] | None = None) -> int:
        observed["argv"] = argv
        return 0

    monkeypatch.setattr(overlay_module, "main", fake_overlay_main)

    overlay_args = ["--size", "42", "--margin-right", "24", "--parent-pid", "123"]
    result = macos_app._dispatch_cli_args(["-m", "ptarmigan_flow.activity_overlay", *overlay_args])

    assert result == 0
    assert observed["argv"] == overlay_args


def test_dispatch_cli_args_ignores_normal_app_launch() -> None:
    assert macos_app._dispatch_cli_args([]) is None


def test_dispatch_cli_args_ignores_unknown_module() -> None:
    assert macos_app._dispatch_cli_args(["-m", "ptarmigan_flow.unknown"]) is None


def test_macos_app_sets_runtime_application_icon() -> None:
    source = _macos_app_source()

    assert "importlib.resources" in source
    assert "NSImage" in source
    assert "setApplicationIconImage_" in source
    assert "APP_ICON_FILE" in source
    assert APP_ICON_FILE == "PtarmiganFlow.icns"


def test_macos_app_wires_in_process_daemon_controller() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow.app_daemon_controller import (" in source
    assert "DaemonController" in source
    assert "build_daemon_from_config" in source
    assert "self.daemon_controller = DaemonController" in source
    assert "lambda: build_daemon_from_config(default_config_path())" in source


def test_macos_app_exposes_start_stop_dictation_actions() -> None:
    source = _macos_app_source()

    assert "Start Dictation" in source
    assert "Stop Dictation" in source
    assert 'self._button("Start Dictation", "startDictation:"' in source
    assert 'self._button("Stop Dictation", "stopDictation:"' in source
    assert "def startDictation_(self, _sender):" in source
    assert "def stopDictation_(self, _sender):" in source
    assert "self.daemon_controller.start()" in source
    assert "self.daemon_controller.stop()" in source


def test_macos_app_auto_starts_daemon_and_stops_on_termination() -> None:
    source = _macos_app_source()

    assert "if report.all_granted:" in source
    assert "self._start_daemon_if_ready(" in source
    assert 'success_message="All permissions granted. Dictation started."' in source
    assert "def applicationWillTerminate_(self, _notification):" in source
    assert "self.daemon_controller.stop()" in source
