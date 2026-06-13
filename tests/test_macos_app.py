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


def test_macos_app_uses_onboarding_flow_for_step_wizard() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow.onboarding_flow import OnboardingFlow" in source
    assert "self.onboarding_flow = OnboardingFlow()" in source
    assert "self.onboarding_flow.current_step" in source
    assert 'if step == "language":' in source
    assert 'elif step == "done":' in source


def test_macos_app_polls_current_permission_step_without_manual_refresh() -> None:
    source = _macos_app_source()

    assert "NSTimer" in source
    assert "scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_" in source
    assert "def pollPermissions_(self, _timer):" in source
    assert "def applicationDidBecomeActive_(self, _notification):" in source
    assert "self._refresh_onboarding_permissions()" in source
    assert '"Refresh"' not in source
    assert "refreshStatus_" not in source


def test_macos_app_language_selection_saves_supported_codes_to_config() -> None:
    source = _macos_app_source()

    assert "load_config" in source
    assert "write_config" in source
    assert '"English"' in source
    assert '"Japanese"' in source
    assert '"Chinese"' in source
    assert 'self._choose_language("en")' in source
    assert 'self._choose_language("ja")' in source
    assert 'self._choose_language("zh")' in source


def test_macos_app_permission_steps_have_allow_and_system_settings_actions() -> None:
    source = _macos_app_source()

    assert '"Allow"' in source
    assert '"Open System Settings"' in source
    assert '"requestMicrophone:"' in source
    assert '"requestAccessibility:"' in source
    assert '"requestInputMonitoring:"' in source
    assert "def openSystemSettings_(self, _sender):" in source
    assert "x-apple.systempreferences:com.apple.preference.security" in source


def test_macos_app_creates_status_bar_menu() -> None:
    source = _macos_app_source()

    assert "NSStatusBar" in source
    assert "NSVariableStatusItemLength" in source
    assert "statusItemWithLength_" in source
    assert "self.status_item" in source
    assert "self.status_menu" in source
    assert '"Dictation Stopped"' in source
    assert '"Start Dictation"' in source
    assert '"Stop Dictation"' in source
    assert '"Settings"' in source
    assert '"Edit Dictionary"' in source
    assert '"Open Config"' in source
    assert '"Login at Startup"' in source
    assert '"Quit"' in source


def test_macos_app_wires_corrections_editor_window() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow.corrections_editor_model import CorrectionsEditorModel" in source
    assert "resolve_dictionary_path" in source
    assert "self.corrections_model = CorrectionsEditorModel.load" in source
    assert "def showDictionaryEditor_(self, _sender):" in source
    assert "self._build_dictionary_window()" in source
    assert "Dictionary Editor" in source
    assert "NSTableView" in source or "dictionary_row_controls" in source
    assert "addExactCorrectionRow:" in source
    assert "addRegexCorrectionRow:" in source
    assert "deleteDictionaryRow:" in source


def test_macos_app_dictionary_editor_validates_and_saves() -> None:
    source = _macos_app_source()

    assert "def saveDictionary_(self, _sender):" in source
    assert "self._sync_dictionary_model_from_controls()" in source
    assert "self.corrections_model.validate()" in source
    assert "self.corrections_model.save(self.dictionary_path)" in source
    assert "Invalid dictionary rule" in source
    assert "Dictionary saved. Restart dictation to apply changes." in source


def test_macos_app_wires_login_item_toggle_with_checkmark() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow import login_item" in source
    assert "def toggleLoginAtStartup_(self, _sender):" in source
    assert "login_item.is_enabled()" in source
    assert "login_item.register()" in source
    assert "login_item.unregister()" in source
    assert "self.login_menu_item.setState_(" in source
    assert "NSControlStateValueOn" in source
    assert "NSControlStateValueOff" in source


def test_macos_app_removes_launchd_buttons_from_app_ui() -> None:
    source = _macos_app_source()

    assert 'self._button("Install Login Startup", "installLaunchAgent:"' not in source
    assert 'self._button("Restart Daemon", "restartLaunchAgent:"' not in source
    assert "def installLaunchAgent_(self, _sender):" not in source
    assert "def restartLaunchAgent_(self, _sender):" not in source
    assert "install_launch_agent" not in source
    assert "restart_launch_agent" not in source
