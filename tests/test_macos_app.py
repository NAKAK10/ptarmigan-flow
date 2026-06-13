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


def test_main_calls_freeze_support_before_dispatch(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        macos_app.multiprocessing,
        "freeze_support",
        lambda: calls.append("freeze_support"),
    )

    def fake_dispatch(_argv):
        calls.append("dispatch")
        return 0

    monkeypatch.setattr(macos_app, "_dispatch_cli_args", fake_dispatch)

    assert macos_app.main() == 0
    # freeze_support must run before any CLI dispatch / GUI launch so that
    # multiprocessing worker children never fall through to the GUI entry point.
    assert calls == ["freeze_support", "dispatch"]


def test_macos_app_sets_runtime_application_icon() -> None:
    source = _macos_app_source()

    assert "importlib.resources" in source
    assert "NSImage" in source
    assert "setApplicationIconImage_" in source
    assert "APP_ICON_FILE" in source
    assert APP_ICON_FILE == "PtarmiganFlow.icns"


def test_macos_app_wires_subprocess_daemon_controller() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow.app_daemon_controller import (" in source
    assert "DaemonController" in source
    assert "daemon_run_command" in source
    assert "build_daemon_from_config" not in source
    assert "self.daemon_controller = DaemonController" in source
    assert "lambda: daemon_run_command(default_config_path())" in source


def test_macos_app_exposes_start_stop_dictation_actions() -> None:
    source = _macos_app_source()

    assert 'self._button(strings["start_dictation_button"], "startDictation:"' in source
    assert 'self._button(strings["stop_dictation_button"], "stopDictation:"' in source
    assert "def startDictation_(self, _sender):" in source
    assert "def stopDictation_(self, _sender):" in source
    assert "self.daemon_controller.start()" in source
    assert "self.daemon_controller.stop()" in source
    assert '"Start Dictation"' not in source
    assert '"Stop Dictation"' not in source


def test_macos_app_auto_starts_daemon_and_stops_on_termination() -> None:
    source = _macos_app_source()

    assert "if report.all_granted:" in source
    assert "self._start_daemon_if_ready(" in source
    assert 'success_message_key="all_permissions_granted_started_message"' in source
    assert "def applicationWillTerminate_(self, _notification):" in source
    assert "self.daemon_controller.stop()" in source


def test_macos_app_guards_auto_start_by_configured_backend_availability() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow.stt import availability" in source
    assert "from ptarmigan_flow.stt.factory import parse_stt_model" in source
    assert "backend, _model_id = parse_stt_model(model_token)" in source
    assert "availability.is_backend_available(backend)" in source
    assert 'strings["model_unavailable_message"].format(model=model_token)' in source
    start_method = source.split("def _start_daemon_if_ready", maxsplit=1)[1].split(
        "def pollPermissions_",
        maxsplit=1,
    )[0]
    assert "if not self._configured_backend_is_available():" in start_method
    assert "self.daemon_controller.start()" in start_method


def test_macos_app_uses_onboarding_flow_for_step_wizard() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow.onboarding_flow import OnboardingFlow" in source
    assert "from ptarmigan_flow import onboarding_flow as onboarding_flow_module" in source
    assert "from ptarmigan_flow import app_relaunch, login_item, onboarding_strings" in source
    assert "self.onboarding_flow = OnboardingFlow()" in source
    assert "self.onboarding_flow.current_step" in source
    assert "self.ui_language" in source
    assert "load_config(default_config_path()).language" in source
    assert "onboarding_strings.strings_for(self.ui_language)" in source
    assert 'if step == "language":' in source
    assert 'elif step == "done":' in source


def test_macos_app_starts_onboarding_from_persisted_language_state() -> None:
    source = _macos_app_source()

    assert "language_was_selected()" in source
    assert "self.onboarding_flow.start(" in source
    assert "report=check_all_permissions()" in source
    assert "language_already_selected=onboarding_flow_module.language_was_selected()" in source


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
    assert 'strings["language_english"]' in source
    assert 'strings["language_japanese"]' in source
    assert 'strings["language_chinese"]' in source
    assert '"Japanese"' not in source
    assert '"Chinese"' not in source
    assert 'self._choose_language("en")' in source
    assert 'self._choose_language("ja")' in source
    assert 'self._choose_language("zh")' in source
    assert "self.ui_language = code" in source
    assert "onboarding_flow_module.mark_language_selected()" in source
    assert "self._render_current_step()" in source


def test_macos_app_does_not_front_completed_onboarding_window_on_launch() -> None:
    source = _macos_app_source()
    build_window = source.split("def _build_window", maxsplit=1)[1].split(
        "@objc.python_method",
        maxsplit=1,
    )[0]
    launch_tail = source.split("delegate = OnboardingController.alloc().init()", maxsplit=1)[
        1
    ].split("app.run()", maxsplit=1)[0]

    assert "self._show_onboarding_window_if_needed()" in source
    assert "if not self.onboarding_flow.is_complete:" in source
    assert "self.window.makeKeyAndOrderFront_(None)" not in build_window
    assert "activateIgnoringOtherApps_" not in build_window
    assert "activateIgnoringOtherApps_" not in launch_tail


def test_macos_app_resolves_onboarding_copy_from_selected_language() -> None:
    source = _macos_app_source()

    assert 'strings["app_setup_title"]' in source
    assert 'strings["choose_language_title"]' in source
    assert 'strings["choose_language_body"]' in source
    assert 'strings["done_title"]' in source
    assert 'strings["done_body"]' in source
    assert 'strings["start_dictation_button"]' in source
    assert 'strings["stop_dictation_button"]' in source
    assert 'strings["settings_button"]' in source
    assert 'strings["login_at_startup_button"]' in source
    assert '"title_key": "microphone_title"' in source
    assert '"body_key": "microphone_body"' in source
    assert '"title_key": "accessibility_title"' in source
    assert '"body_key": "accessibility_body"' in source
    assert '"title_key": "input_monitoring_title"' in source
    assert '"body_key": "input_monitoring_body"' in source
    assert 'strings[config["title_key"]]' in source
    assert 'strings[config["body_key"]]' in source


def test_macos_app_permission_steps_have_allow_and_system_settings_actions() -> None:
    source = _macos_app_source()

    assert 'strings["allow_button"]' in source
    assert 'strings["open_system_settings_button"]' in source
    assert '"requestMicrophone:"' in source
    assert '"requestAccessibility:"' in source
    assert '"requestInputMonitoring:"' in source
    assert "def openSystemSettings_(self, _sender):" in source
    assert "x-apple.systempreferences:com.apple.preference.security" in source


def test_macos_app_wires_restart_only_for_restart_sensitive_permission_steps() -> None:
    source = _macos_app_source()
    permission_renderer = source.split("def _render_permission_step", maxsplit=1)[1].split(
        "@objc.python_method",
        maxsplit=1,
    )[0]

    assert 'if step in {"accessibility", "input_monitoring"}:' in permission_renderer
    assert 'strings["restart_required_note"]' in permission_renderer
    assert 'self._button(strings["restart_app_button"], "restartApp:"' in permission_renderer
    assert "microphone" not in permission_renderer.split("if step in", maxsplit=1)[1].split(
        ":",
        maxsplit=1,
    )[0]


def test_macos_app_restart_action_relaunches_then_terminates_current_process() -> None:
    source = _macos_app_source()

    assert "def restartApp_(self, _sender):" in source
    assert "if app_relaunch.relaunch_app():" in source
    assert "NSApplication.sharedApplication().terminate_(self)" in source


def test_macos_app_creates_status_bar_menu() -> None:
    source = _macos_app_source()

    assert "NSStatusBar" in source
    assert "NSVariableStatusItemLength" in source
    assert "statusItemWithLength_" in source
    assert "self.status_item" in source
    assert "self.status_menu" in source
    assert 'strings["dictation_stopped_menu"]' in source
    assert 'strings["start_dictation_button"]' in source
    assert 'strings["stop_dictation_button"]' in source
    assert 'strings["settings_menu"]' in source
    assert 'strings["edit_dictionary_menu"]' in source
    assert 'strings["login_at_startup_menu"]' in source
    assert 'strings["quit_menu"]' in source
    assert '"Open Config"' not in source


def test_macos_app_wires_settings_form_window() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow.app_settings_model import (" in source
    assert "AppSettingsModel" in source
    assert "NSPopUpButton" in source
    assert "def _build_settings_window(self) -> None:" in source
    assert "def _render_settings_form(self) -> None:" in source
    assert "def saveSettings_(self, _sender):" in source
    assert "AppSettingsModel.load(default_config_path())" in source
    assert "self.settings_model.validate()" in source
    assert "self.settings_model.save(default_config_path())" in source
    assert 'strings["open_config_advanced_button"]' in source
    assert 'self._button(strings["settings_button"], "showSettings:"' in source
    assert 'self._menu_item(strings["settings_menu"], "showSettings:")' in source


def test_macos_app_wires_corrections_editor_window() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow.corrections_editor_model import CorrectionsEditorModel" in source
    assert "resolve_dictionary_path" in source
    assert "self.corrections_model = CorrectionsEditorModel.load" in source
    assert "def showDictionaryEditor_(self, _sender):" in source
    assert "self._build_dictionary_window()" in source
    assert 'strings["dictionary_editor_title"]' in source
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
    assert 'strings["dictionary_invalid_rule_message"].format(' in source
    assert 'strings["dictionary_saved_message"]' in source


def test_macos_app_wires_login_item_toggle_with_checkmark() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow import app_relaunch, login_item, onboarding_strings" in source
    assert "def toggleLoginAtStartup_(self, _sender):" in source
    assert "login_item.is_enabled()" in source
    assert "login_item.register()" in source
    assert "login_item.unregister()" in source
    assert "self.login_menu_item.setState_(" in source
    assert "NSControlStateValueOn" in source
    assert "NSControlStateValueOff" in source


def test_macos_app_routes_user_messages_through_localized_strings() -> None:
    source = _macos_app_source()

    for literal in (
        "Grant all permissions before starting dictation.",
        "Dictation daemon is not running yet.",
        "Could not start dictation:",
        "Dictation started.",
        "All permissions granted. Dictation started.",
        "Dictation stopped.",
        "Could not restart app.",
        "Saved language to",
        "Could not save language:",
        "Opened config:",
        "Login at startup enabled.",
        "Login at startup disabled.",
        "Could not enable login at startup.",
        "Could not disable login at startup.",
        "Dictionary Editor",
        "Exact Rules",
        "Regex Rules",
        "Canonical",
        "Candidates / Patterns (comma-separated)",
        "No rules yet.",
        "Add Exact",
        "Add Regex",
        "Delete",
        "Dictionary saved. Restart dictation to apply changes.",
        "Invalid dictionary rule:",
    ):
        assert literal not in source


def test_macos_app_removes_launchd_buttons_from_app_ui() -> None:
    source = _macos_app_source()

    assert 'self._button("Install Login Startup", "installLaunchAgent:"' not in source
    assert 'self._button("Restart Daemon", "restartLaunchAgent:"' not in source
    assert "def installLaunchAgent_(self, _sender):" not in source
    assert "def restartLaunchAgent_(self, _sender):" not in source
    assert "install_launch_agent" not in source
    assert "restart_launch_agent" not in source
