from __future__ import annotations

import sys
from pathlib import Path

import ptarmigan_flow.cli as cli_module
from ptarmigan_flow import macos_app
from ptarmigan_flow.app_icon import APP_ICON_FILE

ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _macos_app_source() -> str:
    return _source("src/ptarmigan_flow/macos_app.py")


def _web_ui_source() -> str:
    return _source("src/ptarmigan_flow/web_ui.py")


def _web_bridge_source() -> str:
    return _source("src/ptarmigan_flow/web_bridge.py")


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
    assert calls == ["freeze_support", "dispatch"]


def test_macos_app_sets_runtime_application_icon() -> None:
    source = _macos_app_source()

    assert "importlib.resources" in source
    assert "NSImage" in source
    assert "setApplicationIconImage_" in source
    assert "APP_ICON_FILE" in source
    assert APP_ICON_FILE == "PtarmiganFlow.icns"


def test_macos_app_uses_wkwebview_host_instead_of_fixed_coordinate_appkit_ui() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow.web_ui import WebUIController" in source
    assert "self.web_ui = WebUIController" in source
    assert "self.web_ui.show(route=\"onboarding\")" in source
    assert "self.web_ui.show(route=\"settings\")" in source
    assert "self.web_ui.show(route=\"dictionary\")" in source
    for removed in (
        "NSButton",
        "NSTextField",
        "NSPopUpButton",
        "NSProgressIndicator",
        "NSMakeRect(0, 0, 640, 430)",
        "dictionary_row_controls",
        "settings_model_popup",
        "download_progress_indicator",
        "def _label(",
        "def _button(",
    ):
        assert removed not in source


def test_web_ui_hosts_wkwebview_and_message_bridge() -> None:
    source = _web_ui_source()

    assert "WKWebView" in source
    assert "WKWebViewConfiguration" in source
    assert "WKUserContentController" in source
    assert 'addScriptMessageHandler_name_(self, "bridge")' in source
    assert "userContentController_didReceiveScriptMessage_" in source
    assert "window.app.dispatch(" in source
    assert "evaluateJavaScript_completionHandler_" in source
    assert "loadFileURL_allowingReadAccessToURL_" in source
    assert "importlib.resources.files" in source


def test_macos_app_creates_native_status_bar_menu() -> None:
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


def test_macos_app_wires_web_bridge_to_existing_logic() -> None:
    source = _macos_app_source()

    assert "from ptarmigan_flow.web_bridge import BridgeDependencies, WebBridgeDispatcher" in source
    assert "self.bridge = WebBridgeDispatcher" in source
    assert "BridgeDependencies(" in source
    assert "config_path=default_config_path" in source
    assert "check_permissions=check_all_permissions" in source
    assert "available_model_entries=availability.available_model_entries" in source
    assert "is_model_downloaded=model_download.is_model_downloaded" in source
    assert "resolve_dictionary_path=self._dictionary_path" in source
    assert "request_permission=self._request_permission" in source
    assert "open_system_settings=self._open_system_settings" in source
    assert "start_dictation=self._start_daemon_if_ready" in source
    assert "stop_dictation=self._stop_daemon" in source
    assert "login_is_enabled=login_item.is_enabled" in source
    assert "login_register=login_item.register" in source
    assert "login_unregister=login_item.unregister" in source
    assert "restart_app=self._restart_app" in source


def test_web_bridge_is_pyobjc_independent() -> None:
    source = _web_bridge_source()

    for forbidden in ("AppKit", "Foundation", "WebKit", "objc"):
        assert forbidden not in source
    assert "def handle_action(action: str, payload: dict)" in source
    assert "class WebBridgeDispatcher" in source
    assert "class BridgeDependencies" in source


def test_macos_app_starts_onboarding_from_persisted_language_state() -> None:
    source = _macos_app_source()

    assert "language_was_selected()" in source
    assert "self.onboarding_flow.start(" in source
    assert "report=check_all_permissions()" in source
    assert "language_already_selected=onboarding_flow_module.language_was_selected()" in source
    assert "self.bridge.set_onboarding_flow(self.onboarding_flow)" in source


def test_macos_app_polls_permissions_through_subprocess_and_pushes_changes() -> None:
    source = _macos_app_source()

    assert "NSTimer" in source
    assert 'schedule_timer(1.75, self, "pollPermissions:", None, True)' in source
    assert "check_all_permissions_subprocess" in source
    assert "report = check_all_permissions_subprocess()" in source
    assert "if report is None:" in source
    assert "report = check_all_permissions()" in source
    assert "threading.Thread(" in source
    assert "daemon=True" in source
    assert "performSelectorOnMainThread_withObject_waitUntilDone_" in source
    assert "def applyPermissionCheckResult_(self, payload):" in source
    assert 'self._push_event("permissionsChanged", self._state_payload())' in source


def test_macos_app_permission_refresh_auto_advances_and_starts_daemon_when_ready() -> None:
    source = _macos_app_source()
    refresh_method = source.split("def _refresh_onboarding_permissions", maxsplit=1)[1].split(
        "@objc.python_method",
        maxsplit=1,
    )[0]

    assert "before_step = self.onboarding_flow.current_step" in refresh_method
    assert "self.onboarding_flow.refresh(report)" in refresh_method
    assert "after_step = self.onboarding_flow.current_step" in refresh_method
    assert "if after_step != before_step:" in refresh_method
    assert "if report.all_granted:" in refresh_method
    assert "self._start_daemon_if_ready(" in refresh_method
    assert 'success_message_key="all_permissions_granted_started_message"' in refresh_method


def test_macos_app_downloads_model_with_jsonl_child_process_and_pushes_progress() -> None:
    source = _macos_app_source()

    assert "import json" in source
    assert "subprocess.Popen(" in source
    assert "sys.executable" in source
    assert '"-m"' in source
    assert '"ptarmigan_flow.cli"' in source
    assert '"download-model"' in source
    assert "stdout=subprocess.PIPE" in source
    assert "stderr=subprocess.STDOUT" in source
    assert "text=True" in source
    assert "target=self._read_model_download_progress" in source
    assert "json.loads(line)" in source
    assert 'self._push_event("downloadProgress", payload)' in source
    assert 'self._push_event("daemonState", self._state_payload())' in source
    assert "NSProgressIndicator" not in source


def test_macos_app_guards_daemon_start_by_permissions_backend_and_model_download() -> None:
    source = _macos_app_source()
    start_method = source.split("def _start_daemon_if_ready", maxsplit=1)[1].split(
        "@objc.python_method",
        maxsplit=1,
    )[0]

    assert "check_all_permissions()" in start_method
    assert "if not report.all_granted:" in start_method
    assert "if not self._configured_backend_is_available():" in start_method
    assert "model_token = self._configured_model_token()" in start_method
    assert "model_download.is_model_downloaded(model_token)" in start_method
    assert "self._start_model_download(model_token, success_message_key)" in start_method
    assert "self.daemon_controller.start()" in start_method
    assert "self._push_daemon_state()" in start_method


def test_macos_app_exposes_menu_actions_to_web_routes_and_bridge_side_effects() -> None:
    source = _macos_app_source()

    assert "def startDictation_(self, _sender):" in source
    assert "def stopDictation_(self, _sender):" in source
    assert "def showSettings_(self, _sender):" in source
    assert "def showDictionaryEditor_(self, _sender):" in source
    assert "def toggleLoginAtStartup_(self, _sender):" in source
    assert "def restartApp_(self, _sender):" in source
    assert "def _restart_app(self) -> bool:" in source
    assert "self._start_daemon_if_ready()" in source
    assert "self._stop_daemon()" in source
    assert "self._set_route(\"settings\")" in source
    assert "self._set_route(\"dictionary\")" in source
    assert "self._toggle_login()" in source
    assert "NSApplication.sharedApplication().terminate_(self)" in source


def test_macos_app_wires_login_item_toggle_with_checkmark() -> None:
    source = _macos_app_source()

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
