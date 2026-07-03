from __future__ import annotations

import sys
import types
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


def _webui_app_source() -> str:
    return _source("src/ptarmigan_flow/webui/app.js")


def _capture_app_delegate(monkeypatch, daemon_controller_cls):
    captured: dict[str, object] = {
        "global_handlers": [],
        "local_handlers": [],
        "global_monitors": [],
        "local_monitors": [],
        "removed_monitors": [],
        "timers": [],
    }

    class FakeNSObject:
        @classmethod
        def alloc(cls):
            return cls.__new__(cls)

        def init(self):
            return self

    class FakeApp:
        def setApplicationIconImage_(self, _image) -> None:
            pass

        def setActivationPolicy_(self, _policy) -> None:
            pass

        def setDelegate_(self, delegate) -> None:
            captured["delegate"] = delegate

        def run(self) -> None:
            pass

    fake_app = FakeApp()

    class FakeNSApplication:
        @staticmethod
        def sharedApplication():
            return fake_app

    class FakeNSImage:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithData_(self, _data):
            return self

    class FakeNSData:
        @staticmethod
        def dataWithBytes_length_(data, _length):
            return data

    class FakeNSEvent:
        @staticmethod
        def addGlobalMonitorForEventsMatchingMask_handler_(_mask, handler):
            captured["global_handlers"].append(handler)
            monitor = object()
            captured["global_monitors"].append(monitor)
            return monitor

        @staticmethod
        def addLocalMonitorForEventsMatchingMask_handler_(_mask, handler):
            captured["local_handlers"].append(handler)
            monitor = object()
            captured["local_monitors"].append(monitor)
            return monitor

        @staticmethod
        def removeMonitor_(monitor) -> None:
            captured["removed_monitors"].append(monitor)

    class FakeTimer:
        def __init__(self, interval, target, selector, user_info, repeats) -> None:
            self.interval = interval
            self.target = target
            self.selector = selector
            self.user_info = user_info
            self.repeats = repeats
            self.invalidated = False

        def invalidate(self) -> None:
            self.invalidated = True

    class FakeNSTimer:
        @staticmethod
        def scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval,
            target,
            selector,
            user_info,
            repeats,
        ):
            timer = FakeTimer(interval, target, selector, user_info, repeats)
            captured["timers"].append(timer)
            return timer

    fake_objc = types.SimpleNamespace(
        python_method=lambda method: method,
        super=lambda cls, instance: super(cls, instance),
    )
    fake_appkit = types.SimpleNamespace(
        NSApplication=FakeNSApplication,
        NSApplicationActivationPolicyAccessory=0,
        NSControlStateValueOff=0,
        NSControlStateValueOn=1,
        NSEvent=FakeNSEvent,
        NSEventMaskFlagsChanged=1,
        NSImage=FakeNSImage,
        NSMenu=object,
        NSMenuItem=object,
        NSStatusBar=object,
        NSVariableStatusItemLength=0,
        NSWorkspace=object,
    )
    fake_foundation = types.SimpleNamespace(
        NSURL=object,
        NSData=FakeNSData,
        NSObject=FakeNSObject,
        NSTimer=FakeNSTimer,
    )
    fake_web_ui = types.SimpleNamespace(WebUIController=object)

    monkeypatch.setitem(sys.modules, "objc", fake_objc)
    monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)
    monkeypatch.setitem(sys.modules, "Foundation", fake_foundation)
    monkeypatch.setitem(sys.modules, "ptarmigan_flow.web_ui", fake_web_ui)
    monkeypatch.setattr(
        "ptarmigan_flow.logging_setup.configure_app_file_logging",
        lambda _level: "/tmp/ptarmigan-flow-app.log",
    )
    monkeypatch.setattr(macos_app, "InProcessDaemonController", daemon_controller_cls)

    assert macos_app._run_appkit_app() == 0
    return captured


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


def test_combine_permission_reports_prefers_in_process_accessibility_and_input_monitoring() -> None:
    subprocess_report = macos_app.PermissionReport(
        microphone=False,
        accessibility=False,
        input_monitoring=False,
    )

    accessibility_combined = macos_app._combine_permission_reports(
        subprocess_report,
        accessibility_in_process=True,
        input_monitoring_in_process=False,
    )
    input_monitoring_combined = macos_app._combine_permission_reports(
        subprocess_report,
        accessibility_in_process=False,
        input_monitoring_in_process=True,
    )

    assert accessibility_combined == macos_app.PermissionReport(
        microphone=False,
        accessibility=True,
        input_monitoring=False,
    )
    assert input_monitoring_combined == macos_app.PermissionReport(
        microphone=False,
        accessibility=False,
        input_monitoring=True,
    )


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


def test_macos_app_creates_native_status_bar_menu_without_manual_voice_input_controls() -> None:
    source = _macos_app_source()
    build_status_item = source.split("def _build_status_item", maxsplit=1)[1].split(
        "@objc.python_method",
        maxsplit=1,
    )[0]
    update_status_menu = source.split("def _update_status_menu", maxsplit=1)[1].split(
        "@objc.python_method",
        maxsplit=1,
    )[0]

    assert "NSStatusBar" in source
    assert "NSVariableStatusItemLength" in source
    assert "statusItemWithLength_" in source
    assert "self.status_item" in source
    assert "self.status_menu" in source
    assert 'strings["dictation_stopped_menu"]' in source
    assert "start_menu_item" not in build_status_item
    assert "stop_menu_item" not in build_status_item
    assert "startDictation:" not in build_status_item
    assert "stopDictation:" not in build_status_item
    assert 'strings["start_dictation_button"]' not in build_status_item
    assert 'strings["stop_dictation_button"]' not in build_status_item
    assert "start_menu_item" not in update_status_menu
    assert "stop_menu_item" not in update_status_menu
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
    assert "mark_hotkey_confirmed=onboarding_flow_module.mark_hotkey_confirmed" in source


def test_macos_app_uses_in_process_daemon_controller() -> None:
    source = _macos_app_source()

    assert (
        "from ptarmigan_flow.app_daemon_controller import (\n"
        "    InProcessDaemonController,\n"
        ")"
    ) in source
    assert "daemon_run_command" not in source
    assert "    DaemonController,\n" not in source
    assert "self.daemon_controller = InProcessDaemonController(default_config_path())" in source


def test_macos_app_installs_main_thread_nsevent_hotkey_monitor() -> None:
    source = _macos_app_source()
    init_body = source.split("def init(self):", maxsplit=1)[1].split(
        "def applicationDidFinishLaunching_",
        maxsplit=1,
    )[0]
    install_body = source.split("def _install_hotkey_event_monitor(self) -> None:", maxsplit=1)[
        1
    ].split("def _remove_hotkey_event_monitor(self) -> None:", maxsplit=1)[0]
    remove_body = source.split("def _remove_hotkey_event_monitor(self) -> None:", maxsplit=1)[
        1
    ].split("@objc.python_method", maxsplit=1)[0]
    start_method = source.split("def _start_daemon_if_ready", maxsplit=1)[1].split(
        "@objc.python_method",
        maxsplit=1,
    )[0]
    stop_method = source.split("def _stop_daemon(self) -> None:", maxsplit=1)[1].split(
        "@objc.python_method",
        maxsplit=1,
    )[0]

    assert "from ptarmigan_flow.hotkey_monitor import macos_keycode_for_hotkey" in source
    assert "NSEvent" in source
    assert "NSEventMaskFlagsChanged" in source
    assert "_HOTKEY_KEYCODE_TO_MODIFIER_FLAG = {" in source
    assert "60: 1 << 17" in source
    assert "self._hotkey_global_monitor = None" in init_body
    assert "self._hotkey_local_monitor = None" in init_body
    assert "key_name = str(self.bridge._load_config().hotkey.key)" in install_body
    assert "macos_keycode_for_hotkey(key_name)" in install_body
    assert "flag_bit = _HOTKEY_KEYCODE_TO_MODIFIER_FLAG.get(keycode)" in install_body
    assert "event.keyCode()" in install_body
    assert "event.modifierFlags()" in install_body
    assert "self.daemon_controller.notify_hotkey_press()" in install_body
    assert "self.daemon_controller.notify_hotkey_release()" in install_body
    assert "NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(" in install_body
    assert "NSEvent.addLocalMonitorForEventsMatchingMask_handler_(" in install_body
    assert "NSEvent.removeMonitor_(monitor)" in remove_body
    assert "if not self.daemon_controller.is_running:" in start_method
    assert start_method.index("self._install_hotkey_event_monitor()") < start_method.rindex(
        "self._push_daemon_state()"
    )
    assert stop_method.index("self.daemon_controller.stop()") < stop_method.index(
        "self._remove_hotkey_event_monitor()"
    )
    assert stop_method.index("self._remove_hotkey_event_monitor()") < stop_method.index(
        "self._push_daemon_state()"
    )


def test_macos_app_right_shift_nsevent_handler_notifies_daemon_controller(monkeypatch) -> None:
    class FakeDaemonController:
        def __init__(self, _config_path) -> None:
            self.is_running = True
            self.events: list[str] = []

        def start(self) -> None:
            self.is_running = True

        def stop(self) -> None:
            self.is_running = False

        def notify_hotkey_press(self) -> None:
            self.events.append("press")

        def notify_hotkey_release(self) -> None:
            self.events.append("release")

    class FakeEvent:
        def __init__(self, keycode: int, flags: int) -> None:
            self._keycode = keycode
            self._flags = flags

        def keyCode(self) -> int:  # noqa: N802 - mirrors NSEvent
            return self._keycode

        def modifierFlags(self) -> int:  # noqa: N802 - mirrors NSEvent
            return self._flags

    captured = _capture_app_delegate(monkeypatch, FakeDaemonController)
    controller = captured["delegate"]
    controller.bridge = types.SimpleNamespace(
        _load_config=lambda: types.SimpleNamespace(hotkey=types.SimpleNamespace(key="right_shift"))
    )

    controller._install_hotkey_event_monitor()

    global_handler = captured["global_handlers"][0]
    local_handler = captured["local_handlers"][0]
    wrong_key_event = FakeEvent(59, 1 << 17)
    press_event = FakeEvent(60, 1 << 17)
    release_event = FakeEvent(60, 0)

    global_handler(wrong_key_event)
    global_handler(press_event)
    global_handler(release_event)
    local_handler(press_event)
    local_handler(release_event)

    assert controller.daemon_controller.events == [
        "press",
        "release",
        "press",
        "release",
    ]
    assert local_handler(wrong_key_event) is wrong_key_event


def test_macos_app_reconciles_async_daemon_failure_after_monitor_install(
    monkeypatch,
) -> None:
    class FakeDaemonController:
        def __init__(self, _config_path) -> None:
            self._is_running = False
            self.last_error: Exception | None = None
            self.start_calls = 0

        @property
        def is_running(self) -> bool:
            return self._is_running

        def start(self) -> None:
            self.start_calls += 1
            self._is_running = True

        def fail_later(self) -> None:
            self._is_running = False
            self.last_error = RuntimeError("daemon boot failed after thread start")

        def stop(self) -> None:
            self._is_running = False

        def notify_hotkey_press(self) -> None:
            pass

        def notify_hotkey_release(self) -> None:
            pass

    captured = _capture_app_delegate(monkeypatch, FakeDaemonController)
    controller = captured["delegate"]
    pushed: list[tuple[str, dict[str, object]]] = []
    controller.web_ui = types.SimpleNamespace(
        push_event=lambda event, payload: pushed.append((event, payload))
    )
    controller.bridge = types.SimpleNamespace(
        _load_config=lambda: types.SimpleNamespace(
            hotkey=types.SimpleNamespace(key="right_shift")
        ),
        handle_action=lambda _action, _payload: {
            "language": "en",
            "daemon_running": controller.daemon_controller.is_running,
        },
    )
    controller._configured_backend_is_available = lambda: True
    controller._configured_model_token = lambda: "moonshine/base"
    monkeypatch.setattr(macos_app.model_download, "is_model_downloaded", lambda _token: True)

    result = controller._start_daemon_if_ready(
        macos_app.PermissionReport(
            microphone=True,
            accessibility=True,
            input_monitoring=True,
        )
    )
    controller.daemon_controller.fail_later()

    assert result is None
    assert controller.daemon_controller.start_calls == 1
    assert len(captured["timers"]) == 1
    assert captured["timers"][0].selector == "pollDaemonStatus:"

    controller.pollDaemonStatus_(captured["timers"][0])

    assert controller._hotkey_global_monitor is None
    assert controller._hotkey_local_monitor is None
    assert captured["timers"][0].invalidated is True
    assert captured["removed_monitors"] == [
        captured["global_monitors"][0],
        captured["local_monitors"][0],
    ]
    assert pushed[-1][0] == "daemonState"
    assert pushed[-1][1]["daemon_running"] is False
    assert (
        pushed[-1][1]["daemon_error_message"]
        == "Could not start voice input: daemon boot failed after thread start"
    )


def test_macos_app_surfaces_daemon_error_when_start_returns_not_running(
    monkeypatch,
) -> None:
    class FakeDaemonController:
        def __init__(self, _config_path) -> None:
            self.is_running = False
            self.last_error: Exception | None = None
            self.start_calls = 0

        def start(self) -> None:
            self.start_calls += 1
            self.last_error = RuntimeError("daemon boot failed")

        def stop(self) -> None:
            self.is_running = False

        def notify_hotkey_press(self) -> None:
            pass

        def notify_hotkey_release(self) -> None:
            pass

    captured = _capture_app_delegate(monkeypatch, FakeDaemonController)
    controller = captured["delegate"]
    pushed: list[tuple[str, dict[str, object]]] = []
    controller.web_ui = types.SimpleNamespace(
        push_event=lambda event, payload: pushed.append((event, payload))
    )
    controller.bridge = types.SimpleNamespace(
        handle_action=lambda _action, _payload: {
            "language": "en",
            "daemon_running": False,
        }
    )
    controller._configured_backend_is_available = lambda: True
    controller._configured_model_token = lambda: "moonshine/base"
    monkeypatch.setattr(macos_app.model_download, "is_model_downloaded", lambda _token: True)

    result = controller._start_daemon_if_ready(
        macos_app.PermissionReport(
            microphone=True,
            accessibility=True,
            input_monitoring=True,
        )
    )

    assert controller.daemon_controller.start_calls == 1
    assert isinstance(result, str)
    assert pushed[-1][0] == "daemonState"
    assert pushed[-1][1]["daemon_error_message"] == result
    assert "Could not start voice input" in result
    assert "daemon boot failed" in result


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
    assert "hotkey_was_confirmed()" in source
    assert "self.onboarding_flow.start(" in source
    assert "report=check_all_permissions()" in source
    assert "language_already_selected=onboarding_flow_module.language_was_selected()" in source
    assert "hotkey_already_confirmed=onboarding_flow_module.hotkey_was_confirmed()" in source
    assert "self.bridge.set_onboarding_flow(self.onboarding_flow)" in source


def test_macos_app_polls_permissions_through_subprocess_and_pushes_changes() -> None:
    source = _macos_app_source()

    assert "NSTimer" in source
    assert 'schedule_timer(1.75, self, "pollPermissions:", None, True)' in source
    assert "check_all_permissions_subprocess" in source
    assert "report = check_all_permissions_subprocess()" in source
    assert "check_accessibility_permission()" in source
    assert "check_input_monitoring_permission()" in source
    assert "_combine_permission_reports(" in source
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

    assert "self._last_permission_report or self._combined_permission_report()" in start_method
    assert "if not report.all_granted:" in start_method
    assert "if not self._configured_backend_is_available():" in start_method
    assert "model_token = self._configured_model_token()" in start_method
    assert "model_download.is_model_downloaded(model_token)" in start_method
    assert "self._start_model_download(model_token, success_message_key)" in start_method
    assert "self.daemon_controller.start()" in start_method
    assert "self._push_daemon_state()" in start_method


def test_macos_app_download_done_event_starts_daemon_only_for_configured_model(
    monkeypatch,
) -> None:
    class FakeDaemonController:
        def __init__(self, _config_path) -> None:
            self.is_running = False

        def start(self) -> None:
            self.is_running = True

        def stop(self) -> None:
            self.is_running = False

        def notify_hotkey_press(self) -> None:
            pass

        def notify_hotkey_release(self) -> None:
            pass

    captured = _capture_app_delegate(monkeypatch, FakeDaemonController)
    controller = captured["delegate"]
    controller.web_ui = types.SimpleNamespace(push_event=lambda _event, _payload: None)
    controller._configured_model_token = lambda: "moonshine/base"

    start_daemon_calls: list[dict[str, object]] = []
    push_daemon_state_calls: list[None] = []
    controller._start_daemon_if_ready = lambda **kwargs: start_daemon_calls.append(kwargs)
    controller._push_daemon_state = lambda: push_daemon_state_calls.append(None)

    controller.applyModelDownloadProgress_(
        {
            "type": "done",
            "model": "moonshine/base",
            "success_message_key": "voice_input_started_message",
        }
    )

    assert start_daemon_calls == [{"success_message_key": "voice_input_started_message"}]
    assert push_daemon_state_calls == []

    controller.applyModelDownloadProgress_(
        {
            "type": "done",
            "model": "granite:ibm/other",
            "success_message_key": "voice_input_started_message",
        }
    )

    assert start_daemon_calls == [{"success_message_key": "voice_input_started_message"}]
    assert push_daemon_state_calls == [None]


def test_macos_app_surfaces_missing_permission_message_when_daemon_start_is_blocked() -> None:
    source = _macos_app_source()
    start_method = source.split("def _start_daemon_if_ready", maxsplit=1)[1].split(
        "@objc.python_method",
        maxsplit=1,
    )[0]

    assert "self._missing_permissions_message(report)" in start_method
    assert "self._daemon_error_message" in source
    assert '"daemon_error_message"' in source
    assert 'strings["grant_permissions_message"]' in source
    assert 'strings["accessibility_title"]' in source
    assert 'strings["input_monitoring_title"]' in source


def test_macos_app_daemon_start_reuses_latest_combined_permission_report(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeNSObject:
        @classmethod
        def alloc(cls):
            return cls.__new__(cls)

        def init(self):
            return self

    class FakeApp:
        def setApplicationIconImage_(self, _image) -> None:
            pass

        def setActivationPolicy_(self, _policy) -> None:
            pass

        def setDelegate_(self, delegate) -> None:
            captured["delegate"] = delegate

        def run(self) -> None:
            pass

    fake_app = FakeApp()

    class FakeNSApplication:
        @staticmethod
        def sharedApplication():
            return fake_app

    class FakeNSImage:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithData_(self, _data):
            return self

    class FakeNSData:
        @staticmethod
        def dataWithBytes_length_(data, _length):
            return data

    class FakeNSEvent:
        @staticmethod
        def addGlobalMonitorForEventsMatchingMask_handler_(_mask, _handler):
            return object()

        @staticmethod
        def addLocalMonitorForEventsMatchingMask_handler_(_mask, _handler):
            return object()

        @staticmethod
        def removeMonitor_(_monitor) -> None:
            return None

    class FakeTimer:
        def invalidate(self) -> None:
            pass

    class FakeNSTimer:
        @staticmethod
        def scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _interval,
            _target,
            _selector,
            _user_info,
            _repeats,
        ):
            return FakeTimer()

    class FakeInProcessDaemonController:
        def __init__(self, _config_path) -> None:
            self.is_running = False
            self.start_calls = 0

        def start(self) -> None:
            self.start_calls += 1
            self.is_running = True

        def stop(self) -> None:
            self.is_running = False

    fake_objc = types.SimpleNamespace(
        python_method=lambda method: method,
        super=lambda cls, instance: super(cls, instance),
    )
    fake_appkit = types.SimpleNamespace(
        NSApplication=FakeNSApplication,
        NSApplicationActivationPolicyAccessory=0,
        NSControlStateValueOff=0,
        NSControlStateValueOn=1,
        NSEvent=FakeNSEvent,
        NSEventMaskFlagsChanged=1,
        NSImage=FakeNSImage,
        NSMenu=object,
        NSMenuItem=object,
        NSStatusBar=object,
        NSVariableStatusItemLength=0,
        NSWorkspace=object,
    )
    fake_foundation = types.SimpleNamespace(
        NSURL=object,
        NSData=FakeNSData,
        NSObject=FakeNSObject,
        NSTimer=FakeNSTimer,
    )
    fake_web_ui = types.SimpleNamespace(WebUIController=object)

    monkeypatch.setitem(sys.modules, "objc", fake_objc)
    monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)
    monkeypatch.setitem(sys.modules, "Foundation", fake_foundation)
    monkeypatch.setitem(sys.modules, "ptarmigan_flow.web_ui", fake_web_ui)
    monkeypatch.setattr(
        "ptarmigan_flow.logging_setup.configure_app_file_logging",
        lambda _level: "/tmp/ptarmigan-flow-app.log",
    )
    monkeypatch.setattr(macos_app, "InProcessDaemonController", FakeInProcessDaemonController)
    monkeypatch.setattr(
        macos_app,
        "check_all_permissions",
        lambda: macos_app.PermissionReport(
            microphone=True,
            accessibility=False,
            input_monitoring=True,
        ),
    )
    monkeypatch.setattr(macos_app.model_download, "is_model_downloaded", lambda _token: True)

    assert macos_app._run_appkit_app() == 0
    controller = captured["delegate"]
    controller._last_permission_report = macos_app.PermissionReport(
        microphone=True,
        accessibility=True,
        input_monitoring=True,
    )
    pushed_errors: list[str] = []
    pushed_states: list[str] = []
    controller._missing_permissions_message = lambda _report: "grant all permissions"
    controller._push_daemon_error = lambda message: pushed_errors.append(message)
    controller._push_daemon_state = lambda: pushed_states.append("daemonState")
    controller._configured_backend_is_available = lambda: True
    controller._configured_model_token = lambda: "moonshine/base"

    result = controller._start_daemon_if_ready()

    assert result is None
    assert pushed_errors == []
    assert controller.daemon_controller.start_calls == 1
    assert pushed_states == ["daemonState"]


def test_webui_done_screen_omits_manual_voice_input_start_and_stop_buttons() -> None:
    source = _webui_app_source()
    done_template = source.split('if (current === "done")', maxsplit=1)[1].split(
        "return renderPermissionStep(current, dots);",
        maxsplit=1,
    )[0]

    assert 'data-action="start"' not in done_template
    assert 'data-action="stop"' not in done_template
    assert 't("start_dictation_button")' not in done_template
    assert 't("stop_dictation_button")' not in done_template
    assert 'data-route="settings"' in done_template
    assert 'data-action="toggle-login"' in done_template


def test_webui_onboarding_includes_hotkey_confirmation_step() -> None:
    source = _webui_app_source()

    assert (
        '["language", "hotkey", "microphone", "accessibility", "input_monitoring", "done"]'
        in source
    )
    assert 'if (current === "hotkey")' in source
    assert 't("hotkey_confirm_title")' in source
    assert 't("hotkey_confirm_body")' in source
    assert 't("hotkey_select_label")' in source
    assert 'id="onboarding-hotkey"' in source
    assert 'data-action="confirm-hotkey"' in source
    assert 'bridge("confirmHotkey", { hotkey: selectedHotkey })' in source
    for hotkey in (
        "right_cmd",
        "left_cmd",
        "right_shift",
        "left_shift",
        "right_alt",
        "left_alt",
        "right_ctrl",
        "left_ctrl",
    ):
        assert f'"{hotkey}"' in source


def test_macos_app_exposes_menu_actions_to_web_routes_and_bridge_side_effects() -> None:
    source = _macos_app_source()

    assert "def startDictation_(self, _sender):" not in source
    assert "def stopDictation_(self, _sender):" not in source
    assert "def showSettings_(self, _sender):" in source
    assert "def showDictionaryEditor_(self, _sender):" in source
    assert "def toggleLoginAtStartup_(self, _sender):" in source
    assert "def restartApp_(self, _sender):" in source
    assert "def _restart_app(self) -> bool:" in source
    assert "self._set_route(\"settings\")" in source
    assert "self._set_route(\"dictionary\")" in source
    assert "self._toggle_login()" in source
    assert "NSApplication.sharedApplication().terminate_(self)" in source


def test_macos_app_restart_and_quit_use_plain_termination_paths() -> None:
    source = _macos_app_source()
    restart_body = source.split("def _restart_app(self) -> bool:", maxsplit=1)[1].split(
        "def pollPermissions_",
        maxsplit=1,
    )[0]
    quit_body = source.split("def quit_(self, _sender):", maxsplit=1)[1].split(
        "app = NSApplication.sharedApplication()",
        maxsplit=1,
    )[0]
    restart_lines = [
        line.strip()
        for line in restart_body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    quit_lines = [
        line.strip()
        for line in quit_body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert restart_lines == [
        "if app_relaunch.relaunch_app():",
        "NSApplication.sharedApplication().terminate_(self)",
        "return True",
        "return False",
    ]
    assert quit_lines == ["NSApplication.sharedApplication().terminate_(self)"]


def test_macos_app_shutdown_runs_plain_cleanup_steps() -> None:
    source = _macos_app_source()
    terminate_parts = source.split(
        "def applicationWillTerminate_(self, _notification):",
        maxsplit=1,
    )
    terminate_body = terminate_parts[1].split("def applicationDidBecomeActive_", maxsplit=1)[0]
    terminate_lines = [
        line.strip()
        for line in terminate_body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert terminate_lines == [
        "self._stop_permission_timer()",
        "self._terminate_model_download_process()",
        "self._remove_hotkey_event_monitor()",
        "self.daemon_controller.stop()",
    ]


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
