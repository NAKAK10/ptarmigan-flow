"""PyObjC WKWebView app used by notarized macOS release builds."""

from __future__ import annotations

import importlib.resources
import json
import logging
import multiprocessing
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ptarmigan_flow import (
    app_relaunch,
    app_updater,
    login_item,
    onboarding_strings,
    setup_status,
)
from ptarmigan_flow import onboarding_flow as onboarding_flow_module
from ptarmigan_flow.app_daemon_controller import (
    InProcessDaemonController,
)
from ptarmigan_flow.app_icon import APP_ICON_FILE, APP_ICON_RESOURCE_PACKAGE
from ptarmigan_flow.config import (
    default_config_path,
    ensure_config_exists,
)
from ptarmigan_flow.hotkey_monitor import macos_keycode_for_hotkey
from ptarmigan_flow.onboarding_flow import OnboardingFlow
from ptarmigan_flow.permissions import (
    PermissionReport,
    check_accessibility_permission,
    check_all_permissions,
    check_all_permissions_subprocess,
    check_input_monitoring_permission,
    request_accessibility_permission,
    request_input_monitoring_permission,
    request_microphone_permission,
)
from ptarmigan_flow.stt import availability, model_download
from ptarmigan_flow.stt.factory import parse_stt_model
from ptarmigan_flow.transcription_corrections import resolve_dictionary_path
from ptarmigan_flow.web_bridge import BridgeDependencies, WebBridgeDispatcher

APP_NAME = "PtarmiganFlow"
# Delay before the one-shot update check after launch, so startup work settles first.
_STARTUP_UPDATE_CHECK_DELAY_SECONDS = 5.0
_NS_ALERT_FIRST_BUTTON_RETURN = 1000
_UPDATE_LOG_PATH = Path.home() / "Library" / "Logs" / "ptarmigan-flow" / "update.log"
_HOTKEY_KEYCODE_TO_MODIFIER_FLAG = {
    54: 1 << 20,
    55: 1 << 20,
    60: 1 << 17,
    56: 1 << 17,
    61: 1 << 19,
    58: 1 << 19,
    62: 1 << 18,
    59: 1 << 18,
}


def _set_application_icon(app: object, ns_image_cls: object, ns_data_cls: object) -> None:
    try:
        icon_resource = importlib.resources.files(APP_ICON_RESOURCE_PACKAGE).joinpath(APP_ICON_FILE)
        icon_bytes = icon_resource.read_bytes()
    except OSError:
        return

    icon_data = ns_data_cls.dataWithBytes_length_(icon_bytes, len(icon_bytes))
    icon_image = ns_image_cls.alloc().initWithData_(icon_data)
    if icon_image is not None:
        app.setApplicationIconImage_(icon_image)


_EDIT_MENU_ITEMS: tuple[tuple[str, str, str], ...] = (
    ("Undo", "undo:", "z"),
    ("Redo", "redo:", "Z"),
    ("Cut", "cut:", "x"),
    ("Copy", "copy:", "c"),
    ("Paste", "paste:", "v"),
    ("Select All", "selectAll:", "a"),
)


def _install_edit_menu(app: object, ns_menu_cls: object, ns_menu_item_cls: object) -> None:
    """Install a main menu with a standard Edit menu.

    The app is an accessory (menu-bar) app with no visible menu bar, but
    Cmd+C/V/X/A key equivalents are still resolved through the main menu.
    Without this, paste does nothing inside WKWebView text fields.
    """
    main_menu = ns_menu_cls.alloc().init()
    edit_root = ns_menu_item_cls.alloc().initWithTitle_action_keyEquivalent_("Edit", None, "")
    edit_menu = ns_menu_cls.alloc().initWithTitle_("Edit")
    for title, action, key in _EDIT_MENU_ITEMS:
        edit_menu.addItem_(
            ns_menu_item_cls.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
        )
    edit_root.setSubmenu_(edit_menu)
    main_menu.addItem_(edit_root)
    app.setMainMenu_(main_menu)


def open_config() -> Path:
    """Ensure the user config exists and open it with the default macOS editor."""
    config_path = default_config_path()
    ensure_config_exists(config_path)
    if sys.platform == "darwin":
        subprocess.run(["open", str(config_path)], check=False)
    return config_path


def _dispatch_cli_args(argv: list[str]) -> int | None:
    """Route `<bundle exe> -m <module>` argv into the matching in-process entry point.

    The frozen executable stands in for the Python interpreter wherever the
    daemon builds child commands as ``[sys.executable, "-m", <module>, ...]``
    (launchd fallback and the activity-overlay subprocess).
    """
    if len(argv) < 2 or argv[0] != "-m":
        return None

    module_name = argv[1]
    if module_name == "ptarmigan_flow.cli":
        from ptarmigan_flow.cli import main as cli_main

        original_argv = sys.argv[:]
        try:
            sys.argv = [original_argv[0], *argv[2:]]
            return int(cli_main())
        finally:
            sys.argv = original_argv

    if module_name == "ptarmigan_flow.activity_overlay":
        from ptarmigan_flow.activity_overlay import main as overlay_main

        return int(overlay_main(argv[2:]))

    return None


def _combine_permission_reports(
    report: PermissionReport | None,
    *,
    accessibility_in_process: bool,
    input_monitoring_in_process: bool,
) -> PermissionReport | None:
    if report is None:
        return None
    return PermissionReport(
        microphone=report.microphone,
        accessibility=report.accessibility or accessibility_in_process,
        input_monitoring=report.input_monitoring or input_monitoring_in_process,
    )


def _run_appkit_app() -> int:
    from ptarmigan_flow.logging_setup import configure_app_file_logging

    log_path = configure_app_file_logging("DEBUG")
    logging.getLogger(__name__).info("PtarmiganFlow GUI app starting; logging to %s", log_path)

    import objc
    from AppKit import (
        NSAlert,
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSControlStateValueOff,
        NSControlStateValueOn,
        NSEvent,
        NSEventMaskFlagsChanged,
        NSImage,
        NSMenu,
        NSMenuItem,
        NSStatusBar,
        NSVariableStatusItemLength,
        NSWorkspace,
    )
    from Foundation import NSURL, NSData, NSObject, NSTimer

    from ptarmigan_flow.web_ui import WebUIController

    class OnboardingController(NSObject):
        """Menu-bar controller whose presentation layer is WKWebView."""

        # Keep permission names discoverable for packaging smoke tests:
        # Microphone, Accessibility, Input Monitoring.
        _permission_step_config = {
            "microphone": {
                "settings_url": (
                    "x-apple.systempreferences:com.apple.preference.security"
                    "?Privacy_Microphone"
                ),
            },
            "accessibility": {
                "settings_url": (
                    "x-apple.systempreferences:com.apple.preference.security"
                    "?Privacy_Accessibility"
                ),
            },
            "input_monitoring": {
                "settings_url": (
                    "x-apple.systempreferences:com.apple.preference.security"
                    "?Privacy_ListenEvent"
                ),
            },
        }

        def init(self):  # noqa: N802 - PyObjC follows Objective-C naming.
            self = objc.super(OnboardingController, self).init()
            if self is None:
                return None
            self.onboarding_flow = OnboardingFlow()
            self.daemon_controller = InProcessDaemonController(default_config_path())
            self._hotkey_global_monitor = None
            self._hotkey_local_monitor = None
            self._hotkey_capture_active = False
            self._hotkey_capture_started_at: float | None = None
            self.daemon_status_timer = None
            self.permission_timer = None
            self._permission_check_generation = 0
            self._permission_check_in_progress = False
            self._permission_check_thread: threading.Thread | None = None
            self._last_permission_report: PermissionReport | None = None
            # Latest report known to the main thread; drives the "setup required" menu.
            self._setup_permission_report: PermissionReport | None = None
            self._model_download_process: subprocess.Popen[str] | None = None
            self._model_download_success_message_key = "voice_input_started_message"
            self._model_download_thread: threading.Thread | None = None
            self._daemon_error_message: str | None = None
            self._update_check_in_progress = False
            self._update_install_in_progress = False
            # Newer release found by the last successful check; drives the update menu item.
            self._update_result: app_updater.UpdateCheckResult | None = None
            # (strings key, percent text) while an update is downloading/verifying/installing.
            self._update_progress: tuple[str, str] | None = None
            self._update_install_bundle: Path | None = None
            self._update_startup_timer = None
            self.web_ui = None
            self.bridge = WebBridgeDispatcher(
                deps=BridgeDependencies(
                    config_path=default_config_path,
                    check_permissions=check_all_permissions,
                    available_model_entries=availability.available_model_entries,
                    is_model_downloaded=model_download.is_model_downloaded,
                    resolve_dictionary_path=self._dictionary_path,
                    request_permission=self._request_permission,
                    open_system_settings=self._open_system_settings,
                    open_config_file=open_config,
                    start_dictation=self._start_daemon_if_ready,
                    stop_dictation=self._stop_daemon,
                    start_model_download=lambda token: self._start_model_download(
                        token, "voice_input_started_message"
                    ),
                    daemon_is_running=lambda: self.daemon_controller.is_running,
                    login_is_enabled=login_item.is_enabled,
                    login_register=login_item.register,
                    login_unregister=login_item.unregister,
                    restart_app=self._restart_app,
                    mark_language_selected=onboarding_flow_module.mark_language_selected,
                    mark_hotkey_confirmed=onboarding_flow_module.mark_hotkey_confirmed,
                    set_hotkey_capture_active=self._set_hotkey_capture_active,
                )
            )
            return self

        def applicationDidFinishLaunching_(self, _notification):  # noqa: N802
            self._build_status_item()
            self.onboarding_flow.start(
                report=check_all_permissions(),
                language_already_selected=onboarding_flow_module.language_was_selected(),
                hotkey_already_confirmed=onboarding_flow_module.hotkey_was_confirmed(),
            )
            self.bridge.set_onboarding_flow(self.onboarding_flow)
            self.web_ui = WebUIController.alloc().initWithBridge_title_(self.bridge, APP_NAME)
            self._show_onboarding_window_if_needed()
            self._refresh_onboarding_permissions()
            self._start_permission_check()
            self._schedule_startup_update_check()

        def applicationWillTerminate_(self, _notification):  # noqa: N802
            self._stop_permission_timer()
            self._terminate_model_download_process()
            self._remove_hotkey_event_monitor()
            self.daemon_controller.stop()

        def applicationDidBecomeActive_(self, _notification):  # noqa: N802
            self._start_permission_check()

        def applicationShouldHandleReopen_hasVisibleWindows_(  # noqa: N802
            self,
            _application,
            _has_visible_windows,
        ):
            if self.onboarding_flow.is_complete:
                self._set_route("settings")
            else:
                self._set_route("onboarding")
            return True

        @objc.python_method
        def _strings(self) -> dict[str, str]:
            try:
                language = self.bridge.current_language()
            except Exception:
                language = "en"
            return onboarding_strings.strings_for(str(language))

        @objc.python_method
        def _menu_item(self, title: str, action: str | None):
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
            item.setTarget_(self)
            return item

        @objc.python_method
        def _build_status_item(self) -> None:
            strings = self._strings()
            self.status_item = (
                NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
            )
            status_button = self.status_item.button()
            if status_button is not None:
                status_button.setTitle_("Pt")

            self.status_menu = NSMenu.alloc().init()
            self.dictation_status_menu_item = self._menu_item(
                strings["dictation_stopped_menu"],
                None,
            )
            self.dictation_status_menu_item.setEnabled_(False)
            self.status_menu.addItem_(self.dictation_status_menu_item)
            self.open_setup_menu_item = self._menu_item(strings["open_setup_menu"], "showSetup:")
            self.open_setup_menu_item.setHidden_(True)
            self.status_menu.addItem_(self.open_setup_menu_item)
            self.status_menu.addItem_(NSMenuItem.separatorItem())
            self.settings_menu_item = self._menu_item(strings["settings_menu"], "showSettings:")
            self.status_menu.addItem_(self.settings_menu_item)
            self.dictionary_menu_item = self._menu_item(
                strings["edit_dictionary_menu"],
                "showDictionaryEditor:",
            )
            self.status_menu.addItem_(self.dictionary_menu_item)
            self.login_menu_item = self._menu_item(
                strings["login_at_startup_menu"],
                "toggleLoginAtStartup:",
            )
            self.status_menu.addItem_(self.login_menu_item)
            self.status_menu.addItem_(NSMenuItem.separatorItem())
            self.update_available_menu_item = self._menu_item(
                strings["update_available_menu"].format(version=""),
                "installUpdate:",
            )
            self.update_available_menu_item.setHidden_(True)
            self.status_menu.addItem_(self.update_available_menu_item)
            self.check_updates_menu_item = self._menu_item(
                strings["check_updates_menu"],
                "checkForUpdates:",
            )
            self.status_menu.addItem_(self.check_updates_menu_item)
            self.quit_menu_item = self._menu_item(strings["quit_menu"], "quit:")
            self.status_menu.addItem_(self.quit_menu_item)
            self.status_menu.setDelegate_(self)
            self.status_item.setMenu_(self.status_menu)
            self._update_status_menu()
            self._update_login_menu_state()

        @objc.python_method
        def _update_status_menu(self) -> None:
            if not hasattr(self, "status_menu"):
                return
            strings = self._strings()
            menu_state = setup_status.status_menu_state(
                self._missing_setup_items(),
                daemon_running=self.daemon_controller.is_running,
                strings=strings,
            )
            self.dictation_status_menu_item.setTitle_(menu_state.status_title)
            self.open_setup_menu_item.setTitle_(strings["open_setup_menu"])
            self.open_setup_menu_item.setHidden_(not menu_state.show_open_setup)
            status_button = self.status_item.button()
            if status_button is not None:
                status_button.setTitle_(menu_state.button_title)
            self.settings_menu_item.setTitle_(strings["settings_menu"])
            self.dictionary_menu_item.setTitle_(strings["edit_dictionary_menu"])
            self.login_menu_item.setTitle_(strings["login_at_startup_menu"])
            self._update_updater_menu_items(strings)
            self.quit_menu_item.setTitle_(strings["quit_menu"])

        @objc.python_method
        def _update_updater_menu_items(self, strings: dict[str, str]) -> None:
            update_item = self.update_available_menu_item
            result = self._update_result
            if self._update_progress is not None:
                key, percent = self._update_progress
                update_item.setTitle_(strings[key].format(percent=percent).rstrip())
                # Without an action the auto-enabling menu greys the item out.
                update_item.setAction_(None)
                update_item.setHidden_(False)
            elif result is not None and result.release is not None and not result.is_dev_build:
                update_item.setTitle_(
                    strings["update_available_menu"].format(version=result.release.version)
                )
                update_item.setAction_("installUpdate:")
                update_item.setHidden_(False)
            else:
                update_item.setHidden_(True)
            self.check_updates_menu_item.setTitle_(
                strings["update_checking_menu"]
                if self._update_check_in_progress
                else strings["check_updates_menu"]
            )

        @objc.python_method
        def _update_login_menu_state(self) -> None:
            # SMAppService.status() makes backgroundtaskmanagementd re-verify the
            # app signature, so only query it on demand, never from timers (#29).
            if not hasattr(self, "status_menu"):
                return
            login_state = (
                NSControlStateValueOn if login_item.is_enabled() else NSControlStateValueOff
            )
            self.login_menu_item.setState_(login_state)

        @objc.python_method
        def _missing_setup_items(self) -> list[str]:
            return setup_status.missing_setup_items(
                self.onboarding_flow.current_step,
                self._setup_permission_report,
            )

        @objc.python_method
        def _show_onboarding_window_if_needed(self) -> None:
            if not self.onboarding_flow.is_complete and self.web_ui is not None:
                self.web_ui.show(route="onboarding")
                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

        @objc.python_method
        def _set_route(self, route: str) -> None:
            if self.web_ui is None:
                return
            if route == "settings":
                self.web_ui.show(route="settings")
            elif route == "dictionary":
                self.web_ui.show(route="dictionary")
            else:
                self.web_ui.show(route="onboarding")
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

        @objc.python_method
        def _state_payload(self, *, refresh_login: bool = False) -> dict[str, Any]:
            payload = self.bridge.handle_action(
                "getState",
                {"cached_login": not refresh_login},
            )
            if self._daemon_error_message:
                payload["daemon_error_message"] = self._daemon_error_message
            return payload

        @objc.python_method
        def _push_event(self, event: str, payload: dict[str, Any]) -> None:
            if self.web_ui is not None:
                self.web_ui.push_event(event, payload)

        @objc.python_method
        def _push_daemon_state(self) -> None:
            self._reconcile_daemon_status()
            self._update_status_menu()
            self._push_event("daemonState", self._state_payload())

        @objc.python_method
        def _missing_permissions_message(self, report: PermissionReport) -> str:
            strings = self._strings()
            missing_labels: list[str] = []
            if not report.microphone:
                missing_labels.append(strings["microphone_title"])
            if not report.accessibility:
                missing_labels.append(strings["accessibility_title"])
            if not report.input_monitoring:
                missing_labels.append(strings["input_monitoring_title"])
            message = strings["grant_permissions_message"]
            if missing_labels:
                message = f"{message} ({', '.join(missing_labels)})"
            return message

        @objc.python_method
        def _push_daemon_error(self, message: str) -> None:
            self._daemon_error_message = message
            self._push_daemon_state()

        # Auto-heals a stuck capture flag (e.g. the JS side never called
        # endHotkeyCapture because of a crash/navigation edge case) so the
        # dictation hotkey doesn't stay suppressed indefinitely.
        _HOTKEY_CAPTURE_TIMEOUT_SECONDS = 30.0

        @objc.python_method
        def _set_hotkey_capture_active(self, active: bool) -> None:
            self._hotkey_capture_active = bool(active)
            self._hotkey_capture_started_at = time.monotonic() if active else None

        @objc.python_method
        def _is_hotkey_capture_active(self) -> bool:
            if not self._hotkey_capture_active:
                return False
            started_at = self._hotkey_capture_started_at
            if (
                started_at is not None
                and time.monotonic() - started_at > self._HOTKEY_CAPTURE_TIMEOUT_SECONDS
            ):
                self._set_hotkey_capture_active(False)
                return False
            return True

        @objc.python_method
        def _install_hotkey_event_monitor(self) -> None:
            if (
                self._hotkey_global_monitor is not None
                or self._hotkey_local_monitor is not None
            ):
                return
            try:
                key_name = str(self.bridge._load_config().hotkey.key)
            except Exception:
                key_name = "right_cmd"
            keycode = macos_keycode_for_hotkey(key_name)
            if keycode is None:
                logging.getLogger(__name__).warning("No keycode mapping for hotkey %s", key_name)
                return
            flag_bit = _HOTKEY_KEYCODE_TO_MODIFIER_FLAG.get(keycode)

            def _handle(event):
                try:
                    if self._is_hotkey_capture_active():
                        return
                    if int(event.keyCode()) != keycode:
                        return
                    if flag_bit is not None:
                        pressed = bool(int(event.modifierFlags()) & flag_bit)
                    else:
                        pressed = False
                    if pressed:
                        self.daemon_controller.notify_hotkey_press()
                    else:
                        self.daemon_controller.notify_hotkey_release()
                except Exception:
                    logging.getLogger(__name__).exception("Hotkey NSEvent handler failed")

            def _handle_local(event):
                _handle(event)
                return event

            self._hotkey_global_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                NSEventMaskFlagsChanged,
                _handle,
            )
            self._hotkey_local_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                NSEventMaskFlagsChanged,
                _handle_local,
            )
            logging.getLogger(__name__).info(
                "Installed NSEvent hotkey monitor for %s (keycode=%s)",
                key_name,
                keycode,
            )

        @objc.python_method
        def _remove_hotkey_event_monitor(self) -> None:
            for monitor in (self._hotkey_global_monitor, self._hotkey_local_monitor):
                if monitor is not None:
                    try:
                        NSEvent.removeMonitor_(monitor)
                    except Exception:
                        logging.getLogger(__name__).exception("Failed to remove hotkey monitor")
            self._hotkey_global_monitor = None
            self._hotkey_local_monitor = None
            self._stop_daemon_status_timer()

        @objc.python_method
        def _start_daemon_status_timer(self) -> None:
            if self.daemon_status_timer is not None:
                return
            if (
                self._hotkey_global_monitor is None
                and self._hotkey_local_monitor is None
            ):
                return
            schedule_timer = (
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_
            )
            self.daemon_status_timer = schedule_timer(1.75, self, "pollDaemonStatus:", None, True)

        @objc.python_method
        def _stop_daemon_status_timer(self) -> None:
            if self.daemon_status_timer is not None:
                self.daemon_status_timer.invalidate()
                self.daemon_status_timer = None

        @objc.python_method
        def _reconcile_daemon_status(self) -> bool:
            if (
                self._hotkey_global_monitor is None
                and self._hotkey_local_monitor is None
            ):
                return False
            if self.daemon_controller.is_running:
                return False
            last_error = getattr(self.daemon_controller, "last_error", None)
            error = str(last_error) if last_error is not None else None
            if not error:
                error = self._strings()["daemon_not_running_message"]
            self._daemon_error_message = self._strings()["daemon_start_failed_message"].format(
                error=error
            )
            self._remove_hotkey_event_monitor()
            return True

        @objc.python_method
        def _dictionary_path(self) -> Path:
            path, _explicit = resolve_dictionary_path(None)
            return path

        @objc.python_method
        def _stop_permission_timer(self) -> None:
            self._permission_check_generation += 1
            if self.permission_timer is not None:
                self.permission_timer.invalidate()
                self.permission_timer = None

        @objc.python_method
        def _restart_permission_timer(self) -> None:
            self._stop_permission_timer()
            if self.onboarding_flow.current_step not in self._permission_step_config:
                return
            schedule_timer = (
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_
            )
            self.permission_timer = schedule_timer(1.75, self, "pollPermissions:", None, True)

        @objc.python_method
        def _start_permission_check(self) -> None:
            if self.onboarding_flow.current_step not in self._permission_step_config:
                return
            if self._permission_check_in_progress:
                return
            self._permission_check_in_progress = True
            generation = self._permission_check_generation
            thread = threading.Thread(
                target=self._check_permissions_in_background,
                args=(generation,),
                daemon=True,
            )
            self._permission_check_thread = thread
            try:
                thread.start()
            except RuntimeError:
                self._permission_check_in_progress = False

        @objc.python_method
        def _combined_permission_report(self) -> PermissionReport:
            report = check_all_permissions_subprocess()
            if report is None:
                report = check_all_permissions()
            return _combine_permission_reports(
                report,
                accessibility_in_process=check_accessibility_permission(),
                input_monitoring_in_process=check_input_monitoring_permission(),
            )

        @objc.python_method
        def _check_permissions_in_background(self, generation: int) -> None:
            report = self._combined_permission_report()
            payload = {"generation": generation, "report": report}
            try:
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyPermissionCheckResult:",
                    payload,
                    False,
                )
            except Exception:
                self._permission_check_in_progress = False

        def applyPermissionCheckResult_(self, payload):  # noqa: N802
            self._permission_check_in_progress = False
            try:
                generation = int(payload["generation"])
                report = payload["report"]
            except Exception:
                return
            if generation != self._permission_check_generation:
                return
            if self.onboarding_flow.current_step not in self._permission_step_config:
                return
            if not isinstance(report, PermissionReport):
                return
            self._last_permission_report = report
            self._refresh_onboarding_permissions(report)

        @objc.python_method
        def _refresh_onboarding_permissions(
            self,
            report: PermissionReport | None = None,
        ) -> PermissionReport:
            if report is None:
                report = check_all_permissions()
            self._setup_permission_report = report
            before_step = self.onboarding_flow.current_step
            self.onboarding_flow.refresh(report)
            after_step = self.onboarding_flow.current_step
            if after_step != before_step:
                self._show_onboarding_window_if_needed()
            self._push_event("permissionsChanged", self._state_payload())
            self._restart_permission_timer()
            self._update_status_menu()
            if report.all_granted:
                if self.onboarding_flow.is_complete:
                    self._start_daemon_if_ready(
                        report,
                        success_message_key="all_permissions_granted_started_message",
                    )
            return report

        @objc.python_method
        def _terminate_model_download_process(self) -> None:
            process = self._model_download_process
            if process is None:
                return
            if process.poll() is None:
                process.terminate()
            self._model_download_process = None

        @objc.python_method
        def _configured_model_token(self) -> str | None:
            try:
                config = self.bridge._load_config()
                model_token = str(config.stt.model)
                parse_stt_model(model_token)
            except Exception:
                self._push_daemon_state()
                return None
            return model_token

        @objc.python_method
        def _configured_backend_is_available(self) -> bool:
            try:
                config = self.bridge._load_config()
                model_token = str(config.stt.model)
                backend, _model_id = parse_stt_model(model_token)
            except Exception:
                self._push_daemon_state()
                return False
            if availability.is_backend_available(backend):
                return True
            self._push_daemon_state()
            return False

        @objc.python_method
        def _start_model_download(self, model_token: str, success_message_key: str) -> bool:
            process = self._model_download_process
            if process is not None and process.poll() is None:
                return False

            self._model_download_success_message_key = success_message_key
            self._push_event(
                "downloadProgress",
                {"type": "progress", "model": model_token, "fraction": None},
            )
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "ptarmigan_flow.cli",
                        "download-model",
                        "--model",
                        model_token,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:
                self._push_event(
                    "downloadProgress",
                    {"type": "error", "model": model_token, "message": str(exc)},
                )
                self._push_daemon_state()
                return True

            self._model_download_process = process
            thread = threading.Thread(
                target=self._read_model_download_progress,
                args=(process, model_token, success_message_key),
                daemon=True,
            )
            self._model_download_thread = thread
            try:
                thread.start()
            except RuntimeError as exc:
                self._terminate_model_download_process()
                self._push_event(
                    "downloadProgress",
                    {"type": "error", "model": model_token, "message": str(exc)},
                )
                self._push_daemon_state()
            return True

        @objc.python_method
        def _read_model_download_progress(
            self,
            process: subprocess.Popen[str],
            model_token: str,
            success_message_key: str,
        ) -> None:
            saw_terminal_event = False
            stream = process.stdout
            if stream is not None:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    payload["model"] = model_token
                    payload["success_message_key"] = success_message_key
                    if payload.get("type") in {"done", "error"}:
                        saw_terminal_event = True
                    self._send_model_download_event(payload)
            return_code = process.wait()
            if saw_terminal_event:
                return
            if return_code == 0:
                self._send_model_download_event(
                    {
                        "type": "done",
                        "model": model_token,
                        "success_message_key": success_message_key,
                    }
                )
                return
            self._send_model_download_event(
                {
                    "type": "error",
                    "model": model_token,
                    "message": f"download-model exited with status {return_code}",
                    "success_message_key": success_message_key,
                }
            )

        @objc.python_method
        def _send_model_download_event(self, payload: dict[str, object]) -> None:
            try:
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyModelDownloadProgress:",
                    payload,
                    False,
                )
            except Exception:
                return

        def applyModelDownloadProgress_(self, payload):  # noqa: N802
            try:
                event_type = str(payload.get("type", ""))
            except Exception:
                return
            self._push_event("downloadProgress", payload)
            if event_type == "done":
                self._model_download_process = None
                success_key = str(
                    payload.get(
                        "success_message_key",
                        self._model_download_success_message_key,
                    )
                )
                downloaded_model = str(payload.get("model", ""))
                if downloaded_model == self._configured_model_token():
                    self._start_daemon_if_ready(success_message_key=success_key)
                else:
                    self._push_daemon_state()
                return
            if event_type == "error":
                self._model_download_process = None
                self._push_event("daemonState", self._state_payload())

        @objc.python_method
        def _start_daemon_if_ready(
            self,
            report: PermissionReport | None = None,
            *,
            success_message_key: str = "voice_input_started_message",
        ) -> str | None:
            if report is None:
                report = self._last_permission_report or self._combined_permission_report()
            self._setup_permission_report = report
            if not report.all_granted:
                message = self._missing_permissions_message(report)
                self._push_daemon_error(message)
                return message
            self._daemon_error_message = None
            if not self._configured_backend_is_available():
                self._push_daemon_state()
                return None
            model_token = self._configured_model_token()
            if model_token is None:
                self._push_daemon_state()
                return None
            if not model_download.is_model_downloaded(model_token):
                self._start_model_download(model_token, success_message_key)
                self._push_daemon_state()
                return None
            if not self.daemon_controller.is_running:
                try:
                    self.daemon_controller.start()
                except Exception as exc:
                    message = self._strings()["daemon_start_failed_message"].format(error=exc)
                    self._push_daemon_error(message)
                    return message
            if not self.daemon_controller.is_running:
                last_error = getattr(self.daemon_controller, "last_error", None)
                error = str(last_error) if last_error is not None else None
                if not error:
                    error = self._strings()["daemon_not_running_message"]
                message = self._strings()["daemon_start_failed_message"].format(error=error)
                self._push_daemon_error(message)
                return message
            self._install_hotkey_event_monitor()
            self._start_daemon_status_timer()
            self._push_daemon_state()
            return None

        @objc.python_method
        def _stop_daemon(self) -> None:
            self.daemon_controller.stop()
            self._remove_hotkey_event_monitor()
            self._push_daemon_state()

        @objc.python_method
        def _request_permission(self, kind: str) -> None:
            if kind == "microphone":
                request_microphone_permission()
            elif kind == "accessibility":
                request_accessibility_permission()
            elif kind == "input_monitoring":
                request_input_monitoring_permission()
            self._start_permission_check()

        @objc.python_method
        def _open_system_settings(self, kind: str) -> None:
            settings_url = self._permission_step_config.get(kind, {}).get("settings_url")
            if settings_url is None:
                return
            url = NSURL.URLWithString_(settings_url)
            if url is not None:
                NSWorkspace.sharedWorkspace().openURL_(url)

        @objc.python_method
        def _current_app_bundle(self) -> Path | None:
            if not getattr(sys, "frozen", False):
                return None
            return app_relaunch.current_app_bundle_path()

        @objc.python_method
        def _current_app_version(self) -> str:
            return app_updater.running_app_version(self._current_app_bundle())

        @objc.python_method
        def _schedule_startup_update_check(self) -> None:
            # Only release builds check on their own; one shot, never polled.
            if not getattr(sys, "frozen", False):
                return
            if not app_updater.is_release_version(self._current_app_version()):
                return
            schedule_timer = (
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_
            )
            self._update_startup_timer = schedule_timer(
                _STARTUP_UPDATE_CHECK_DELAY_SECONDS,
                self,
                "startupUpdateCheck:",
                None,
                False,
            )

        @objc.python_method
        def _start_update_check(self, *, manual: bool) -> None:
            if self._update_check_in_progress or self._update_install_in_progress:
                return
            self._update_check_in_progress = True
            self._update_status_menu()
            thread = threading.Thread(
                target=self._check_updates_in_background,
                args=(manual,),
                daemon=True,
            )
            try:
                thread.start()
            except RuntimeError:
                self._update_check_in_progress = False
                self._update_status_menu()

        @objc.python_method
        def _check_updates_in_background(self, manual: bool) -> None:
            bundle = self._current_app_bundle()
            current_version = app_updater.running_app_version(bundle)
            try:
                result = app_updater.check_for_update(
                    current_version,
                    frozen=bool(getattr(sys, "frozen", False)),
                    bundle_path=bundle,
                )
            except Exception as exc:  # never let the worker thread die silently
                result = app_updater.UpdateCheckResult(
                    current_version=current_version,
                    error=str(exc),
                )
            try:
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyUpdateCheckResult:",
                    {"manual": manual, "result": result},
                    False,
                )
            except Exception:
                self._update_check_in_progress = False

        def applyUpdateCheckResult_(self, payload):  # noqa: N802
            self._update_check_in_progress = False
            try:
                manual = bool(payload["manual"])
                result = payload["result"]
            except Exception:
                return
            if not isinstance(result, app_updater.UpdateCheckResult):
                return
            if result.error is None:
                self._update_result = result if result.update_available else None
            else:
                logging.getLogger(__name__).warning("Update check failed: %s", result.error)
            self._update_status_menu()
            if manual:
                self._show_update_check_result(result)

        @objc.python_method
        def _run_alert(self, title: str, body: str, buttons: list[str]) -> int:
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            alert = NSAlert.alloc().init()
            alert.setMessageText_(title)
            alert.setInformativeText_(body)
            for button in buttons:
                alert.addButtonWithTitle_(button)
            return int(alert.runModal()) - _NS_ALERT_FIRST_BUTTON_RETURN

        @objc.python_method
        def _open_release_page(self, result: app_updater.UpdateCheckResult | None) -> None:
            page = result.release_page_url if result is not None else app_updater.RELEASES_PAGE_URL
            url = NSURL.URLWithString_(page)
            if url is not None:
                NSWorkspace.sharedWorkspace().openURL_(url)

        @objc.python_method
        def _show_update_check_result(self, result: app_updater.UpdateCheckResult) -> None:
            strings = self._strings()
            if result.error is not None:
                self._run_alert(
                    strings["update_check_failed_title"],
                    result.error,
                    [strings["update_ok_button"]],
                )
                return
            if not result.update_available:
                self._run_alert(
                    strings["update_up_to_date_title"],
                    strings["update_up_to_date_body"].format(version=result.current_version),
                    [strings["update_ok_button"]],
                )
                return
            self._confirm_update(result)

        @objc.python_method
        def _update_blocker_message(
            self,
            result: app_updater.UpdateCheckResult,
            strings: dict[str, str],
        ) -> str:
            blocker = result.install_blocker
            if blocker == app_updater.INSTALL_BLOCK_TRANSLOCATED:
                return strings["update_blocked_translocated"]
            if blocker == app_updater.INSTALL_BLOCK_NOT_WRITABLE:
                bundle = self._current_app_bundle()
                location = str(bundle.parent) if bundle is not None else ""
                return strings["update_blocked_not_writable"].format(path=location)
            if blocker == app_updater.INSTALL_BLOCK_NO_ASSET:
                return strings["update_blocked_no_asset"]
            return strings["update_blocked_dev_build"]

        @objc.python_method
        def _confirm_update(self, result: app_updater.UpdateCheckResult) -> None:
            if result.release is None:
                return
            strings = self._strings()
            version = result.release.version
            if result.is_dev_build:
                choice = self._run_alert(
                    strings["update_dev_build_title"],
                    strings["update_dev_build_body"].format(
                        current=result.current_version,
                        version=version,
                    ),
                    [strings["update_release_notes_button"], strings["update_ok_button"]],
                )
                if choice == 0:
                    self._open_release_page(result)
                return
            title = strings["update_available_title"].format(version=version)
            if not result.can_install:
                choice = self._run_alert(
                    title,
                    self._update_blocker_message(result, strings),
                    [strings["update_release_notes_button"], strings["update_later_button"]],
                )
                if choice == 0:
                    self._open_release_page(result)
                return
            choice = self._run_alert(
                title,
                strings["update_available_body"].format(
                    current=result.current_version,
                    version=version,
                ),
                [
                    strings["update_button"],
                    strings["update_release_notes_button"],
                    strings["update_later_button"],
                ],
            )
            if choice == 0:
                self._start_update_install(result)
            elif choice == 1:
                self._open_release_page(result)

        @objc.python_method
        def _start_update_install(self, result: app_updater.UpdateCheckResult) -> None:
            if self._update_install_in_progress:
                return
            bundle = self._current_app_bundle()
            if (
                bundle is None
                or not result.can_install
                or not app_updater.bundle_location_is_writable(bundle)
            ):
                self._open_release_page(result)
                return
            self._update_install_in_progress = True
            self._update_install_bundle = bundle
            self._update_progress = ("update_downloading_menu", "")
            self._update_status_menu()
            thread = threading.Thread(
                target=self._prepare_update_in_background,
                args=(result, bundle),
                daemon=True,
            )
            try:
                thread.start()
            except RuntimeError as exc:
                self._fail_update(str(exc))

        @objc.python_method
        def _post_update_event(self, payload: dict[str, object]) -> None:
            try:
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyUpdateEvent:",
                    payload,
                    False,
                )
            except Exception:
                logging.getLogger(__name__).exception("Could not post update event")

        @objc.python_method
        def _prepare_update_in_background(
            self,
            result: app_updater.UpdateCheckResult,
            bundle: Path,
        ) -> None:
            last_percent = {"value": -1}

            def on_progress(downloaded: int, total: int | None) -> None:
                if not total:
                    return
                percent = min(100, int(downloaded * 100 / total))
                if percent == last_percent["value"]:
                    return
                last_percent["value"] = percent
                self._post_update_event(
                    {"kind": "progress", "key": "update_downloading_menu", "percent": f"{percent}%"}
                )

            def on_verify() -> None:
                self._post_update_event(
                    {"kind": "progress", "key": "update_verifying_menu", "percent": ""}
                )

            try:
                prepared = app_updater.prepare_update(
                    result,
                    bundle,
                    progress=on_progress,
                    on_verify=on_verify,
                )
            except Exception as exc:
                logging.getLogger(__name__).warning("Update preparation failed: %s", exc)
                self._post_update_event({"kind": "failed", "error": str(exc)})
                return
            self._post_update_event({"kind": "prepared", "prepared": prepared})

        def applyUpdateEvent_(self, payload):  # noqa: N802
            try:
                kind = payload["kind"]
            except Exception:
                return
            if kind == "progress":
                self._update_progress = (str(payload["key"]), str(payload["percent"]))
                self._update_status_menu()
                return
            if kind == "prepared":
                self._install_prepared_update(payload["prepared"])
                return
            if kind == "failed":
                self._fail_update(str(payload.get("error") or ""))

        @objc.python_method
        def _install_prepared_update(self, prepared: app_updater.PreparedUpdate) -> None:
            bundle = self._update_install_bundle
            self._update_progress = ("update_installing_menu", "")
            self._update_status_menu()
            try:
                if bundle is None:
                    raise app_updater.UpdateError("app bundle location is unknown")
                app_updater.launch_install_helper(prepared, bundle, log_path=_UPDATE_LOG_PATH)
            except Exception as exc:
                app_updater.discard_prepared_update(prepared)
                self._fail_update(str(exc))
                return
            # The helper waits for this process to exit before swapping the bundle.
            NSApplication.sharedApplication().terminate_(self)

        @objc.python_method
        def _fail_update(self, error: str) -> None:
            self._update_install_in_progress = False
            self._update_progress = None
            self._update_install_bundle = None
            self._update_status_menu()
            strings = self._strings()
            self._run_alert(
                strings["update_failed_title"],
                strings["update_failed_body"].format(error=error),
                [strings["update_ok_button"]],
            )

        def startupUpdateCheck_(self, _timer):  # noqa: N802
            self._update_startup_timer = None
            self._start_update_check(manual=False)

        def checkForUpdates_(self, _sender):  # noqa: N802
            self._start_update_check(manual=True)

        def installUpdate_(self, _sender):  # noqa: N802
            if self._update_result is not None and not self._update_install_in_progress:
                self._confirm_update(self._update_result)

        @objc.python_method
        def _toggle_login(self) -> None:
            if login_item.is_enabled():
                login_item.unregister()
            else:
                login_item.register()
            self._update_status_menu()
            self._update_login_menu_state()
            self._push_event("daemonState", self._state_payload(refresh_login=True))

        @objc.python_method
        def _restart_app(self) -> bool:
            if app_relaunch.relaunch_app():
                NSApplication.sharedApplication().terminate_(self)
                return True
            return False

        def pollPermissions_(self, _timer):  # noqa: N802
            self._start_permission_check()

        def pollDaemonStatus_(self, _timer):  # noqa: N802
            if self._reconcile_daemon_status():
                self._push_daemon_state()
                return
            self._update_status_menu()

        def menuNeedsUpdate_(self, _menu):  # noqa: N802
            # Language/hotkey steps advance inside the web bridge; refresh lazily here.
            self._update_status_menu()
            self._update_login_menu_state()

        def showSetup_(self, _sender):  # noqa: N802
            self._set_route("onboarding")

        def showSettings_(self, _sender):  # noqa: N802
            self._set_route("settings")

        def showDictionaryEditor_(self, _sender):  # noqa: N802
            self._set_route("dictionary")

        def toggleLoginAtStartup_(self, _sender):  # noqa: N802
            self._toggle_login()

        def restartApp_(self, _sender):  # noqa: N802
            self._restart_app()

        def quit_(self, _sender):  # noqa: N802
            NSApplication.sharedApplication().terminate_(self)

    app = NSApplication.sharedApplication()
    _set_application_icon(app, NSImage, NSData)
    _install_edit_menu(app, NSMenu, NSMenuItem)
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    delegate = OnboardingController.alloc().init()
    app.setDelegate_(delegate)
    app.run()
    return 0


def main() -> int:
    """Run the native onboarding app."""
    # Must run before anything else: in the PyInstaller bundle the STT backend
    # spawns multiprocessing workers by re-executing this frozen binary. Without
    # freeze_support() those children would fall through to _run_appkit_app() and
    # launch a second GUI instead of running their worker, breaking app startup.
    multiprocessing.freeze_support()

    cli_result = _dispatch_cli_args(sys.argv[1:])
    if cli_result is not None:
        return cli_result

    if sys.platform != "darwin":
        print(f"{APP_NAME} onboarding requires macOS.", file=sys.stderr)
        return 2
    try:
        return _run_appkit_app()
    except ImportError as exc:
        print(f"{APP_NAME} onboarding requires PyObjC: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
