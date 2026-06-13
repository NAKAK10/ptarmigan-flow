"""PyObjC onboarding app used by notarized macOS release builds."""

from __future__ import annotations

import importlib.resources
import subprocess
import sys
from pathlib import Path

from ptarmigan_flow.app_daemon_controller import (
    DaemonController,
    build_daemon_from_config,
)
from ptarmigan_flow.app_icon import APP_ICON_FILE, APP_ICON_RESOURCE_PACKAGE
from ptarmigan_flow.config import (
    default_config_path,
    ensure_config_exists,
    load_config,
    write_config,
)
from ptarmigan_flow.launchd import install_launch_agent, restart_launch_agent
from ptarmigan_flow.onboarding_flow import OnboardingFlow
from ptarmigan_flow.permissions import (
    PermissionReport,
    check_all_permissions,
    request_accessibility_permission,
    request_input_monitoring_permission,
    request_microphone_permission,
)

APP_NAME = "PtarmiganFlow"


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


def _run_appkit_app() -> int:
    import objc
    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyRegular,
        NSBackingStoreBuffered,
        NSButton,
        NSImage,
        NSTextField,
        NSWindow,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskMiniaturizable,
        NSWindowStyleMaskTitled,
        NSWorkspace,
    )
    from Foundation import NSURL, NSData, NSMakeRect, NSObject, NSTimer

    class OnboardingController(NSObject):
        """Native step-by-step onboarding window."""

        content_view: object
        message_label: object
        permission_timer: object | None
        window: object

        _permission_step_config = {
            "microphone": {
                "title": "Microphone Access",
                "body": "Allow PtarmiganFlow to capture audio while you hold the hotkey.",
                "request_action": "requestMicrophone:",
                "settings_url": (
                    "x-apple.systempreferences:com.apple.preference.security"
                    "?Privacy_Microphone"
                ),
            },
            "accessibility": {
                "title": "Accessibility Access",
                "body": (
                    "Allow PtarmiganFlow to control the active text field for dictation output."
                ),
                "request_action": "requestAccessibility:",
                "settings_url": (
                    "x-apple.systempreferences:com.apple.preference.security"
                    "?Privacy_Accessibility"
                ),
            },
            "input_monitoring": {
                "title": "Input Monitoring",
                "body": "Allow PtarmiganFlow to detect the push-to-talk hotkey.",
                "request_action": "requestInputMonitoring:",
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
            self.permission_timer = None
            self.daemon_controller = DaemonController(
                lambda: build_daemon_from_config(default_config_path())
            )
            return self

        def applicationDidFinishLaunching_(self, _notification):  # noqa: N802
            self._build_window()
            self._render_current_step()
            self._refresh_onboarding_permissions()

        def applicationWillTerminate_(self, _notification):  # noqa: N802
            self._stop_permission_timer()
            self.daemon_controller.stop()

        def applicationDidBecomeActive_(self, _notification):  # noqa: N802
            self._refresh_onboarding_permissions()

        @objc.python_method
        def _label(self, text: str, x: float, y: float, w: float, h: float, *, size: float = 14.0):
            label = NSTextField.labelWithString_(text)
            label.setFrame_(NSMakeRect(x, y, w, h))
            label.setFont_(label.font().fontWithSize_(size))
            return label

        @objc.python_method
        def _button(self, title: str, action: str, x: float, y: float, w: float):
            button = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, w, 32))
            button.setTitle_(title)
            button.setTarget_(self)
            button.setAction_(action)
            return button

        @objc.python_method
        def _build_window(self) -> None:
            style = (
                NSWindowStyleMaskTitled
                | NSWindowStyleMaskClosable
                | NSWindowStyleMaskMiniaturizable
            )
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, 640, 430),
                style,
                NSBackingStoreBuffered,
                False,
            )
            self.window.setTitle_(APP_NAME)
            self.content_view = self.window.contentView()
            self.window.center()
            self.window.makeKeyAndOrderFront_(None)

        @objc.python_method
        def _clear_content_view(self) -> None:
            for subview in list(self.content_view.subviews()):
                subview.removeFromSuperview()

        @objc.python_method
        def _render_current_step(self) -> None:
            self._clear_content_view()
            self.content_view.addSubview_(
                self._label("PtarmiganFlow setup", 28, 374, 580, 32, size=22.0)
            )
            step = self.onboarding_flow.current_step
            if step == "language":
                self._render_language_step()
            elif step == "done":
                self._render_done_step()
            else:
                self._render_permission_step(step)
            self.message_label = self._label("", 36, 52, 560, 32)
            self.content_view.addSubview_(self.message_label)
            self._restart_permission_timer()

        @objc.python_method
        def _render_language_step(self) -> None:
            self.content_view.addSubview_(
                self._label("Choose Language", 36, 306, 560, 30, size=20.0)
            )
            self.content_view.addSubview_(
                self._label(
                    "Select the transcription language to save into your config.",
                    36,
                    270,
                    560,
                    24,
                )
            )
            self.content_view.addSubview_(self._button("English", "chooseEnglish:", 36, 214, 150))
            self.content_view.addSubview_(
                self._button("Japanese", "chooseJapanese:", 204, 214, 150)
            )
            self.content_view.addSubview_(self._button("Chinese", "chooseChinese:", 372, 214, 150))

        @objc.python_method
        def _render_permission_step(self, step: str) -> None:
            config = self._permission_step_config[step]
            self.content_view.addSubview_(
                self._label(
                    config["title"],
                    36,
                    306,
                    560,
                    30,
                    size=20.0,
                )
            )
            self.content_view.addSubview_(
                self._label(
                    config["body"],
                    28,
                    270,
                    580,
                    24,
                )
            )
            self.content_view.addSubview_(
                self._button("Allow", config["request_action"], 36, 214, 150)
            )
            self.content_view.addSubview_(
                self._button("Open System Settings", "openSystemSettings:", 204, 214, 190)
            )

        @objc.python_method
        def _render_done_step(self) -> None:
            self.content_view.addSubview_(
                self._label("Ready to Dictate", 36, 306, 560, 30, size=20.0)
            )
            self.content_view.addSubview_(
                self._label(
                    "Setup is complete. Start dictation now or open the config file.",
                    36,
                    270,
                    560,
                    24,
                )
            )
            self.content_view.addSubview_(
                self._button("Start Dictation", "startDictation:", 36, 214, 150)
            )
            self.content_view.addSubview_(
                self._button("Stop Dictation", "stopDictation:", 204, 214, 140)
            )
            self.content_view.addSubview_(self._button("Open Config", "openConfig:", 362, 214, 128))
            self.content_view.addSubview_(
                self._button("Install Login Startup", "installLaunchAgent:", 36, 160, 178)
            )
            self.content_view.addSubview_(
                self._button("Restart Daemon", "restartLaunchAgent:", 232, 160, 150)
            )

        @objc.python_method
        def _stop_permission_timer(self) -> None:
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
            self.permission_timer = schedule_timer(1.0, self, "pollPermissions:", None, True)

        @objc.python_method
        def _set_message(self, message: str) -> None:
            self.message_label.setStringValue_(message)

        @objc.python_method
        def _save_language(self, code: str) -> Path:
            config_path = default_config_path()
            ensure_config_exists(config_path)
            config = load_config(config_path)
            config.language = code
            write_config(config_path, config)
            return config_path

        @objc.python_method
        def _choose_language(self, code: str) -> None:
            try:
                config_path = self._save_language(code)
                self.onboarding_flow.choose_language(code)
            except Exception as exc:
                self._set_message(f"Could not save language: {exc}")
                return
            self._render_current_step()
            self._set_message(f"Saved language to {config_path}.")
            self._refresh_onboarding_permissions()

        @objc.python_method
        def _refresh_onboarding_permissions(self) -> PermissionReport:
            report = check_all_permissions()
            before_step = self.onboarding_flow.current_step
            self.onboarding_flow.refresh(report)
            after_step = self.onboarding_flow.current_step
            if after_step != before_step:
                self._render_current_step()
            if report.all_granted:
                if self.onboarding_flow.is_complete:
                    self._start_daemon_if_ready(
                        report,
                        success_message="All permissions granted. Dictation started.",
                    )
            return report

        @objc.python_method
        def _start_daemon_if_ready(
            self,
            report: PermissionReport | None = None,
            *,
            success_message: str = "Dictation started.",
        ) -> None:
            if report is None:
                report = check_all_permissions()
            if not report.all_granted:
                self._set_message("Grant all permissions before starting dictation.")
                return
            if not self.daemon_controller.is_running:
                self.daemon_controller.start()
            if self.daemon_controller.is_running:
                self._set_message(success_message)
                return
            error = self.daemon_controller.last_error
            if error is None:
                self._set_message("Dictation daemon is not running yet.")
            else:
                self._set_message(f"Could not start dictation: {error}")

        def pollPermissions_(self, _timer):  # noqa: N802
            self._refresh_onboarding_permissions()

        def chooseEnglish_(self, _sender):  # noqa: N802
            self._choose_language("en")

        def chooseJapanese_(self, _sender):  # noqa: N802
            self._choose_language("ja")

        def chooseChinese_(self, _sender):  # noqa: N802
            self._choose_language("zh")

        def requestMicrophone_(self, _sender):  # noqa: N802
            request_microphone_permission()
            self._refresh_onboarding_permissions()

        def requestAccessibility_(self, _sender):  # noqa: N802
            request_accessibility_permission()
            self._refresh_onboarding_permissions()

        def requestInputMonitoring_(self, _sender):  # noqa: N802
            request_input_monitoring_permission()
            self._refresh_onboarding_permissions()

        def openSystemSettings_(self, _sender):  # noqa: N802
            step = self.onboarding_flow.current_step
            settings_url = self._permission_step_config.get(step, {}).get("settings_url")
            if settings_url is None:
                return
            url = NSURL.URLWithString_(settings_url)
            if url is not None:
                NSWorkspace.sharedWorkspace().openURL_(url)

        def startDictation_(self, _sender):  # noqa: N802
            self._start_daemon_if_ready()

        def stopDictation_(self, _sender):  # noqa: N802
            self.daemon_controller.stop()
            self._set_message("Dictation stopped.")

        def installLaunchAgent_(self, _sender):  # noqa: N802
            try:
                plist_path = install_launch_agent(default_config_path())
            except Exception as exc:
                self._set_message(f"Could not install login startup: {exc}")
                return
            self._set_message(f"Installed login startup: {plist_path}")

        def restartLaunchAgent_(self, _sender):  # noqa: N802
            try:
                restarted = restart_launch_agent()
            except Exception as exc:
                self._set_message(f"Could not restart daemon: {exc}")
                return
            message = "Daemon restarted." if restarted else "Login startup is not installed yet."
            self._set_message(message)

        def openConfig_(self, _sender):  # noqa: N802
            config_path = open_config()
            self._set_message(f"Opened config: {config_path}")

    app = NSApplication.sharedApplication()
    _set_application_icon(app, NSImage, NSData)
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    delegate = OnboardingController.alloc().init()
    app.setDelegate_(delegate)
    app.activateIgnoringOtherApps_(True)
    app.run()
    return 0


def main() -> int:
    """Run the native onboarding app."""
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
