"""PyObjC onboarding app used by notarized macOS release builds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ptarmigan_flow.config import default_config_path, ensure_config_exists
from ptarmigan_flow.launchd import install_launch_agent, restart_launch_agent
from ptarmigan_flow.permissions import (
    PermissionReport,
    check_all_permissions,
    request_accessibility_permission,
    request_input_monitoring_permission,
    request_microphone_permission,
)

APP_NAME = "PtarmiganFlow"


def open_config() -> Path:
    """Ensure the user config exists and open it with the default macOS editor."""
    config_path = default_config_path()
    ensure_config_exists(config_path)
    if sys.platform == "darwin":
        subprocess.run(["open", str(config_path)], check=False)
    return config_path


def _permission_status(report: PermissionReport, key: str) -> str:
    granted = bool(getattr(report, key))
    return "OK" if granted else "Missing"


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
        NSTextField,
        NSWindow,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskMiniaturizable,
        NSWindowStyleMaskTitled,
    )
    from Foundation import NSMakeRect, NSObject

    class OnboardingController(NSObject):
        """Small native window for permissions, launch setup, and config access."""

        status_labels: dict[str, object]
        message_label: object
        window: object

        def init(self):  # noqa: N802 - PyObjC follows Objective-C naming.
            self = objc.super(OnboardingController, self).init()
            if self is None:
                return None
            self.status_labels = {}
            return self

        def applicationDidFinishLaunching_(self, _notification):  # noqa: N802
            self._build_window()
            self.refreshStatus_(None)

        def _label(self, text: str, x: float, y: float, w: float, h: float, *, size: float = 14.0):
            label = NSTextField.labelWithString_(text)
            label.setFrame_(NSMakeRect(x, y, w, h))
            label.setFont_(label.font().fontWithSize_(size))
            return label

        def _button(self, title: str, action: str, x: float, y: float, w: float):
            button = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, w, 32))
            button.setTitle_(title)
            button.setTarget_(self)
            button.setAction_(action)
            return button

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
            content = self.window.contentView()
            content.addSubview_(self._label("PtarmiganFlow setup", 28, 374, 580, 32, size=22.0))
            content.addSubview_(
                self._label(
                    "Grant permissions, install login startup, and open config from one place.",
                    28,
                    342,
                    580,
                    24,
                )
            )

            rows = [
                ("Microphone", "microphone", "requestMicrophone:", 292),
                ("Accessibility", "accessibility", "requestAccessibility:", 238),
                ("Input Monitoring", "input_monitoring", "requestInputMonitoring:", 184),
            ]
            for title, key, action, y in rows:
                content.addSubview_(self._label(title, 36, y + 5, 180, 24))
                status = self._label("Checking", 216, y + 5, 120, 24)
                self.status_labels[key] = status
                content.addSubview_(status)
                content.addSubview_(self._button("Allow", action, 366, y, 110))

            content.addSubview_(self._button("Refresh", "refreshStatus:", 500, 184, 94))
            content.addSubview_(
                self._button("Install Login Startup", "installLaunchAgent:", 36, 104, 178)
            )
            content.addSubview_(
                self._button("Restart Daemon", "restartLaunchAgent:", 232, 104, 150)
            )
            content.addSubview_(self._button("Open Config", "openConfig:", 400, 104, 128))
            self.message_label = self._label("", 36, 52, 560, 32)
            content.addSubview_(self.message_label)
            self.window.center()
            self.window.makeKeyAndOrderFront_(None)

        def _set_message(self, message: str) -> None:
            self.message_label.setStringValue_(message)

        def refreshStatus_(self, _sender):  # noqa: N802
            report = check_all_permissions()
            for key, label in self.status_labels.items():
                label.setStringValue_(_permission_status(report, key))
            self._set_message("Permission status refreshed.")

        def requestMicrophone_(self, _sender):  # noqa: N802
            request_microphone_permission()
            self.refreshStatus_(None)

        def requestAccessibility_(self, _sender):  # noqa: N802
            request_accessibility_permission()
            self.refreshStatus_(None)

        def requestInputMonitoring_(self, _sender):  # noqa: N802
            request_input_monitoring_permission()
            self.refreshStatus_(None)

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
