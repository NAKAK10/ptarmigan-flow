"""PyObjC onboarding app used by notarized macOS release builds."""

from __future__ import annotations

import importlib.resources
import multiprocessing
import subprocess
import sys
from pathlib import Path

from ptarmigan_flow import app_relaunch, login_item, onboarding_strings
from ptarmigan_flow.app_daemon_controller import (
    DaemonController,
    daemon_run_command,
)
from ptarmigan_flow.app_icon import APP_ICON_FILE, APP_ICON_RESOURCE_PACKAGE
from ptarmigan_flow.app_settings_model import (
    SUPPORTED_HOTKEYS,
    SUPPORTED_LANGUAGES,
    SUPPORTED_OUTPUT_MODES,
    AppSettingsModel,
)
from ptarmigan_flow.config import (
    default_config_path,
    ensure_config_exists,
    load_config,
    write_config,
)
from ptarmigan_flow.corrections_editor_model import CorrectionsEditorModel
from ptarmigan_flow.onboarding_flow import OnboardingFlow
from ptarmigan_flow.permissions import (
    PermissionReport,
    check_all_permissions,
    request_accessibility_permission,
    request_input_monitoring_permission,
    request_microphone_permission,
)
from ptarmigan_flow.stt import availability
from ptarmigan_flow.stt.factory import parse_stt_model
from ptarmigan_flow.transcription_corrections import resolve_dictionary_path

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
        NSApplicationActivationPolicyAccessory,
        NSBackingStoreBuffered,
        NSButton,
        NSControlStateValueOff,
        NSControlStateValueOn,
        NSImage,
        NSMenu,
        NSMenuItem,
        NSPopUpButton,
        NSStatusBar,
        NSTextField,
        NSVariableStatusItemLength,
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
        dictionary_content_view: object
        dictionary_message_label: object
        dictionary_path: Path
        dictionary_row_controls: list[dict[str, object]]
        dictionary_window: object | None
        dictionary_menu_item: object
        dictation_status_menu_item: object
        login_menu_item: object
        message_label: object
        permission_timer: object | None
        quit_menu_item: object
        settings_content_view: object
        settings_hotkey_popup: object
        settings_language_popup: object
        settings_menu_item: object
        settings_message_label: object
        settings_model: AppSettingsModel
        settings_model_popup: object
        settings_output_mode_popup: object
        settings_window: object | None
        start_menu_item: object
        status_item: object
        status_menu: object
        stop_menu_item: object
        ui_language: str
        window: object

        # Keep the macOS permission name discoverable for packaging smoke tests: Input Monitoring.
        _permission_step_config = {
            "microphone": {
                "title_key": "microphone_title",
                "body_key": "microphone_body",
                "request_action": "requestMicrophone:",
                "settings_url": (
                    "x-apple.systempreferences:com.apple.preference.security"
                    "?Privacy_Microphone"
                ),
            },
            "accessibility": {
                "title_key": "accessibility_title",
                "body_key": "accessibility_body",
                "request_action": "requestAccessibility:",
                "settings_url": (
                    "x-apple.systempreferences:com.apple.preference.security"
                    "?Privacy_Accessibility"
                ),
            },
            "input_monitoring": {
                "title_key": "input_monitoring_title",
                "body_key": "input_monitoring_body",
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
            try:
                self.ui_language = load_config(default_config_path()).language
            except Exception:
                self.ui_language = "en"
            self.daemon_controller = DaemonController(
                lambda: daemon_run_command(default_config_path())
            )
            self.dictionary_path, _explicit = resolve_dictionary_path(None)
            self.corrections_model = CorrectionsEditorModel()
            self.dictionary_window = None
            self.dictionary_row_controls = []
            self.settings_window = None
            return self

        def applicationDidFinishLaunching_(self, _notification):  # noqa: N802
            self._build_status_item()
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
        def _menu_item(self, title: str, action: str | None):
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
            item.setTarget_(self)
            return item

        @objc.python_method
        def _strings(self) -> dict[str, str]:
            return onboarding_strings.strings_for(self.ui_language)

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
            self.start_menu_item = self._menu_item(
                strings["start_dictation_button"],
                "startDictation:",
            )
            self.status_menu.addItem_(self.start_menu_item)
            self.stop_menu_item = self._menu_item(
                strings["stop_dictation_button"],
                "stopDictation:",
            )
            self.status_menu.addItem_(self.stop_menu_item)
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
            self.quit_menu_item = self._menu_item(strings["quit_menu"], "quit:")
            self.status_menu.addItem_(self.quit_menu_item)
            self.status_item.setMenu_(self.status_menu)
            self._update_status_menu()

        @objc.python_method
        def _update_status_menu(self) -> None:
            if not hasattr(self, "status_menu"):
                return
            strings = self._strings()
            is_running = self.daemon_controller.is_running
            self.dictation_status_menu_item.setTitle_(
                strings["dictation_running_menu"]
                if is_running
                else strings["dictation_stopped_menu"]
            )
            self.start_menu_item.setTitle_(strings["start_dictation_button"])
            self.stop_menu_item.setTitle_(strings["stop_dictation_button"])
            self.settings_menu_item.setTitle_(strings["settings_menu"])
            self.dictionary_menu_item.setTitle_(strings["edit_dictionary_menu"])
            self.login_menu_item.setTitle_(strings["login_at_startup_menu"])
            self.quit_menu_item.setTitle_(strings["quit_menu"])
            self.start_menu_item.setEnabled_(not is_running)
            self.stop_menu_item.setEnabled_(is_running)
            login_state = (
                NSControlStateValueOn if login_item.is_enabled() else NSControlStateValueOff
            )
            self.login_menu_item.setState_(login_state)

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
            strings = self._strings()
            self.content_view.addSubview_(
                self._label(strings["app_setup_title"], 28, 374, 580, 32, size=22.0)
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
            strings = self._strings()
            self.content_view.addSubview_(
                self._label(strings["choose_language_title"], 36, 306, 560, 30, size=20.0)
            )
            self.content_view.addSubview_(
                self._label(
                    strings["choose_language_body"],
                    36,
                    270,
                    560,
                    24,
                )
            )
            self.content_view.addSubview_(
                self._button(strings["language_english"], "chooseEnglish:", 36, 214, 150)
            )
            self.content_view.addSubview_(
                self._button(strings["language_japanese"], "chooseJapanese:", 204, 214, 150)
            )
            self.content_view.addSubview_(
                self._button(strings["language_chinese"], "chooseChinese:", 372, 214, 150)
            )

        @objc.python_method
        def _render_permission_step(self, step: str) -> None:
            config = self._permission_step_config[step]
            strings = self._strings()
            self.content_view.addSubview_(
                self._label(
                    strings[config["title_key"]],
                    36,
                    306,
                    560,
                    30,
                    size=20.0,
                )
            )
            self.content_view.addSubview_(
                self._label(
                    strings[config["body_key"]],
                    28,
                    270,
                    580,
                    24,
                )
            )
            self.content_view.addSubview_(
                self._button(strings["allow_button"], config["request_action"], 36, 214, 150)
            )
            self.content_view.addSubview_(
                self._button(
                    strings["open_system_settings_button"],
                    "openSystemSettings:",
                    204,
                    214,
                    190,
                )
            )
            if step in {"accessibility", "input_monitoring"}:
                self.content_view.addSubview_(
                    self._label(strings["restart_required_note"], 36, 166, 560, 38)
                )
                self.content_view.addSubview_(
                    self._button(strings["restart_app_button"], "restartApp:", 36, 124, 150)
                )

        @objc.python_method
        def _render_done_step(self) -> None:
            strings = self._strings()
            self.content_view.addSubview_(
                self._label(strings["done_title"], 36, 306, 560, 30, size=20.0)
            )
            self.content_view.addSubview_(
                self._label(
                    strings["done_body"],
                    36,
                    270,
                    560,
                    24,
                )
            )
            self.content_view.addSubview_(
                self._button(strings["start_dictation_button"], "startDictation:", 36, 214, 150)
            )
            self.content_view.addSubview_(
                self._button(strings["stop_dictation_button"], "stopDictation:", 204, 214, 140)
            )
            self.content_view.addSubview_(
                self._button(strings["settings_button"], "showSettings:", 362, 214, 128)
            )
            self.content_view.addSubview_(
                self._button(
                    strings["login_at_startup_button"],
                    "toggleLoginAtStartup:",
                    36,
                    160,
                    178,
                )
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
            if hasattr(self, "message_label"):
                self.message_label.setStringValue_(message)

        @objc.python_method
        def _text_field(self, text: str, x: float, y: float, w: float, h: float = 26):
            field = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
            field.setStringValue_(text)
            return field

        @objc.python_method
        def _popup(
            self,
            items: list[tuple[str, str]],
            selected_value: str,
            x: float,
            y: float,
            w: float,
        ):
            popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
                NSMakeRect(x, y, w, 28),
                False,
            )
            selected_item = None
            for title, value in items:
                popup.addItemWithTitle_(title)
                item = popup.lastItem()
                if item is not None:
                    item.setRepresentedObject_(value)
                    if value == selected_value:
                        selected_item = item
            if selected_item is not None:
                popup.selectItem_(selected_item)
            elif items:
                popup.selectItemAtIndex_(0)
            return popup

        @objc.python_method
        def _selected_popup_value(self, popup: object) -> str:
            selected_item = popup.selectedItem()
            if selected_item is None:
                return ""
            value = selected_item.representedObject()
            if value is not None:
                return str(value)
            return str(selected_item.title())

        @objc.python_method
        def _settings_model_items(self) -> list[tuple[str, str]]:
            return [
                (f"{entry.label} ({entry.token})", entry.token)
                for entry in availability.available_model_entries()
            ]

        @objc.python_method
        def _settings_language_items(self) -> list[tuple[str, str]]:
            strings = self._strings()
            labels = {
                "en": strings["language_english"],
                "ja": strings["language_japanese"],
                "zh": strings["language_chinese"],
            }
            return [(labels[code], code) for code in SUPPORTED_LANGUAGES]

        @objc.python_method
        def _settings_hotkey_items(self) -> list[tuple[str, str]]:
            return [(hotkey, hotkey) for hotkey in SUPPORTED_HOTKEYS]

        @objc.python_method
        def _settings_output_mode_items(self) -> list[tuple[str, str]]:
            strings = self._strings()
            labels = {
                "direct_typing": strings["output_direct_typing"],
                "clipboard_paste": strings["output_clipboard_paste"],
            }
            return [(labels[mode], mode) for mode in SUPPORTED_OUTPUT_MODES]

        @objc.python_method
        def _build_settings_window(self) -> None:
            if self.settings_window is not None:
                return
            style = (
                NSWindowStyleMaskTitled
                | NSWindowStyleMaskClosable
                | NSWindowStyleMaskMiniaturizable
            )
            self.settings_window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, 620, 390),
                style,
                NSBackingStoreBuffered,
                False,
            )
            self.settings_content_view = self.settings_window.contentView()
            self.settings_window.center()

        @objc.python_method
        def _clear_settings_view(self) -> None:
            for subview in list(self.settings_content_view.subviews()):
                subview.removeFromSuperview()

        @objc.python_method
        def _set_settings_message(self, message: str) -> None:
            if hasattr(self, "settings_message_label"):
                self.settings_message_label.setStringValue_(message)

        @objc.python_method
        def _settings_field_labels(self, fields: list[str]) -> str:
            strings = self._strings()
            labels = {
                "model": strings["settings_model_label"],
                "language": strings["settings_language_label"],
                "hotkey": strings["settings_hotkey_label"],
                "output_mode": strings["settings_output_mode_label"],
            }
            return ", ".join(labels.get(field, field) for field in fields)

        @objc.python_method
        def _render_settings_form(self) -> None:
            self._clear_settings_view()
            strings = self._strings()
            self.settings_window.setTitle_(strings["settings_window_title"])
            self.settings_content_view.addSubview_(
                self._label(strings["settings_window_title"], 28, 338, 560, 30, size=22.0)
            )

            self.settings_content_view.addSubview_(
                self._label(strings["settings_model_label"], 34, 284, 160, 24)
            )
            model_items = self._settings_model_items()
            self.settings_model_popup = self._popup(
                model_items,
                self.settings_model.model,
                210,
                280,
                360,
            )
            self.settings_content_view.addSubview_(self.settings_model_popup)
            if not model_items:
                self.settings_content_view.addSubview_(
                    self._label(strings["settings_no_models_message"], 210, 250, 360, 24)
                )

            self.settings_content_view.addSubview_(
                self._label(strings["settings_language_label"], 34, 224, 160, 24)
            )
            self.settings_language_popup = self._popup(
                self._settings_language_items(),
                self.settings_model.language,
                210,
                220,
                220,
            )
            self.settings_content_view.addSubview_(self.settings_language_popup)

            self.settings_content_view.addSubview_(
                self._label(strings["settings_hotkey_label"], 34, 164, 160, 24)
            )
            self.settings_hotkey_popup = self._popup(
                self._settings_hotkey_items(),
                self.settings_model.hotkey,
                210,
                160,
                220,
            )
            self.settings_content_view.addSubview_(self.settings_hotkey_popup)

            self.settings_content_view.addSubview_(
                self._label(strings["settings_output_mode_label"], 34, 104, 160, 24)
            )
            self.settings_output_mode_popup = self._popup(
                self._settings_output_mode_items(),
                self.settings_model.output_mode,
                210,
                100,
                220,
            )
            self.settings_content_view.addSubview_(self.settings_output_mode_popup)

            self.settings_content_view.addSubview_(
                self._button(strings["open_config_advanced_button"], "openConfig:", 34, 46, 230)
            )
            self.settings_content_view.addSubview_(
                self._button(strings["settings_save_button"], "saveSettings:", 480, 46, 90)
            )
            self.settings_message_label = self._label("", 34, 16, 536, 24)
            self.settings_content_view.addSubview_(self.settings_message_label)

        @objc.python_method
        def _build_dictionary_window(self) -> None:
            if self.dictionary_window is not None:
                return
            style = (
                NSWindowStyleMaskTitled
                | NSWindowStyleMaskClosable
                | NSWindowStyleMaskMiniaturizable
            )
            self.dictionary_window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, 760, 620),
                style,
                NSBackingStoreBuffered,
                False,
            )
            self.dictionary_content_view = self.dictionary_window.contentView()
            self.dictionary_window.center()

        @objc.python_method
        def _clear_dictionary_view(self) -> None:
            for subview in list(self.dictionary_content_view.subviews()):
                subview.removeFromSuperview()

        @objc.python_method
        def _load_dictionary_model(self) -> None:
            self.dictionary_path, _explicit = resolve_dictionary_path(None)
            self.corrections_model = CorrectionsEditorModel.load(self.dictionary_path)

        @objc.python_method
        def _render_dictionary_editor(self) -> None:
            self._clear_dictionary_view()
            strings = self._strings()
            self.dictionary_row_controls = []
            self.dictionary_window.setTitle_(strings["dictionary_editor_title"])
            self.dictionary_content_view.addSubview_(
                self._label(strings["dictionary_editor_title"], 28, 570, 700, 30, size=22.0)
            )
            self.dictionary_content_view.addSubview_(
                self._label(str(self.dictionary_path), 30, 542, 700, 22)
            )
            y = self._render_dictionary_section(
                "exact",
                strings["dictionary_exact_rules_title"],
                self.corrections_model.exact,
                492,
            )
            self._render_dictionary_section(
                "regex",
                strings["dictionary_regex_rules_title"],
                self.corrections_model.regex,
                y - 26,
            )
            self.dictionary_content_view.addSubview_(
                self._button(
                    strings["dictionary_add_exact_button"],
                    "addExactCorrectionRow:",
                    30,
                    56,
                    118,
                )
            )
            self.dictionary_content_view.addSubview_(
                self._button(
                    strings["dictionary_add_regex_button"],
                    "addRegexCorrectionRow:",
                    162,
                    56,
                    118,
                )
            )
            self.dictionary_content_view.addSubview_(
                self._button(strings["dictionary_save_button"], "saveDictionary:", 626, 56, 90)
            )
            self.dictionary_message_label = self._label("", 30, 22, 700, 24)
            self.dictionary_content_view.addSubview_(self.dictionary_message_label)

        @objc.python_method
        def _render_dictionary_section(
            self,
            section: str,
            title: str,
            entries: dict[str, list[str]],
            y: float,
        ) -> float:
            strings = self._strings()
            self.dictionary_content_view.addSubview_(self._label(title, 30, y, 200, 24, size=17.0))
            y -= 30
            self.dictionary_content_view.addSubview_(
                self._label(strings["dictionary_canonical_label"], 32, y, 180, 20)
            )
            self.dictionary_content_view.addSubview_(
                self._label(strings["dictionary_candidates_patterns_label"], 240, y, 360, 20)
            )
            y -= 32
            if not entries:
                self.dictionary_content_view.addSubview_(
                    self._label(strings["dictionary_no_rules"], 32, y, 300, 24)
                )
                return y - 38
            for key, values in entries.items():
                key_field = self._text_field(key, 30, y, 190)
                values_field = self._text_field(", ".join(values), 238, y, 360)
                delete_button = self._button(
                    strings["dictionary_delete_button"],
                    "deleteDictionaryRow:",
                    616,
                    y - 3,
                    88,
                )
                delete_button.setTag_(len(self.dictionary_row_controls))
                self.dictionary_content_view.addSubview_(key_field)
                self.dictionary_content_view.addSubview_(values_field)
                self.dictionary_content_view.addSubview_(delete_button)
                self.dictionary_row_controls.append(
                    {
                        "section": section,
                        "key_field": key_field,
                        "values_field": values_field,
                    }
                )
                y -= 36
            return y - 20

        @objc.python_method
        def _set_dictionary_message(self, message: str) -> None:
            if hasattr(self, "dictionary_message_label"):
                self.dictionary_message_label.setStringValue_(message)

        @objc.python_method
        def _split_dictionary_values(self, text: str) -> list[str]:
            return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]

        @objc.python_method
        def _sync_dictionary_model_from_controls(self) -> None:
            exact: dict[str, list[str]] = {}
            regex: dict[str, list[str]] = {}
            for row in self.dictionary_row_controls:
                section = row["section"]
                key = str(row["key_field"].stringValue())
                values = self._split_dictionary_values(str(row["values_field"].stringValue()))
                if section == "exact":
                    exact[key] = values
                else:
                    regex[key] = values
            self.corrections_model = CorrectionsEditorModel(exact=exact, regex=regex)

        @objc.python_method
        def _new_dictionary_key(self, section: str) -> str:
            strings = self._strings()
            if section == "exact":
                table = self.corrections_model.exact
                base = strings["dictionary_new_exact_rule"]
            else:
                table = self.corrections_model.regex
                base = strings["dictionary_new_regex_rule"]
            if base not in table:
                return base
            index = 2
            while f"{base} {index}" in table:
                index += 1
            return f"{base} {index}"

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
            strings = self._strings()
            try:
                config_path = self._save_language(code)
                self.ui_language = code
                self.onboarding_flow.choose_language(code)
            except Exception as exc:
                self._set_message(strings["language_save_failed_message"].format(error=exc))
                return
            self._render_current_step()
            self._update_status_menu()
            self._set_message(strings["language_saved_message"].format(path=config_path))
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
                        success_message_key="all_permissions_granted_started_message",
                    )
            return report

        @objc.python_method
        def _start_daemon_if_ready(
            self,
            report: PermissionReport | None = None,
            *,
            success_message_key: str = "voice_input_started_message",
        ) -> None:
            strings = self._strings()
            if report is None:
                report = check_all_permissions()
            if not report.all_granted:
                self._set_message(strings["grant_permissions_message"])
                self._update_status_menu()
                return
            if not self._configured_backend_is_available():
                self._update_status_menu()
                return
            if not self.daemon_controller.is_running:
                try:
                    self.daemon_controller.start()
                except Exception as exc:
                    self._set_message(
                        strings["daemon_start_failed_message"].format(error=exc)
                    )
                    self._update_status_menu()
                    return
            if self.daemon_controller.is_running:
                self._set_message(strings[success_message_key])
                self._update_status_menu()
                return
            error = self.daemon_controller.last_error
            if error is None:
                self._set_message(strings["daemon_not_running_message"])
            else:
                self._set_message(strings["daemon_start_failed_message"].format(error=error))
            self._update_status_menu()

        @objc.python_method
        def _configured_backend_is_available(self) -> bool:
            strings = self._strings()
            try:
                config = load_config(default_config_path())
                model_token = str(config.stt.model)
                backend, _model_id = parse_stt_model(model_token)
            except Exception as exc:
                self._set_message(strings["daemon_start_failed_message"].format(error=exc))
                return False
            if availability.is_backend_available(backend):
                return True
            self._set_message(strings["model_unavailable_message"].format(model=model_token))
            return False

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

        def restartApp_(self, _sender):  # noqa: N802
            if app_relaunch.relaunch_app():
                NSApplication.sharedApplication().terminate_(self)
                return
            self._set_message(self._strings()["restart_failed_message"])

        def startDictation_(self, _sender):  # noqa: N802
            self._start_daemon_if_ready()

        def stopDictation_(self, _sender):  # noqa: N802
            self.daemon_controller.stop()
            self._set_message(self._strings()["dictation_stopped_message"])
            self._update_status_menu()

        def showSettings_(self, _sender):  # noqa: N802
            self._build_settings_window()
            strings = self._strings()
            try:
                self.settings_model = AppSettingsModel.load(default_config_path())
                load_error = None
            except Exception as exc:
                self.settings_model = AppSettingsModel(
                    model="",
                    language=self.ui_language,
                    hotkey="right_cmd",
                    output_mode="direct_typing",
                )
                load_error = exc
            self._render_settings_form()
            if load_error is not None:
                self._set_settings_message(
                    strings["settings_load_failed_message"].format(error=load_error)
                )
            self.settings_window.makeKeyAndOrderFront_(None)
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

        def saveSettings_(self, _sender):  # noqa: N802
            strings = self._strings()
            self.settings_model = AppSettingsModel(
                model=self._selected_popup_value(self.settings_model_popup),
                language=self._selected_popup_value(self.settings_language_popup),
                hotkey=self._selected_popup_value(self.settings_hotkey_popup),
                output_mode=self._selected_popup_value(self.settings_output_mode_popup),
            )
            errors = self.settings_model.validate()
            if errors:
                self._set_settings_message(
                    strings["settings_validation_error"].format(
                        fields=self._settings_field_labels(errors),
                    )
                )
                return
            try:
                self.settings_model.save(default_config_path())
            except Exception as exc:
                self._set_settings_message(
                    strings["settings_save_failed_message"].format(error=exc)
                )
                return
            self.ui_language = self.settings_model.language
            self._update_status_menu()
            self._render_current_step()
            self._render_settings_form()
            self._set_settings_message(strings["settings_saved_message"])
            self._set_message(strings["settings_saved_message"])

        def showDictionaryEditor_(self, _sender):  # noqa: N802
            self._build_dictionary_window()
            try:
                self._load_dictionary_model()
                load_error = None
            except Exception as exc:
                self.corrections_model = CorrectionsEditorModel()
                load_error = exc
            self._render_dictionary_editor()
            if load_error is not None:
                self._set_dictionary_message(
                    self._strings()["dictionary_load_failed_message"].format(error=load_error)
                )
            self.dictionary_window.makeKeyAndOrderFront_(None)
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

        def addExactCorrectionRow_(self, _sender):  # noqa: N802
            self._sync_dictionary_model_from_controls()
            self.corrections_model.add_exact(
                self._new_dictionary_key("exact"),
                [self._strings()["dictionary_default_candidate"]],
            )
            self._render_dictionary_editor()

        def addRegexCorrectionRow_(self, _sender):  # noqa: N802
            self._sync_dictionary_model_from_controls()
            self.corrections_model.add_regex(
                self._new_dictionary_key("regex"),
                [self._strings()["dictionary_default_pattern"]],
            )
            self._render_dictionary_editor()

        def deleteDictionaryRow_(self, sender):  # noqa: N802
            index = int(sender.tag())
            self._sync_dictionary_model_from_controls()
            if index >= len(self.dictionary_row_controls):
                self._render_dictionary_editor()
                return
            row = self.dictionary_row_controls[index]
            key = str(row["key_field"].stringValue())
            if row["section"] == "exact":
                self.corrections_model.remove_exact(key)
            else:
                self.corrections_model.remove_regex(key)
            self._render_dictionary_editor()

        def saveDictionary_(self, _sender):  # noqa: N802
            strings = self._strings()
            self._sync_dictionary_model_from_controls()
            errors = self.corrections_model.validate()
            if errors:
                error = errors[0]
                self._set_dictionary_message(
                    strings["dictionary_invalid_rule_message"].format(
                        section=error.section,
                        key=error.key,
                        pattern=error.pattern,
                        message=error.message,
                    )
                )
                return
            try:
                self.corrections_model.save(self.dictionary_path)
            except Exception as exc:
                self._set_dictionary_message(
                    strings["dictionary_save_failed_message"].format(error=exc)
                )
                return
            self._set_dictionary_message(strings["dictionary_saved_message"])
            self._set_message(strings["dictionary_saved_message"])

        def openConfig_(self, _sender):  # noqa: N802
            config_path = open_config()
            message = self._strings()["config_opened_message"].format(path=config_path)
            self._set_message(message)
            self._set_settings_message(message)

        def toggleLoginAtStartup_(self, _sender):  # noqa: N802
            strings = self._strings()
            if login_item.is_enabled():
                changed = login_item.unregister()
                message = (
                    strings["login_disabled_message"]
                    if changed
                    else strings["login_disable_failed_message"]
                )
            else:
                changed = login_item.register()
                message = (
                    strings["login_enabled_message"]
                    if changed
                    else strings["login_enable_failed_message"]
                )
            self._set_message(message)
            self._update_status_menu()

        def quit_(self, _sender):  # noqa: N802
            NSApplication.sharedApplication().terminate_(self)

    app = NSApplication.sharedApplication()
    _set_application_icon(app, NSImage, NSData)
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    delegate = OnboardingController.alloc().init()
    app.setDelegate_(delegate)
    app.activateIgnoringOtherApps_(True)
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
