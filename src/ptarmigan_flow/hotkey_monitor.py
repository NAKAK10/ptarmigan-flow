"""Global hotkey listener."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable

from pynput import keyboard

LOGGER = logging.getLogger(__name__)


SPECIAL_KEYS: dict[str, keyboard.Key] = {
    "right_cmd": keyboard.Key.cmd_r,
    "left_cmd": keyboard.Key.cmd,
    "right_shift": keyboard.Key.shift_r,
    "left_shift": keyboard.Key.shift,
    "right_alt": keyboard.Key.alt_r,
    "left_alt": keyboard.Key.alt_l,
    "right_ctrl": keyboard.Key.ctrl_r,
    "left_ctrl": keyboard.Key.ctrl_l,
}

_MACOS_KEYCODE_BY_NAME: dict[str, int] = {
    "right_cmd": 54,
    "left_cmd": 55,
    "right_shift": 60,
    "left_shift": 56,
    "right_alt": 61,
    "left_alt": 58,
    "right_ctrl": 62,
    "left_ctrl": 59,
}


def macos_keycode_for_hotkey(key_name: str) -> int | None:
    return _MACOS_KEYCODE_BY_NAME.get(key_name.strip().lower())


class HotkeyMonitor:
    """Monitor one key globally and emit press/release callbacks."""

    def __init__(
        self,
        key_name: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        *,
        max_hold_seconds: float | None = None,
    ) -> None:
        self.key_name = key_name
        self._target_key = self._parse_key_name(key_name)
        self._on_press_callback = on_press
        self._on_release_callback = on_release
        self._max_hold_seconds = max_hold_seconds if (max_hold_seconds or 0) > 0 else None
        self._pressed = False
        self._lock = threading.Lock()
        self._release_timer: threading.Timer | None = None
        self._macos_keycode = _MACOS_KEYCODE_BY_NAME.get(self.key_name.strip().lower())
        self._hid_key_state_reader = self._build_hid_key_state_reader()
        self._hid_polling_active = False
        self._hid_polling_thread = None
        self._listener_started: bool = False
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)

    @staticmethod
    def _build_hid_key_state_reader() -> Callable[[int], bool] | None:
        try:
            from Quartz import CGEventSourceKeyState, kCGEventSourceStateHIDSystemState
        except Exception:
            return None

        def _reader(keycode: int) -> bool:
            return bool(CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, keycode))

        return _reader

    def _cancel_release_timer(self) -> None:
        if self._release_timer is None:
            return
        self._release_timer.cancel()
        self._release_timer = None

    def _schedule_release_timer(self) -> None:
        self._cancel_release_timer()
        if self._max_hold_seconds is None:
            return
        timer = threading.Timer(self._max_hold_seconds, self._force_release_if_stuck)
        timer.daemon = True
        self._release_timer = timer
        timer.start()

    @staticmethod
    def _parse_key_name(key_name: str) -> keyboard.Key | keyboard.KeyCode:
        name = key_name.strip().lower()
        if name in SPECIAL_KEYS:
            return SPECIAL_KEYS[name]

        if len(name) == 1:
            return keyboard.KeyCode.from_char(name)

        supported = ", ".join(sorted(SPECIAL_KEYS))
        raise ValueError(
            f"Unsupported key '{key_name}'. "
            f"Supported: {supported} or single characters"
        )

    def _matches(self, key: keyboard.Key | keyboard.KeyCode | None) -> bool:
        return key == self._target_key

    def _register_press(self) -> None:
        fired = False
        with self._lock:
            if not self._pressed:
                self._pressed = True
                self._schedule_release_timer()
                fired = True
        if fired:
            LOGGER.info("Hotkey press detected: %s", self.key_name)
            self._on_press_callback()

    def _register_release(self) -> None:
        fired = False
        with self._lock:
            if self._pressed:
                self._pressed = False
                self._cancel_release_timer()
                fired = True
        if fired:
            LOGGER.info("Hotkey release detected: %s", self.key_name)
            self._on_release_callback()

    def notify_press(self) -> None:
        """Feed an externally-detected press (e.g. from an NSEvent monitor)."""
        self._register_press()

    def notify_release(self) -> None:
        """Feed an externally-detected release (e.g. from an NSEvent monitor)."""
        self._register_release()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        recovered_stuck_release = False
        with self._lock:
            if not self._matches(key):
                return
            if self._pressed:
                # Recover from a missed release event: synthesize one before
                # accepting the new press so daemon state can recover quickly.
                recovered_stuck_release = True
                self._pressed = False
                self._cancel_release_timer()
            self._pressed = True
            self._schedule_release_timer()
        if recovered_stuck_release:
            LOGGER.warning("Recovered hotkey state after missed release event: %s", self.key_name)
            self._on_release_callback()
        LOGGER.debug("Hotkey down: %s", self.key_name)
        self._on_press_callback()

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        with self._lock:
            if not self._matches(key) or not self._pressed:
                return
            self._pressed = False
            self._cancel_release_timer()
        LOGGER.debug("Hotkey up: %s", self.key_name)
        self._on_release_callback()

    def _force_release_if_stuck(self) -> None:
        with self._lock:
            if not self._pressed:
                return
            self._pressed = False
            self._release_timer = None
        LOGGER.warning(
            "Hotkey release fallback triggered after %.2fs: %s",
            self._max_hold_seconds or 0.0,
            self.key_name,
        )
        self._on_release_callback()

    def start(self) -> None:
        """Start listening in background thread."""
        if "AppKit" not in sys.modules:
            try:
                self._listener.start()
                self._listener_started = True
            except Exception:
                LOGGER.warning("pynput listener failed to start; relying on HID polling")
        self._start_hid_polling()

    def _start_hid_polling(self) -> None:
        if self._macos_keycode is None or self._hid_key_state_reader is None:
            return
        if self._hid_polling_active:
            return
        self._hid_polling_active = True
        self._hid_polling_thread = threading.Thread(
            target=self._hid_poll_loop,
            name="hid-hotkey-poll",
            daemon=True,
        )
        self._hid_polling_thread.start()

    def _hid_poll_loop(self) -> None:
        prev = False
        while self._hid_polling_active:
            physical = self._physical_pressed_state()
            if physical is True and not prev:
                self._register_press()
                prev = True
            elif physical is False and prev:
                self._register_release()
                prev = False
            time.sleep(0.05)

    def _physical_pressed_state(self) -> bool | None:
        if self._macos_keycode is None or self._hid_key_state_reader is None:
            return None
        try:
            return self._hid_key_state_reader(self._macos_keycode)
        except Exception:
            return None

    def physical_pressed_state(self) -> bool | None:
        """Return current physical pressed state when available (macOS Quartz)."""
        return self._physical_pressed_state()

    def is_pressed(self) -> bool:
        with self._lock:
            return self._pressed

    def stop(self) -> None:
        """Stop listener."""
        self._hid_polling_active = False
        with self._lock:
            self._pressed = False
            self._cancel_release_timer()
        if self._listener_started:
            self._listener.stop()

    def join(self) -> None:
        """Block until listener exits."""
        if self._listener_started:
            self._listener.join()
