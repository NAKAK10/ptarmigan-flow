from __future__ import annotations

import sys

import ptarmigan_flow.hotkey_monitor as hotkey_monitor
from ptarmigan_flow.hotkey_monitor import HotkeyMonitor


class _FakeListener:
    def __init__(self, on_press, on_release):
        self.on_press = on_press
        self.on_release = on_release

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def join(self) -> None:
        return None


class _CountingListener(_FakeListener):
    def __init__(self, on_press, on_release):
        super().__init__(on_press, on_release)
        self.start_calls = 0
        self.stop_calls = 0
        self.join_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def join(self) -> None:
        self.join_calls += 1


def test_start_uses_pynput_listener_when_only_appkit_is_loaded(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "AppKit", object())
    monkeypatch.delitem(sys.modules, "ptarmigan_flow.macos_app", raising=False)
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _CountingListener)

    hid_polling_started = 0

    def fake_start_hid_polling(self) -> None:
        nonlocal hid_polling_started
        hid_polling_started += 1

    monkeypatch.setattr(HotkeyMonitor, "_start_hid_polling", fake_start_hid_polling)

    monitor = HotkeyMonitor("a", on_press=lambda: None, on_release=lambda: None)

    monitor.start()

    assert monitor._listener.start_calls == 1
    assert monitor._listener_started is True
    assert hid_polling_started == 1


def test_start_skips_pynput_listener_when_explicitly_disabled(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "AppKit", object())
    monkeypatch.delitem(sys.modules, "ptarmigan_flow.macos_app", raising=False)
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _CountingListener)

    hid_polling_started = 0

    def fake_start_hid_polling(self) -> None:
        nonlocal hid_polling_started
        hid_polling_started += 1

    monkeypatch.setattr(HotkeyMonitor, "_start_hid_polling", fake_start_hid_polling)

    monitor = HotkeyMonitor(
        "a",
        on_press=lambda: None,
        on_release=lambda: None,
        use_pynput_listener=False,
    )

    monitor.start()

    assert monitor._listener.start_calls == 0
    assert monitor._listener_started is False
    assert hid_polling_started == 1


def test_stop_and_join_ignore_pynput_listener_before_it_starts(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _CountingListener)

    monitor = HotkeyMonitor("a", on_press=lambda: None, on_release=lambda: None)

    monitor.stop()
    monitor.join()

    assert monitor._listener.stop_calls == 0
    assert monitor._listener.join_calls == 0


def test_notify_press_calls_callback_once_and_sets_pressed_state(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _FakeListener)
    presses = 0

    def on_press() -> None:
        nonlocal presses
        presses += 1

    monitor = HotkeyMonitor("a", on_press=on_press, on_release=lambda: None)

    monitor.notify_press()

    assert presses == 1
    assert monitor.is_pressed() is True


def test_notify_release_calls_callback_and_clears_pressed_state(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _FakeListener)
    releases = 0

    def on_release() -> None:
        nonlocal releases
        releases += 1

    monitor = HotkeyMonitor("a", on_press=lambda: None, on_release=on_release)

    monitor.notify_press()
    monitor.notify_release()

    assert releases == 1
    assert monitor.is_pressed() is False


def test_notify_press_is_idempotent_while_pressed(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _FakeListener)
    presses = 0

    def on_press() -> None:
        nonlocal presses
        presses += 1

    monitor = HotkeyMonitor("a", on_press=on_press, on_release=lambda: None)

    monitor.notify_press()
    monitor.notify_press()

    assert presses == 1
    assert monitor.is_pressed() is True


def test_macos_keycode_for_hotkey_returns_supported_mapping() -> None:
    assert hotkey_monitor.macos_keycode_for_hotkey("right_shift") == 60
    assert hotkey_monitor.macos_keycode_for_hotkey(" unknown ") is None


def test_force_release_recovers_stuck_pressed_state(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _FakeListener)

    pressed = 0
    released = 0

    def on_press() -> None:
        nonlocal pressed
        pressed += 1

    def on_release() -> None:
        nonlocal released
        released += 1

    monitor = HotkeyMonitor("a", on_press=on_press, on_release=on_release, max_hold_seconds=1.0)

    monitor._on_press(monitor._target_key)
    monitor._force_release_if_stuck()
    monitor._on_release(monitor._target_key)

    assert pressed == 1
    assert released == 1


def test_stop_clears_pressed_state(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _FakeListener)

    pressed = 0

    def on_press() -> None:
        nonlocal pressed
        pressed += 1

    monitor = HotkeyMonitor("a", on_press=on_press, on_release=lambda: None, max_hold_seconds=1.0)

    monitor._on_press(monitor._target_key)
    monitor.stop()
    monitor._on_press(monitor._target_key)

    assert pressed == 2


def test_duplicate_press_recovers_missed_release(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _FakeListener)

    pressed = 0
    released = 0

    def on_press() -> None:
        nonlocal pressed
        pressed += 1

    def on_release() -> None:
        nonlocal released
        released += 1

    monitor = HotkeyMonitor("a", on_press=on_press, on_release=on_release, max_hold_seconds=1.0)

    monitor._on_press(monitor._target_key)
    monitor._on_press(monitor._target_key)

    assert pressed == 2
    assert released == 1


def test_is_pressed_reflects_hotkey_state(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _FakeListener)

    monitor = HotkeyMonitor(
        "a", on_press=lambda: None, on_release=lambda: None, max_hold_seconds=1.0
    )
    assert monitor.is_pressed() is False

    monitor._on_press(monitor._target_key)
    assert monitor.is_pressed() is True

    monitor._on_release(monitor._target_key)
    assert monitor.is_pressed() is False


def test_is_pressed_prefers_physical_state_when_available(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _FakeListener)

    monitor = HotkeyMonitor(
        "a", on_press=lambda: None, on_release=lambda: None, max_hold_seconds=1.0
    )
    monitor._on_press(monitor._target_key)
    monitor._physical_pressed_state = lambda: True
    assert monitor.is_pressed() is True

    monitor._physical_pressed_state = lambda: False
    assert monitor.is_pressed() is True


def test_is_pressed_ignores_physical_false(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _FakeListener)

    monitor = HotkeyMonitor(
        "a", on_press=lambda: None, on_release=lambda: None, max_hold_seconds=1.0
    )
    monitor._physical_pressed_state = lambda: False
    monitor._on_press(monitor._target_key)

    assert monitor.is_pressed() is True


def test_on_press_accepts_event_even_when_physical_released(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.hotkey_monitor.keyboard.Listener", _FakeListener)

    pressed = 0

    def on_press() -> None:
        nonlocal pressed
        pressed += 1

    monitor = HotkeyMonitor("a", on_press=on_press, on_release=lambda: None, max_hold_seconds=1.0)
    monitor._physical_pressed_state = lambda: False

    monitor._on_press(monitor._target_key)

    assert pressed == 1
    assert monitor.is_pressed() is True
