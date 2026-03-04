"""Right-bottom activity overlay process for recording/transcription state."""

from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Protocol

_POLL_INTERVAL_SECONDS = 0.04


@dataclass(slots=True)
class OverlayConfig:
    size: int
    margin_right: int
    margin_bottom: int
    parent_pid: int


class OverlayWindowPort(Protocol):
    def show_recording(self) -> None: ...

    def show_processing(self) -> None: ...

    def hide(self) -> None: ...

    def close(self) -> None: ...

    def pump_events(self, timeout_seconds: float) -> None: ...


class AppKitOverlayWindow(OverlayWindowPort):
    """Cocoa overlay window that avoids Dock icon and focus stealing."""

    def __init__(self, config: OverlayConfig) -> None:
        try:
            import AppKit  # type: ignore
            import Foundation  # type: ignore
        except Exception as exc:  # pragma: no cover - platform/optional dependency
            raise RuntimeError("Cocoa/AppKit runtime is not available") from exc

        self._appkit = AppKit
        self._foundation = Foundation
        self._config = config

        self._app = self._appkit.NSApplication.sharedApplication()
        accessory_policy = getattr(self._appkit, "NSApplicationActivationPolicyAccessory", 1)
        self._app.setActivationPolicy_(accessory_policy)

        style_mask = int(getattr(self._appkit, "NSWindowStyleMaskBorderless", 0))
        style_mask |= int(getattr(self._appkit, "NSWindowStyleMaskNonactivatingPanel", 0))

        self._panel = self._appkit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0.0, 0.0), (float(self._config.size), float(self._config.size))),
            style_mask,
            int(getattr(self._appkit, "NSBackingStoreBuffered", 2)),
            False,
        )
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(self._appkit.NSColor.clearColor())
        self._panel.setHasShadow_(False)
        self._panel.setIgnoresMouseEvents_(True)
        if hasattr(self._panel, "setFloatingPanel_"):
            self._panel.setFloatingPanel_(True)
        if hasattr(self._panel, "setHidesOnDeactivate_"):
            self._panel.setHidesOnDeactivate_(False)
        if hasattr(self._panel, "setBecomesKeyOnlyIfNeeded_"):
            self._panel.setBecomesKeyOnlyIfNeeded_(False)

        level = getattr(self._appkit, "NSStatusWindowLevel", None)
        if level is None:
            level = getattr(self._appkit, "NSFloatingWindowLevel", 3)
        self._panel.setLevel_(int(level))

        collection_behavior = 0
        collection_behavior |= int(getattr(self._appkit, "NSWindowCollectionBehaviorCanJoinAllSpaces", 0))
        collection_behavior |= int(getattr(self._appkit, "NSWindowCollectionBehaviorStationary", 0))
        collection_behavior |= int(getattr(self._appkit, "NSWindowCollectionBehaviorIgnoresCycle", 0))
        if collection_behavior and hasattr(self._panel, "setCollectionBehavior_"):
            self._panel.setCollectionBehavior_(collection_behavior)

        content_view = self._appkit.NSView.alloc().initWithFrame_(
            ((0.0, 0.0), (float(self._config.size), float(self._config.size)))
        )
        content_view.setWantsLayer_(True)
        self._panel.setContentView_(content_view)

        self._layer = content_view.layer()
        self._layer.setCornerRadius_(float(self._config.size) / 2.0)
        self._layer.setMasksToBounds_(True)

        self._position_bottom_right()
        self.hide()

    def _position_bottom_right(self) -> None:
        screen = self._appkit.NSScreen.mainScreen()
        if screen is None:
            frame = ((0.0, 0.0), (float(self._config.size), float(self._config.size)))
            self._panel.setFrame_display_(frame, True)
            return

        (origin_x, origin_y), (width, _height) = screen.visibleFrame()
        pos_x = float(origin_x + width - self._config.size - self._config.margin_right)
        pos_y = float(origin_y + self._config.margin_bottom)
        frame = ((pos_x, pos_y), (float(self._config.size), float(self._config.size)))
        self._panel.setFrame_display_(frame, True)

    def _set_fill(self, *, r: float, g: float, b: float, a: float) -> None:
        color = self._appkit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a)
        self._layer.setBackgroundColor_(color.CGColor())

    def _set_border(self, *, width: float, r: float, g: float, b: float, a: float) -> None:
        self._layer.setBorderWidth_(width)
        color = self._appkit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a)
        self._layer.setBorderColor_(color.CGColor())

    def show_recording(self) -> None:
        self._set_fill(r=0.86, g=0.21, b=0.27, a=0.95)
        self._set_border(width=0.0, r=1.0, g=1.0, b=1.0, a=1.0)
        self._position_bottom_right()
        self._panel.orderFrontRegardless()

    def show_processing(self) -> None:
        self._set_fill(r=0.04, g=0.45, b=0.52, a=0.95)
        self._set_border(width=max(1.0, self._config.size / 10.0), r=0.89, g=0.98, b=0.99, a=1.0)
        self._position_bottom_right()
        self._panel.orderFrontRegardless()

    def hide(self) -> None:
        self._panel.orderOut_(None)

    def close(self) -> None:
        self._panel.close()

    def pump_events(self, timeout_seconds: float) -> None:
        timeout = max(0.0, float(timeout_seconds))
        mode = getattr(self._foundation, "NSDefaultRunLoopMode", "kCFRunLoopDefaultMode")
        mask = getattr(self._appkit, "NSEventMaskAny", (1 << 64) - 1)
        deadline = self._foundation.NSDate.dateWithTimeIntervalSinceNow_(timeout)
        event = self._app.nextEventMatchingMask_untilDate_inMode_dequeue_(mask, deadline, mode, True)
        if event is not None:
            self._app.sendEvent_(event)
        self._app.updateWindows()


class OverlayRuntime:
    """Command runtime independent from UI backend."""

    def __init__(self, *, window: OverlayWindowPort, parent_pid: int) -> None:
        self._window = window
        self._commands: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._parent_pid = int(parent_pid)

    def enqueue_command(self, command: str) -> None:
        normalized = command.strip().upper()
        if normalized:
            self._commands.put(normalized)

    def request_exit(self) -> None:
        self._commands.put("EXIT")

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _safe_hide_and_close(self) -> None:
        try:
            self._window.hide()
        except Exception:
            pass
        try:
            self._window.close()
        except Exception:
            pass

    def process_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return

            if command == "SHOW_RECORDING":
                self._window.show_recording()
            elif command == "SHOW_PROCESSING":
                self._window.show_processing()
            elif command == "HIDE":
                self._window.hide()
            elif command == "EXIT":
                self._stop_event.set()
                self._safe_hide_and_close()
                return

    def check_parent_alive(self) -> None:
        if self._stop_event.is_set():
            return
        if os.getppid() != self._parent_pid:
            self.request_exit()

    def run(self) -> int:
        try:
            while not self._stop_event.is_set():
                self.process_commands()
                if self._stop_event.is_set():
                    break
                self.check_parent_alive()
                self._window.pump_events(_POLL_INTERVAL_SECONDS)
            return 0
        finally:
            self._safe_hide_and_close()


def _stdin_reader(runtime: OverlayRuntime) -> None:
    while not runtime.is_stopped():
        line = sys.stdin.readline()
        if line == "":
            runtime.request_exit()
            return
        runtime.enqueue_command(line)


def _parse_args(argv: list[str]) -> OverlayConfig:
    parser = argparse.ArgumentParser(description="Moonshine Flow activity overlay")
    parser.add_argument("--size", type=int, default=42)
    parser.add_argument("--margin-right", type=int, default=24)
    parser.add_argument("--margin-bottom", type=int, default=24)
    parser.add_argument("--parent-pid", type=int, default=os.getppid())
    args = parser.parse_args(argv)
    return OverlayConfig(
        size=max(16, int(args.size)),
        margin_right=max(0, int(args.margin_right)),
        margin_bottom=max(0, int(args.margin_bottom)),
        parent_pid=int(args.parent_pid),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    try:
        window = AppKitOverlayWindow(args)
    except Exception:
        return 1

    runtime = OverlayRuntime(window=window, parent_pid=args.parent_pid)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: runtime.request_exit())
    signal.signal(signal.SIGINT, lambda _signum, _frame: runtime.request_exit())

    reader = threading.Thread(target=_stdin_reader, args=(runtime,), daemon=True, name="activity-overlay-stdin")
    reader.start()
    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())
