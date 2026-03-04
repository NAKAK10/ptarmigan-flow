"""Right-bottom activity overlay process for recording/transcription state."""

from __future__ import annotations

import argparse
import math
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Protocol

_POLL_INTERVAL_SECONDS = 0.04
_HIDE_FADE_SECONDS = 0.16
_RING_PHASE_OFFSET_SECONDS = 0.45
_ANIMATION_REPEAT_COUNT = 1_000_000_000.0


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

        try:
            import QuartzCore  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            QuartzCore = None

        self._appkit = AppKit
        self._foundation = Foundation
        self._quartz_core = QuartzCore
        self._config = config
        self._hide_deadline_monotonic: float | None = None
        self._animation_started_at_monotonic = time.monotonic()
        self._mode = "hidden"
        self._is_visible = False

        self._app = self._appkit.NSApplication.sharedApplication()
        accessory_policy = getattr(self._appkit, "NSApplicationActivationPolicyAccessory", 1)
        self._app.setActivationPolicy_(accessory_policy)

        # Keep current config compatibility, but render a tighter visual footprint.
        requested_size = float(self._config.size)
        self._hud_size = max(14.0, requested_size * 0.68)
        self._padding = max(2.0, requested_size * 0.04)
        self._panel_size = self._hud_size + (self._padding * 2.0)

        style_mask = int(getattr(self._appkit, "NSWindowStyleMaskBorderless", 0))
        style_mask |= int(getattr(self._appkit, "NSWindowStyleMaskNonactivatingPanel", 0))

        self._panel = self._appkit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0.0, 0.0), (self._panel_size, self._panel_size)),
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
        collection_behavior |= int(
            getattr(self._appkit, "NSWindowCollectionBehaviorCanJoinAllSpaces", 0)
        )
        collection_behavior |= int(getattr(self._appkit, "NSWindowCollectionBehaviorStationary", 0))
        collection_behavior |= int(
            getattr(self._appkit, "NSWindowCollectionBehaviorIgnoresCycle", 0)
        )
        if collection_behavior and hasattr(self._panel, "setCollectionBehavior_"):
            self._panel.setCollectionBehavior_(collection_behavior)

        content_view = self._appkit.NSView.alloc().initWithFrame_(
            ((0.0, 0.0), (self._panel_size, self._panel_size))
        )
        content_view.setWantsLayer_(True)
        self._panel.setContentView_(content_view)

        self._root_layer = content_view.layer()
        self._root_layer.setBackgroundColor_(self._color(0.0, 0.0, 0.0, 0.0))
        self._root_layer.setCornerRadius_(self._panel_size / 2.0)
        self._root_layer.setMasksToBounds_(False)

        self._glass_layer = self._new_layer()
        self._glass_layer.setFrame_(
            ((self._padding, self._padding), (self._hud_size, self._hud_size))
        )
        self._glass_layer.setCornerRadius_(self._hud_size / 2.0)
        self._glass_layer.setBorderWidth_(1.0)
        self._glass_layer.setMasksToBounds_(False)
        self._root_layer.addSublayer_(self._glass_layer)

        self._glow_layer = self._new_layer()
        self._glow_layer.setFrame_(
            ((self._padding, self._padding), (self._hud_size, self._hud_size))
        )
        self._glow_layer.setCornerRadius_(self._hud_size / 2.0)
        self._glow_layer.setBackgroundColor_(self._color(1.0, 1.0, 1.0, 0.0))
        self._glow_layer.setHidden_(True)
        self._root_layer.addSublayer_(self._glow_layer)

        self._ring_a_base_size = self._hud_size * 0.66
        ring_a_origin = self._padding + ((self._hud_size - self._ring_a_base_size) / 2.0)
        self._ring_a_layer = self._new_layer()
        self._ring_a_layer.setFrame_(
            ((ring_a_origin, ring_a_origin), (self._ring_a_base_size, self._ring_a_base_size))
        )
        self._ring_a_layer.setCornerRadius_(self._ring_a_base_size / 2.0)
        self._ring_a_layer.setBackgroundColor_(self._color(1.0, 1.0, 1.0, 0.0))
        self._ring_a_layer.setHidden_(True)
        self._root_layer.addSublayer_(self._ring_a_layer)

        self._ring_b_base_size = self._hud_size * 0.84
        ring_b_origin = self._padding + ((self._hud_size - self._ring_b_base_size) / 2.0)
        self._ring_b_layer = self._new_layer()
        self._ring_b_layer.setFrame_(
            ((ring_b_origin, ring_b_origin), (self._ring_b_base_size, self._ring_b_base_size))
        )
        self._ring_b_layer.setCornerRadius_(self._ring_b_base_size / 2.0)
        self._ring_b_layer.setBackgroundColor_(self._color(1.0, 1.0, 1.0, 0.0))
        self._ring_b_layer.setHidden_(True)
        self._root_layer.addSublayer_(self._ring_b_layer)

        self._pulse_base_size = self._hud_size * 0.58
        pulse_origin = self._padding + ((self._hud_size - self._pulse_base_size) / 2.0)
        self._pulse_layer = self._new_layer()
        self._pulse_layer.setFrame_(
            ((pulse_origin, pulse_origin), (self._pulse_base_size, self._pulse_base_size))
        )
        self._pulse_layer.setCornerRadius_(self._pulse_base_size / 2.0)
        self._pulse_layer.setBackgroundColor_(self._color(1.0, 1.0, 1.0, 0.0))
        self._pulse_layer.setHidden_(True)
        self._root_layer.addSublayer_(self._pulse_layer)

        self._core_base_size = self._hud_size * 0.40
        core_origin = self._padding + ((self._hud_size - self._core_base_size) / 2.0)
        self._core_layer = self._new_layer()
        self._core_layer.setFrame_(
            ((core_origin, core_origin), (self._core_base_size, self._core_base_size))
        )
        self._core_layer.setCornerRadius_(self._core_base_size / 2.0)
        self._core_layer.setHidden_(True)
        self._root_layer.addSublayer_(self._core_layer)

        self._apply_idle_visuals()
        self._clear_animations()

        self._root_layer.setOpacity_(1.0)

        self._position_bottom_right()
        self.hide()

    @property
    def _animations_supported(self) -> bool:
        return self._quartz_core is not None and hasattr(self._quartz_core, "CABasicAnimation")

    def _new_layer(self):
        if self._quartz_core is not None and hasattr(self._quartz_core, "CALayer"):
            return self._quartz_core.CALayer.layer()
        return self._root_layer.__class__.layer()

    def _color(self, r: float, g: float, b: float, a: float):
        return self._appkit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a).CGColor()

    def _timing_function(self, name: str):
        if self._quartz_core is None:
            return None
        mapping = {
            "ease_out": "kCAMediaTimingFunctionEaseOut",
            "ease_in_out": "kCAMediaTimingFunctionEaseInEaseOut",
            "linear": "kCAMediaTimingFunctionLinear",
        }
        constant_name = mapping.get(name, "kCAMediaTimingFunctionEaseInEaseOut")
        function_name = getattr(self._quartz_core, constant_name, None) or "easeInEaseOut"
        try:
            return self._quartz_core.CAMediaTimingFunction.functionWithName_(function_name)
        except Exception:
            return None

    def _make_animation(
        self,
        *,
        key_path: str,
        from_value: float,
        to_value: float,
        duration: float,
        timing: str = "ease_in_out",
        autoreverses: bool = False,
        delay: float = 0.0,
        repeat: bool = True,
    ):
        if not self._animations_supported:
            return None
        animation = self._quartz_core.CABasicAnimation.animationWithKeyPath_(key_path)
        animation.setFromValue_(float(from_value))
        animation.setToValue_(float(to_value))
        animation.setDuration_(max(0.01, float(duration)))
        animation.setAutoreverses_(bool(autoreverses))
        animation.setRepeatCount_(_ANIMATION_REPEAT_COUNT if repeat else 0.0)
        if delay > 0.0:
            begin_time = float(self._quartz_core.CACurrentMediaTime() + delay)
            animation.setBeginTime_(begin_time)
            fill_mode = getattr(self._quartz_core, "kCAFillModeBoth", None)
            if fill_mode is not None:
                animation.setFillMode_(fill_mode)
            animation.setRemovedOnCompletion_(False)
        timing_function = self._timing_function(timing)
        if timing_function is not None:
            animation.setTimingFunction_(timing_function)
        return animation

    @staticmethod
    def _remove_animation(layer, key: str) -> None:
        try:
            layer.removeAnimationForKey_(key)
        except Exception:
            pass

    @staticmethod
    def _add_animation(layer, animation, key: str) -> None:
        if animation is None:
            return
        try:
            layer.addAnimation_forKey_(animation, key)
        except Exception:
            pass

    def _clear_animations(self) -> None:
        for layer in (
            self._root_layer,
            self._glass_layer,
            self._glow_layer,
            self._ring_a_layer,
            self._ring_b_layer,
            self._pulse_layer,
            self._core_layer,
        ):
            try:
                layer.removeAllAnimations()
            except Exception:
                pass
        self._remove_animation(self._root_layer, "overlay.fade.out")
        self._remove_animation(self._ring_a_layer, "processing.ring.a.rotation")
        self._remove_animation(self._ring_b_layer, "processing.ring.b.rotation")

    def _apply_idle_visuals(self) -> None:
        self._glass_layer.setBackgroundColor_(self._color(0.11, 0.13, 0.17, 0.58))
        self._glass_layer.setBorderColor_(self._color(0.88, 0.95, 1.0, 0.20))
        self._glass_layer.setShadowColor_(self._color(0.0, 0.0, 0.0, 1.0))
        self._glass_layer.setShadowOpacity_(0.18)
        self._glass_layer.setShadowRadius_(6.0)
        self._glass_layer.setShadowOffset_((0.0, -1.0))
        self._glow_layer.setHidden_(True)
        self._ring_a_layer.setHidden_(True)
        self._ring_b_layer.setHidden_(True)
        self._pulse_layer.setHidden_(True)
        self._core_layer.setHidden_(True)

    def _prepare_show(self) -> None:
        self._hide_deadline_monotonic = None
        self._remove_animation(self._root_layer, "overlay.fade.out")
        self._root_layer.setOpacity_(1.0)
        self._position_bottom_right()
        self._panel.orderFrontRegardless()
        self._is_visible = True

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._animation_started_at_monotonic = time.monotonic()

    def _set_circle_size(self, layer, diameter: float) -> None:
        bounded = max(2.0, float(diameter))
        origin = self._padding + ((self._hud_size - bounded) / 2.0)
        layer.setFrame_(((origin, origin), (bounded, bounded)))
        layer.setCornerRadius_(bounded / 2.0)

    @staticmethod
    def _set_layer_opacity(layer, opacity: float) -> None:
        layer.setOpacity_(max(0.0, min(1.0, float(opacity))))

    def _tick_fallback_animation(self) -> None:
        if self._animations_supported or not self._is_visible:
            return
        mode = self._mode
        now = time.monotonic()
        elapsed = max(0.0, now - self._animation_started_at_monotonic)
        tau = math.tau

        if mode == "recording":
            phase = 0.5 * (1.0 + math.sin((elapsed / 0.82) * tau))
            self._pulse_layer.setHidden_(False)
            self._core_layer.setHidden_(False)
            self._set_circle_size(
                self._pulse_layer,
                self._pulse_base_size * (0.90 + (phase * 0.62)),
            )
            self._set_circle_size(
                self._core_layer,
                self._core_base_size * (0.95 + (phase * 0.13)),
            )
            self._set_layer_opacity(self._pulse_layer, 0.60 - (phase * 0.55))
            self._set_layer_opacity(self._glow_layer, 0.22 + (phase * 0.32))
            self._set_layer_opacity(self._core_layer, 0.80 + (phase * 0.20))
            return

        if mode == "processing":
            cycle = 1.2
            phase_a = (elapsed % cycle) / cycle
            phase_b = ((elapsed + _RING_PHASE_OFFSET_SECONDS) % cycle) / cycle
            core_phase = 0.5 * (1.0 + math.sin((elapsed / 0.66) * tau))
            self._ring_a_layer.setHidden_(False)
            self._ring_b_layer.setHidden_(False)
            self._core_layer.setHidden_(False)

            self._set_circle_size(
                self._ring_a_layer,
                self._ring_a_base_size * (0.78 + (phase_a * 0.52)),
            )
            self._set_circle_size(
                self._ring_b_layer,
                self._ring_b_base_size * (0.78 + (phase_b * 0.52)),
            )
            self._set_layer_opacity(self._ring_a_layer, max(0.06, 0.74 - (phase_a * 0.72)))
            self._set_layer_opacity(self._ring_b_layer, max(0.06, 0.74 - (phase_b * 0.72)))
            self._set_circle_size(
                self._core_layer,
                self._core_base_size * (0.95 + (core_phase * 0.14)),
            )
            self._set_layer_opacity(self._core_layer, 0.84 + (core_phase * 0.16))
            self._set_layer_opacity(self._glow_layer, 0.18 + (core_phase * 0.24))
            return

        if mode == "hiding":
            deadline = self._hide_deadline_monotonic
            if deadline is None:
                return
            start = deadline - _HIDE_FADE_SECONDS
            progress = 1.0 - ((deadline - now) / _HIDE_FADE_SECONDS)
            progress = max(0.0, min(1.0, progress))
            if now <= start:
                progress = 0.0
            self._root_layer.setOpacity_(1.0 - progress)

    def _start_recording_animation(self) -> None:
        self._clear_animations()
        self._prepare_show()
        self._set_mode("recording")
        self._glass_layer.setBackgroundColor_(self._color(0.19, 0.08, 0.12, 0.72))
        self._glass_layer.setBorderColor_(self._color(1.0, 0.74, 0.78, 0.45))
        self._glass_layer.setShadowOpacity_(0.32)
        self._glass_layer.setShadowRadius_(14.0)
        self._glass_layer.setShadowOffset_((0.0, 0.0))

        self._glow_layer.setHidden_(False)
        self._glow_layer.setBackgroundColor_(self._color(1.0, 0.32, 0.44, 0.16))
        self._glow_layer.setShadowColor_(self._color(1.0, 0.26, 0.42, 1.0))
        self._glow_layer.setShadowOpacity_(0.85)
        self._glow_layer.setShadowRadius_(10.0)
        self._glow_layer.setShadowOffset_((0.0, 0.0))

        self._ring_a_layer.setHidden_(True)
        self._ring_b_layer.setHidden_(True)

        self._pulse_layer.setHidden_(False)
        self._pulse_layer.setBorderWidth_(2.0)
        self._pulse_layer.setBorderColor_(self._color(1.0, 0.51, 0.61, 0.64))
        self._pulse_layer.setOpacity_(0.0)

        self._core_layer.setHidden_(False)
        self._core_layer.setBackgroundColor_(self._color(1.0, 0.28, 0.41, 0.95))
        self._core_layer.setShadowColor_(self._color(1.0, 0.35, 0.47, 1.0))
        self._core_layer.setShadowOpacity_(0.70)
        self._core_layer.setShadowRadius_(8.0)
        self._core_layer.setShadowOffset_((0.0, 0.0))

        self._add_animation(
            self._pulse_layer,
            self._make_animation(
                key_path="transform.scale",
                from_value=0.82,
                to_value=1.55,
                duration=1.05,
                timing="ease_out",
            ),
            "recording.pulse.scale",
        )
        self._add_animation(
            self._pulse_layer,
            self._make_animation(
                key_path="opacity",
                from_value=0.70,
                to_value=0.0,
                duration=1.05,
                timing="ease_out",
            ),
            "recording.pulse.opacity",
        )
        self._add_animation(
            self._core_layer,
            self._make_animation(
                key_path="transform.scale",
                from_value=1.0,
                to_value=1.12,
                duration=0.76,
                autoreverses=True,
            ),
            "recording.core.scale",
        )
        self._add_animation(
            self._glow_layer,
            self._make_animation(
                key_path="opacity",
                from_value=0.28,
                to_value=0.56,
                duration=0.76,
                autoreverses=True,
            ),
            "recording.glow.opacity",
        )
        self._tick_fallback_animation()

    def _start_processing_animation(self) -> None:
        self._clear_animations()
        self._prepare_show()
        self._set_mode("processing")
        self._glass_layer.setBackgroundColor_(self._color(0.07, 0.16, 0.20, 0.70))
        self._glass_layer.setBorderColor_(self._color(0.64, 0.98, 1.0, 0.42))
        self._glass_layer.setShadowOpacity_(0.30)
        self._glass_layer.setShadowRadius_(12.0)
        self._glass_layer.setShadowOffset_((0.0, 0.0))

        self._glow_layer.setHidden_(False)
        self._glow_layer.setBackgroundColor_(self._color(0.19, 0.86, 0.91, 0.14))
        self._glow_layer.setShadowColor_(self._color(0.30, 0.95, 1.0, 1.0))
        self._glow_layer.setShadowOpacity_(0.74)
        self._glow_layer.setShadowRadius_(9.0)
        self._glow_layer.setShadowOffset_((0.0, 0.0))

        self._ring_a_layer.setHidden_(False)
        self._ring_a_layer.setBorderWidth_(2.0)
        self._ring_a_layer.setBorderColor_(self._color(0.56, 0.98, 1.0, 0.80))
        self._ring_a_layer.setOpacity_(0.0)

        self._ring_b_layer.setHidden_(False)
        self._ring_b_layer.setBorderWidth_(2.0)
        self._ring_b_layer.setBorderColor_(self._color(0.53, 0.96, 1.0, 0.52))
        self._ring_b_layer.setOpacity_(0.0)

        self._pulse_layer.setHidden_(True)

        self._core_layer.setHidden_(False)
        self._core_layer.setBackgroundColor_(self._color(0.41, 0.96, 1.0, 0.92))
        self._core_layer.setShadowColor_(self._color(0.35, 0.97, 1.0, 1.0))
        self._core_layer.setShadowOpacity_(0.72)
        self._core_layer.setShadowRadius_(8.0)
        self._core_layer.setShadowOffset_((0.0, 0.0))

        for layer, delay, key_prefix in (
            (self._ring_a_layer, 0.0, "processing.ring.a"),
            (self._ring_b_layer, _RING_PHASE_OFFSET_SECONDS, "processing.ring.b"),
        ):
            self._add_animation(
                layer,
                self._make_animation(
                    key_path="transform.scale",
                    from_value=0.78,
                    to_value=1.22,
                    duration=1.25,
                    timing="ease_out",
                    delay=delay,
                ),
                f"{key_prefix}.scale",
            )
            self._add_animation(
                layer,
                self._make_animation(
                    key_path="opacity",
                    from_value=0.72,
                    to_value=0.05,
                    duration=1.25,
                    timing="ease_out",
                    delay=delay,
                ),
                f"{key_prefix}.opacity",
            )
            self._add_animation(
                layer,
                self._make_animation(
                    key_path="transform.rotation.z",
                    from_value=0.0,
                    to_value=math.tau,
                    duration=1.5 if layer is self._ring_a_layer else 1.9,
                    timing="linear",
                    delay=delay,
                ),
                f"{key_prefix}.rotation",
            )
        self._add_animation(
            self._core_layer,
            self._make_animation(
                key_path="transform.scale",
                from_value=0.96,
                to_value=1.08,
                duration=0.66,
                autoreverses=True,
            ),
            "processing.core.scale",
        )
        self._add_animation(
            self._glow_layer,
            self._make_animation(
                key_path="opacity",
                from_value=0.24,
                to_value=0.48,
                duration=0.66,
                autoreverses=True,
            ),
            "processing.glow.opacity",
        )
        self._tick_fallback_animation()

    def _position_bottom_right(self) -> None:
        screen = self._appkit.NSScreen.mainScreen()
        if screen is None:
            frame = ((0.0, 0.0), (self._panel_size, self._panel_size))
            self._panel.setFrame_display_(frame, True)
            return

        (origin_x, origin_y), (width, _height) = screen.visibleFrame()
        pos_x = float(origin_x + width - self._panel_size - self._config.margin_right)
        pos_y = float(origin_y + self._config.margin_bottom)
        frame = ((pos_x, pos_y), (self._panel_size, self._panel_size))
        self._panel.setFrame_display_(frame, True)

    def _flush_pending_hide_if_needed(self) -> None:
        deadline = self._hide_deadline_monotonic
        if deadline is None or time.monotonic() < deadline:
            return
        self._hide_deadline_monotonic = None
        self._apply_idle_visuals()
        self._root_layer.setOpacity_(1.0)
        self._panel.orderOut_(None)
        self._set_mode("hidden")
        self._is_visible = False

    def show_recording(self) -> None:
        self._start_recording_animation()

    def show_processing(self) -> None:
        self._start_processing_animation()

    def hide(self) -> None:
        self._clear_animations()
        if self._is_visible:
            self._set_mode("hiding")
            self._hide_deadline_monotonic = time.monotonic() + _HIDE_FADE_SECONDS
            if not self._animations_supported:
                return
            animation = self._make_animation(
                key_path="opacity",
                from_value=1.0,
                to_value=0.0,
                duration=_HIDE_FADE_SECONDS,
                timing="ease_out",
                repeat=False,
            )
            self._add_animation(self._root_layer, animation, "overlay.fade.out")
            return
        self._set_mode("hidden")
        self._hide_deadline_monotonic = None
        self._apply_idle_visuals()
        self._root_layer.setOpacity_(1.0)
        self._panel.orderOut_(None)
        self._is_visible = False

    def close(self) -> None:
        self._hide_deadline_monotonic = None
        self._set_mode("hidden")
        self._panel.close()

    def pump_events(self, timeout_seconds: float) -> None:
        timeout = max(0.0, float(timeout_seconds))
        mode = getattr(self._foundation, "NSDefaultRunLoopMode", "kCFRunLoopDefaultMode")
        mask = getattr(self._appkit, "NSEventMaskAny", (1 << 64) - 1)
        deadline = self._foundation.NSDate.dateWithTimeIntervalSinceNow_(timeout)
        event = self._app.nextEventMatchingMask_untilDate_inMode_dequeue_(
            mask,
            deadline,
            mode,
            True,
        )
        if event is not None:
            self._app.sendEvent_(event)
        self._app.updateWindows()
        self._tick_fallback_animation()
        self._flush_pending_hide_if_needed()


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

    reader = threading.Thread(
        target=_stdin_reader,
        args=(runtime,),
        daemon=True,
        name="activity-overlay-stdin",
    )
    reader.start()
    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())
