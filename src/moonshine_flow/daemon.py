"""Main daemon orchestration."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from moonshine_flow.activity_indicator import create_activity_indicator
from moonshine_flow.audio_recorder import AudioRecorder
from moonshine_flow.config import AppConfig
from moonshine_flow.domain.transcription_session import append_only_delta, has_sufficient_new_audio
from moonshine_flow.hotkey_monitor import HotkeyMonitor
from moonshine_flow.output_injector import OutputInjector
from moonshine_flow.ports.runtime import (
    ActivityIndicatorPort,
    AudioInputPort,
    SpeechToTextPort,
    TextOutputPort,
)
from moonshine_flow.stt.factory import create_stt_backend, parse_stt_model
from moonshine_flow.text_processing.interfaces import TextPostProcessor

LOGGER = logging.getLogger(__name__)
_HOTKEY_COOLDOWN_SECONDS = 0.25
_RECORDING_STALE_GRACE_SECONDS = 0.5
_RECORDING_STARTUP_GRACE_SECONDS = 0.35
_LIVE_INPUT_STOP_LOCK_TIMEOUT_SECONDS = 0.35
_LIVE_INPUT_MIN_NEW_AUDIO_SECONDS = 0.1
_MAIN_LOOP_IDLE_SLEEP_SECONDS = 0.2
_MAIN_LOOP_ACTIVE_SLEEP_SECONDS = 0.05


@dataclass(slots=True)
class _QueuedAudio:
    audio: np.ndarray
    emitted_prefix: str = ""


class MoonshineFlowDaemon:
    """Hold-to-record, release-to-transcribe daemon."""

    def __init__(
        self,
        config: AppConfig,
        post_processor: TextPostProcessor | None = None,
        *,
        enable_streaming: bool = True,
        activity_indicator: ActivityIndicatorPort | None = None,
    ) -> None:
        self.config = config
        self._enable_streaming = enable_streaming
        self._stop_event = threading.Event()
        self._audio_queue: queue.Queue[_QueuedAudio] = queue.Queue()
        self._state_lock = threading.Lock()
        self._live_input_lock = threading.Lock()
        self._transcription_in_progress = False
        self._last_release_at_monotonic = 0.0
        self._recording_stale_since_monotonic: float | None = None
        self._recording_started_at_monotonic: float | None = None
        self._pending_stop_timer: threading.Timer | None = None
        self._pending_stop_id: int | None = None
        self._next_stop_id = 0
        self._live_emitted_text = ""
        self._live_last_snapshot_samples = 0
        self._live_stop_requested = False
        self._hotkey_not_pressed_since_monotonic: float | None = None

        self.recorder: AudioInputPort = AudioRecorder(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            dtype=config.audio.dtype,
            max_record_seconds=config.audio.max_record_seconds,
            input_device=config.audio.input_device,
            input_device_policy=str(config.audio.input_device_policy),
        )
        self.transcriber: SpeechToTextPort = create_stt_backend(
            config,
            post_processor=post_processor,
        )
        supports_realtime_input = getattr(self.transcriber, "supports_realtime_input", None)
        self._supports_realtime_input = bool(
            callable(supports_realtime_input) and supports_realtime_input()
        )
        self.injector: TextOutputPort = OutputInjector(
            mode=config.output.mode.value,
            paste_shortcut=config.output.paste_shortcut,
        )
        self.activity_indicator: ActivityIndicatorPort = (
            activity_indicator
            if activity_indicator is not None
            else create_activity_indicator(config)
        )
        self.hotkey = HotkeyMonitor(
            key_name=config.hotkey.key,
            on_press=self._on_hotkey_down,
            on_release=self._on_hotkey_up,
            max_hold_seconds=float(config.audio.max_record_seconds) + 1.0,
        )

        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="moonshine-worker",
        )

    def _show_recording_indicator(self) -> None:
        try:
            self.activity_indicator.show_recording()
        except Exception:
            LOGGER.debug("Failed to show recording activity indicator", exc_info=True)

    def _show_processing_indicator(self) -> None:
        try:
            self.activity_indicator.show_processing()
        except Exception:
            LOGGER.debug("Failed to show processing activity indicator", exc_info=True)

    def _hide_activity_indicator(self) -> None:
        try:
            self.activity_indicator.hide()
        except Exception:
            LOGGER.debug("Failed to hide activity indicator", exc_info=True)

    def _close_activity_indicator(self) -> None:
        try:
            self.activity_indicator.close()
        except Exception:
            LOGGER.debug("Failed to close activity indicator", exc_info=True)

    def _on_hotkey_down(self) -> None:
        if self._stop_event.is_set():
            return
        with self._state_lock:
            canceled = self._cancel_pending_stop_locked()
            if canceled:
                LOGGER.debug("Canceled delayed stop because hotkey was pressed again")
            if self._transcription_in_progress:
                LOGGER.info("Ignored hotkey press while transcription is in progress")
                return
            cooldown_until = self._last_release_at_monotonic + _HOTKEY_COOLDOWN_SECONDS
            self._hotkey_not_pressed_since_monotonic = None
        if self.recorder.is_recording:
            return
        if time.monotonic() < cooldown_until:
            LOGGER.debug("Ignored hotkey press during cooldown window")
            return

        try:
            self.recorder.start()
            now = time.monotonic()
            if self._live_input_enabled():
                with self._state_lock:
                    self._reset_live_state_locked()
            with self._state_lock:
                self._recording_started_at_monotonic = now
                self._recording_stale_since_monotonic = None
            self._show_recording_indicator()
            LOGGER.info("Recording started")
        except Exception:
            LOGGER.exception("Failed to start recording")

    def _on_hotkey_up(self) -> None:
        if self._stop_event.is_set() or not self.recorder.is_recording:
            return
        with self._state_lock:
            self._hotkey_not_pressed_since_monotonic = None

        release_tail_seconds = self._effective_release_tail_seconds()
        if release_tail_seconds <= 0.0:
            self._stop_recording_and_queue_audio(reason="hotkey-release")
            return

        with self._state_lock:
            self._cancel_pending_stop_locked()
            self._next_stop_id += 1
            stop_id = self._next_stop_id
            timer = threading.Timer(
                release_tail_seconds,
                self._on_delayed_stop_timer,
                args=(stop_id,),
            )
            timer.daemon = True
            self._pending_stop_id = stop_id
            self._pending_stop_timer = timer

        LOGGER.debug(
            "Scheduled delayed stop %.3fs after release (stop_id=%s)",
            release_tail_seconds,
            stop_id,
        )
        timer.start()

    def _effective_release_tail_seconds(self) -> float:
        configured = float(self.config.audio.release_tail_seconds)
        try:
            model_token = str(getattr(getattr(self.config, "stt", None), "model", ""))
            prefix, _model_id = parse_stt_model(model_token)
        except Exception:
            return configured

        # Keep explicit user overrides, but default realtime STT to zero extra tail.
        if prefix in {"voxtral", "vllm"} and abs(configured - 0.25) < 1e-9:
            return 0.0
        return configured

    def _cancel_pending_stop_locked(self) -> bool:
        timer = self._pending_stop_timer
        self._pending_stop_timer = None
        self._pending_stop_id = None
        if timer is None:
            return False
        timer.cancel()
        return True

    def _on_delayed_stop_timer(self, stop_id: int) -> None:
        with self._state_lock:
            if self._pending_stop_id != stop_id:
                return
            self._pending_stop_id = None
            self._pending_stop_timer = None
        self._stop_recording_and_queue_audio(reason="delayed-hotkey-release")

    def _live_input_enabled(self) -> bool:
        return self._enable_streaming and self._supports_realtime_input

    def _reset_live_state_locked(self) -> None:
        self._live_emitted_text = ""
        self._live_last_snapshot_samples = 0
        self._live_stop_requested = False

    def _stop_recording_and_queue_audio(self, *, reason: str) -> None:
        if self._stop_event.is_set() or not self.recorder.is_recording:
            return

        emitted_prefix = ""
        with self._state_lock:
            if self._live_input_enabled():
                self._live_stop_requested = True

        lock_acquired = self._live_input_lock.acquire(timeout=_LIVE_INPUT_STOP_LOCK_TIMEOUT_SECONDS)
        if not lock_acquired:
            LOGGER.warning(
                "Live input lock was busy during stop; stopping without live lock (%s)",
                reason,
            )
        try:
            audio = self.recorder.stop()
        except Exception:
            LOGGER.exception("Failed to stop recording (%s)", reason)
            try:
                self.recorder.close()
            except Exception:
                LOGGER.debug(
                    "Failed to force-close recorder after stop failure (%s)",
                    reason,
                    exc_info=True,
                )
            self._hide_activity_indicator()
            return
        finally:
            with self._state_lock:
                self._last_release_at_monotonic = time.monotonic()
                self._recording_stale_since_monotonic = None
                self._recording_started_at_monotonic = None
                self._hotkey_not_pressed_since_monotonic = None
                if self._live_input_enabled():
                    emitted_prefix = self._live_emitted_text
                    self._reset_live_state_locked()
            if lock_acquired:
                self._live_input_lock.release()

        if audio.size == 0:
            LOGGER.info("Skipped empty audio capture (%s)", reason)
            self._hide_activity_indicator()
            return

        if emitted_prefix:
            LOGGER.debug(
                "Captured live emitted prefix before queueing final transcription (%s chars)",
                len(emitted_prefix),
            )
        self._audio_queue.put(_QueuedAudio(audio=audio, emitted_prefix=emitted_prefix))
        self._show_processing_indicator()
        LOGGER.info("Queued audio for transcription (%s)", reason)

    def _process_live_input_tick(self) -> None:
        if not self._live_input_enabled():
            return
        if self._stop_event.is_set() or not self.recorder.is_recording:
            return
        if not self._live_input_lock.acquire(blocking=False):
            return

        try:
            if self._stop_event.is_set() or not self.recorder.is_recording:
                return
            with self._state_lock:
                if self._transcription_in_progress or self._live_stop_requested:
                    return
                emitted = self._live_emitted_text
                last_snapshot_samples = self._live_last_snapshot_samples

            audio = self.recorder.snapshot()
            total_samples = int(audio.shape[0]) if audio.ndim else 0
            if total_samples <= 0:
                return
            if not has_sufficient_new_audio(
                total_samples=total_samples,
                last_snapshot_samples=last_snapshot_samples,
                sample_rate=self.config.audio.sample_rate,
                min_new_audio_seconds=_LIVE_INPUT_MIN_NEW_AUDIO_SECONDS,
            ):
                return

            for update in self.transcriber.transcribe_stream(audio, self.config.audio.sample_rate):
                with self._state_lock:
                    if (
                        self._live_stop_requested
                        or self._stop_event.is_set()
                        or not self.recorder.is_recording
                    ):
                        break
                delta = self._append_only_delta(emitted, update)
                if not delta:
                    continue
                self.injector.inject(delta)
                emitted += delta
                with self._state_lock:
                    self._live_emitted_text = emitted
            with self._state_lock:
                if not self._live_stop_requested:
                    self._live_emitted_text = emitted
                    self._live_last_snapshot_samples = total_samples
        except Exception:
            LOGGER.exception("Realtime input tick failed")
        finally:
            self._live_input_lock.release()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            with self._state_lock:
                self._transcription_in_progress = True
            self._show_processing_indicator()
            try:
                # Serialize transcriber access with live input tick
                # to avoid concurrent GPU execution.
                with self._live_input_lock:
                    audio = item.audio
                    if self._enable_streaming:
                        # Live input already emitted partial text while key was held.
                        # Use one final transcript after release
                        # to avoid re-emitting duplicated chunks.
                        if item.emitted_prefix:
                            text = self.transcriber.transcribe(audio, self.config.audio.sample_rate)
                            if text:
                                delta = self._append_only_delta(item.emitted_prefix, text)
                                if delta:
                                    self.injector.inject(delta)
                                else:
                                    LOGGER.info("Transcription result was empty")
                            else:
                                LOGGER.info("Transcription result was empty")
                        else:
                            emitted = item.emitted_prefix
                            last_injected_delta = ""
                            for update in self.transcriber.transcribe_stream(
                                audio, self.config.audio.sample_rate
                            ):
                                delta = self._append_only_delta(emitted, update)
                                if not delta:
                                    continue
                                if delta == last_injected_delta:
                                    LOGGER.debug("Skipping duplicate streaming delta")
                                    continue
                                self.injector.inject(delta)
                                emitted += delta
                                last_injected_delta = delta
                            if not emitted.strip():
                                LOGGER.info("Transcription result was empty")
                    else:
                        text = self.transcriber.transcribe(audio, self.config.audio.sample_rate)
                        if text:
                            delta = self._append_only_delta(item.emitted_prefix, text)
                            if delta:
                                self.injector.inject(delta)
                            else:
                                LOGGER.info("Transcription result was empty")
                        else:
                            LOGGER.info("Transcription result was empty")
            except Exception:
                LOGGER.exception("Transcription pipeline failed")
            finally:
                with self._state_lock:
                    self._transcription_in_progress = False
                self._audio_queue.task_done()
                if self._audio_queue.empty() and not self.recorder.is_recording:
                    self._hide_activity_indicator()

    @staticmethod
    def _append_only_delta(previous: str, current: str) -> str:
        delta = append_only_delta(previous, current)
        if not delta and current != previous:
            if not current.startswith(previous):
                common_prefix_len = 0
                max_common = min(len(previous), len(current))
                while (
                    common_prefix_len < max_common
                    and previous[common_prefix_len] == current[common_prefix_len]
                ):
                    common_prefix_len += 1
                if len(previous) - common_prefix_len <= 4:
                    LOGGER.debug("Applying tolerant delta for non-monotonic tail rewrite")
                else:
                    LOGGER.debug("Skipping non-monotonic streaming update")
        return delta

    def _effective_hotkey_release_reconcile_seconds(self) -> float:
        return max(0.0, float(getattr(self.config.audio, "hotkey_release_reconcile_seconds", 0.25)))

    def _recover_missed_hotkey_release_if_needed(self) -> None:
        if not self.recorder.is_recording:
            with self._state_lock:
                self._hotkey_not_pressed_since_monotonic = None
            return

        now = time.monotonic()
        is_pressed = getattr(self.hotkey, "is_pressed", None)
        if not callable(is_pressed):
            return
        try:
            hotkey_pressed = bool(is_pressed())
        except Exception:
            LOGGER.debug("Failed to read hotkey pressed state", exc_info=True)
            return

        if hotkey_pressed:
            with self._state_lock:
                self._hotkey_not_pressed_since_monotonic = None
            return

        with self._state_lock:
            if self._pending_stop_id is not None:
                self._hotkey_not_pressed_since_monotonic = None
                return
            stale_since = self._hotkey_not_pressed_since_monotonic
            if stale_since is None:
                self._hotkey_not_pressed_since_monotonic = now
                return
            stale_for = now - stale_since
        reconcile_seconds = self._effective_hotkey_release_reconcile_seconds()
        if reconcile_seconds > 0.0 and stale_for < reconcile_seconds:
            return

        LOGGER.warning(
            "Hotkey release callback appears missed; reconciling after %.2fs",
            stale_for,
        )
        self._stop_recording_and_queue_audio(reason="hotkey-release-reconciled")
        with self._state_lock:
            self._hotkey_not_pressed_since_monotonic = None

    def _recover_stale_recording_if_needed(self) -> None:
        if not self.recorder.is_recording:
            with self._state_lock:
                self._recording_stale_since_monotonic = None
                self._recording_started_at_monotonic = None
            return
        if self.recorder.is_stream_active():
            with self._state_lock:
                self._recording_stale_since_monotonic = None
            return

        now = time.monotonic()
        snapshot_samples = 0
        try:
            snapshot = self.recorder.snapshot()
            snapshot_samples = int(snapshot.shape[0]) if snapshot.ndim else 0
        except Exception:
            LOGGER.debug(
                "Failed to inspect recorder snapshot while stream looked inactive",
                exc_info=True,
            )
        with self._state_lock:
            if self._recording_started_at_monotonic is None:
                self._recording_started_at_monotonic = now
            recording_for = now - self._recording_started_at_monotonic
            if recording_for < _RECORDING_STARTUP_GRACE_SECONDS:
                self._recording_stale_since_monotonic = None
                LOGGER.debug(
                    "Audio stream reported inactive during startup grace (%.2fs/%.2fs, samples=%s)",
                    recording_for,
                    _RECORDING_STARTUP_GRACE_SECONDS,
                    snapshot_samples,
                )
                return
            stale_since = self._recording_stale_since_monotonic
            if stale_since is None:
                self._recording_stale_since_monotonic = now
                LOGGER.debug(
                    "Detected inactive audio stream while recording; waiting for recovery grace "
                    "(samples=%s)",
                    snapshot_samples,
                )
                return
            stale_for = now - stale_since
        if stale_for < _RECORDING_STALE_GRACE_SECONDS:
            return

        LOGGER.warning(
            "Recovering recorder after %.2fs inactive stream while recording "
            "(recording_for=%.2fs, samples=%s)",
            stale_for,
            recording_for,
            snapshot_samples,
        )
        try:
            with self._state_lock:
                self._cancel_pending_stop_locked()
            self.recorder.close()
        except Exception:
            LOGGER.exception("Failed to recover recorder from stale recording state")
        finally:
            with self._state_lock:
                self._recording_stale_since_monotonic = None
                self._recording_started_at_monotonic = None
                self._last_release_at_monotonic = now

    def _release_idle_transcriber_resources_if_needed(self) -> None:
        if self.recorder.is_recording:
            return
        with self._state_lock:
            if self._transcription_in_progress:
                return
        if not self._audio_queue.empty():
            return
        maybe_release = getattr(self.transcriber, "maybe_release_idle_resources", None)
        if not callable(maybe_release):
            return
        try:
            maybe_release()
        except Exception:
            LOGGER.debug("Failed to release idle STT resources cleanly", exc_info=True)

    def _transcriber_uses_external_server(self) -> bool:
        summary_fn = getattr(self.transcriber, "backend_summary", None)
        if not callable(summary_fn):
            return False
        try:
            summary = str(summary_fn())
        except Exception:
            return False
        return "backend=vllm-realtime" in summary

    def run_forever(self) -> None:
        """Run daemon until stop() is called."""
        LOGGER.info("Moonshine Flow daemon starting (%s)", self.transcriber.backend_summary())
        if not self._transcriber_uses_external_server():
            LOGGER.info(
                "🚀 Backend session active (no external server): %s",
                self.transcriber.backend_summary(),
            )
        self._worker.start()
        self.hotkey.start()

        try:
            while not self._stop_event.is_set():
                self._process_live_input_tick()
                self._recover_missed_hotkey_release_if_needed()
                self._recover_stale_recording_if_needed()
                self._release_idle_transcriber_resources_if_needed()
                if self._live_input_enabled() and self.recorder.is_recording:
                    time.sleep(_MAIN_LOOP_ACTIVE_SLEEP_SECONDS)
                else:
                    time.sleep(_MAIN_LOOP_IDLE_SLEEP_SECONDS)
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop daemon components."""
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        self._hide_activity_indicator()
        if not self._transcriber_uses_external_server():
            LOGGER.info(
                "💨 Backend session stopped (no external server): %s",
                self.transcriber.backend_summary(),
            )
        with self._live_input_lock:
            with self._state_lock:
                self._cancel_pending_stop_locked()
                self._reset_live_state_locked()
                self._hotkey_not_pressed_since_monotonic = None
        try:
            self.hotkey.stop()
        except Exception:
            LOGGER.debug("Failed to stop hotkey listener cleanly", exc_info=True)

        try:
            self.recorder.close()
        except Exception:
            LOGGER.debug("Failed to close recorder cleanly", exc_info=True)

        close = getattr(self.transcriber, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                LOGGER.debug("Failed to close transcriber cleanly", exc_info=True)
        self._close_activity_indicator()

        LOGGER.info("Moonshine Flow daemon stopped")
