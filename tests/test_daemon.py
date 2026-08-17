from __future__ import annotations

import io
import logging
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

sys.modules.setdefault(
    "sounddevice",
    SimpleNamespace(
        InputStream=object,
        default=SimpleNamespace(device=(None, None)),
        query_devices=lambda: [],
    ),
)
sys.modules.setdefault(
    "pynput",
    SimpleNamespace(
        keyboard=SimpleNamespace(
            Key=SimpleNamespace(
                cmd_r="cmd_r",
                cmd="cmd",
                shift_r="shift_r",
                shift="shift",
                alt_r="alt_r",
                alt_l="alt_l",
                ctrl_r="ctrl_r",
                ctrl_l="ctrl_l",
            ),
            KeyCode=SimpleNamespace(from_char=lambda char: char),
            Listener=lambda **_kwargs: SimpleNamespace(
                start=lambda: None,
                stop=lambda: None,
                join=lambda: None,
            ),
        )
    ),
)

import ptarmigan_flow.daemon as daemon_module  # noqa: E402
from ptarmigan_flow.config import AppConfig  # noqa: E402
from ptarmigan_flow.ports.runtime import BackendWarmState  # noqa: E402
from ptarmigan_flow.stt.runtime_backend import (  # noqa: E402
    SpeechToTextRequestTimeoutError,
    STTRecoverySummary,
)


class _FakeRecorder:
    last_input_device_policy: str | None = None

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        dtype: str,
        max_record_seconds: int,
        input_device: str | int | None = None,
        input_device_policy: str = "playback_friendly",
    ) -> None:
        del sample_rate, channels, dtype, max_record_seconds, input_device
        _FakeRecorder.last_input_device_policy = input_device_policy
        self.is_recording = False
        self.stream_active = True
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0
        self.snapshot_calls = 0
        self.snapshot_audio = np.array([[0.1], [0.2]], dtype=np.float32)

    def start(self) -> None:
        self.start_calls += 1
        self.is_recording = True
        self.stream_active = True

    def stop(self) -> np.ndarray:
        self.stop_calls += 1
        self.is_recording = False
        self.stream_active = False
        return np.array([[0.25], [0.5]], dtype=np.float32)

    def close(self) -> None:
        self.close_calls += 1
        self.is_recording = False
        self.stream_active = False

    def is_stream_active(self) -> bool:
        return self.stream_active

    def snapshot(self) -> np.ndarray:
        self.snapshot_calls += 1
        return self.snapshot_audio.copy()


class _FakeTranscriber:
    def __init__(self, **kwargs) -> None:
        del kwargs
        self.calls: list[np.ndarray] = []
        self.idle_release_calls = 0
        self.warmup_calls = 0
        self.warmup_started = threading.Event()
        self.warmup_unblock = threading.Event()
        self.warmup_unblock.set()
        self._warm_state = BackendWarmState(
            resource_mode="in_process",
            ready=True,
            warmed=True,
            warmup_running=False,
            supports_keydown_warmup=False,
            last_activity_at_monotonic=time.monotonic(),
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        del sample_rate
        self.calls.append(audio)
        return "hello"

    def transcribe_stream(self, audio: np.ndarray, sample_rate: int):
        yield self.transcribe(audio, sample_rate)

    def supports_realtime_input(self) -> bool:
        return False

    def warm_state(self) -> BackendWarmState:
        return self._warm_state

    def warmup_for_low_latency(self) -> None:
        self.warmup_calls += 1
        self.warmup_started.set()
        self.warmup_unblock.wait(timeout=1.0)
        self._warm_state = BackendWarmState(
            resource_mode=self._warm_state.resource_mode,
            ready=True,
            warmed=True,
            warmup_running=False,
            supports_keydown_warmup=self._warm_state.supports_keydown_warmup,
            last_activity_at_monotonic=time.monotonic(),
        )

    def backend_summary(self) -> str:
        return "fake-backend"

    def maybe_release_idle_resources(self) -> None:
        self.idle_release_calls += 1

    def close(self) -> None:
        return None


class _FakeInjector:
    def __init__(self, **kwargs) -> None:
        del kwargs
        self.injected: list[str] = []

    def inject(self, text: str) -> None:
        self.injected.append(text)


class _FakeHotkeyMonitor:
    def __init__(
        self,
        key_name: str,
        on_press,
        on_release,
        *,
        max_hold_seconds: float | None = None,
        use_pynput_listener: bool = True,
    ) -> None:
        del key_name, max_hold_seconds
        self.on_press = on_press
        self.on_release = on_release
        self.use_pynput_listener = use_pynput_listener
        self.started = False
        self.pressed = False
        self.physical_state: bool | None = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def join(self) -> None:
        return None

    def is_pressed(self) -> bool:
        return self.pressed

    def notify_press(self) -> None:
        if self.pressed:
            return
        self.pressed = True
        self.on_press()

    def notify_release(self) -> bool:
        if not self.pressed:
            return False
        self.pressed = False
        self.on_release()
        return True

    def physical_pressed_state(self) -> bool | None:
        return self.physical_state


class _FakeActivityIndicator:
    def __init__(self) -> None:
        self.show_recording_calls = 0
        self.show_processing_calls = 0
        self.hide_calls = 0
        self.close_calls = 0

    def show_recording(self) -> None:
        self.show_recording_calls += 1

    def show_processing(self) -> None:
        self.show_processing_calls += 1

    def hide(self) -> None:
        self.hide_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _FakeTimer:
    instances: list[_FakeTimer] = []

    def __init__(self, interval, function, args=None, kwargs=None):
        self.interval = interval
        self.function = function
        self.args = tuple(args or ())
        self.kwargs = dict(kwargs or {})
        self.daemon = False
        self.started = False
        self.canceled = False
        _FakeTimer.instances.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.canceled = True

    def fire(self) -> None:
        if self.canceled:
            return
        self.function(*self.args, **self.kwargs)


def _reset_fake_timer() -> None:
    _FakeTimer.instances = []


def _build_daemon(
    monkeypatch,
    *,
    config: AppConfig | None = None,
    use_pynput_listener: bool | None = None,
) -> daemon_module.PtarmiganFlowDaemon:
    monkeypatch.setattr(daemon_module, "AudioRecorder", _FakeRecorder)
    monkeypatch.setattr(
        daemon_module,
        "create_runtime_stt_backend",
        lambda *_args, **_kwargs: _FakeTranscriber(),
    )
    monkeypatch.setattr(daemon_module, "OutputInjector", _FakeInjector)
    monkeypatch.setattr(daemon_module, "HotkeyMonitor", _FakeHotkeyMonitor)
    monkeypatch.setattr(
        daemon_module,
        "create_activity_indicator",
        lambda *_args, **_kwargs: _FakeActivityIndicator(),
    )
    kwargs = (
        {"use_pynput_listener": use_pynput_listener}
        if use_pynput_listener is not None
        else {}
    )
    return daemon_module.PtarmiganFlowDaemon(config or AppConfig(), **kwargs)


def _timeout_error(*, request_kind: str = "transcribe") -> SpeechToTextRequestTimeoutError:
    return SpeechToTextRequestTimeoutError(
        STTRecoverySummary(
            failure_kind="timeout",
            request_kind=request_kind,
            request_id=1,
            generation=1,
            backend_summary="backend=fake-backend",
            audio_seconds=1.0,
            timeout_seconds=0.1,
            started_at_monotonic=1.0,
            ended_at_monotonic=1.1,
            warm_state=None,
            restart_succeeded=True,
        )
    )


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_hotkey_down_ignored_while_transcription_busy(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    with daemon._state_lock:
        daemon._transcription_in_progress = True

    daemon._on_hotkey_down()

    assert daemon.recorder.start_calls == 0


def test_daemon_passes_explicit_pynput_listener_setting_to_hotkey_monitor(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch, use_pynput_listener=False)

    assert daemon.hotkey.use_pynput_listener is False


def test_hotkey_down_respects_cooldown(monkeypatch) -> None:
    config = AppConfig()
    config.audio.release_tail_seconds = 0.0
    daemon = _build_daemon(monkeypatch, config=config)
    monotonic_values = iter([10.0, 10.1, 10.4])
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: next(monotonic_values))

    daemon.recorder.is_recording = True
    daemon._on_hotkey_up()
    assert daemon.recorder.stop_calls == 1

    daemon._on_hotkey_down()
    assert daemon.recorder.start_calls == 0

    daemon._on_hotkey_down()
    assert daemon.recorder.start_calls == 1


def test_hotkey_down_shows_recording_indicator(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)

    daemon._on_hotkey_down()

    assert daemon.recorder.start_calls == 1
    assert daemon.activity_indicator.show_recording_calls == 1


def test_hotkey_down_starts_keydown_warmup_for_cold_backend(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon.transcriber._warm_state = BackendWarmState(
        resource_mode="in_process",
        ready=True,
        warmed=False,
        warmup_running=False,
        supports_keydown_warmup=True,
        last_activity_at_monotonic=None,
    )

    daemon._on_hotkey_down()

    assert daemon.transcriber.warmup_started.wait(timeout=1.0) is True
    thread = daemon._keydown_warmup_thread
    if thread is not None:
        thread.join(timeout=1.0)
    assert daemon.transcriber.warmup_calls == 1


def test_keydown_warmup_skips_when_backend_is_recently_warm(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon.recorder.is_recording = True
    daemon.transcriber._warm_state = BackendWarmState(
        resource_mode="in_process",
        ready=True,
        warmed=True,
        warmup_running=False,
        supports_keydown_warmup=True,
        last_activity_at_monotonic=100.0,
    )
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: 150.0)

    daemon._maybe_start_keydown_warmup()

    assert daemon.transcriber.warmup_calls == 0
    assert daemon._keydown_warmup_thread is None


def test_keydown_warmup_does_not_create_duplicate_workers(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon.recorder.is_recording = True
    daemon.transcriber._warm_state = BackendWarmState(
        resource_mode="in_process",
        ready=True,
        warmed=False,
        warmup_running=False,
        supports_keydown_warmup=True,
        last_activity_at_monotonic=None,
    )
    daemon.transcriber.warmup_unblock.clear()

    daemon._maybe_start_keydown_warmup()
    assert daemon.transcriber.warmup_started.wait(timeout=1.0) is True

    daemon._maybe_start_keydown_warmup()

    thread = daemon._keydown_warmup_thread
    assert thread is not None
    daemon.transcriber.warmup_unblock.set()
    thread.join(timeout=1.0)
    assert daemon.transcriber.warmup_calls == 1


def test_keydown_warmup_aborts_if_recording_stops_before_lock(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon.recorder.is_recording = True
    daemon.transcriber._warm_state = BackendWarmState(
        resource_mode="in_process",
        ready=True,
        warmed=False,
        warmup_running=False,
        supports_keydown_warmup=True,
        last_activity_at_monotonic=None,
    )
    daemon._live_input_lock.acquire()

    daemon._maybe_start_keydown_warmup()
    thread = daemon._keydown_warmup_thread
    assert thread is not None

    daemon.recorder.is_recording = False
    daemon._live_input_lock.release()
    thread.join(timeout=1.0)

    assert daemon.transcriber.warmup_calls == 0


def test_hotkey_up_schedules_delayed_stop(monkeypatch) -> None:
    _reset_fake_timer()
    monkeypatch.setattr(daemon_module.threading, "Timer", _FakeTimer)
    daemon = _build_daemon(monkeypatch)
    daemon.recorder.is_recording = True

    daemon._on_hotkey_up()

    assert len(_FakeTimer.instances) == 1
    timer = _FakeTimer.instances[0]
    assert timer.started is True
    assert timer.interval == 0.25
    assert daemon.recorder.stop_calls == 0

    timer.fire()

    assert daemon.recorder.stop_calls == 1
    assert daemon._audio_queue.qsize() == 1


def test_hotkey_up_uses_short_tail_floor_for_default_realtime_model(monkeypatch) -> None:
    _reset_fake_timer()
    monkeypatch.setattr(daemon_module.threading, "Timer", _FakeTimer)
    config = AppConfig()
    config.stt.model = "voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602"
    daemon = _build_daemon(monkeypatch, config=config)
    daemon.recorder.is_recording = True

    daemon._on_hotkey_up()

    assert len(_FakeTimer.instances) == 1
    timer = _FakeTimer.instances[0]
    assert timer.started is True
    assert timer.interval == 0.15
    assert daemon.recorder.stop_calls == 0

    timer.fire()

    assert daemon.recorder.stop_calls == 1
    assert daemon._audio_queue.qsize() == 1


def test_effective_release_tail_keeps_explicit_override_for_realtime(monkeypatch) -> None:
    config = AppConfig()
    config.stt.model = "voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602"
    config.audio.release_tail_seconds = 0.1
    daemon = _build_daemon(monkeypatch, config=config)

    assert daemon._effective_release_tail_seconds() == 0.1


def test_effective_release_tail_keeps_default_for_granite(monkeypatch) -> None:
    config = AppConfig()
    config.stt.model = "granite:ibm-granite/granite-4.0-1b-speech"
    daemon = _build_daemon(monkeypatch, config=config)

    assert daemon._effective_release_tail_seconds() == 0.25


def test_hotkey_down_cancels_pending_delayed_stop(monkeypatch) -> None:
    _reset_fake_timer()
    monkeypatch.setattr(daemon_module.threading, "Timer", _FakeTimer)
    daemon = _build_daemon(monkeypatch)
    daemon.recorder.is_recording = True

    daemon._on_hotkey_up()
    assert len(_FakeTimer.instances) == 1
    timer = _FakeTimer.instances[0]

    daemon._on_hotkey_down()

    assert timer.canceled is True
    assert daemon.recorder.stop_calls == 0


def test_stop_cancels_pending_delayed_stop(monkeypatch) -> None:
    _reset_fake_timer()
    monkeypatch.setattr(daemon_module.threading, "Timer", _FakeTimer)
    daemon = _build_daemon(monkeypatch)
    daemon.recorder.is_recording = True

    daemon._on_hotkey_up()
    timer = _FakeTimer.instances[0]

    daemon.stop()

    assert timer.canceled is True
    assert daemon.recorder.close_calls == 1
    assert daemon.activity_indicator.hide_calls >= 1
    assert daemon.activity_indicator.close_calls == 1


def test_recover_stale_recording_closes_recorder(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    monotonic_values = iter([1.0, 1.2, 1.8, 2.4])
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: next(monotonic_values))

    daemon.recorder.is_recording = True
    daemon.recorder.stream_active = False

    daemon._recover_stale_recording_if_needed()
    daemon._recover_stale_recording_if_needed()
    daemon._recover_stale_recording_if_needed()
    assert daemon.recorder.close_calls == 0

    daemon._recover_stale_recording_if_needed()
    assert daemon.recorder.close_calls == 1
    assert daemon.recorder.is_recording is False


def test_live_input_tick_injects_delta_for_realtime_backend(monkeypatch) -> None:
    config = AppConfig()
    config.audio.sample_rate = 10
    daemon = _build_daemon(monkeypatch, config=config)
    daemon._supports_realtime_input = True
    daemon.recorder.is_recording = True
    daemon.recorder.snapshot_audio = np.arange(4, dtype=np.float32).reshape(-1, 1)

    daemon.transcriber.transcribe_stream = lambda *_args, **_kwargs: iter(["he"])

    daemon._process_live_input_tick()

    assert daemon.recorder.snapshot_calls == 1
    assert daemon.injector.injected == ["he"]
    assert daemon._live_emitted_text == "he"
    assert daemon._live_last_snapshot_samples == 4


def test_live_input_tick_skips_when_stop_requested(monkeypatch) -> None:
    config = AppConfig()
    config.audio.sample_rate = 10
    daemon = _build_daemon(monkeypatch, config=config)
    daemon._supports_realtime_input = True
    daemon.recorder.is_recording = True
    daemon.recorder.snapshot_audio = np.arange(4, dtype=np.float32).reshape(-1, 1)
    with daemon._state_lock:
        daemon._live_stop_requested = True

    daemon.transcriber.transcribe_stream = lambda *_args, **_kwargs: iter(["he"])

    daemon._process_live_input_tick()

    assert daemon.recorder.snapshot_calls == 0
    assert daemon.injector.injected == []
    assert daemon._live_emitted_text == ""


def test_stop_recording_queues_emitted_prefix_in_live_input_mode(monkeypatch) -> None:
    config = AppConfig()
    config.audio.release_tail_seconds = 0.0
    daemon = _build_daemon(monkeypatch, config=config)
    daemon._supports_realtime_input = True
    daemon.recorder.is_recording = True
    with daemon._state_lock:
        daemon._live_emitted_text = "he"
        daemon._live_last_snapshot_samples = 10

    daemon._stop_recording_and_queue_audio(reason="hotkey-release")

    item = daemon._audio_queue.get_nowait()
    assert item.emitted_prefix == "he"
    assert item.audio.shape == (2, 1)
    assert daemon._live_emitted_text == ""
    assert daemon._live_last_snapshot_samples == 0


def test_stop_recording_queued_audio_shows_processing_indicator(monkeypatch) -> None:
    config = AppConfig()
    config.audio.release_tail_seconds = 0.0
    daemon = _build_daemon(monkeypatch, config=config)
    daemon.recorder.is_recording = True

    daemon._stop_recording_and_queue_audio(reason="hotkey-release")

    assert daemon._audio_queue.qsize() == 1
    assert daemon.activity_indicator.show_processing_calls == 1


def test_worker_streaming_skips_already_emitted_prefix(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon.transcriber.transcribe_stream = lambda *_args, **_kwargs: iter(["hello"])
    daemon._audio_queue.put(
        SimpleNamespace(
            audio=np.array([[0.25], [0.5]], dtype=np.float32),
            emitted_prefix="he",
        )
    )

    worker = threading.Thread(target=daemon._worker_loop, daemon=True)
    worker.start()
    daemon._audio_queue.join()
    daemon._stop_event.set()
    worker.join(timeout=1.0)

    assert daemon.injector.injected == ["llo"]
    assert daemon.activity_indicator.hide_calls == 1


def test_worker_runs_final_transcription_after_keydown_warmup(monkeypatch) -> None:
    config = AppConfig()
    config.audio.release_tail_seconds = 0.0
    daemon = _build_daemon(monkeypatch, config=config)
    daemon.recorder.is_recording = True
    daemon.transcriber._warm_state = BackendWarmState(
        resource_mode="in_process",
        ready=True,
        warmed=False,
        warmup_running=False,
        supports_keydown_warmup=True,
        last_activity_at_monotonic=None,
    )

    daemon._maybe_start_keydown_warmup()
    assert daemon.transcriber.warmup_started.wait(timeout=1.0) is True
    warmup_thread = daemon._keydown_warmup_thread
    if warmup_thread is not None:
        warmup_thread.join(timeout=1.0)

    daemon._stop_recording_and_queue_audio(reason="hotkey-release")

    worker = threading.Thread(target=daemon._worker_loop, daemon=True)
    worker.start()
    daemon._audio_queue.join()
    daemon._stop_event.set()
    worker.join(timeout=1.0)

    assert daemon.transcriber.warmup_calls == 1
    assert len(daemon.transcriber.calls) == 1
    assert daemon.injector.injected == ["hello"]


def test_worker_waits_for_live_input_lock_before_transcribing(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon._audio_queue.put(
        SimpleNamespace(
            audio=np.array([[0.25], [0.5]], dtype=np.float32),
            emitted_prefix="",
        )
    )

    daemon._live_input_lock.acquire()
    worker = threading.Thread(target=daemon._worker_loop, daemon=True)
    worker.start()
    time.sleep(0.05)

    assert daemon.transcriber.calls == []

    daemon._live_input_lock.release()
    daemon._audio_queue.join()
    daemon._stop_event.set()
    worker.join(timeout=1.0)

    assert len(daemon.transcriber.calls) == 1


def test_indicator_stays_visible_until_inject_finishes(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon.transcriber.transcribe_stream = lambda *_args, **_kwargs: iter(["hello"])
    daemon._audio_queue.put(
        SimpleNamespace(
            audio=np.array([[0.25], [0.5]], dtype=np.float32),
            emitted_prefix="",
        )
    )
    entered = threading.Event()
    unblock = threading.Event()

    def _blocking_inject(text: str) -> None:
        entered.set()
        unblock.wait(timeout=1.0)
        daemon.injector.injected.append(text)

    daemon.injector.inject = _blocking_inject
    worker = threading.Thread(target=daemon._worker_loop, daemon=True)
    worker.start()

    assert entered.wait(timeout=1.0) is True
    assert daemon.activity_indicator.hide_calls == 0

    unblock.set()
    daemon._audio_queue.join()
    daemon._stop_event.set()
    worker.join(timeout=1.0)

    assert daemon.activity_indicator.hide_calls == 1


def test_recover_missed_hotkey_release_stops_recording(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon.recorder.is_recording = True
    daemon.hotkey.pressed = False

    monotonic_values = iter([1.0, 1.4, 1.41])
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: next(monotonic_values))

    daemon._recover_missed_hotkey_release_if_needed()
    assert daemon._audio_queue.qsize() == 0

    daemon._recover_missed_hotkey_release_if_needed()
    assert daemon.recorder.stop_calls == 1
    assert daemon._audio_queue.qsize() == 1


def test_recover_missed_hotkey_release_clears_hotkey_state_for_next_press(
    monkeypatch,
) -> None:
    config = AppConfig()
    config.audio.release_tail_seconds = 0.0
    daemon = _build_daemon(monkeypatch, config=config)
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: clock.now)

    daemon.hotkey.notify_press()
    assert daemon.recorder.start_calls == 1
    assert daemon.hotkey.is_pressed() is True

    daemon.hotkey.pressed = False
    clock.now = 10.1
    daemon._recover_missed_hotkey_release_if_needed()
    clock.now = 10.5
    daemon._recover_missed_hotkey_release_if_needed()

    assert daemon.hotkey.is_pressed() is False
    assert daemon.recorder.stop_calls == 1

    clock.now = 10.8
    daemon.hotkey.notify_press()

    assert daemon.recorder.start_calls == 2


def test_recover_missed_hotkey_release_prefers_physical_pressed_state(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon.recorder.is_recording = True
    daemon.hotkey.pressed = False
    daemon.hotkey.physical_state = True

    monotonic_values = iter([1.0, 1.4])
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: next(monotonic_values))

    daemon._recover_missed_hotkey_release_if_needed()
    daemon._recover_missed_hotkey_release_if_needed()

    assert daemon.recorder.stop_calls == 0
    assert daemon._audio_queue.qsize() == 0


def test_recover_missed_hotkey_release_preserves_pending_delayed_stop_after_notify_release(
    monkeypatch,
) -> None:
    _reset_fake_timer()
    monkeypatch.setattr(daemon_module.threading, "Timer", _FakeTimer)
    daemon = _build_daemon(monkeypatch)
    daemon.recorder.is_recording = True
    daemon.hotkey.pressed = False
    clock = SimpleNamespace(now=1.0)
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: clock.now)

    def notify_release_already_consumed() -> bool:
        daemon._on_hotkey_up()
        return False

    daemon.hotkey.notify_release = notify_release_already_consumed

    daemon._recover_missed_hotkey_release_if_needed()
    clock.now = 1.4
    daemon._recover_missed_hotkey_release_if_needed()

    assert len(_FakeTimer.instances) == 1
    timer = _FakeTimer.instances[0]
    assert timer.started is True
    assert daemon.recorder.stop_calls == 0
    assert daemon._audio_queue.qsize() == 0

    timer.fire()

    assert daemon.recorder.stop_calls == 1
    assert daemon._audio_queue.qsize() == 1


def test_recover_missed_hotkey_release_transient_physical_false_keeps_recording(
    monkeypatch,
) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon.recorder.is_recording = True
    daemon.hotkey.pressed = True
    daemon.hotkey.physical_state = False
    clock = SimpleNamespace(now=1.0)
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: clock.now)

    daemon._recover_missed_hotkey_release_if_needed()

    daemon.hotkey.physical_state = True
    clock.now = 1.4
    daemon._recover_missed_hotkey_release_if_needed()

    assert daemon.hotkey.is_pressed() is True
    assert daemon.recorder.stop_calls == 0
    assert daemon._audio_queue.qsize() == 0
    assert daemon._hotkey_not_pressed_since_monotonic is None


def test_recover_missed_hotkey_release_sustained_physical_false_keeps_recording_after_debounce(
    monkeypatch,
) -> None:
    config = AppConfig()
    config.audio.release_tail_seconds = 0.0
    daemon = _build_daemon(monkeypatch, config=config)
    daemon.recorder.is_recording = True
    daemon.hotkey.pressed = True
    daemon.hotkey.physical_state = False

    clock = SimpleNamespace(now=1.0)
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: clock.now)

    daemon._recover_missed_hotkey_release_if_needed()
    clock.now = 1.4
    daemon._recover_missed_hotkey_release_if_needed()

    assert daemon.hotkey.is_pressed() is True
    assert daemon.recorder.stop_calls == 0
    assert daemon._audio_queue.qsize() == 0
    assert daemon._hotkey_not_pressed_since_monotonic is None


def test_append_only_delta_tolerates_non_monotonic_tail() -> None:
    delta = daemon_module.PtarmiganFlowDaemon._append_only_delta("hellp", "hello world")
    assert delta == "o world"


def test_append_only_delta_keeps_phrase_overlap_without_aggressive_trim() -> None:
    previous = "同じ情報が2度入力される場合があるのでその対策を行います"
    current = (
        "同じ情報が2度入力される場合があるのでその対策を行います"
        "場合があるのでその対策を行ってください"
    )
    delta = daemon_module.PtarmiganFlowDaemon._append_only_delta(previous, current)
    assert delta == "場合があるのでその対策を行ってください"


def test_worker_uses_final_transcribe_once_when_emitted_prefix_exists(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon.transcriber.transcribe_stream = lambda *_args, **_kwargs: iter(["WRONG"])
    daemon.transcriber.transcribe = lambda *_args, **_kwargs: "hello"
    daemon._audio_queue.put(
        SimpleNamespace(
            audio=np.array([[0.25], [0.5]], dtype=np.float32),
            emitted_prefix="he",
        )
    )

    worker = threading.Thread(target=daemon._worker_loop, daemon=True)
    worker.start()
    daemon._audio_queue.join()
    daemon._stop_event.set()
    worker.join(timeout=1.0)

    assert daemon.injector.injected == ["llo"]


def test_worker_skips_consecutive_duplicate_streaming_delta(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon._audio_queue.put(
        SimpleNamespace(
            audio=np.array([[0.25], [0.5]], dtype=np.float32),
            emitted_prefix="",
        )
    )
    daemon.transcriber.transcribe_stream = lambda *_args, **_kwargs: iter(["a", "b", "c"])
    daemon._append_only_delta = lambda *_args, **_kwargs: "dup"  # type: ignore[method-assign]

    worker = threading.Thread(target=daemon._worker_loop, daemon=True)
    worker.start()
    daemon._audio_queue.join()
    daemon._stop_event.set()
    worker.join(timeout=1.0)

    assert daemon.injector.injected == ["dup"]


def test_daemon_passes_audio_input_device_policy_to_recorder(monkeypatch) -> None:
    config = AppConfig()
    config.audio.input_device_policy = "external_preferred"
    _build_daemon(monkeypatch, config=config)

    assert _FakeRecorder.last_input_device_policy == "external_preferred"


def test_release_idle_transcriber_resources_when_idle(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon._release_idle_transcriber_resources_if_needed()

    assert daemon.transcriber.idle_release_calls == 1


def test_release_idle_transcriber_resources_skips_when_busy(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)

    daemon.recorder.is_recording = True
    daemon._release_idle_transcriber_resources_if_needed()
    assert daemon.transcriber.idle_release_calls == 0

    daemon.recorder.is_recording = False
    with daemon._state_lock:
        daemon._transcription_in_progress = True
    daemon._release_idle_transcriber_resources_if_needed()
    assert daemon.transcriber.idle_release_calls == 0

    with daemon._state_lock:
        daemon._transcription_in_progress = False
    daemon._audio_queue.put(object())
    daemon._release_idle_transcriber_resources_if_needed()
    assert daemon.transcriber.idle_release_calls == 0


def test_stop_recording_defers_final_transcription_when_live_lock_is_busy(monkeypatch) -> None:
    config = AppConfig()
    config.audio.release_tail_seconds = 0.0
    daemon = _build_daemon(monkeypatch, config=config)
    daemon._supports_realtime_input = True
    daemon.recorder.is_recording = True
    with daemon._state_lock:
        daemon._live_emitted_text = "latest-prefix"

    monkeypatch.setattr(daemon_module, "_LIVE_INPUT_STOP_LOCK_TIMEOUT_SECONDS", 0.01)
    daemon._live_input_lock.acquire()
    try:
        daemon._stop_recording_and_queue_audio(reason="hotkey-release-reconciled")
    finally:
        if daemon._live_input_lock.locked():
            daemon._live_input_lock.release()

    assert daemon.recorder.stop_calls == 1
    assert daemon._audio_queue.qsize() == 0
    assert daemon.activity_indicator.show_processing_calls == 1
    assert daemon._pending_final_audio is not None
    assert daemon._pending_final_audio.emitted_prefix == "latest-prefix"


def test_stop_recording_defers_final_transcription_when_non_realtime_lock_is_busy(
    monkeypatch,
) -> None:
    config = AppConfig()
    config.audio.release_tail_seconds = 0.0
    daemon = _build_daemon(monkeypatch, config=config)
    daemon.recorder.is_recording = True

    monkeypatch.setattr(daemon_module, "_LIVE_INPUT_STOP_LOCK_TIMEOUT_SECONDS", 0.01)
    daemon._live_input_lock.acquire()
    try:
        daemon._stop_recording_and_queue_audio(reason="hotkey-release-reconciled")
    finally:
        if daemon._live_input_lock.locked():
            daemon._live_input_lock.release()

    assert daemon.recorder.stop_calls == 1
    assert daemon._audio_queue.qsize() == 0
    assert daemon.activity_indicator.show_processing_calls == 1
    assert daemon._pending_final_audio is not None
    assert daemon._pending_final_audio.emitted_prefix == ""
    assert daemon._pending_final_audio_reason == "hotkey-release-reconciled"
    assert daemon._pending_final_audio_created_at_monotonic is not None


def test_flush_pending_final_audio_queues_after_live_lock_releases(monkeypatch) -> None:
    config = AppConfig()
    config.audio.release_tail_seconds = 0.0
    daemon = _build_daemon(monkeypatch, config=config)
    daemon._supports_realtime_input = True
    daemon.recorder.is_recording = True
    with daemon._state_lock:
        daemon._live_emitted_text = "latest-prefix"

    monkeypatch.setattr(daemon_module, "_LIVE_INPUT_STOP_LOCK_TIMEOUT_SECONDS", 0.01)
    daemon._live_input_lock.acquire()
    daemon._stop_recording_and_queue_audio(reason="hotkey-release-reconciled")
    daemon._live_input_lock.release()

    daemon._flush_pending_final_audio_if_ready()

    item = daemon._audio_queue.get_nowait()
    assert item.emitted_prefix == "latest-prefix"
    assert daemon._pending_final_audio is None
    assert daemon._pending_final_audio_reason is None


def test_flush_pending_final_audio_drops_stale_audio_and_allows_next_recording(
    monkeypatch,
) -> None:
    daemon = _build_daemon(monkeypatch)
    with daemon._state_lock:
        daemon._pending_final_audio = SimpleNamespace(
            audio=np.array([[0.25], [0.5]], dtype=np.float32),
            emitted_prefix="",
        )
        daemon._pending_final_audio_reason = "hotkey-release-reconciled"
        daemon._pending_final_audio_created_at_monotonic = 1.0

    monkeypatch.setattr(daemon_module, "_PENDING_FINAL_AUDIO_TIMEOUT_SECONDS", 0.1, raising=False)
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: 1.2)

    daemon._flush_pending_final_audio_if_ready()

    assert daemon._audio_queue.qsize() == 0
    assert daemon._pending_final_audio is None
    assert daemon._pending_final_audio_reason is None
    assert daemon._pending_final_audio_created_at_monotonic is None
    assert daemon.activity_indicator.hide_calls == 1

    daemon._on_hotkey_down()

    assert daemon.recorder.start_calls == 1
    assert daemon.activity_indicator.show_recording_calls == 1


def test_hotkey_down_ignores_press_while_final_transcription_is_pending(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    with daemon._state_lock:
        daemon._pending_final_audio = SimpleNamespace(audio=np.zeros((2, 1)), emitted_prefix="")

    daemon._on_hotkey_down()

    assert daemon.recorder.start_calls == 0


def test_stop_recording_logs_info_when_keydown_warmup_holds_lock(monkeypatch, caplog) -> None:
    config = AppConfig()
    config.audio.release_tail_seconds = 0.0
    daemon = _build_daemon(monkeypatch, config=config)
    daemon.recorder.is_recording = True
    daemon.transcriber._warm_state = BackendWarmState(
        resource_mode="in_process",
        ready=True,
        warmed=False,
        warmup_running=False,
        supports_keydown_warmup=True,
        last_activity_at_monotonic=None,
    )

    monkeypatch.setattr(daemon_module, "_LIVE_INPUT_STOP_LOCK_TIMEOUT_SECONDS", 0.01)
    daemon.transcriber.warmup_unblock.clear()
    daemon._maybe_start_keydown_warmup()
    assert daemon.transcriber.warmup_started.wait(timeout=1.0) is True

    with caplog.at_level(logging.INFO, logger=daemon_module.__name__):
        daemon._stop_recording_and_queue_audio(reason="hotkey-release-reconciled")

    daemon.transcriber.warmup_unblock.set()
    thread = daemon._keydown_warmup_thread
    if thread is not None:
        thread.join(timeout=1.0)

    assert any(
        "Keydown warmup still held transcriber lock during stop" in entry.message
        for entry in caplog.records
    )


def test_worker_defers_audio_instead_of_blocking_forever_when_live_lock_is_busy(
    monkeypatch,
) -> None:
    daemon = _build_daemon(monkeypatch)
    daemon._audio_queue.put(
        SimpleNamespace(
            audio=np.array([[0.25], [0.5]], dtype=np.float32),
            emitted_prefix="",
        )
    )

    monkeypatch.setattr(
        daemon_module,
        "_WORKER_LIVE_INPUT_LOCK_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    daemon._live_input_lock.acquire()
    worker = threading.Thread(target=daemon._worker_loop, daemon=True)
    try:
        worker.start()
        time.sleep(0.05)

        assert daemon._audio_queue.unfinished_tasks == 0
        assert daemon._pending_final_audio is not None
        assert daemon._pending_final_audio_reason == "worker-live-lock-timeout"
        assert daemon._transcription_in_progress is False
    finally:
        if daemon._live_input_lock.locked():
            daemon._live_input_lock.release()
        daemon._stop_event.set()
        worker.join(timeout=1.0)


def test_stop_recording_force_closes_recorder_when_stop_fails(monkeypatch) -> None:
    config = AppConfig()
    config.audio.release_tail_seconds = 0.0
    daemon = _build_daemon(monkeypatch, config=config)
    daemon.recorder.is_recording = True

    def _raising_stop() -> np.ndarray:
        daemon.recorder.stop_calls += 1
        raise RuntimeError("stop failed")

    daemon.recorder.stop = _raising_stop  # type: ignore[method-assign]

    daemon._stop_recording_and_queue_audio(reason="hotkey-release")

    assert daemon.recorder.stop_calls == 1
    assert daemon.recorder.close_calls == 1
    assert daemon.recorder.is_recording is False
    assert daemon.activity_indicator.hide_calls == 1
    assert daemon._audio_queue.qsize() == 0


def test_worker_recovers_from_recoverable_stt_failure(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    stderr = _TTYBuffer()
    monkeypatch.setattr(daemon_module.sys, "stderr", stderr)

    def _raise_timeout(*_args, **_kwargs) -> str:
        raise _timeout_error()

    daemon.transcriber.transcribe = _raise_timeout  # type: ignore[method-assign]
    daemon._audio_queue.put(
        SimpleNamespace(
            audio=np.array([[0.25], [0.5]], dtype=np.float32),
            emitted_prefix="he",
        )
    )

    worker = threading.Thread(target=daemon._worker_loop, daemon=True)
    worker.start()
    daemon._audio_queue.join()
    daemon._stop_event.set()
    worker.join(timeout=1.0)

    assert daemon.injector.injected == []
    assert daemon.activity_indicator.hide_calls == 1
    assert daemon._transcription_in_progress is False
    assert len(daemon._recoverable_stt_failures) == 1
    assert "Recovered from STT timeout" in stderr.getvalue()
