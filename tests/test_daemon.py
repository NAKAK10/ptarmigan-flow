from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np

import moonshine_flow.daemon as daemon_module
from moonshine_flow.config import AppConfig


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

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        del sample_rate
        self.calls.append(audio)
        return "hello"

    def transcribe_stream(self, audio: np.ndarray, sample_rate: int):
        yield self.transcribe(audio, sample_rate)

    def supports_realtime_input(self) -> bool:
        return False

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
    ) -> None:
        del key_name, max_hold_seconds
        self.on_press = on_press
        self.on_release = on_release
        self.started = False
        self.pressed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def join(self) -> None:
        return None

    def is_pressed(self) -> bool:
        return self.pressed


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
) -> daemon_module.MoonshineFlowDaemon:
    monkeypatch.setattr(daemon_module, "AudioRecorder", _FakeRecorder)
    monkeypatch.setattr(
        daemon_module,
        "create_stt_backend",
        lambda *_args, **_kwargs: _FakeTranscriber(),
    )
    monkeypatch.setattr(daemon_module, "OutputInjector", _FakeInjector)
    monkeypatch.setattr(daemon_module, "HotkeyMonitor", _FakeHotkeyMonitor)
    monkeypatch.setattr(
        daemon_module,
        "create_activity_indicator",
        lambda *_args, **_kwargs: _FakeActivityIndicator(),
    )
    return daemon_module.MoonshineFlowDaemon(config or AppConfig())


def test_hotkey_down_ignored_while_transcription_busy(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    with daemon._state_lock:
        daemon._transcription_in_progress = True

    daemon._on_hotkey_down()

    assert daemon.recorder.start_calls == 0


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


def test_hotkey_up_skips_delay_for_default_realtime_model(monkeypatch) -> None:
    _reset_fake_timer()
    monkeypatch.setattr(daemon_module.threading, "Timer", _FakeTimer)
    config = AppConfig()
    config.stt.model = "voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602"
    daemon = _build_daemon(monkeypatch, config=config)
    daemon.recorder.is_recording = True

    daemon._on_hotkey_up()

    assert len(_FakeTimer.instances) == 0
    assert daemon.recorder.stop_calls == 1
    assert daemon._audio_queue.qsize() == 1


def test_effective_release_tail_keeps_explicit_override_for_realtime(monkeypatch) -> None:
    config = AppConfig()
    config.stt.model = "voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602"
    config.audio.release_tail_seconds = 0.1
    daemon = _build_daemon(monkeypatch, config=config)

    assert daemon._effective_release_tail_seconds() == 0.1


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


def test_append_only_delta_tolerates_non_monotonic_tail() -> None:
    delta = daemon_module.MoonshineFlowDaemon._append_only_delta("hellp", "hello world")
    assert delta == "o world"


def test_append_only_delta_keeps_phrase_overlap_without_aggressive_trim() -> None:
    previous = "同じ情報が2度入力される場合があるのでその対策を行います"
    current = (
        "同じ情報が2度入力される場合があるのでその対策を行います"
        "場合があるのでその対策を行ってください"
    )
    delta = daemon_module.MoonshineFlowDaemon._append_only_delta(previous, current)
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


def test_stop_recording_falls_back_when_live_lock_is_busy(monkeypatch) -> None:
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

    item = daemon._audio_queue.get_nowait()
    assert daemon.recorder.stop_calls == 1
    assert item.emitted_prefix == "latest-prefix"


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
