from __future__ import annotations

import numpy as np

import moonshine_flow.daemon as daemon_module
from moonshine_flow.config import AppConfig


class _FakeRecorder:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        dtype: str,
        max_record_seconds: int,
        input_device: str | int | None = None,
    ) -> None:
        del sample_rate, channels, dtype, max_record_seconds, input_device
        self.is_recording = False
        self.stream_active = True
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0

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


class _FakeTranscriber:
    def __init__(self, **kwargs) -> None:
        del kwargs
        self.calls: list[np.ndarray] = []

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        del sample_rate
        self.calls.append(audio)
        return "hello"

    def transcribe_stream(self, audio: np.ndarray, sample_rate: int):
        yield self.transcribe(audio, sample_rate)

    def backend_summary(self) -> str:
        return "fake-backend"

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

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def join(self) -> None:
        return None


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
    monkeypatch.setattr(daemon_module, "create_stt_backend", lambda *_args, **_kwargs: _FakeTranscriber())
    monkeypatch.setattr(daemon_module, "OutputInjector", _FakeInjector)
    monkeypatch.setattr(daemon_module, "HotkeyMonitor", _FakeHotkeyMonitor)
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


def test_recover_stale_recording_closes_recorder(monkeypatch) -> None:
    daemon = _build_daemon(monkeypatch)
    monotonic_values = iter([1.0, 1.2, 1.8])
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: next(monotonic_values))

    daemon.recorder.is_recording = True
    daemon.recorder.stream_active = False

    daemon._recover_stale_recording_if_needed()
    daemon._recover_stale_recording_if_needed()
    assert daemon.recorder.close_calls == 0

    daemon._recover_stale_recording_if_needed()
    assert daemon.recorder.close_calls == 1
    assert daemon.recorder.is_recording is False
