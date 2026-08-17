from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pytest

import ptarmigan_flow.audio_recorder as audio_recorder_module
from ptarmigan_flow.audio_recorder import AudioRecorder


class _FakeStream:
    created = 0
    started = 0
    stopped = 0
    closed = 0
    last_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeStream.last_kwargs = kwargs
        _FakeStream.created += 1

    def start(self) -> None:
        _FakeStream.started += 1

    def stop(self) -> None:
        _FakeStream.stopped += 1

    def close(self) -> None:
        _FakeStream.closed += 1


class _StateAwareStream:
    created = 0
    instances: list[_StateAwareStream] = []

    def __init__(self, **kwargs):
        del kwargs
        _StateAwareStream.created += 1
        self.active = False
        self.closed = False
        _StateAwareStream.instances.append(self)

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.closed = True
        self.active = False


def _reset_fake_stream() -> None:
    _FakeStream.created = 0
    _FakeStream.started = 0
    _FakeStream.stopped = 0
    _FakeStream.closed = 0
    _FakeStream.last_kwargs = None


def _reset_state_aware_stream() -> None:
    _StateAwareStream.created = 0
    _StateAwareStream.instances = []


def test_recorder_opens_and_closes_per_recording(monkeypatch) -> None:
    _reset_fake_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
    )

    recorder.start()
    recorder.stop()
    recorder.start()
    recorder.stop()

    assert _FakeStream.created == 2
    assert _FakeStream.started == 2
    assert _FakeStream.stopped == 2
    assert _FakeStream.closed == 2


def test_recorder_resolves_configured_input_device_name_to_current_index(monkeypatch) -> None:
    _reset_fake_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.query_devices",
        lambda: [
            {"name": "Built-in Output", "max_input_channels": 0},
            {"name": "MacBook Air Microphone", "max_input_channels": 1},
        ],
    )
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.default",
        type("DefaultDevice", (), {"device": (1, 1)})(),
    )

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
        input_device="MacBook Air Microphone",
    )

    recorder.start()
    recorder.stop()

    assert _FakeStream.created == 1
    assert _FakeStream.started == 1
    assert _FakeStream.stopped == 1
    assert _FakeStream.closed == 1
    assert _FakeStream.last_kwargs is not None
    assert _FakeStream.last_kwargs["device"] == 1


def test_missing_configured_input_device_name_falls_back_to_policy(monkeypatch, caplog) -> None:
    _reset_fake_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.query_devices",
        lambda: [
            {"name": "MacBook Air Microphone", "max_input_channels": 1},
            {"name": "Keiju's AirPods", "max_input_channels": 1},
        ],
    )
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.default",
        type("DefaultDevice", (), {"device": (1, 1)})(),
    )

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
        input_device="Missing Microphone",
        input_device_policy="playback_friendly",
    )

    with caplog.at_level(logging.WARNING):
        recorder.start()
        recorder.stop()

    assert _FakeStream.last_kwargs is not None
    assert _FakeStream.last_kwargs["device"] == 0
    assert "Configured input device 'Missing Microphone' is unavailable" in caplog.text


def test_invalid_legacy_numeric_input_device_falls_back_to_policy(monkeypatch, caplog) -> None:
    _reset_fake_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.query_devices",
        lambda: [
            {"name": "MacBook Air Microphone", "max_input_channels": 1},
            {"name": "USB Desk Mic", "max_input_channels": 1},
        ],
    )
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.default",
        type("DefaultDevice", (), {"device": (0, 0)})(),
    )

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
        input_device=7,
        input_device_policy="external_preferred",
    )

    with caplog.at_level(logging.WARNING):
        recorder.start()
        recorder.stop()

    assert _FakeStream.last_kwargs is not None
    assert _FakeStream.last_kwargs["device"] == 1
    assert "Configured input device index 7 is unavailable" in caplog.text


def test_configured_input_device_name_warns_when_query_fails(monkeypatch, caplog) -> None:
    _reset_fake_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.query_devices",
        lambda: (_ for _ in ()).throw(RuntimeError("query failed")),
    )

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
        input_device="USB Desk Mic",
        input_device_policy="playback_friendly",
    )

    with caplog.at_level(logging.WARNING):
        recorder.start()
        recorder.stop()

    assert _FakeStream.last_kwargs is not None
    assert "device" not in _FakeStream.last_kwargs
    assert (
        "Configured input device 'USB Desk Mic' could not be resolved because audio device query failed"
        in caplog.text
    )
    assert "falling back to playback-friendly input selection" in caplog.text


def test_configured_input_device_index_warns_when_query_fails(monkeypatch, caplog) -> None:
    _reset_fake_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.query_devices",
        lambda: (_ for _ in ()).throw(RuntimeError("query failed")),
    )

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
        input_device=7,
        input_device_policy="system_default",
    )

    with caplog.at_level(logging.WARNING):
        recorder.start()
        recorder.stop()

    assert _FakeStream.last_kwargs is not None
    assert "device" not in _FakeStream.last_kwargs
    assert (
        "Configured input device index 7 could not be resolved because audio device query failed"
        in caplog.text
    )
    assert "falling back to system default input" in caplog.text


def test_stop_captures_final_callback_frames(monkeypatch) -> None:
    class _CallbackOnStopStream:
        def __init__(self, **kwargs):
            self._callback = kwargs["callback"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            tail = np.array([[0.25], [0.5]], dtype=np.float32)
            self._callback(tail, 2, None, 0)

        def close(self) -> None:
            return None

    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _CallbackOnStopStream)

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
    )

    recorder.start()
    merged = recorder.stop()

    assert merged.shape == (2, 1)
    assert np.allclose(merged[:, 0], np.array([0.25, 0.5], dtype=np.float32))


def test_stop_drains_late_callback_frames_before_dispose(monkeypatch) -> None:
    class _DelayedDeliveryStream:
        def __init__(self, **kwargs):
            self._callback = kwargs["callback"]
            self.stopped = False
            self.closed = False

        def start(self) -> None:
            return None

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _DelayedDeliveryStream)

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
    )

    recorder.start()

    tail = np.array([[0.75]], dtype=np.float32)

    def _deliver_late_frame() -> None:
        time.sleep(0.03)
        recorder._callback(tail, 1, None, 0)

    thread = threading.Thread(target=_deliver_late_frame)
    thread.start()
    merged = recorder.stop()
    thread.join()

    assert merged.shape[0] >= 1
    assert np.isclose(merged[-1, 0], 0.75)


def test_recorder_uses_system_default_when_policy_is_system_default(monkeypatch) -> None:
    _reset_fake_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
        input_device_policy="system_default",
    )

    recorder.start()
    recorder.stop()

    assert _FakeStream.last_kwargs is not None
    assert "device" not in _FakeStream.last_kwargs


def test_playback_friendly_policy_avoids_bluetooth_default_input(monkeypatch) -> None:
    _reset_fake_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.query_devices",
        lambda: [
            {"name": "MacBook Air Microphone", "max_input_channels": 1},
            {"name": "Keiju's AirPods", "max_input_channels": 1},
        ],
    )
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.default",
        type("DefaultDevice", (), {"device": (1, 1)})(),
    )

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
        input_device_policy="playback_friendly",
    )

    recorder.start()
    recorder.stop()

    assert _FakeStream.last_kwargs is not None
    assert _FakeStream.last_kwargs["device"] == 0


def test_playback_friendly_policy_keeps_non_bluetooth_default_input(monkeypatch) -> None:
    _reset_fake_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.query_devices",
        lambda: [{"name": "MacBook Air Microphone", "max_input_channels": 1}],
    )
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.default",
        type("DefaultDevice", (), {"device": (0, 0)})(),
    )

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
        input_device_policy="playback_friendly",
    )

    recorder.start()
    recorder.stop()

    assert _FakeStream.last_kwargs is not None
    assert "device" not in _FakeStream.last_kwargs


def test_playback_friendly_policy_handles_missing_default_input_index(monkeypatch) -> None:
    _reset_fake_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.query_devices",
        lambda: [{"name": "MacBook Air Microphone", "max_input_channels": 1}],
    )
    monkeypatch.setattr(
        "ptarmigan_flow.audio_recorder.sd.default",
        type("DefaultDevice", (), {"device": (None, None)})(),
    )

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
        input_device_policy="playback_friendly",
    )

    recorder.start()
    recorder.stop()

    assert _FakeStream.last_kwargs is not None
    assert _FakeStream.last_kwargs["device"] == 0


def test_callback_stop_resets_recording_state(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=1,
    )

    recorder.start()

    chunk = np.ones((16000, 1), dtype=np.float32)
    with pytest.raises(audio_recorder_module.sd.CallbackStop):
        recorder._callback(chunk, 16000, None, 0)

    assert recorder.is_recording is False


def test_callback_stop_uses_cumulative_frames_and_resets_on_restart(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)

    recorder = AudioRecorder(
        sample_rate=4,
        channels=1,
        dtype="float32",
        max_record_seconds=1,
    )

    recorder.start()
    recorder._callback(np.ones((2, 1), dtype=np.float32), 2, None, 0)
    assert recorder.is_recording is True

    with pytest.raises(audio_recorder_module.sd.CallbackStop):
        recorder._callback(np.ones((2, 1), dtype=np.float32), 2, None, 0)

    assert recorder.is_recording is False
    merged = recorder.stop()
    assert merged.shape == (4, 1)

    recorder.start()
    recorder._callback(np.ones((2, 1), dtype=np.float32), 2, None, 0)

    assert recorder.is_recording is True
    merged = recorder.stop()
    assert merged.shape == (2, 1)


def test_start_reopens_stale_stream_after_callback_stop(monkeypatch) -> None:
    _reset_state_aware_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _StateAwareStream)

    recorder = AudioRecorder(
        sample_rate=10,
        channels=1,
        dtype="float32",
        max_record_seconds=1,
    )

    recorder.start()
    assert recorder._stream is not None
    first_stream = recorder._stream

    chunk = np.ones((10, 1), dtype=np.float32)
    with pytest.raises(audio_recorder_module.sd.CallbackStop):
        recorder._callback(chunk, 10, None, 0)

    assert recorder.is_recording is False
    first_stream.active = False

    recorder.start()
    assert _StateAwareStream.created == 2
    assert first_stream.closed is True
    assert recorder._stream is not first_stream
    assert recorder.is_recording is True


def test_is_stream_active_reflects_stream_state(monkeypatch) -> None:
    _reset_state_aware_stream()
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _StateAwareStream)

    recorder = AudioRecorder(
        sample_rate=10,
        channels=1,
        dtype="float32",
        max_record_seconds=1,
    )
    assert recorder.is_stream_active() is False

    recorder.start()
    assert recorder.is_stream_active() is True

    assert recorder._stream is not None
    recorder._stream.active = False
    assert recorder.is_stream_active() is False


def test_snapshot_returns_buffer_without_stopping(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.audio_recorder.sd.InputStream", _FakeStream)

    recorder = AudioRecorder(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        max_record_seconds=30,
    )
    recorder.start()
    recorder._callback(np.array([[0.1], [0.2]], dtype=np.float32), 2, None, 0)

    snap = recorder.snapshot()
    assert recorder.is_recording is True
    assert snap.shape == (2, 1)
    assert np.allclose(snap[:, 0], np.array([0.1, 0.2], dtype=np.float32))

    merged = recorder.stop()
    assert np.allclose(merged[:, 0], np.array([0.1, 0.2], dtype=np.float32))
