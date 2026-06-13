from __future__ import annotations

import numpy as np
import pytest

from ptarmigan_flow.config import AppConfig
from ptarmigan_flow.stt.runtime_backend import (
    IsolatedSpeechToTextBackend,
    SpeechToTextChildCrashedError,
    SpeechToTextRequestTimeoutError,
    create_runtime_stt_backend,
)
from ptarmigan_flow.stt.vllm_realtime import VLLMRealtimeSTTBackend

_ECHO_FACTORY = "ptarmigan_flow.stt._test_support:build_echo_backend"
_TIMEOUT_FACTORY = "ptarmigan_flow.stt._test_support:build_timeout_backend"
_CRASH_FACTORY = "ptarmigan_flow.stt._test_support:build_crash_backend"
_TEST_CHILD_STARTUP_TIMEOUT_SECONDS = 5.0


class _SuffixProcessor:
    def __init__(self, suffix: str) -> None:
        self._suffix = suffix

    def apply(self, text: str) -> str:
        return f"{text}{self._suffix}" if text else text


def _config(model: str = "granite:test-model") -> AppConfig:
    config = AppConfig()
    config.stt.model = model
    return config


def test_create_runtime_stt_backend_isolates_local_backends() -> None:
    backend = create_runtime_stt_backend(_config("granite:test-model"))
    try:
        assert isinstance(backend, IsolatedSpeechToTextBackend)
    finally:
        backend.close()


def test_create_runtime_stt_backend_keeps_vllm_direct() -> None:
    backend = create_runtime_stt_backend(
        _config("vllm:mistralai/Voxtral-Mini-4B-Realtime-2602")
    )
    assert isinstance(backend, VLLMRealtimeSTTBackend)


def test_isolated_backend_applies_parent_post_processing(monkeypatch) -> None:
    monkeypatch.setattr(
        "ptarmigan_flow.stt.runtime_backend._CHILD_STARTUP_TIMEOUT_SECONDS",
        _TEST_CHILD_STARTUP_TIMEOUT_SECONDS,
    )
    backend = IsolatedSpeechToTextBackend(
        _config(),
        backend_prefix="granite",
        model_id="test-model",
        post_processor=_SuffixProcessor("!"),
        backend_factory_spec=_ECHO_FACTORY,
    )
    audio = np.zeros((8, 1), dtype=np.float32)

    try:
        assert backend.preflight_model() == "test-echo"
        assert backend.transcribe(audio, 16000) == "samples=8!"
        assert list(backend.transcribe_stream(audio, 16000)) == ["samples=8!"]
        warm_state = backend.warm_state()
        assert warm_state.ready is True
        assert warm_state.warmed is True
    finally:
        backend.close()


def test_isolated_backend_times_out_and_accepts_next_request(monkeypatch) -> None:
    monkeypatch.setattr(
        "ptarmigan_flow.stt.runtime_backend._CHILD_STARTUP_TIMEOUT_SECONDS",
        _TEST_CHILD_STARTUP_TIMEOUT_SECONDS,
    )
    monkeypatch.setattr(
        "ptarmigan_flow.stt.runtime_backend._MIN_TRANSCRIPTION_TIMEOUT_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        "ptarmigan_flow.stt.runtime_backend._MAX_TRANSCRIPTION_TIMEOUT_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        "ptarmigan_flow.stt.runtime_backend._TRANSCRIPTION_TIMEOUT_PADDING_SECONDS",
        0.0,
    )
    backend = IsolatedSpeechToTextBackend(
        _config(),
        backend_prefix="granite",
        model_id="test-model",
        backend_factory_spec=_TIMEOUT_FACTORY,
    )
    slow_audio = np.zeros((16000, 1), dtype=np.float32)
    fast_audio = np.zeros((64, 1), dtype=np.float32)

    try:
        backend.preflight_model()
        with pytest.raises(SpeechToTextRequestTimeoutError) as exc_info:
            backend.transcribe(slow_audio, 16000)
        assert exc_info.value.summary.restart_succeeded is True
        assert backend.transcribe(fast_audio, 16000) == "samples=64"
    finally:
        backend.close()


def test_isolated_backend_recovers_after_child_crash(monkeypatch) -> None:
    monkeypatch.setattr(
        "ptarmigan_flow.stt.runtime_backend._CHILD_STARTUP_TIMEOUT_SECONDS",
        _TEST_CHILD_STARTUP_TIMEOUT_SECONDS,
    )
    backend = IsolatedSpeechToTextBackend(
        _config(),
        backend_prefix="granite",
        model_id="test-model",
        backend_factory_spec=_CRASH_FACTORY,
    )
    crash_audio = np.zeros((16000, 1), dtype=np.float32)
    fast_audio = np.zeros((32, 1), dtype=np.float32)

    try:
        backend.preflight_model()
        with pytest.raises(SpeechToTextChildCrashedError) as exc_info:
            backend.transcribe(crash_audio, 16000)
        assert exc_info.value.summary.restart_succeeded is True
        assert backend.transcribe(fast_audio, 16000) == "samples=32"
    finally:
        backend.close()
