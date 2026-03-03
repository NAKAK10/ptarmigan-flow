from __future__ import annotations

import numpy as np

from moonshine_flow.stt.vllm_realtime import VLLMRealtimeBackendSettings, VLLMRealtimeSTTBackend


class _FakeServerManager:
    websocket_url = "ws://127.0.0.1:8000/v1/realtime?intent=transcription"

    def ensure_started(self, _model_id: str) -> str:
        return "http://127.0.0.1:8000"

    def stop(self) -> None:
        return None


def _make_backend(trailing_silence_seconds: float) -> VLLMRealtimeSTTBackend:
    settings = VLLMRealtimeBackendSettings(
        model_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
        language="ja",
        trailing_silence_seconds=trailing_silence_seconds,
    )
    return VLLMRealtimeSTTBackend(settings, server_manager=_FakeServerManager())


def test_append_trailing_silence_keeps_length_when_zero() -> None:
    backend = _make_backend(0.0)
    audio = np.array([0.25, 0.5], dtype=np.float32)
    out = backend._append_trailing_silence(audio, sample_rate=16000)
    assert out.shape == audio.shape


def test_append_trailing_silence_extends_audio() -> None:
    backend = _make_backend(0.25)
    audio = np.array([0.25, 0.5], dtype=np.float32)
    out = backend._append_trailing_silence(audio, sample_rate=10)
    assert out.shape[0] == 4


def test_append_trailing_silence_clamps_negative_to_zero() -> None:
    backend = _make_backend(-1.0)
    audio = np.array([0.25, 0.5], dtype=np.float32)
    out = backend._append_trailing_silence(audio, sample_rate=16000)
    assert out.shape == audio.shape

