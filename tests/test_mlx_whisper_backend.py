from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

from moonshine_flow.stt.mlx_whisper import MLXWhisperBackendSettings, MLXWhisperSTTBackend


def test_transcribe_uses_mlx_whisper_module(monkeypatch, tmp_path) -> None:
    fake_mod = ModuleType("mlx_whisper")

    calls: dict[str, object] = {}

    def fake_transcribe(path: str, *, path_or_hf_repo: str, language: str):
        calls["path"] = path
        calls["model"] = path_or_hf_repo
        calls["language"] = language
        return {"text": " こんにちは "}

    fake_mod.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.chdir(tmp_path)

    settings = MLXWhisperBackendSettings(
        model_id="mlx-community/whisper-large-v3-turbo",
        language="ja",
        trailing_silence_seconds=0.0,
    )
    backend = MLXWhisperSTTBackend(settings)

    text = backend.transcribe(np.array([0.1, 0.2], dtype=np.float32), sample_rate=16000)

    assert text == "こんにちは"
    assert calls["model"] == "mlx-community/whisper-large-v3-turbo"
    assert calls["language"] == "ja"


def test_transcribe_stream_yields_single_final_text(monkeypatch, tmp_path) -> None:
    fake_mod = ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda *_args, **_kwargs: {"text": " hello "}
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    monkeypatch.chdir(tmp_path)

    settings = MLXWhisperBackendSettings(
        model_id="mlx-community/whisper-large-v3-turbo",
        language="en",
        trailing_silence_seconds=0.0,
    )
    backend = MLXWhisperSTTBackend(settings)

    updates = list(backend.transcribe_stream(np.array([0.1], dtype=np.float32), sample_rate=16000))
    assert updates == ["hello"]

