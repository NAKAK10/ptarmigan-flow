from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

from moonshine_flow.stt.voxtral_transformers import (
    VoxtralTransformersSettings,
    VoxtralTransformersSTTBackend,
)


def test_voxtral_transcribe_with_fake_transformers(monkeypatch) -> None:
    transformers_mod = ModuleType("transformers")
    mistral_common_mod = ModuleType("mistral_common")

    class _FakeInputs(dict):
        def to(self, *_args, **_kwargs):
            return self

    class _FakeProcessor:
        feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16000})()

        @classmethod
        def from_pretrained(cls, _repo_id: str):
            return cls()

        def __call__(self, _audio, **_kwargs):
            return _FakeInputs(input_values=np.array([[0.1, 0.2]], dtype=np.float32))

        def batch_decode(self, _outputs, skip_special_tokens: bool = True):
            del skip_special_tokens
            return [" hello world "]

    class _FakeModel:
        device = "cpu"
        dtype = np.float32

        @classmethod
        def from_pretrained(cls, _repo_id: str, **_kwargs):
            return cls()

        def generate(self, **_inputs):
            return np.array([[1, 2, 3]], dtype=np.int32)

    transformers_mod.AutoProcessor = _FakeProcessor
    transformers_mod.VoxtralRealtimeForConditionalGeneration = _FakeModel

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "mistral_common", mistral_common_mod)

    settings = VoxtralTransformersSettings(
        model_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
        language="en",
        trailing_silence_seconds=0.0,
    )
    backend = VoxtralTransformersSTTBackend(settings)

    text = backend.transcribe(np.array([0.1, 0.2], dtype=np.float32), sample_rate=16000)
    assert text == "hello world"


def test_voxtral_preflight_falls_back_without_accelerate(monkeypatch) -> None:
    transformers_mod = ModuleType("transformers")
    mistral_common_mod = ModuleType("mistral_common")
    model_load_calls: list[dict[str, object]] = []

    class _FakeProcessor:
        feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16000})()

        @classmethod
        def from_pretrained(cls, _repo_id: str):
            return cls()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, _repo_id: str, **kwargs):
            model_load_calls.append(dict(kwargs))
            if "device_map" in kwargs:
                raise RuntimeError(
                    "Using a `device_map`, `tp_plan`, `torch.device` context manager or "
                    "setting `torch.set_default_device(device)` requires `accelerate`."
                )
            return cls()

    transformers_mod.AutoProcessor = _FakeProcessor
    transformers_mod.VoxtralRealtimeForConditionalGeneration = _FakeModel

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "mistral_common", mistral_common_mod)

    settings = VoxtralTransformersSettings(
        model_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
        language="en",
        trailing_silence_seconds=0.0,
    )
    backend = VoxtralTransformersSTTBackend(settings)
    backend.preflight_model()

    assert model_load_calls == [{"device_map": "auto"}, {}]
