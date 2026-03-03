from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

from moonshine_flow.stt.voxtral_mlx import VoxtralMLXSettings, VoxtralMLXSTTBackend


def test_voxtral_mlx_preflight_and_transcribe(monkeypatch) -> None:
    voxmlx_mod = ModuleType("voxmlx")
    voxmlx_generate_mod = ModuleType("voxmlx.generate")
    mistral_common_mod = ModuleType("mistral_common")
    mistral_common_tokens_mod = ModuleType("mistral_common.tokens")
    mistral_common_tokenizers_mod = ModuleType("mistral_common.tokens.tokenizers")
    mistral_common_base_mod = ModuleType("mistral_common.tokens.tokenizers.base")

    calls: dict[str, object] = {}

    class _FakeTokenizer:
        bos_id = 1
        eos_id = 2

        @staticmethod
        def get_special_token(_name: str) -> int:
            return 999

        @staticmethod
        def decode(_tokens, special_token_policy=None) -> str:
            del special_token_policy
            return " hello world "

    def _fake_load_model(model_path: str):
        calls["model_path"] = model_path
        return object(), _FakeTokenizer(), {}

    def _fake_generate(
        _model,
        _audio_path,
        prompt_tokens,
        *,
        n_delay_tokens: int,
        temperature: float,
        eos_token_id: int,
        **_kwargs,
    ):
        calls["n_delay_tokens"] = n_delay_tokens
        calls["temperature"] = temperature
        calls["eos_token_id"] = eos_token_id
        calls["prompt_len"] = len(prompt_tokens)
        return [10, 11]

    class _SpecialTokenPolicy:
        IGNORE = object()

    voxmlx_mod.load_model = _fake_load_model
    voxmlx_generate_mod.generate = _fake_generate
    mistral_common_base_mod.SpecialTokenPolicy = _SpecialTokenPolicy

    monkeypatch.setitem(sys.modules, "voxmlx", voxmlx_mod)
    monkeypatch.setitem(sys.modules, "voxmlx.generate", voxmlx_generate_mod)
    monkeypatch.setitem(sys.modules, "mistral_common", mistral_common_mod)
    monkeypatch.setitem(sys.modules, "mistral_common.tokens", mistral_common_tokens_mod)
    monkeypatch.setitem(
        sys.modules,
        "mistral_common.tokens.tokenizers",
        mistral_common_tokenizers_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "mistral_common.tokens.tokenizers.base",
        mistral_common_base_mod,
    )

    settings = VoxtralMLXSettings(
        model_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
        language="ja",
        trailing_silence_seconds=0.0,
    )
    backend = VoxtralMLXSTTBackend(settings)
    assert backend.preflight_model() == "voxtral-mlx"

    text = backend.transcribe(np.array([0.1, -0.2, 0.3], dtype=np.float32), sample_rate=16000)
    assert text == "hello world"
    assert calls["model_path"] == "mlx-community/Voxtral-Mini-4B-Realtime-6bit"
    assert calls["n_delay_tokens"] == 6
    assert calls["temperature"] == 0.0
    assert calls["eos_token_id"] == 2
    assert int(calls["prompt_len"]) > 30
