from types import SimpleNamespace

import pytest

import moonshine_flow.stt.factory as factory_module
from moonshine_flow.stt.factory import create_stt_backend, parse_stt_model


def _config(model_token: str, *, trailing_silence_seconds: float = 1.0):
    return SimpleNamespace(
        stt=SimpleNamespace(model=model_token),
        language="ja",
        model=SimpleNamespace(device="mps"),
        audio=SimpleNamespace(trailing_silence_seconds=trailing_silence_seconds),
    )


def test_parse_stt_model_supports_prefixed_tokens() -> None:
    assert parse_stt_model("moonshine:base") == ("moonshine", "base")
    assert parse_stt_model("voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602") == (
        "voxtral",
        "mistralai/Voxtral-Mini-4B-Realtime-2602",
    )
    assert parse_stt_model("vllm:mistralai/Voxtral-Mini-4B-Realtime-2602") == (
        "vllm",
        "mistralai/Voxtral-Mini-4B-Realtime-2602",
    )


def test_parse_stt_model_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError):
        parse_stt_model("moonshine")


def test_create_stt_backend_builds_moonshine_backend() -> None:
    backend = create_stt_backend(_config("moonshine:base"))
    assert backend.__class__.__name__ == "MoonshineSTTBackend"


def test_create_stt_backend_builds_vllm_backend() -> None:
    backend = create_stt_backend(_config("vllm:mistralai/Voxtral-Mini-4B-Realtime-2602"))
    assert backend.__class__.__name__ == "VLLMRealtimeSTTBackend"
    assert backend._settings.trailing_silence_seconds == 0.0


def test_create_stt_backend_builds_voxtral_backend(monkeypatch) -> None:
    # Keep deterministic regardless of host machine/packages.
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(factory_module.platform, "machine", lambda: "x86_64")
    backend = create_stt_backend(_config("voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602"))
    assert backend.__class__.__name__ == "VoxtralTransformersSTTBackend"
    assert backend._settings.trailing_silence_seconds == 0.0


def test_create_stt_backend_prefers_voxmlx_on_macos_arm64(monkeypatch) -> None:
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory_module.platform, "machine", lambda: "arm64")
    backend = create_stt_backend(_config("voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602"))
    assert backend.__class__.__name__ == "VoxtralMLXSTTBackend"
    assert backend._settings.trailing_silence_seconds == 0.0


def test_create_stt_backend_keeps_explicit_realtime_trailing_silence(monkeypatch) -> None:
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(factory_module.platform, "machine", lambda: "x86_64")
    backend = create_stt_backend(
        _config(
            "voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602",
            trailing_silence_seconds=0.2,
        )
    )
    assert backend._settings.trailing_silence_seconds == 0.2


def test_create_stt_backend_builds_mlx_backend_on_macos_arm64(monkeypatch) -> None:
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory_module.platform, "machine", lambda: "arm64")
    backend = create_stt_backend(_config("mlx:mlx-community/whisper-large-v3-turbo"))
    assert backend.__class__.__name__ == "MLXWhisperSTTBackend"


def test_create_stt_backend_rejects_mlx_backend_on_unsupported_platform(monkeypatch) -> None:
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(factory_module.platform, "machine", lambda: "x86_64")
    with pytest.raises(ValueError, match="supported only on macOS arm64"):
        create_stt_backend(_config("mlx:mlx-community/whisper-large-v3-turbo"))


def test_create_stt_backend_rejects_unknown_backend_prefix() -> None:
    with pytest.raises(ValueError, match="Unsupported STT backend"):
        create_stt_backend(_config("unknown:model"))
