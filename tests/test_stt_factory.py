from types import SimpleNamespace

import pytest

import ptarmigan_flow.stt.factory as factory_module
from ptarmigan_flow.stt.factory import create_stt_backend, parse_stt_model
from ptarmigan_flow.stt.model_families import WHISPER_HF_MODEL_ID, WHISPER_MLX_MODEL_ID


def _config(
    model_token: str,
    *,
    trailing_silence_seconds: float = 1.0,
    idle_shutdown_seconds: float = 30.0,
    max_record_seconds: int = 30,
    startup_preset: str = "off",
):
    return SimpleNamespace(
        stt=SimpleNamespace(
            model=model_token,
            idle_shutdown_seconds=idle_shutdown_seconds,
            vllm=SimpleNamespace(startup_preset=startup_preset),
        ),
        language="ja",
        model=SimpleNamespace(device="mps"),
        audio=SimpleNamespace(
            trailing_silence_seconds=trailing_silence_seconds,
            max_record_seconds=max_record_seconds,
        ),
    )


def test_parse_stt_model_supports_prefixed_tokens() -> None:
    assert parse_stt_model("moonshine:base") == ("moonshine", "base")
    assert parse_stt_model("granite:ibm-granite/granite-4.0-1b-speech") == (
        "granite",
        "ibm-granite/granite-4.0-1b-speech",
    )
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
    assert backend._settings.trailing_silence_seconds == 0.2
    assert backend._settings.idle_shutdown_seconds == 30.0
    assert backend._settings.startup_preset == "off"
    assert backend._settings.max_model_len == 2048


def test_create_stt_backend_propagates_vllm_startup_preset() -> None:
    backend = create_stt_backend(
        _config(
            "vllm:mistralai/Voxtral-Mini-4B-Realtime-2602",
            startup_preset="balanced",
        )
    )
    assert backend._settings.startup_preset == "balanced"


def test_create_stt_backend_derives_vllm_max_model_len_for_60_seconds() -> None:
    backend = create_stt_backend(
        _config(
            "vllm:mistralai/Voxtral-Mini-4B-Realtime-2602",
            max_record_seconds=60,
        )
    )
    assert backend._settings.max_model_len == 4096


def test_create_stt_backend_derives_vllm_max_model_len_above_60_seconds() -> None:
    backend = create_stt_backend(
        _config(
            "vllm:mistralai/Voxtral-Mini-4B-Realtime-2602",
            max_record_seconds=61,
        )
    )
    assert backend._settings.max_model_len == 8192


def test_create_stt_backend_builds_granite_transformers_backend(monkeypatch) -> None:
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(factory_module.platform, "machine", lambda: "x86_64")
    backend = create_stt_backend(_config("granite:ibm-granite/granite-4.0-1b-speech"))
    assert backend.__class__.__name__ == "GraniteTransformersSTTBackend"
    assert backend._settings.trailing_silence_seconds == 1.0


def test_create_stt_backend_builds_granite_mlx_backend_on_macos_arm64(monkeypatch) -> None:
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory_module.platform, "machine", lambda: "arm64")
    backend = create_stt_backend(_config("granite:ibm-granite/granite-4.0-1b-speech"))
    assert backend.__class__.__name__ == "GraniteMLXSTTBackend"
    assert backend._settings.trailing_silence_seconds == 1.0


def test_create_stt_backend_builds_voxtral_backend(monkeypatch) -> None:
    # Keep deterministic regardless of host machine/packages.
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(factory_module.platform, "machine", lambda: "x86_64")
    backend = create_stt_backend(_config("voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602"))
    assert backend.__class__.__name__ == "VoxtralTransformersSTTBackend"
    assert backend._settings.trailing_silence_seconds == 0.2


def test_create_stt_backend_prefers_voxmlx_on_macos_arm64(monkeypatch) -> None:
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory_module.platform, "machine", lambda: "arm64")
    backend = create_stt_backend(_config("voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602"))
    assert backend.__class__.__name__ == "VoxtralMLXSTTBackend"
    assert backend._settings.trailing_silence_seconds == 0.2


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
    backend = create_stt_backend(_config(f"mlx:{WHISPER_HF_MODEL_ID}"))
    assert backend.__class__.__name__ == "MLXWhisperSTTBackend"
    assert WHISPER_MLX_MODEL_ID in backend.backend_summary()


def test_create_stt_backend_rejects_mlx_backend_on_unsupported_platform(monkeypatch) -> None:
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(factory_module.platform, "machine", lambda: "x86_64")
    with pytest.raises(ValueError, match="supported only on macOS arm64"):
        create_stt_backend(_config(f"mlx:{WHISPER_HF_MODEL_ID}"))


def test_create_stt_backend_accepts_legacy_mlx_whisper_model_id(monkeypatch) -> None:
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory_module.platform, "machine", lambda: "arm64")
    backend = create_stt_backend(_config("mlx:mlx-community/whisper-large-v3-turbo"))
    assert backend.__class__.__name__ == "MLXWhisperSTTBackend"
    assert WHISPER_MLX_MODEL_ID in backend.backend_summary()


def test_create_stt_backend_rejects_unknown_backend_prefix() -> None:
    with pytest.raises(ValueError, match="Unsupported STT backend"):
        create_stt_backend(_config("unknown:model"))


def test_runtime_status_for_moonshine_reports_no_external_server() -> None:
    backend = create_stt_backend(_config("moonshine:base"))
    assert backend.runtime_status().startswith("🚀 Backend ready (no external server):")


def test_runtime_status_for_vllm_reports_external_server_state() -> None:
    backend = create_stt_backend(_config("vllm:mistralai/Voxtral-Mini-4B-Realtime-2602"))
    assert backend.runtime_status().startswith("💨 External server stopped:")
