"""Factory for STT backends."""

from __future__ import annotations

import platform

from moonshine_flow.stt.base import SpeechToTextBackend
from moonshine_flow.stt.mlx_whisper import MLXWhisperBackendSettings, MLXWhisperSTTBackend
from moonshine_flow.stt.moonshine import MoonshineSTTBackend
from moonshine_flow.stt.vllm_realtime import VLLMRealtimeBackendSettings, VLLMRealtimeSTTBackend
from moonshine_flow.stt.voxtral_mlx import VoxtralMLXSettings, VoxtralMLXSTTBackend
from moonshine_flow.stt.voxtral_transformers import (
    VoxtralTransformersSettings,
    VoxtralTransformersSTTBackend,
)
from moonshine_flow.text_processing.interfaces import NoopTextPostProcessor, TextPostProcessor


def parse_stt_model(model: str) -> tuple[str, str]:
    token = model.strip()
    if not token:
        raise ValueError("stt.model must not be empty")
    if ":" not in token:
        raise ValueError(
            "stt.model must use '<backend>:<model>' format "
            "(example: moonshine:base, mlx:mlx-community/whisper-large-v3-turbo, "
            "voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602, "
            "vllm:mistralai/Voxtral-Mini-4B-Realtime-2602)"
        )
    prefix, model_id = token.split(":", 1)
    prefix = prefix.strip().lower()
    model_id = model_id.strip()
    if not prefix or not model_id:
        raise ValueError("stt.model must include both backend prefix and model identifier")
    return prefix, model_id


def create_stt_backend(
    config: object,
    *,
    post_processor: TextPostProcessor | None = None,
) -> SpeechToTextBackend:
    stt_cfg = getattr(config, "stt", None)
    model_token = str(getattr(stt_cfg, "model", "")).strip()
    prefix, model_id = parse_stt_model(model_token)
    processor = post_processor or NoopTextPostProcessor()

    if prefix == "moonshine":
        if model_id not in {"tiny", "base"}:
            raise ValueError("moonshine model must be one of: tiny, base")
        model_cfg = getattr(config, "model", None)
        audio_cfg = getattr(config, "audio", None)
        language = str(getattr(config, "language", "en"))
        device = str(getattr(model_cfg, "device", "mps"))
        trailing_silence_seconds = float(getattr(audio_cfg, "trailing_silence_seconds", 1.0))
        return MoonshineSTTBackend(
            model_size=model_id,
            language=language,
            device=device,
            trailing_silence_seconds=trailing_silence_seconds,
            post_processor=processor,
        )

    if prefix == "vllm":
        language = str(getattr(config, "language", "en")).strip().lower() or "en"
        audio_cfg = getattr(config, "audio", None)
        trailing_silence_seconds = _effective_trailing_silence_seconds_for_realtime(audio_cfg)
        settings = VLLMRealtimeBackendSettings(
            model_id=model_id,
            language=language,
            trailing_silence_seconds=trailing_silence_seconds,
        )
        return VLLMRealtimeSTTBackend(settings, post_processor=processor)

    if prefix == "voxtral":
        language = str(getattr(config, "language", "en")).strip().lower() or "en"
        audio_cfg = getattr(config, "audio", None)
        trailing_silence_seconds = _effective_trailing_silence_seconds_for_realtime(audio_cfg)
        if _is_macos_arm64():
            settings = VoxtralMLXSettings(
                model_id=model_id,
                language=language,
                trailing_silence_seconds=trailing_silence_seconds,
            )
            return VoxtralMLXSTTBackend(settings, post_processor=processor)
        settings = VoxtralTransformersSettings(
            model_id=model_id,
            language=language,
            trailing_silence_seconds=trailing_silence_seconds,
        )
        return VoxtralTransformersSTTBackend(settings, post_processor=processor)

    if prefix == "mlx":
        system = platform.system().strip().lower()
        machine = platform.machine().strip().lower()
        if system != "darwin" or machine not in {"arm64", "aarch64"}:
            raise ValueError("mlx backend is supported only on macOS arm64")
        language = str(getattr(config, "language", "en")).strip().lower() or "en"
        audio_cfg = getattr(config, "audio", None)
        trailing_silence_seconds = float(getattr(audio_cfg, "trailing_silence_seconds", 1.0))
        settings = MLXWhisperBackendSettings(
            model_id=model_id,
            language=language,
            trailing_silence_seconds=trailing_silence_seconds,
        )
        return MLXWhisperSTTBackend(settings, post_processor=processor)

    raise ValueError(f"Unsupported STT backend prefix: {prefix}")


def _is_macos_arm64() -> bool:
    system = platform.system().strip().lower()
    machine = platform.machine().strip().lower()
    return system == "darwin" and machine in {"arm64", "aarch64"}


def _effective_trailing_silence_seconds_for_realtime(audio_cfg: object | None) -> float:
    configured = float(getattr(audio_cfg, "trailing_silence_seconds", 1.0))
    # Keep backward compatibility for explicit user overrides while making realtime
    # models low-latency by default.
    if abs(configured - 1.0) < 1e-9:
        return 0.0
    return configured
