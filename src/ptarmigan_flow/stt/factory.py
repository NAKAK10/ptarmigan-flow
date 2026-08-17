"""Factory for STT backends."""

from __future__ import annotations

import platform

from ptarmigan_flow.stt.base import SpeechToTextBackend
from ptarmigan_flow.stt.model_families import WHISPER_HF_MODEL_ID
from ptarmigan_flow.text_processing.interfaces import NoopTextPostProcessor, TextPostProcessor


def parse_stt_model(model: str) -> tuple[str, str]:
    token = model.strip()
    if not token:
        raise ValueError("stt.model must not be empty")
    if ":" not in token:
        raise ValueError(
            "stt.model must use '<backend>:<model>' format "
            f"(example: moonshine:base, mlx:{WHISPER_HF_MODEL_ID}, "
            "granite:ibm-granite/granite-4.0-1b-speech, "
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
        from ptarmigan_flow.stt.moonshine import MoonshineSTTBackend

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
        from ptarmigan_flow.stt.vllm_realtime import (
            VLLMRealtimeBackendSettings,
            VLLMRealtimeSTTBackend,
        )

        language = str(getattr(config, "language", "en")).strip().lower() or "en"
        audio_cfg = getattr(config, "audio", None)
        stt_cfg = getattr(config, "stt", None)
        stt_vllm_cfg = getattr(stt_cfg, "vllm", None)
        trailing_silence_seconds = _effective_trailing_silence_seconds_for_realtime(audio_cfg)
        settings = VLLMRealtimeBackendSettings(
            model_id=model_id,
            language=language,
            trailing_silence_seconds=trailing_silence_seconds,
            idle_shutdown_seconds=float(getattr(stt_cfg, "idle_shutdown_seconds", 30.0)),
            startup_preset=str(getattr(stt_vllm_cfg, "startup_preset", "off")),
            max_model_len=_vllm_max_model_len_for_recording_seconds(
                int(getattr(audio_cfg, "max_record_seconds", 30))
            ),
        )
        return VLLMRealtimeSTTBackend(settings, post_processor=processor)

    if prefix == "voxtral":
        language = str(getattr(config, "language", "en")).strip().lower() or "en"
        audio_cfg = getattr(config, "audio", None)
        trailing_silence_seconds = _effective_trailing_silence_seconds_for_realtime(audio_cfg)
        if _is_macos_arm64():
            from ptarmigan_flow.stt.voxtral_mlx import VoxtralMLXSettings, VoxtralMLXSTTBackend

            settings = VoxtralMLXSettings(
                model_id=model_id,
                language=language,
                trailing_silence_seconds=trailing_silence_seconds,
            )
            return VoxtralMLXSTTBackend(settings, post_processor=processor)
        from ptarmigan_flow.stt.voxtral_transformers import (
            VoxtralTransformersSettings,
            VoxtralTransformersSTTBackend,
        )

        settings = VoxtralTransformersSettings(
            model_id=model_id,
            language=language,
            trailing_silence_seconds=trailing_silence_seconds,
        )
        return VoxtralTransformersSTTBackend(settings, post_processor=processor)

    if prefix == "granite":
        language = str(getattr(config, "language", "en")).strip().lower() or "en"
        audio_cfg = getattr(config, "audio", None)
        trailing_silence_seconds = float(getattr(audio_cfg, "trailing_silence_seconds", 1.0))
        if _is_macos_arm64():
            from ptarmigan_flow.stt.granite_mlx import GraniteMLXSettings, GraniteMLXSTTBackend

            settings = GraniteMLXSettings(
                model_id=model_id,
                language=language,
                trailing_silence_seconds=trailing_silence_seconds,
            )
            return GraniteMLXSTTBackend(settings, post_processor=processor)
        from ptarmigan_flow.stt.granite_transformers import (
            GraniteTransformersSettings,
            GraniteTransformersSTTBackend,
        )

        settings = GraniteTransformersSettings(
            model_id=model_id,
            language=language,
            trailing_silence_seconds=trailing_silence_seconds,
        )
        return GraniteTransformersSTTBackend(settings, post_processor=processor)

    if prefix == "mlx":
        system = platform.system().strip().lower()
        machine = platform.machine().strip().lower()
        if system != "darwin" or machine not in {"arm64", "aarch64"}:
            raise ValueError("mlx backend is supported only on macOS arm64")
        from ptarmigan_flow.stt.mlx_whisper import (
            MLXWhisperBackendSettings,
            MLXWhisperSTTBackend,
        )

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
    # models low-latency by default. A hard 0.0 truncates trailing syllables, so
    # keep a small non-zero floor.
    if abs(configured - 1.0) < 1e-9:
        return 0.2
    return configured


def _vllm_max_model_len_for_recording_seconds(max_record_seconds: int) -> int:
    # Approximate speech tokenization budget at ~3000 tokens per minute.
    if max_record_seconds <= 30:
        return 2048
    if max_record_seconds <= 60:
        return 4096
    return 8192
