"""Canonical model ids and family-specific model resolution helpers."""

from __future__ import annotations

from ptarmigan_flow.stt.haas_artifact import HAAS_MODEL_ID as HAAS_HF_MODEL_ID

VOXTRAL_HF_MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
VOXTRAL_MLX_MODEL_ID = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"
GRANITE_HF_MODEL_ID = "ibm-granite/granite-4.0-1b-speech"
GRANITE_MLX_MODEL_ID = "mlx-community/granite-4.0-1b-speech-8bit"
WHISPER_HF_MODEL_ID = "openai/whisper-large-v3-turbo"
WHISPER_MLX_MODEL_ID = "mlx-community/whisper-large-v3-turbo"


def validate_stt_language(model: str, language: str) -> None:
    """Validate Haas's fixed language without changing any user's settings."""
    prefix, _separator, model_id = model.strip().partition(":")
    if prefix.strip().lower() == "haas":
        if model_id.strip() != HAAS_HF_MODEL_ID:
            raise ValueError(f"Haas supports only haas:{HAAS_HF_MODEL_ID} (Q4_K)")
        if language.strip().lower() not in {"ja", "auto"}:
            raise ValueError(
                "Haas is Japanese-only; select Japanese in Settings or set language='ja' "
                "in config.toml. Use `pflow config language` before selecting Haas. "
                "language='auto' is also accepted for Haas and still recognizes Japanese only."
            )
    elif language.strip().lower() == "auto":
        raise ValueError(
            "language='auto' is no longer supported for this backend; use an explicit "
            "value like 'ja' or 'en' with `pflow config language`."
        )


def resolve_voxtral_mlx_model_id(model_id: str) -> str:
    normalized = model_id.strip()
    if normalized.lower() == VOXTRAL_HF_MODEL_ID.lower():
        return VOXTRAL_MLX_MODEL_ID
    return normalized


def resolve_granite_mlx_model_id(model_id: str) -> str:
    normalized = model_id.strip()
    if normalized.lower() == GRANITE_HF_MODEL_ID.lower():
        return GRANITE_MLX_MODEL_ID
    return normalized


def resolve_whisper_mlx_model_id(model_id: str) -> str:
    normalized = model_id.strip()
    if normalized.lower() == WHISPER_HF_MODEL_ID.lower():
        return WHISPER_MLX_MODEL_ID
    return normalized


def resolve_runtime_model_id(
    *,
    prefix: str,
    model_id: str,
    macos_arm64: bool,
) -> str:
    normalized_prefix = prefix.strip().lower()
    normalized_model_id = model_id.strip()
    if normalized_prefix == "haas" and normalized_model_id != HAAS_HF_MODEL_ID:
        raise ValueError(f"Haas supports only haas:{HAAS_HF_MODEL_ID} (Q4_K)")
    if normalized_prefix == "voxtral" and macos_arm64:
        return resolve_voxtral_mlx_model_id(normalized_model_id)
    if normalized_prefix == "granite" and macos_arm64:
        return resolve_granite_mlx_model_id(normalized_model_id)
    if normalized_prefix == "mlx" and macos_arm64:
        return resolve_whisper_mlx_model_id(normalized_model_id)
    return normalized_model_id
