"""Runtime availability checks for optional STT backends."""

from __future__ import annotations

import importlib.util

from ptarmigan_flow.stt.haas_runtime import SPEECH_CPP_PATH_ENV, find_speech_executable
from ptarmigan_flow.stt.model_catalog import CatalogEntry, verified_model_entries

_BackendVariant = tuple[str, tuple[str, ...]]

_BACKEND_VARIANTS: dict[str, tuple[_BackendVariant, ...]] = {
    # Keep the preset selectable even before the external runtime is installed.
    "haas": (("ptarmigan_flow.stt.haas", ("numpy", "huggingface_hub")),),
    "moonshine": (("ptarmigan_flow.stt.moonshine", ("moonshine_voice",)),),
    "granite": (
        # mlx-audio's Granite model loads its tokenizer via transformers (no torch).
        ("ptarmigan_flow.stt.granite_mlx", ("mlx_audio", "transformers")),
        ("ptarmigan_flow.stt.granite_transformers", ("transformers", "torch")),
    ),
    # mlx_whisper imports numba and scipy at package import time (timing.py).
    "mlx": (("ptarmigan_flow.stt.mlx_whisper", ("mlx_whisper", "numba", "scipy")),),
    "voxtral": (
        ("ptarmigan_flow.stt.voxtral_mlx", ("voxmlx", "mistral_common")),
        ("ptarmigan_flow.stt.voxtral_transformers", ("transformers", "mistral_common")),
    ),
}


def backend_runtime_guidance(backend: str) -> str | None:
    """Cheap discovery only; version and model checks run in background preflight."""
    if backend.strip().lower() != "haas":
        return None
    try:
        if find_speech_executable() is not None:
            return None
    except (OSError, RuntimeError) as exc:
        return str(exc)
    return (
        "Install external speech.cpp >=0.9.4 from "
        "https://github.com/nyosegawa/speech.cpp/releases. "
        f"Set {SPEECH_CPP_PATH_ENV} to the speech executable if needed; "
        "~/.local/bin, /opt/homebrew/bin and /usr/local/bin are also searched."
    )


def is_backend_available(backend: str) -> bool:
    """Return whether the app build can import a backend's representative module."""
    normalized = backend.strip().lower()
    if normalized == "vllm":
        return True
    variants = _BACKEND_VARIANTS.get(normalized)
    if variants is None:
        return False
    for module_name, dependency_modules in variants:
        try:
            if importlib.util.find_spec(module_name) is None:
                continue
            if all(importlib.util.find_spec(name) is not None for name in dependency_modules):
                return True
        except Exception:
            return False
    return False


def available_model_entries() -> list[CatalogEntry]:
    """Return verified catalog models whose backend exists in this app build."""
    return [entry for entry in verified_model_entries() if is_backend_available(entry.backend)]
