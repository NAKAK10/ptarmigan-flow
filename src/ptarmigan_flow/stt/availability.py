"""Runtime availability checks for optional STT backends."""

from __future__ import annotations

import importlib.util

from ptarmigan_flow.stt.model_catalog import CatalogEntry, verified_model_entries

_BackendVariant = tuple[str, tuple[str, ...]]

_BACKEND_VARIANTS: dict[str, tuple[_BackendVariant, ...]] = {
    "moonshine": (("ptarmigan_flow.stt.moonshine", ("moonshine_voice",)),),
    "granite": (
        ("ptarmigan_flow.stt.granite_mlx", ("mlx_audio",)),
        ("ptarmigan_flow.stt.granite_transformers", ("transformers", "torch")),
    ),
    "mlx": (("ptarmigan_flow.stt.mlx_whisper", ("mlx_whisper",)),),
    "voxtral": (
        ("ptarmigan_flow.stt.voxtral_mlx", ("voxmlx", "mistral_common")),
        ("ptarmigan_flow.stt.voxtral_transformers", ("transformers", "mistral_common")),
    ),
}


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
    return [
        entry
        for entry in verified_model_entries()
        if is_backend_available(entry.backend)
    ]
