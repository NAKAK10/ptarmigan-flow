"""Runtime availability checks for optional STT backends."""

from __future__ import annotations

import importlib.util

from ptarmigan_flow.stt.model_catalog import CatalogEntry, verified_model_entries

_BACKEND_MODULES: dict[str, tuple[str, ...]] = {
    "moonshine": ("ptarmigan_flow.stt.moonshine",),
    "granite": (
        "ptarmigan_flow.stt.granite_mlx",
        "ptarmigan_flow.stt.granite_transformers",
    ),
    "mlx": ("ptarmigan_flow.stt.mlx_whisper",),
    "voxtral": (
        "ptarmigan_flow.stt.voxtral_mlx",
        "ptarmigan_flow.stt.voxtral_transformers",
    ),
}


def is_backend_available(backend: str) -> bool:
    """Return whether the app build can import a backend's representative module."""
    normalized = backend.strip().lower()
    if normalized == "vllm":
        return True
    module_names = _BACKEND_MODULES.get(normalized)
    if module_names is None:
        return False
    for module_name in module_names:
        try:
            if importlib.util.find_spec(module_name) is not None:
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
