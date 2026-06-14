from __future__ import annotations

from dataclasses import dataclass

from ptarmigan_flow.stt import availability


@dataclass(frozen=True)
class _Entry:
    token: str
    backend: str


def test_is_backend_available_uses_representative_modules(monkeypatch) -> None:
    requested: list[str] = []
    available_modules = {
        "ptarmigan_flow.stt.moonshine",
        "moonshine_voice",
        "ptarmigan_flow.stt.granite_transformers",
        "transformers",
        "torch",
        "ptarmigan_flow.stt.voxtral_mlx",
        "voxmlx",
        "mistral_common",
    }

    def fake_find_spec(module_name: str):
        requested.append(module_name)
        if module_name in available_modules:
            return object()
        return None

    monkeypatch.setattr(availability.importlib.util, "find_spec", fake_find_spec)

    assert availability.is_backend_available("moonshine") is True
    assert availability.is_backend_available("granite") is True
    assert availability.is_backend_available("mlx") is False
    assert availability.is_backend_available("voxtral") is True
    assert availability.is_backend_available("vllm") is True
    assert availability.is_backend_available("unknown") is False
    assert requested == [
        "ptarmigan_flow.stt.moonshine",
        "moonshine_voice",
        "ptarmigan_flow.stt.granite_mlx",
        "ptarmigan_flow.stt.granite_transformers",
        "transformers",
        "torch",
        "ptarmigan_flow.stt.mlx_whisper",
        "ptarmigan_flow.stt.voxtral_mlx",
        "voxmlx",
        "mistral_common",
    ]


def test_is_backend_available_requires_runtime_package_for_backend_variant(monkeypatch) -> None:
    available_modules = {
        "ptarmigan_flow.stt.moonshine",
        "moonshine_voice",
        "ptarmigan_flow.stt.granite_mlx",
        "ptarmigan_flow.stt.mlx_whisper",
        "ptarmigan_flow.stt.voxtral_mlx",
    }

    def fake_find_spec(module_name: str):
        if module_name in available_modules:
            return object()
        return None

    monkeypatch.setattr(availability.importlib.util, "find_spec", fake_find_spec)

    assert availability.is_backend_available("moonshine") is True
    assert availability.is_backend_available("granite") is False
    assert availability.is_backend_available("mlx") is False
    assert availability.is_backend_available("voxtral") is False


def test_is_backend_available_returns_false_when_find_spec_raises(monkeypatch) -> None:
    def broken_find_spec(_module_name: str):
        raise ModuleNotFoundError("missing optional dependency")

    monkeypatch.setattr(availability.importlib.util, "find_spec", broken_find_spec)

    assert availability.is_backend_available("moonshine") is False


def test_available_model_entries_filters_verified_catalog_by_backend(monkeypatch) -> None:
    entries = [
        _Entry(token="moonshine:tiny", backend="moonshine"),
        _Entry(token="granite:ibm-granite/granite-4.0-1b-speech", backend="granite"),
        _Entry(token="mlx:mlx-community/whisper-large-v3-turbo", backend="mlx"),
    ]

    monkeypatch.setattr(availability, "verified_model_entries", lambda: entries)
    monkeypatch.setattr(
        availability,
        "is_backend_available",
        lambda backend: backend in {"moonshine", "mlx"},
    )

    assert availability.available_model_entries() == [entries[0], entries[2]]


def test_available_model_entries_filters_mlx_models_without_runtime_packages(monkeypatch) -> None:
    entries = [
        _Entry(token="moonshine:tiny", backend="moonshine"),
        _Entry(token="moonshine:base", backend="moonshine"),
        _Entry(token="granite:ibm-granite/granite-4.0-1b-speech", backend="granite"),
        _Entry(token="mlx:mlx-community/whisper-large-v3-turbo", backend="mlx"),
        _Entry(token="voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602", backend="voxtral"),
    ]
    available_modules = {
        "ptarmigan_flow.stt.moonshine",
        "moonshine_voice",
        "ptarmigan_flow.stt.granite_mlx",
        "ptarmigan_flow.stt.mlx_whisper",
        "ptarmigan_flow.stt.voxtral_mlx",
    }

    def fake_find_spec(module_name: str):
        if module_name in available_modules:
            return object()
        return None

    monkeypatch.setattr(availability, "verified_model_entries", lambda: entries)
    monkeypatch.setattr(availability.importlib.util, "find_spec", fake_find_spec)

    assert availability.available_model_entries() == entries[:2]
