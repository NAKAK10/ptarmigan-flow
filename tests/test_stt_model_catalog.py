from __future__ import annotations

import json
import urllib.request
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from ptarmigan_flow.stt.model_catalog import (
    HUB_SEARCH_BACKENDS,
    CatalogEntry,
    search_hub_models,
    verified_model_entries,
)
from ptarmigan_flow.stt.model_families import (
    GRANITE_HF_MODEL_ID,
    VOXTRAL_HF_MODEL_ID,
    WHISPER_HF_MODEL_ID,
)


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _install_fake_urlopen(monkeypatch, payload: object) -> list[urllib.request.Request]:
    captured: list[urllib.request.Request] = []

    def _fake_urlopen(request: urllib.request.Request, timeout: float = 0.0) -> _FakeResponse:
        captured.append(request)
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    return captured


def test_verified_model_entries_tokens_in_order() -> None:
    entries = verified_model_entries()

    assert [entry.token for entry in entries] == [
        "moonshine:tiny",
        "moonshine:base",
        f"granite:{GRANITE_HF_MODEL_ID}",
        f"mlx:{WHISPER_HF_MODEL_ID}",
        f"voxtral:{VOXTRAL_HF_MODEL_ID}",
    ]
    assert all(entry.verified for entry in entries)
    assert all(entry.source == "preset" for entry in entries)
    assert all(entry.description for entry in entries)


def test_hub_search_backends_exclude_moonshine() -> None:
    assert HUB_SEARCH_BACKENDS == ("granite", "mlx", "voxtral", "vllm")


def test_search_hub_models_builds_expected_url(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    captured = _install_fake_urlopen(monkeypatch, [])

    search_hub_models("whisper ja", backend="mlx", limit=7)

    assert len(captured) == 1
    request = captured[0]
    split = urlsplit(request.full_url)
    assert split.scheme == "https"
    assert split.netloc == "huggingface.co"
    assert split.path == "/api/models"
    query = parse_qs(split.query)
    assert query["search"] == ["whisper ja"]
    assert query["pipeline_tag"] == ["automatic-speech-recognition"]
    assert query["limit"] == ["7"]
    assert request.get_header("User-agent") == "ptarmigan-flow"
    assert request.get_header("Authorization") is None


def test_search_hub_models_filters_private_and_gated(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    payload = [
        {"id": "openai/whisper-large-v3-turbo", "private": False, "gated": False,
         "downloads": 123},
        {"id": "secret/private-model", "private": True, "gated": False},
        {"id": "vendor/gated-model", "private": False, "gated": "auto"},
        {"id": "vendor/open-model", "private": False},
        {"id": "", "private": False},
        "not-a-dict",
    ]
    _install_fake_urlopen(monkeypatch, payload)

    entries = search_hub_models("whisper", backend="mlx")

    assert [entry.token for entry in entries] == [
        "mlx:openai/whisper-large-v3-turbo",
        "mlx:vendor/open-model",
    ]
    assert all(isinstance(entry, CatalogEntry) for entry in entries)
    assert all(entry.verified is False for entry in entries)
    assert all(entry.source == "hub" for entry in entries)
    assert all("unverified Hub result" in entry.description for entry in entries)
    assert "123 downloads" in entries[0].description


def test_search_hub_models_adds_authorization_header_when_hf_token_set(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    captured = _install_fake_urlopen(monkeypatch, [])

    search_hub_models("voxtral", backend="voxtral")

    assert captured[0].get_header("Authorization") == "Bearer hf_dummy"


def test_search_hub_models_raises_runtime_error_on_http_error(monkeypatch) -> None:
    def _raise_http_error(request: object, timeout: float = 0.0) -> object:
        raise HTTPError("https://huggingface.co/api/models", 500, "Server Error", None, None)

    monkeypatch.setattr("urllib.request.urlopen", _raise_http_error)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        search_hub_models("whisper", backend="mlx")


def test_search_hub_models_mentions_hf_token_on_rate_limit(monkeypatch) -> None:
    def _raise_http_error(request: object, timeout: float = 0.0) -> object:
        raise HTTPError("https://huggingface.co/api/models", 429, "Too Many Requests", None, None)

    monkeypatch.setattr("urllib.request.urlopen", _raise_http_error)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        search_hub_models("whisper", backend="mlx")
