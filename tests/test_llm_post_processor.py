from __future__ import annotations

import moonshine_flow.text_processing.llm as llm_module
from moonshine_flow.text_processing.llm import (
    LLMClientError,
    LLMCorrectionSettings,
    LLMPostProcessor,
    LMStudioClient,
    OllamaClient,
)


def _settings(provider: str) -> LLMCorrectionSettings:
    return LLMCorrectionSettings(
        provider=provider,
        base_url="http://localhost:11434",
        model="test-model",
        timeout_seconds=2.5,
        max_input_chars=500,
        api_key=None,
        enabled_tools=False,
        language="ja",
    )


def test_ollama_client_correct_parses_response(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_http_json_request(*, method, url, timeout_seconds, payload=None, api_key=None):
        captured["method"] = method
        captured["url"] = url
        captured["timeout_seconds"] = timeout_seconds
        captured["payload"] = payload
        captured["api_key"] = api_key
        return {"response": " corrected "}

    monkeypatch.setattr(llm_module, "_http_json_request", fake_http_json_request)

    client = OllamaClient(_settings("ollama"))
    corrected = client.correct("input text")

    assert corrected == "corrected"
    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["timeout_seconds"] == 2.5
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "test-model"
    assert payload["stream"] is False
    assert "truncated at the end" in payload["prompt"]
    assert "Output language must be 'ja'" in payload["prompt"]


def test_lmstudio_client_correct_parses_response(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_http_json_request(*, method, url, timeout_seconds, payload=None, api_key=None):
        captured["method"] = method
        captured["url"] = url
        captured["timeout_seconds"] = timeout_seconds
        captured["payload"] = payload
        captured["api_key"] = api_key
        return {"choices": [{"message": {"content": " revised text "}}]}

    monkeypatch.setattr(llm_module, "_http_json_request", fake_http_json_request)

    client = LMStudioClient(_settings("lmstudio"))
    corrected = client.correct("input text")

    assert corrected == "revised text"
    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["timeout_seconds"] == 2.5
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "test-model"
    assert isinstance(payload["messages"], list)
    assert payload["messages"][0]["role"] == "system"
    assert "truncated at the end" in payload["messages"][0]["content"]
    assert "Output language must be 'ja'" in payload["messages"][0]["content"]


def test_llm_post_processor_falls_back_and_short_circuits_after_failure(monkeypatch) -> None:
    class _FailingClient:
        def __init__(self) -> None:
            self.calls = 0

        def preflight(self) -> None:
            return None

        def correct(self, text: str) -> str:
            del text
            self.calls += 1
            raise LLMClientError("boom")

    failing_client = _FailingClient()
    monkeypatch.setattr(llm_module, "_build_client", lambda _settings: failing_client)
    monotonic_values = iter([10.0, 10.1])
    monkeypatch.setattr(llm_module.time, "monotonic", lambda: next(monotonic_values))

    processor = LLMPostProcessor(_settings("ollama"))
    first = processor.apply("  こんにちは  ")
    second = processor.apply("こんにちは")

    assert first == "こんにちは"
    assert second == "こんにちは"
    assert failing_client.calls == 1


def test_llm_post_processor_preflight_delegates_to_client(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.preflight_calls = 0

        def preflight(self) -> None:
            self.preflight_calls += 1

        def correct(self, text: str) -> str:
            return text

    client = _Client()
    monkeypatch.setattr(llm_module, "_build_client", lambda _settings: client)

    processor = LLMPostProcessor(_settings("ollama"))
    processor.preflight()

    assert client.preflight_calls == 1
