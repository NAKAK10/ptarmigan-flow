"""LLM-based transcript post-processing."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from moonshine_flow.text_processing.interfaces import TextPostProcessor
from moonshine_flow.text_processing.normalizer import normalize_transcript_text

LOGGER = logging.getLogger(__name__)
_CIRCUIT_BREAK_SECONDS = 30.0
_BASE_SYSTEM_INSTRUCTION = (
    "You are a transcription corrector. "
    "Fix only obvious recognition errors and punctuation. "
    "If the transcript appears truncated at the end, complete only the shortest natural ending. "
    "Do not add new facts, entities, or extra sentences. "
    "Keep the original meaning. Return only corrected text."
)


class LLMClientError(RuntimeError):
    """Raised when the LLM endpoint cannot provide a usable response."""


@dataclass(slots=True)
class LLMCorrectionSettings:
    provider: str
    base_url: str
    model: str
    timeout_seconds: float
    max_input_chars: int
    api_key: str | None
    enabled_tools: bool
    language: str


class LLMClient(Protocol):
    def preflight(self) -> None:
        """Validate endpoint/model availability."""

    def correct(self, text: str) -> str:
        """Return corrected text."""


def _build_system_instruction(language: str) -> str:
    normalized = language.strip() or "en"
    return (
        f"{_BASE_SYSTEM_INSTRUCTION} "
        f"Output language must be '{normalized}'. "
        "Do not switch to another language."
    )


def _http_json_request(
    *,
    method: str,
    url: str,
    timeout_seconds: float,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    body = None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, method=method, headers=headers)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        if detail:
            raise LLMClientError(f"HTTP {exc.code}: {detail}") from exc
        raise LLMClientError(f"HTTP {exc.code}") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise LLMClientError(f"Connection failed: {reason}") from exc
    except TimeoutError as exc:
        raise LLMClientError("Request timed out") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMClientError("Endpoint returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMClientError("Endpoint returned unexpected payload shape")
    return parsed


class OllamaClient:
    def __init__(self, settings: LLMCorrectionSettings) -> None:
        self._base_url = settings.base_url.rstrip("/") + "/"
        self._model = settings.model
        self._timeout_seconds = settings.timeout_seconds
        self._api_key = settings.api_key
        self._system_instruction = _build_system_instruction(settings.language)

    def _url(self, path: str) -> str:
        return urljoin(self._base_url, path.lstrip("/"))

    def preflight(self) -> None:
        payload = _http_json_request(
            method="GET",
            url=self._url("/api/tags"),
            timeout_seconds=self._timeout_seconds,
            api_key=self._api_key,
        )
        models = payload.get("models")
        if not isinstance(models, list):
            return
        names = {
            str(item.get("name", "")).strip()
            for item in models
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }
        if names and self._model not in names:
            raise LLMClientError(f"Model '{self._model}' not found in Ollama")

    def correct(self, text: str) -> str:
        payload = _http_json_request(
            method="POST",
            url=self._url("/api/generate"),
            timeout_seconds=self._timeout_seconds,
            api_key=self._api_key,
            payload={
                "model": self._model,
                "stream": False,
                "prompt": f"{self._system_instruction}\n\nInput:\n{text}\n\nCorrected text:",
                "options": {"temperature": 0},
            },
        )
        response_text = payload.get("response")
        if not isinstance(response_text, str):
            raise LLMClientError("Ollama response is missing 'response' text")
        stripped = response_text.strip()
        if not stripped:
            raise LLMClientError("Ollama returned empty response")
        return stripped


class LMStudioClient:
    def __init__(self, settings: LLMCorrectionSettings) -> None:
        self._base_url = settings.base_url.rstrip("/") + "/"
        self._model = settings.model
        self._timeout_seconds = settings.timeout_seconds
        self._api_key = settings.api_key
        self._system_instruction = _build_system_instruction(settings.language)

    def _url(self, path: str) -> str:
        return urljoin(self._base_url, path.lstrip("/"))

    def preflight(self) -> None:
        payload = _http_json_request(
            method="GET",
            url=self._url("/v1/models"),
            timeout_seconds=self._timeout_seconds,
            api_key=self._api_key,
        )
        models = payload.get("data")
        if not isinstance(models, list):
            return
        names = {
            str(item.get("id", "")).strip()
            for item in models
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
        if names and self._model not in names:
            raise LLMClientError(f"Model '{self._model}' not found in LM Studio")

    def correct(self, text: str) -> str:
        payload = _http_json_request(
            method="POST",
            url=self._url("/v1/chat/completions"),
            timeout_seconds=self._timeout_seconds,
            api_key=self._api_key,
            payload={
                "model": self._model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": self._system_instruction},
                    {"role": "user", "content": text},
                ],
            },
        )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMClientError("LM Studio response is missing 'choices'")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMClientError("LM Studio response has invalid 'choices[0]'")
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                stripped = content.strip()
                if stripped:
                    return stripped
        text_fallback = first.get("text")
        if isinstance(text_fallback, str):
            stripped = text_fallback.strip()
            if stripped:
                return stripped
        raise LLMClientError("LM Studio response is missing output text")


def _build_client(settings: LLMCorrectionSettings) -> LLMClient:
    provider = settings.provider.strip().lower()
    if provider == "ollama":
        return OllamaClient(settings)
    if provider == "lmstudio":
        return LMStudioClient(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.provider}")


class LLMPostProcessor(TextPostProcessor):
    """Best-effort LLM correction with fallback to the original text."""

    def __init__(self, settings: LLMCorrectionSettings) -> None:
        self._settings = settings
        self._client = _build_client(settings)
        self._disabled_until = 0.0

    def preflight(self) -> None:
        self._client.preflight()

    def apply(self, text: str) -> str:
        normalized = normalize_transcript_text(text)
        if not normalized:
            return ""

        now = time.monotonic()
        if now < self._disabled_until:
            return normalized

        limited = normalized[: self._settings.max_input_chars]
        try:
            corrected = self._client.correct(limited)
        except LLMClientError as exc:
            self._disabled_until = now + _CIRCUIT_BREAK_SECONDS
            LOGGER.warning(
                "LLM correction failed; using raw transcript for %.0fs (%s)",
                _CIRCUIT_BREAK_SECONDS,
                exc,
            )
            return normalized

        corrected_normalized = normalize_transcript_text(corrected)
        if not corrected_normalized:
            return normalized
        return corrected_normalized
