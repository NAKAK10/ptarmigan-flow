"""Structured catalog of STT models: verified presets and Hugging Face Hub search."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from ptarmigan_flow.stt.model_families import (
    GRANITE_HF_MODEL_ID,
    VOXTRAL_HF_MODEL_ID,
    WHISPER_HF_MODEL_ID,
)

HUB_SEARCH_BACKENDS = ("granite", "mlx", "voxtral", "vllm")

_HUB_API_URL = "https://huggingface.co/api/models"
_USER_AGENT = "ptarmigan-flow"


@dataclass(slots=True, frozen=True)
class CatalogEntry:
    """One selectable STT model."""

    token: str
    backend: str
    label: str
    description: str
    verified: bool
    source: str


def verified_model_entries() -> list[CatalogEntry]:
    """Return the verified preset models in menu order."""
    return [
        CatalogEntry(
            token="moonshine:tiny",
            backend="moonshine",
            label="Moonshine tiny",
            description="Fastest, smallest footprint; English-focused.",
            verified=True,
            source="preset",
        ),
        CatalogEntry(
            token="moonshine:base",
            backend="moonshine",
            label="Moonshine base",
            description="Lightweight and quick; English-focused.",
            verified=True,
            source="preset",
        ),
        CatalogEntry(
            token=f"granite:{GRANITE_HF_MODEL_ID}",
            backend="granite",
            label="Granite 4.0 1B Speech",
            description="Default choice; good Japanese/English accuracy.",
            verified=True,
            source="preset",
        ),
        CatalogEntry(
            token=f"mlx:{WHISPER_HF_MODEL_ID}",
            backend="mlx",
            label="Whisper large-v3-turbo (MLX)",
            description="Multilingual Whisper; strong general accuracy.",
            verified=True,
            source="preset",
        ),
        CatalogEntry(
            token=f"voxtral:{VOXTRAL_HF_MODEL_ID}",
            backend="voxtral",
            label="Voxtral Mini 4B Realtime",
            description="Realtime-capable multilingual model.",
            verified=True,
            source="preset",
        ),
    ]


def _hub_repo_id(item: dict) -> str:
    for key in ("id", "modelId"):
        repo_id = str(item.get(key, "") or "").strip()
        if repo_id:
            return repo_id
    return ""


def _hub_entry_is_public(item: dict) -> bool:
    if item.get("private"):
        return False
    gated = item.get("gated")
    if gated not in (None, False):
        return False
    return True


def search_hub_models(
    query: str,
    *,
    backend: str,
    limit: int = 20,
    timeout: float = 10.0,
) -> list[CatalogEntry]:
    """Search Hugging Face Hub for ASR models and return them as unverified entries."""
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    params = urlencode(
        {
            "search": query,
            "pipeline_tag": "automatic-speech-recognition",
            "limit": limit,
        }
    )
    url = f"{_HUB_API_URL}?{params}"
    headers = {"User-Agent": _USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, method="GET", headers=headers)

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError(
                "Hugging Face Hub request failed (HTTP 429: rate limited). "
                "Set the HF_TOKEN environment variable to raise the limit."
            ) from exc
        raise RuntimeError(f"Hugging Face Hub request failed (HTTP {exc.code})") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Hugging Face Hub connection failed: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Hugging Face Hub request timed out") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Hugging Face Hub returned invalid JSON") from exc
    if not isinstance(payload, list):
        raise RuntimeError("Hugging Face Hub returned an unexpected payload")

    entries: list[CatalogEntry] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if not _hub_entry_is_public(item):
            continue
        repo_id = _hub_repo_id(item)
        if not repo_id:
            continue
        downloads = item.get("downloads")
        description = "unverified Hub result"
        if isinstance(downloads, int):
            description = f"unverified Hub result; {downloads} downloads"
        entries.append(
            CatalogEntry(
                token=f"{backend}:{repo_id}",
                backend=backend,
                label=repo_id,
                description=description,
                verified=False,
                source="hub",
            )
        )
    return entries
