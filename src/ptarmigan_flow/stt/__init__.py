"""Speech-to-text backends."""

from ptarmigan_flow.stt.base import SpeechToTextBackend
from ptarmigan_flow.stt.factory import create_stt_backend, parse_stt_model
from ptarmigan_flow.stt.model_catalog import (
    HUB_SEARCH_BACKENDS,
    CatalogEntry,
    search_hub_models,
    verified_model_entries,
)

__all__ = [
    "HUB_SEARCH_BACKENDS",
    "CatalogEntry",
    "SpeechToTextBackend",
    "create_stt_backend",
    "parse_stt_model",
    "search_hub_models",
    "verified_model_entries",
]

