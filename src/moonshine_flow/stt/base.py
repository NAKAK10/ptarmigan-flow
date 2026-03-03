"""Shared interfaces for STT backends."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np


class SpeechToTextBackend(Protocol):
    """Speech-to-text backend protocol."""

    def preflight_model(self) -> str:
        """Ensure backend/model is ready and return backend identifier."""

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return a final transcript for one audio segment."""

    def transcribe_stream(self, audio: np.ndarray, sample_rate: int) -> Iterator[str]:
        """Yield cumulative transcript updates for one audio segment."""

    def backend_summary(self) -> str:
        """Return a short backend summary for diagnostics."""

    def close(self) -> None:
        """Release backend resources."""

