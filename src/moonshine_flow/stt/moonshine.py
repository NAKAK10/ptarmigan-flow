"""Moonshine STT backend wrapper."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from moonshine_flow.stt.base import SpeechToTextBackend
from moonshine_flow.transcriber import MoonshineTranscriber


class MoonshineSTTBackend(MoonshineTranscriber, SpeechToTextBackend):
    """Moonshine backend with a streaming-compatible interface."""

    def transcribe_stream(self, audio: np.ndarray, sample_rate: int) -> Iterator[str]:
        text = self.transcribe(audio, sample_rate)
        if text:
            yield text

