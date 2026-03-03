"""Moonshine transcription adapter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from moonshine_flow.text_processing.interfaces import NoopTextPostProcessor, TextPostProcessor
from moonshine_flow.text_processing.normalizer import normalize_transcript_text

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DeviceResolution:
    """Resolved inference device details."""

    requested: str
    actual: str


class MoonshineTranscriber:
    """Transcribe audio with moonshine-voice models."""

    def __init__(
        self,
        model_size: str,
        language: str,
        device: str,
        trailing_silence_seconds: float = 1.0,
        post_processor: TextPostProcessor | None = None,
    ) -> None:
        self.model_size = model_size.strip().lower()
        self.language = language.strip().lower()
        self.device_resolution = self._resolve_device(device)
        self.trailing_silence_seconds = max(0.0, min(1.0, float(trailing_silence_seconds)))

        self._backend = "moonshine-voice"
        self._resolved_language = self._resolve_language(self.language)
        self._resolved_model_arch = self.model_size
        self._resolved_model_path = ""
        self._transcriber: Any | None = None
        self._post_processor = post_processor or NoopTextPostProcessor()

    @staticmethod
    def _resolve_device(requested: str) -> DeviceResolution:
        # moonshine-voice Python API does not expose device selection.
        return DeviceResolution(requested=requested, actual="auto")

    @staticmethod
    def _resolve_language(language: str) -> str:
        if not language:
            return "en"
        return language

    @staticmethod
    def _normalize_audio(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        return np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)

    @staticmethod
    def _stringify_transcript(transcript: Any) -> str:
        lines = getattr(transcript, "lines", None)
        if not lines:
            return ""
        parts = [getattr(line, "text", "").strip() for line in lines]
        text = " ".join(part for part in parts if part).strip()
        return MoonshineTranscriber._normalize_transcript_text(text)

    @staticmethod
    def _normalize_transcript_text(text: str) -> str:
        return normalize_transcript_text(text)

    @staticmethod
    def _resolve_model_arch(model_size: str):
        from moonshine_voice.moonshine_api import ModelArch

        mapping = {
            "tiny": ModelArch.TINY,
            "base": ModelArch.BASE,
        }
        return mapping.get(model_size, ModelArch.BASE)

    def _ensure_transcriber(self) -> None:
        if self._transcriber is not None:
            return
        self.preflight_model()

    def preflight_model(self) -> str:
        """Ensure model is downloaded and backend is ready."""
        from moonshine_voice import get_model_for_language
        from moonshine_voice.transcriber import Transcriber

        wanted_arch = self._resolve_model_arch(self.model_size)

        try:
            model_path, model_arch = get_model_for_language(
                wanted_language=self._resolved_language,
                wanted_model_arch=wanted_arch,
            )
        except ValueError as exc:
            LOGGER.warning(
                "Requested model arch '%s' not available for language '%s'; "
                "using language default model (%s)",
                self.model_size,
                self._resolved_language,
                exc,
            )
            model_path, model_arch = get_model_for_language(
                wanted_language=self._resolved_language,
                wanted_model_arch=None,
            )

        self._resolved_model_path = str(model_path)
        self._resolved_model_arch = str(getattr(model_arch, "name", self.model_size)).lower()
        self._transcriber = Transcriber(model_path=model_path, model_arch=model_arch)

        # Smoke test so startup fails early when model/bootstrap is broken.
        probe = np.zeros(3200, dtype=np.float32)
        transcript = self._transcriber.transcribe_without_streaming(
            probe.tolist(),
            sample_rate=16000,
        )
        self._stringify_transcript(transcript)

        return self._backend

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a mono/stereo waveform to text."""
        if audio.size == 0:
            return ""

        self._ensure_transcriber()
        assert self._transcriber is not None

        normalized = self._normalize_audio(audio)
        trailing_silence_samples = int(sample_rate * self.trailing_silence_seconds)
        if trailing_silence_samples > 0:
            normalized = np.concatenate(
                (normalized, np.zeros(trailing_silence_samples, dtype=np.float32))
            )
        transcript = self._transcriber.transcribe_without_streaming(
            normalized.tolist(),
            sample_rate=sample_rate,
        )
        return self._post_processor.apply(self._stringify_transcript(transcript))

    def backend_summary(self) -> str:
        """Return a short backend summary for diagnostics."""
        return (
            f"backend={self._backend} "
            f"model={self._resolved_model_arch} "
            f"language={self._resolved_language} "
            f"device={self.device_resolution.actual}"
        )

    def close(self) -> None:
        """Release backend resources if they are open."""
        if self._transcriber is None:
            return
        close_fn = getattr(self._transcriber, "close", None)
        if callable(close_fn):
            close_fn()
        self._transcriber = None
