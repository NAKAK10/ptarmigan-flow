"""Voxtral STT backend implemented with Transformers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from moonshine_flow.stt.base import SpeechToTextBackend
from moonshine_flow.text_processing.interfaces import NoopTextPostProcessor, TextPostProcessor
from moonshine_flow.text_processing.normalizer import normalize_transcript_text

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VoxtralTransformersSettings:
    model_id: str
    language: str
    trailing_silence_seconds: float


class VoxtralTransformersSTTBackend(SpeechToTextBackend):
    """Transcribe audio using `transformers` Voxtral realtime model."""

    def __init__(
        self,
        settings: VoxtralTransformersSettings,
        *,
        post_processor: TextPostProcessor | None = None,
    ) -> None:
        self._settings = settings
        self._post_processor = post_processor or NoopTextPostProcessor()
        self._processor: Any | None = None
        self._model: Any | None = None
        self._target_sample_rate = 16000

    @staticmethod
    def _ensure_dependencies() -> tuple[Any, Any]:
        try:
            from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "transformers with Voxtral support is required for voxtral backend"
            ) from exc

        try:
            import mistral_common  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "mistral-common[audio] is required for voxtral backend"
            ) from exc

        return AutoProcessor, VoxtralRealtimeForConditionalGeneration

    def preflight_model(self) -> str:
        if self._processor is not None and self._model is not None:
            return "voxtral-transformers"

        AutoProcessor, VoxtralRealtimeForConditionalGeneration = self._ensure_dependencies()
        self._processor = AutoProcessor.from_pretrained(self._settings.model_id)
        self._model = self._load_model(VoxtralRealtimeForConditionalGeneration)
        feature_extractor = getattr(self._processor, "feature_extractor", None)
        sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
        self._target_sample_rate = sampling_rate if sampling_rate > 0 else 16000
        return "voxtral-transformers"

    def _ensure_ready(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        self.preflight_model()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""
        self._ensure_ready()
        assert self._processor is not None and self._model is not None

        mono = self._to_mono_float32(audio)
        mono = self._append_trailing_silence(mono, sample_rate=sample_rate)
        if sample_rate != self._target_sample_rate:
            mono = self._resample_linear(
                mono,
                src_rate=sample_rate,
                dst_rate=self._target_sample_rate,
            )

        try:
            inputs = self._processor(
                mono,
                sampling_rate=self._target_sample_rate,
                return_tensors="pt",
            )
        except TypeError:
            inputs = self._processor(
                mono,
                return_tensors="pt",
            )

        model_device = getattr(self._model, "device", None)
        model_dtype = getattr(self._model, "dtype", None)

        to_fn = getattr(inputs, "to", None)
        if callable(to_fn) and model_device is not None:
            try:
                if model_dtype is not None:
                    inputs = inputs.to(model_device, dtype=model_dtype)
                else:
                    inputs = inputs.to(model_device)
            except TypeError:
                inputs = inputs.to(model_device)

        outputs = self._model.generate(**inputs)
        decoded_outputs = self._processor.batch_decode(outputs, skip_special_tokens=True)
        text = decoded_outputs[0] if decoded_outputs else ""
        normalized = normalize_transcript_text(str(text))
        if not normalized:
            return ""
        return self._post_processor.apply(normalized)

    def transcribe_stream(self, audio: np.ndarray, sample_rate: int) -> Iterator[str]:
        text = self.transcribe(audio, sample_rate)
        if text:
            yield text

    @staticmethod
    def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            return np.mean(audio, axis=1).astype(np.float32, copy=False)
        return audio.astype(np.float32, copy=False)

    def _append_trailing_silence(self, audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        trailing = max(0.0, min(1.0, float(self._settings.trailing_silence_seconds)))
        trailing_samples = int(sample_rate * trailing)
        if trailing_samples <= 0:
            return audio
        return np.concatenate((audio, np.zeros(trailing_samples, dtype=np.float32)))

    @staticmethod
    def _resample_linear(audio: np.ndarray, *, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate <= 0 or dst_rate <= 0 or audio.size == 0:
            return audio
        dst_len = int(round(audio.size * (dst_rate / src_rate)))
        if dst_len <= 1:
            return audio
        src_x = np.linspace(0.0, 1.0, num=audio.size, endpoint=True)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=True)
        return np.interp(dst_x, src_x, audio).astype(np.float32, copy=False)

    def _load_model(self, model_class: Any) -> Any:
        try:
            return model_class.from_pretrained(
                self._settings.model_id,
                device_map="auto",
            )
        except Exception as exc:
            if not self._is_accelerate_required_error(exc):
                raise
            LOGGER.warning(
                "accelerate is not installed; loading Voxtral without device_map=auto"
            )
            return model_class.from_pretrained(self._settings.model_id)

    @staticmethod
    def _is_accelerate_required_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "accelerate" in message and "device_map" in message

    def backend_summary(self) -> str:
        return (
            "backend=voxtral-transformers "
            f"model={self._settings.model_id} "
            f"language={self._settings.language}"
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
