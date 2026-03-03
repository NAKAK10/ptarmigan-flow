"""MLX Whisper STT backend."""

from __future__ import annotations

import os
import tempfile
import wave
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from moonshine_flow.stt.base import SpeechToTextBackend
from moonshine_flow.text_processing.interfaces import NoopTextPostProcessor, TextPostProcessor
from moonshine_flow.text_processing.normalizer import normalize_transcript_text

_TARGET_SAMPLE_RATE = 16000


@dataclass(slots=True)
class MLXWhisperBackendSettings:
    model_id: str
    language: str
    trailing_silence_seconds: float


class MLXWhisperSTTBackend(SpeechToTextBackend):
    """Transcribe audio with mlx-whisper models."""

    def __init__(
        self,
        settings: MLXWhisperBackendSettings,
        *,
        post_processor: TextPostProcessor | None = None,
    ) -> None:
        self._settings = settings
        self._post_processor = post_processor or NoopTextPostProcessor()
        self._ready = False

    @staticmethod
    def _ensure_dependency():
        try:
            import mlx_whisper  # type: ignore
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "mlx-whisper package is required for mlx STT backend"
            ) from exc
        return mlx_whisper

    def preflight_model(self) -> str:
        self._ensure_dependency()
        self._ready = True
        return "mlx-whisper"

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self.preflight_model()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""
        self._ensure_ready()
        mlx_whisper = self._ensure_dependency()
        wav_path = self._prepare_temp_wav(audio, sample_rate=sample_rate)
        try:
            result = mlx_whisper.transcribe(
                wav_path,
                path_or_hf_repo=self._settings.model_id,
                language=self._settings.language,
            )
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        text = ""
        if isinstance(result, dict):
            value = result.get("text")
            if isinstance(value, str):
                text = value
        elif isinstance(result, str):
            text = result

        normalized = normalize_transcript_text(text)
        if not normalized:
            return ""
        return self._post_processor.apply(normalized)

    def transcribe_stream(self, audio: np.ndarray, sample_rate: int) -> Iterator[str]:
        text = self.transcribe(audio, sample_rate)
        if text:
            yield text

    def _prepare_temp_wav(self, audio: np.ndarray, *, sample_rate: int) -> str:
        mono = self._to_mono_float32(audio)
        mono = self._append_trailing_silence(mono, sample_rate=sample_rate)
        if sample_rate != _TARGET_SAMPLE_RATE:
            mono = self._resample_linear(mono, src_rate=sample_rate, dst_rate=_TARGET_SAMPLE_RATE)
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            path = temp.name
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(_TARGET_SAMPLE_RATE)
            wav_file.writeframes(pcm16.tobytes())
        return path

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

    def backend_summary(self) -> str:
        return (
            "backend=mlx-whisper "
            f"model={self._settings.model_id} "
            f"language={self._settings.language}"
        )

    def close(self) -> None:
        return None

