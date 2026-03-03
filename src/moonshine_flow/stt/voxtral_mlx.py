"""Voxtral STT backend implemented with voxmlx (MLX)."""

from __future__ import annotations

import os
import tempfile
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from moonshine_flow.stt.base import SpeechToTextBackend
from moonshine_flow.text_processing.interfaces import NoopTextPostProcessor, TextPostProcessor
from moonshine_flow.text_processing.normalizer import normalize_transcript_text

_TARGET_SAMPLE_RATE = 16000
_DEFAULT_MLX_VOXTRAL_MODEL_ID = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"
_HF_VOXTRAL_MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"


@dataclass(slots=True)
class VoxtralMLXSettings:
    model_id: str
    language: str
    trailing_silence_seconds: float


class VoxtralMLXSTTBackend(SpeechToTextBackend):
    """Transcribe audio with voxmlx-backed Voxtral realtime model."""

    def __init__(
        self,
        settings: VoxtralMLXSettings,
        *,
        post_processor: TextPostProcessor | None = None,
    ) -> None:
        self._settings = settings
        self._post_processor = post_processor or NoopTextPostProcessor()
        self._ready = False

        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._generate: Any | None = None
        self._special_token_policy: Any | None = None
        self._prompt_tokens: list[int] = []
        self._n_delay_tokens = 6
        self._resolved_model_id = self._resolve_model_id(settings.model_id)

    @staticmethod
    def _ensure_dependencies() -> tuple[Any, Any, Any]:
        try:
            import voxmlx
            from voxmlx.generate import generate
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("voxmlx package is required for voxtral backend on macOS") from exc

        try:
            from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("mistral-common is required for voxmlx backend") from exc

        return voxmlx, generate, SpecialTokenPolicy

    @staticmethod
    def _resolve_model_id(model_id: str) -> str:
        normalized = model_id.strip()
        if normalized == _HF_VOXTRAL_MODEL_ID:
            return _DEFAULT_MLX_VOXTRAL_MODEL_ID
        return normalized

    @staticmethod
    def _build_prompt_tokens(tokenizer: Any) -> tuple[list[int], int]:
        n_left_pad_tokens = 32
        n_delay_tokens = 6
        streaming_pad = tokenizer.get_special_token("[STREAMING_PAD]")
        prefix_len = n_left_pad_tokens + n_delay_tokens
        tokens = [tokenizer.bos_id] + [streaming_pad] * prefix_len
        return tokens, n_delay_tokens

    def preflight_model(self) -> str:
        if self._ready:
            return "voxtral-mlx"
        voxmlx, generate, special_token_policy = self._ensure_dependencies()
        model, tokenizer, _config = voxmlx.load_model(self._resolved_model_id)
        prompt_tokens, n_delay_tokens = self._build_prompt_tokens(tokenizer)
        self._model = model
        self._tokenizer = tokenizer
        self._generate = generate
        self._special_token_policy = special_token_policy
        self._prompt_tokens = prompt_tokens
        self._n_delay_tokens = n_delay_tokens
        self._ready = True
        return "voxtral-mlx"

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self.preflight_model()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""
        self._ensure_ready()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._generate is not None
        assert self._special_token_policy is not None

        wav_path = self._prepare_temp_wav(audio, sample_rate=sample_rate)
        try:
            output_tokens = self._generate(
                self._model,
                wav_path,
                self._prompt_tokens,
                n_delay_tokens=self._n_delay_tokens,
                temperature=0.0,
                eos_token_id=self._tokenizer.eos_id,
            )
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        text = self._tokenizer.decode(
            output_tokens,
            special_token_policy=self._special_token_policy.IGNORE,
        )
        normalized = normalize_transcript_text(str(text))
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
            "backend=voxtral-mlx "
            f"model={self._resolved_model_id} "
            f"language={self._settings.language}"
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._generate = None
        self._special_token_policy = None
        self._prompt_tokens = []
        self._ready = False
