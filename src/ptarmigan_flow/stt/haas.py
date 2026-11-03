"""Japanese Haas Q4_K backend using an externally installed speech.cpp executable."""

from __future__ import annotations

import json
import math
import tempfile
import threading
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from ptarmigan_flow.ports.runtime import BackendWarmState, format_backend_warm_state
from ptarmigan_flow.stt.base import SpeechToTextBackend
from ptarmigan_flow.stt.haas_artifact import HAAS_MODEL_ID, HAAS_Q4_K_ARTIFACT
from ptarmigan_flow.stt.haas_runtime import SpeechCppRuntime
from ptarmigan_flow.text_processing.interfaces import NoopTextPostProcessor, TextPostProcessor
from ptarmigan_flow.text_processing.normalizer import normalize_transcript_text


@dataclass(frozen=True, slots=True)
class HaasBackendSettings:
    model_id: str = HAAS_MODEL_ID
    language: str = "ja"
    trailing_silence_seconds: float = 1.0
    device: str = "auto"


class HaasSTTBackend(SpeechToTextBackend):
    """One subprocess per utterance; weights are never retained between requests.

    No downloads occur here. Setup supplies an exact HF snapshot filename or
    populates the pinned cache path used by default.
    """

    def __init__(
        self,
        settings: HaasBackendSettings,
        *,
        model_path: str | Path | None = None,
        post_processor: TextPostProcessor | None = None,
    ) -> None:
        if settings.model_id.strip() != HAAS_MODEL_ID:
            raise ValueError(f"Haas supports only {HAAS_MODEL_ID} (Q4_K)")
        self._language = settings.language.strip().lower()
        if self._language not in {"ja", "auto"}:
            raise ValueError("Haas is Japanese-only; use language='ja' (or 'auto')")
        device = settings.device.strip().lower()
        if device not in {"cpu", "auto", "gpu", "mps"}:
            raise ValueError(
                "Haas model.device must be 'cpu', 'auto', 'gpu' or 'mps' "
                "('mps' uses speech.cpp auto device selection)"
            )
        self._device = "auto" if device == "mps" else device
        if not math.isfinite(settings.trailing_silence_seconds):
            raise ValueError("Haas trailing silence must be finite")
        self._trailing_silence = max(0.0, min(1.0, settings.trailing_silence_seconds))
        self._model_path = (
            Path(model_path).expanduser().absolute()
            if model_path is not None
            else HAAS_Q4_K_ARTIFACT.cached_path().absolute()
        )
        self._post_processor = post_processor or NoopTextPostProcessor()
        self._runtime = SpeechCppRuntime()
        self._lock = threading.RLock()
        self._ready = False
        self._last_activity_at_monotonic: float | None = None

    def preflight_model(self) -> str:
        with self._lock:
            generation = self._runtime.generation
            self._preflight(generation)
        return "speech.cpp-haas"

    def _preflight(self, generation: int) -> None:
        if self._ready:
            return
        self._runtime.check_version(generation=generation)
        HAAS_Q4_K_ARTIFACT.verify_file(self._model_path)
        # Exercise layout, weights, and device initialization, discarding probe
        # text. A successful version/metadata check alone cannot establish these.
        self._invoke(np.zeros(3200, dtype=np.float64), 16000, generation)
        self._ready = True

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""
        mono = self._prepare_audio(audio, sample_rate)
        with self._lock:
            generation = self._runtime.generation
            try:
                self._preflight(generation)
                text = self._invoke(mono, sample_rate, generation)
            except Exception:
                self._ready = False
                raise
            self._last_activity_at_monotonic = time.monotonic()
            normalized = normalize_transcript_text(text)
            return self._post_processor.apply(normalized) if normalized else ""

    @staticmethod
    def _prepare_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, Integral)
            or not 0 < sample_rate <= 2147483647
        ):
            raise ValueError("Haas sample rate must be a positive integer WAVE rate")
        if audio.ndim not in {1, 2} or (audio.ndim == 2 and audio.shape[1] == 0):
            raise ValueError("Haas audio must have shape (samples,) or (samples, channels)")
        if not np.issubdtype(audio.dtype, np.number) or np.iscomplexobj(audio):
            raise ValueError("Haas audio must contain real numeric samples")
        if not np.isfinite(audio).all():
            raise ValueError("Haas audio must contain finite samples")
        # Match the other backends: average channels, then clip to PCM range.
        # Let speech.cpp resample instead of adding a Python DSP dependency.
        samples = audio.astype(np.float64)
        try:
            with np.errstate(over="raise", invalid="raise"):
                mono = samples.mean(axis=1) if audio.ndim == 2 else samples
        except FloatingPointError as exc:
            raise ValueError("Haas audio channel average overflowed") from exc
        return np.clip(mono, -1.0, 1.0)

    def _invoke(self, mono: np.ndarray, sample_rate: int, generation: int) -> str:
        pcm = (mono * 32767.0).astype("<i2")
        padding = int(sample_rate * self._trailing_silence)
        if padding:
            pcm = np.concatenate((pcm, np.zeros(padding, dtype="<i2")))
        timeout = min(60.0, max(20.0, 4.0 * len(pcm) / sample_rate + 10.0))
        with tempfile.TemporaryDirectory(prefix="ptarmigan-haas-") as directory:
            path = Path(directory) / "input.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(int(sample_rate))
                wav.writeframes(pcm.tobytes())
            output = self._runtime.run(
                [
                    "asr",
                    str(self._model_path),
                    "--format",
                    "json",
                    "--device",
                    self._device,
                    "--language",
                    self._language,
                    str(path),
                ],
                timeout_seconds=timeout,
                generation=generation,
            )
            try:
                result = json.loads(output)
            except ValueError as exc:
                raise RuntimeError("speech.cpp returned invalid transcription JSON") from exc
            if (
                not isinstance(result, dict)
                or result.get("file") != str(path)
                or not isinstance(result.get("text"), str)
                or result.get("stop") != "complete"
                or "error" in result
            ):
                raise RuntimeError("speech.cpp returned incomplete or invalid transcription output")
            return result["text"]

    def transcribe_stream(self, audio: np.ndarray, sample_rate: int) -> Iterator[str]:
        text = self.transcribe(audio, sample_rate)
        if text:
            yield text

    def cancel_pending_transcription(self) -> None:
        """Stop the current request without preventing the next utterance."""
        self._runtime.cancel()

    def close(self) -> None:
        self._runtime.close()
        with self._lock:
            self._ready = False
            self._last_activity_at_monotonic = None

    def supports_realtime_input(self) -> bool:
        return False

    def warm_state(self) -> BackendWarmState:
        return BackendWarmState(
            resource_mode="subprocess",
            ready=self._ready,
            warmed=False,
            warmup_running=False,
            supports_keydown_warmup=False,
            last_activity_at_monotonic=self._last_activity_at_monotonic,
        )

    def warmup_for_low_latency(self) -> None:
        return None

    def maybe_release_idle_resources(self) -> None:
        return None

    def backend_summary(self) -> str:
        return (
            f"backend=speech.cpp-haas model={HAAS_MODEL_ID} quantization=Q4_K "
            f"language={self._language} device={self._device}"
        )

    def runtime_status(self) -> str:
        return f"{self.backend_summary()} {format_backend_warm_state(self.warm_state())}"
