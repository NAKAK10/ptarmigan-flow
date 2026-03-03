"""vLLM realtime STT backend."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from moonshine_flow.stt.base import SpeechToTextBackend
from moonshine_flow.stt.server import VLLMServerManager
from moonshine_flow.text_processing.interfaces import NoopTextPostProcessor, TextPostProcessor
from moonshine_flow.text_processing.normalizer import normalize_transcript_text

LOGGER = logging.getLogger(__name__)
_TARGET_SAMPLE_RATE = 16000
_CHUNK_DURATION_SECONDS = 0.2


@dataclass(slots=True)
class VLLMRealtimeBackendSettings:
    model_id: str
    language: str
    trailing_silence_seconds: float


class VLLMRealtimeSTTBackend(SpeechToTextBackend):
    """Realtime transcription backend via local vLLM server."""

    def __init__(
        self,
        settings: VLLMRealtimeBackendSettings,
        *,
        server_manager: VLLMServerManager | None = None,
        post_processor: TextPostProcessor | None = None,
    ) -> None:
        self._settings = settings
        self._server_manager = server_manager or VLLMServerManager()
        self._post_processor = post_processor or NoopTextPostProcessor()
        self._ready = False

    def preflight_model(self) -> str:
        self._server_manager.ensure_started(self._settings.model_id)
        self._ready = True
        return "vllm-realtime"

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self.preflight_model()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        final_text = ""
        for update in self.transcribe_stream(audio, sample_rate):
            final_text = update
        normalized = normalize_transcript_text(final_text)
        if not normalized:
            return ""
        return self._post_processor.apply(normalized)

    def transcribe_stream(self, audio: np.ndarray, sample_rate: int) -> Iterator[str]:
        if audio.size == 0:
            return
        self._ensure_ready()
        pcm16 = self._prepare_pcm16(audio, sample_rate=sample_rate)
        if not pcm16:
            return
        cumulative = ""
        for event in self._stream_events(pcm16):
            update = self._event_to_text(event, cumulative=cumulative)
            if update is None:
                if self._is_done_event(event):
                    break
                continue
            normalized = normalize_transcript_text(update)
            if not normalized:
                continue
            if normalized.startswith(cumulative):
                cumulative = normalized
                yield cumulative
            if self._is_done_event(event):
                break

    def _stream_events(self, pcm16: bytes) -> Iterator[dict[str, object]]:
        try:
            from websockets.sync.client import connect
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "websockets package is required for vLLM realtime backend"
            ) from exc

        ws_url = self._server_manager.websocket_url
        with connect(ws_url, open_timeout=15.0) as websocket:
            self._send_session_start(websocket)
            chunk_bytes = int(_TARGET_SAMPLE_RATE * _CHUNK_DURATION_SECONDS) * 2
            for offset in range(0, len(pcm16), chunk_bytes):
                payload = base64.b64encode(pcm16[offset : offset + chunk_bytes]).decode("ascii")
                websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": payload,
                        }
                    )
                )
            websocket.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            while True:
                raw = websocket.recv()
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    LOGGER.debug("Ignoring non-JSON realtime payload: %r", raw)
                    continue
                if isinstance(event, dict):
                    yield event
                    if self._is_done_event(event):
                        break

    def _send_session_start(self, websocket) -> None:
        websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": self._settings.model_id,
                            "language": self._settings.language,
                        },
                    },
                }
            )
        )

    @staticmethod
    def _event_to_text(event: dict[str, object], *, cumulative: str) -> str | None:
        event_type = str(event.get("type", ""))
        if "error" in event_type.lower():
            message = event.get("error") or event.get("message") or event
            raise RuntimeError(f"Realtime transcription error: {message}")

        delta = event.get("delta")
        if isinstance(delta, str):
            return cumulative + delta
        transcript = event.get("transcript")
        if isinstance(transcript, str):
            return transcript

        item = event.get("item")
        if isinstance(item, dict):
            transcript = item.get("transcript")
            if isinstance(transcript, str):
                return transcript
        return None

    @staticmethod
    def _is_done_event(event: dict[str, object]) -> bool:
        event_type = str(event.get("type", "")).lower()
        return event_type in {
            "transcription.done",
            "response.audio_transcript.done",
            "conversation.item.input_audio_transcription.completed",
            "response.done",
        }

    def _prepare_pcm16(self, audio: np.ndarray, *, sample_rate: int) -> bytes:
        mono = self._to_mono_float32(audio)
        mono = self._append_trailing_silence(mono, sample_rate=sample_rate)
        if sample_rate != _TARGET_SAMPLE_RATE:
            mono = self._resample_linear(mono, src_rate=sample_rate, dst_rate=_TARGET_SAMPLE_RATE)
        clipped = np.clip(mono, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return pcm16.tobytes()

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
            "backend=vllm-realtime "
            f"model={self._settings.model_id} "
            f"language={self._settings.language}"
        )

    def close(self) -> None:
        self._server_manager.stop()

