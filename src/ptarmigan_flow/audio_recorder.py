"""Microphone capture helpers."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
import sounddevice as sd

LOGGER = logging.getLogger(__name__)

_BLUETOOTH_INPUT_KEYWORDS = (
    "airpods",
    "bluetooth",
    "hands-free",
    "handsfree",
    "hfp",
    "headset",
)


class AudioRecorder:
    """Record short microphone audio into memory."""

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        dtype: str,
        max_record_seconds: int,
        input_device: int | str | None = None,
        input_device_policy: str = "playback_friendly",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.max_record_seconds = max_record_seconds
        self.input_device = input_device
        self.input_device_policy = input_device_policy

        self._lock = threading.Lock()
        self._frames: list[np.ndarray] = []
        self._recorded_frames = 0
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._max_frames = self.sample_rate * self.max_record_seconds

    @staticmethod
    def _device_get(device: Any, key: str, default: Any) -> Any:
        getter = getattr(device, "get", None)
        if not callable(getter):
            return default
        return getter(key, default)

    @classmethod
    def _is_input_device(cls, device: Any) -> bool:
        return int(cls._device_get(device, "max_input_channels", 0)) > 0

    @classmethod
    def _is_likely_bluetooth_input(cls, device: Any) -> bool:
        name = str(cls._device_get(device, "name", "")).lower()
        return any(keyword in name for keyword in _BLUETOOTH_INPUT_KEYWORDS)

    @staticmethod
    def _is_stream_active_state(stream: sd.InputStream | None) -> bool:
        if stream is None:
            return False

        try:
            closed = getattr(stream, "closed", False)
        except Exception:
            return False
        if isinstance(closed, bool) and closed:
            return False

        try:
            active = getattr(stream, "active", None)
        except Exception:
            return False
        if isinstance(active, bool):
            return active

        try:
            stopped = getattr(stream, "stopped", None)
        except Exception:
            return False
        if isinstance(stopped, bool):
            return not stopped

        # Fallback for stream doubles in tests that do not expose state.
        return True

    def _dispose_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return

        try:
            stream.stop()
        except Exception:
            LOGGER.debug("Stream was not active during stop", exc_info=True)
        finally:
            try:
                stream.close()
            except Exception:
                LOGGER.debug("Stream close failed", exc_info=True)

    @staticmethod
    def _default_input_index() -> int | None:
        try:
            default_device = getattr(sd, "default", None)
            default_pair = getattr(default_device, "device", None)
            if isinstance(default_pair, (list, tuple)) and default_pair:
                default_index = int(default_pair[0])
                if default_index >= 0:
                    return default_index
        except Exception:
            return None
        return None

    def _query_input_devices(self) -> tuple[list[Any], int | None] | None:
        try:
            return list(sd.query_devices()), self._default_input_index()
        except Exception:
            LOGGER.debug("Failed to query audio devices; using system default input", exc_info=True)
            return None

    def _configured_input_device_fallback_label(self) -> str:
        if self.input_device_policy == "playback_friendly":
            return "playback-friendly input selection"
        if self.input_device_policy == "external_preferred":
            return "external-preferred input selection"
        return "system default input"

    def _warn_configured_input_device_query_failed(self) -> None:
        configured = self.input_device
        fallback = self._configured_input_device_fallback_label()
        if isinstance(configured, str):
            LOGGER.warning(
                "Configured input device '%s' could not be resolved because audio device query failed; "
                "falling back to %s",
                configured,
                fallback,
            )
            return
        if isinstance(configured, int):
            LOGGER.warning(
                "Configured input device index %s could not be resolved because audio device query failed; "
                "falling back to %s",
                configured,
                fallback,
            )
            return
        LOGGER.warning(
            "Configured input device %r could not be resolved because audio device query failed; "
            "falling back to %s",
            configured,
            fallback,
        )

    def _resolve_configured_input_device(self, devices: list[Any]) -> int | None:
        configured = self.input_device
        if configured is None:
            return None

        if isinstance(configured, str):
            wanted_name = configured.strip()
            for index, device in enumerate(devices):
                if not self._is_input_device(device):
                    continue
                name = str(self._device_get(device, "name", f"Device {index}")).strip()
                if name == wanted_name:
                    return index
            LOGGER.warning(
                "Configured input device '%s' is unavailable; falling back to %s",
                configured,
                self._configured_input_device_fallback_label(),
            )
            return None

        if isinstance(configured, int):
            if 0 <= configured < len(devices) and self._is_input_device(devices[configured]):
                return configured
            LOGGER.warning(
                "Configured input device index %s is unavailable; falling back to %s",
                configured,
                self._configured_input_device_fallback_label(),
            )
            return None

        LOGGER.warning(
            "Configured input device %r is unsupported; falling back to %s",
            configured,
            self._configured_input_device_fallback_label(),
        )
        return None

    def _resolve_input_device(self) -> int | None:
        device_query = self._query_input_devices()

        if self.input_device is not None:
            if device_query is None:
                self._warn_configured_input_device_query_failed()
                return None
            devices, default_input_index = device_query
            del default_input_index
            resolved_configured = self._resolve_configured_input_device(devices)
            if resolved_configured is not None:
                return resolved_configured

        if self.input_device_policy == "system_default":
            return None

        if device_query is None:
            return None
        devices, default_input_index = device_query

        if self.input_device_policy == "external_preferred":
            for index, device in enumerate(devices):
                if not self._is_input_device(device):
                    continue
                if index == default_input_index:
                    continue
                return index
            return None

        if self.input_device_policy == "playback_friendly":
            default_input = None
            if default_input_index is not None and 0 <= default_input_index < len(devices):
                default_input = devices[default_input_index]
            if default_input is not None and not self._is_likely_bluetooth_input(default_input):
                return None

            for index, device in enumerate(devices):
                if not self._is_input_device(device):
                    continue
                if self._is_likely_bluetooth_input(device):
                    continue
                return index

            LOGGER.warning(
                "Playback-friendly input policy could not find a non-Bluetooth mic; "
                "using system default"
            )
            return None

        LOGGER.warning(
            "Unknown input_device_policy=%s; using system default",
            self.input_device_policy,
        )
        return None

    def _ensure_stream(self) -> None:
        if self._is_stream_active_state(self._stream):
            return
        if self._stream is not None:
            LOGGER.warning("Detected stale audio input stream; reopening input stream")
            self._dispose_stream()

        stream_kwargs: dict[str, Any] = {
            "samplerate": self.sample_rate,
            "channels": self.channels,
            "dtype": self.dtype,
            "callback": self._callback,
            "blocksize": 512,
            "latency": "low",
        }
        resolved_input_device = self._resolve_input_device()
        if resolved_input_device is not None:
            stream_kwargs["device"] = resolved_input_device
            LOGGER.info("Using input device: %s", resolved_input_device)
        else:
            LOGGER.info("Using system default input device")

        self._stream = sd.InputStream(**stream_kwargs)
        self._stream.start()

    @property
    def is_recording(self) -> bool:
        return self._recording

    def is_stream_active(self) -> bool:
        with self._lock:
            stream = self._stream
        return self._is_stream_active_state(stream)

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del time_info
        if status:
            LOGGER.warning("Audio input status: %s", status)

        with self._lock:
            if not self._recording:
                return
            self._frames.append(indata.copy())
            self._recorded_frames += int(frames)
            if self._recorded_frames >= self._max_frames:
                LOGGER.warning("Reached max recording duration (%ss)", self.max_record_seconds)
                self._recording = False
                raise sd.CallbackStop

    def start(self) -> None:
        """Start recording audio into memory."""
        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._recorded_frames = 0
            self._recording = True

        try:
            self._ensure_stream()
            LOGGER.debug("Audio recording started")
        except Exception:
            with self._lock:
                self._recording = False
            self.close()
            raise

    def _drain_before_stop(self) -> None:
        """Give the driver a brief window to deliver buffered tail frames.

        CoreAudio buffers a few frames beyond what has already reached the
        callback; without this, the trailing syllable(s) of an utterance can
        be lost when the stream is torn down immediately on release. Sleep
        for a short, fixed drain window so this never meaningfully delays
        stop().
        """
        time.sleep(0.06)

    def stop(self) -> np.ndarray:
        """Stop recording and return audio samples."""
        self._drain_before_stop()
        self._dispose_stream()

        with self._lock:
            self._recording = False
            if not self._frames:
                self._recorded_frames = 0
                return np.empty((0, self.channels), dtype=self.dtype)
            merged = np.concatenate(self._frames, axis=0)
            self._frames = []
            self._recorded_frames = 0

        LOGGER.debug("Audio recording stopped: %d samples", merged.shape[0])
        return merged

    def snapshot(self) -> np.ndarray:
        """Return currently buffered audio without stopping recording."""
        with self._lock:
            if not self._frames:
                return np.empty((0, self.channels), dtype=self.dtype)
            return np.concatenate(self._frames, axis=0).copy()

    def close(self) -> None:
        """Close active input stream and reset state."""
        with self._lock:
            self._recording = False
            self._frames = []
            self._recorded_frames = 0

        if self._stream is None:
            return
        self._dispose_stream()
