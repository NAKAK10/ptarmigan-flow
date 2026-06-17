"""Runtime STT backend helpers with optional child-process isolation."""

from __future__ import annotations

import importlib
import multiprocessing
import os
import signal
import threading
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Literal

import numpy as np

from ptarmigan_flow.config import AppConfig
from ptarmigan_flow.ports.runtime import BackendWarmState, format_backend_warm_state
from ptarmigan_flow.stt.base import SpeechToTextBackend
from ptarmigan_flow.stt.factory import create_stt_backend, parse_stt_model
from ptarmigan_flow.text_processing.interfaces import NoopTextPostProcessor, TextPostProcessor

_LOCAL_BACKEND_PREFIXES = frozenset({"moonshine", "granite", "voxtral", "mlx"})
_CONTROL_TIMEOUT_SECONDS = 5.0
_CHILD_STARTUP_TIMEOUT_SECONDS = 10.0
_PREFLIGHT_TIMEOUT_SECONDS = 600.0
_WARMUP_TIMEOUT_SECONDS = 20.0
_MIN_TRANSCRIPTION_TIMEOUT_SECONDS = 20.0
_MAX_TRANSCRIPTION_TIMEOUT_SECONDS = 60.0
_TRANSCRIPTION_TIMEOUT_MULTIPLIER = 4.0
_TRANSCRIPTION_TIMEOUT_PADDING_SECONDS = 10.0
_POLL_SLICE_SECONDS = 0.1


@dataclass(slots=True, frozen=True)
class STTRecoverySummary:
    """Memory-only summary for a recoverable child timeout/crash."""

    failure_kind: Literal["timeout", "crash"]
    request_kind: str
    request_id: int
    generation: int
    backend_summary: str
    audio_seconds: float | None
    timeout_seconds: float | None
    started_at_monotonic: float
    ended_at_monotonic: float
    warm_state: BackendWarmState | None
    restart_succeeded: bool
    restart_error: str | None = None


class RecoverableSpeechToTextError(RuntimeError):
    """Base class for recoverable STT child failures."""

    def __init__(self, summary: STTRecoverySummary) -> None:
        self.summary = summary
        super().__init__(format_stt_recovery_summary(summary))


class SpeechToTextRequestTimeoutError(RecoverableSpeechToTextError):
    """One STT request exceeded the configured timeout budget."""


class SpeechToTextChildCrashedError(RecoverableSpeechToTextError):
    """The isolated STT child exited while a request was running."""


class RemoteSpeechToTextError(RuntimeError):
    """The isolated STT child returned a normal backend exception."""


def create_runtime_stt_backend(
    config: AppConfig,
    *,
    post_processor: TextPostProcessor | None = None,
) -> SpeechToTextBackend:
    """Create the daemon-facing STT backend.

    Local backends run behind a killable child so the daemon can recover from
    wedged inference. External-server backends keep their existing direct path.
    """

    model_token = str(getattr(getattr(config, "stt", None), "model", "")).strip()
    prefix, model_id = parse_stt_model(model_token)
    if prefix not in _LOCAL_BACKEND_PREFIXES:
        return create_stt_backend(config, post_processor=post_processor)
    return IsolatedSpeechToTextBackend(
        config,
        backend_prefix=prefix,
        model_id=model_id,
        post_processor=post_processor,
    )


def format_stt_recovery_summary(summary: STTRecoverySummary) -> str:
    parts = [f"Recovered from STT {summary.failure_kind}"]
    parts.append(f"request={summary.request_kind}")
    if summary.timeout_seconds is not None:
        parts.append(f"budget={summary.timeout_seconds:.1f}s")
    if summary.audio_seconds is not None:
        parts.append(f"audio={summary.audio_seconds:.2f}s")
    parts.append(f"backend={summary.backend_summary}")
    if summary.restart_succeeded:
        parts.append("child=restart-ok")
    elif summary.restart_error:
        parts.append(f"child=restart-failed:{summary.restart_error}")
    else:
        parts.append("child=restart-skipped")
    return " ".join(parts)


class IsolatedSpeechToTextBackend(SpeechToTextBackend):
    """Proxy a local STT backend through a dedicated child process."""

    def __init__(
        self,
        config: AppConfig,
        *,
        backend_prefix: str,
        model_id: str,
        post_processor: TextPostProcessor | None = None,
        backend_factory_spec: str | None = None,
    ) -> None:
        self._config_payload = _dump_config_payload(config)
        self._backend_prefix = backend_prefix
        self._model_id = model_id
        self._post_processor = post_processor or NoopTextPostProcessor()
        self._backend_factory_spec = backend_factory_spec
        self._ctx = multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._process: multiprocessing.Process | None = None
        self._conn: Connection | None = None
        self._closed = False
        self._generation = 0
        self._next_request_id = 0
        self._supports_realtime_input: bool | None = None
        self._last_known_backend_summary = self._summary_hint(config)
        fallback_warm_state = self._fallback_warm_state()
        self._last_known_runtime_status = (
            "🚀 Backend ready (isolated child): "
            f"{self._last_known_backend_summary} "
            f"{format_backend_warm_state(fallback_warm_state)}"
        )
        self._last_known_warm_state = fallback_warm_state

    def _summary_hint(self, config: AppConfig) -> str:
        language = str(getattr(config, "language", "en")).strip().lower() or "en"
        return (
            "backend=isolated-child "
            f"prefix={self._backend_prefix} "
            f"model={self._model_id} "
            f"language={language}"
        )

    def _fallback_warm_state(self) -> BackendWarmState:
        return BackendWarmState(
            resource_mode="child_process",
            ready=False,
            warmed=False,
            warmup_running=False,
            supports_keydown_warmup=self._backend_prefix == "granite",
            last_activity_at_monotonic=None,
        )

    def preflight_model(self) -> str:
        result = self._request(
            "preflight_model",
            timeout_seconds=_PREFLIGHT_TIMEOUT_SECONDS,
            on_recoverable="raise",
        )
        assert isinstance(result, str)
        return result

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""
        timeout_seconds, audio_seconds = self._transcription_budget(audio, sample_rate)
        result = self._request(
            "transcribe",
            payload={"audio": audio, "sample_rate": sample_rate},
            timeout_seconds=timeout_seconds,
            audio_seconds=audio_seconds,
            on_recoverable="raise",
        )
        assert isinstance(result, str)
        return self._post_processor.apply(result) if result else ""

    def transcribe_stream(self, audio: np.ndarray, sample_rate: int) -> Iterator[str]:
        if audio.size == 0:
            return
        timeout_seconds, audio_seconds = self._transcription_budget(audio, sample_rate)
        result = self._request(
            "transcribe_stream",
            payload={"audio": audio, "sample_rate": sample_rate},
            timeout_seconds=timeout_seconds,
            audio_seconds=audio_seconds,
            on_recoverable="raise",
        )
        assert isinstance(result, list)
        for item in result:
            if not isinstance(item, str):
                continue
            processed = self._post_processor.apply(item) if item else ""
            if processed:
                yield processed

    def warm_state(self) -> BackendWarmState:
        result = self._request(
            "warm_state",
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
            on_recoverable="return_cached",
        )
        if isinstance(result, BackendWarmState):
            return result
        return self._last_known_warm_state

    def warmup_for_low_latency(self) -> None:
        self._request(
            "warmup_for_low_latency",
            timeout_seconds=_WARMUP_TIMEOUT_SECONDS,
            on_recoverable="raise",
        )

    def supports_realtime_input(self) -> bool:
        with self._lock:
            if self._supports_realtime_input is None and not self._closed:
                self._ensure_child_locked()
            return bool(self._supports_realtime_input)

    def maybe_release_idle_resources(self) -> None:
        self._request(
            "maybe_release_idle_resources",
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
            on_recoverable="ignore",
        )

    def runtime_status(self) -> str:
        result = self._request(
            "runtime_status",
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
            on_recoverable="return_cached",
        )
        if isinstance(result, str) and result:
            return result
        return self._last_known_runtime_status

    def backend_summary(self) -> str:
        result = self._request(
            "backend_summary",
            timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
            on_recoverable="return_cached",
        )
        if isinstance(result, str) and result:
            return result
        return self._last_known_backend_summary

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process is None and self._conn is None:
                return
            try:
                self._request_locked(
                    "close",
                    timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
                    on_recoverable="ignore",
                )
            finally:
                self._terminate_child_locked()

    def _transcription_budget(self, audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
        total_samples = int(audio.shape[0]) if audio.ndim else 0
        audio_seconds = 0.0
        if sample_rate > 0 and total_samples > 0:
            audio_seconds = total_samples / float(sample_rate)
        timeout_seconds = audio_seconds * _TRANSCRIPTION_TIMEOUT_MULTIPLIER
        timeout_seconds += _TRANSCRIPTION_TIMEOUT_PADDING_SECONDS
        timeout_seconds = max(_MIN_TRANSCRIPTION_TIMEOUT_SECONDS, timeout_seconds)
        timeout_seconds = min(_MAX_TRANSCRIPTION_TIMEOUT_SECONDS, timeout_seconds)
        return timeout_seconds, audio_seconds

    def _request(
        self,
        method: str,
        *,
        payload: dict[str, object] | None = None,
        timeout_seconds: float | None,
        audio_seconds: float | None = None,
        on_recoverable: Literal["raise", "return_cached", "ignore"],
    ) -> object | None:
        with self._lock:
            if self._closed and method != "close":
                return self._cached_response(method)
            return self._request_locked(
                method,
                payload=payload,
                timeout_seconds=timeout_seconds,
                audio_seconds=audio_seconds,
                on_recoverable=on_recoverable,
            )

    def _request_locked(
        self,
        method: str,
        *,
        payload: dict[str, object] | None = None,
        timeout_seconds: float | None,
        audio_seconds: float | None = None,
        on_recoverable: Literal["raise", "return_cached", "ignore"],
    ) -> object | None:
        self._ensure_child_locked()
        request_id = self._next_request_id
        self._next_request_id += 1
        started_at = time.monotonic()
        request = {
            "request_id": request_id,
            "method": method,
            "payload": payload or {},
        }
        try:
            assert self._conn is not None
            self._conn.send(request)
        except Exception:
            return self._handle_recoverable_locked(
                failure_kind="crash",
                method=method,
                request_id=request_id,
                started_at=started_at,
                timeout_seconds=timeout_seconds,
                audio_seconds=audio_seconds,
                on_recoverable=on_recoverable,
            )

        deadline = None if timeout_seconds is None else started_at + timeout_seconds
        watchdog: threading.Timer | None = None
        if timeout_seconds is not None and self._process is not None:
            child_pid = self._process.pid

            def _emergency_kill(pid: int = child_pid) -> None:
                # recv() can block indefinitely after poll() returns True when the
                # child sends partial data. Kill the child to unblock recv().
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            watchdog = threading.Timer(timeout_seconds + 2.0, _emergency_kill)
            watchdog.daemon = True
            watchdog.start()
        try:
            while True:
                try:
                    message = self._poll_message_locked(deadline=deadline)
                except BrokenPipeError:
                    return self._handle_recoverable_locked(
                        failure_kind="crash",
                        method=method,
                        request_id=request_id,
                        started_at=started_at,
                        timeout_seconds=timeout_seconds,
                        audio_seconds=audio_seconds,
                        on_recoverable=on_recoverable,
                    )
                if message is None:
                    return self._handle_recoverable_locked(
                        failure_kind="timeout",
                        method=method,
                        request_id=request_id,
                        started_at=started_at,
                        timeout_seconds=timeout_seconds,
                        audio_seconds=audio_seconds,
                        on_recoverable=on_recoverable,
                    )
                if message.get("type") == "started":
                    self._update_from_child_metadata(message)
                    continue
                if int(message.get("request_id", -1)) != request_id:
                    continue
                if not bool(message.get("ok", False)):
                    error_message = str(message.get("error_message", "Unknown backend error"))
                    remote_traceback = str(message.get("remote_traceback", "")).strip()
                    if remote_traceback:
                        error_message = f"{error_message}\n{remote_traceback}"
                    raise RemoteSpeechToTextError(error_message)
                self._update_from_child_metadata(message)
                return self._decode_result(method, message.get("result"))
        finally:
            if watchdog is not None:
                watchdog.cancel()

    def _ensure_child_locked(self) -> None:
        if self._closed:
            return
        if self._process is not None and self._process.is_alive() and self._conn is not None:
            return
        self._terminate_child_locked()
        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(
            target=_isolated_backend_child_main,
            args=(child_conn, self._config_payload, self._backend_factory_spec),
            name="ptarmigan-stt-child",
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._process = process
        self._conn = parent_conn
        self._generation += 1
        try:
            started = self._poll_message_locked(
                deadline=time.monotonic() + _CHILD_STARTUP_TIMEOUT_SECONDS
            )
        except BrokenPipeError as exc:
            self._terminate_child_locked()
            raise RuntimeError("Isolated STT child exited before startup completed") from exc
        if started is None:
            self._terminate_child_locked()
            raise RuntimeError("Timed out waiting for isolated STT child to start")
        if started.get("type") != "started":
            self._terminate_child_locked()
            raise RuntimeError("Isolated STT child failed to send startup handshake")
        if not bool(started.get("ok", False)):
            error_message = str(started.get("error_message", "Failed to start isolated STT child"))
            remote_traceback = str(started.get("remote_traceback", "")).strip()
            self._terminate_child_locked()
            if remote_traceback:
                error_message = f"{error_message}\n{remote_traceback}"
            raise RuntimeError(error_message)
        self._update_from_child_metadata(started)

    def _poll_message_locked(self, *, deadline: float | None) -> dict[str, object] | None:
        assert self._process is not None
        assert self._conn is not None
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return None
            wait_seconds = _POLL_SLICE_SECONDS
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                wait_seconds = min(wait_seconds, remaining)
            try:
                if self._conn.poll(wait_seconds):
                    message = self._conn.recv()
                    if isinstance(message, dict):
                        return message
                    raise RemoteSpeechToTextError("Isolated STT child returned a malformed message")
            except EOFError:
                raise BrokenPipeError("Isolated STT child closed its control pipe") from None
            if self._process.exitcode is not None:
                raise BrokenPipeError(
                    f"Isolated STT child exited unexpectedly with code {self._process.exitcode}"
                )

    def _handle_recoverable_locked(
        self,
        *,
        failure_kind: Literal["timeout", "crash"],
        method: str,
        request_id: int,
        started_at: float,
        timeout_seconds: float | None,
        audio_seconds: float | None,
        on_recoverable: Literal["raise", "return_cached", "ignore"],
    ) -> object | None:
        summary = STTRecoverySummary(
            failure_kind=failure_kind,
            request_kind=method,
            request_id=request_id,
            generation=self._generation,
            backend_summary=self._last_known_backend_summary,
            audio_seconds=audio_seconds,
            timeout_seconds=timeout_seconds,
            started_at_monotonic=started_at,
            ended_at_monotonic=time.monotonic(),
            warm_state=self._last_known_warm_state,
            restart_succeeded=False,
        )
        restart_error: str | None = None
        if not self._closed:
            try:
                self._terminate_child_locked()
                self._ensure_child_locked()
            except Exception as exc:  # pragma: no cover - restart failure is rare
                restart_error = str(exc)
            else:
                summary = STTRecoverySummary(
                    failure_kind=summary.failure_kind,
                    request_kind=summary.request_kind,
                    request_id=summary.request_id,
                    generation=summary.generation,
                    backend_summary=summary.backend_summary,
                    audio_seconds=summary.audio_seconds,
                    timeout_seconds=summary.timeout_seconds,
                    started_at_monotonic=summary.started_at_monotonic,
                    ended_at_monotonic=summary.ended_at_monotonic,
                    warm_state=summary.warm_state,
                    restart_succeeded=True,
                )
        if restart_error is not None:
            summary = STTRecoverySummary(
                failure_kind=summary.failure_kind,
                request_kind=summary.request_kind,
                request_id=summary.request_id,
                generation=summary.generation,
                backend_summary=summary.backend_summary,
                audio_seconds=summary.audio_seconds,
                timeout_seconds=summary.timeout_seconds,
                started_at_monotonic=summary.started_at_monotonic,
                ended_at_monotonic=summary.ended_at_monotonic,
                warm_state=summary.warm_state,
                restart_succeeded=False,
                restart_error=restart_error,
            )
        if on_recoverable == "ignore":
            return None
        if on_recoverable == "return_cached":
            return self._cached_response(method)
        if failure_kind == "timeout":
            raise SpeechToTextRequestTimeoutError(summary)
        raise SpeechToTextChildCrashedError(summary)

    def _cached_response(self, method: str) -> object | None:
        if method == "supports_realtime_input":
            return bool(self._supports_realtime_input)
        if method == "warm_state":
            return self._last_known_warm_state
        if method == "runtime_status":
            return self._last_known_runtime_status
        if method == "backend_summary":
            return self._last_known_backend_summary
        return None

    def _update_from_child_metadata(self, message: dict[str, object]) -> None:
        supports_realtime = message.get("supports_realtime_input")
        if isinstance(supports_realtime, bool):
            self._supports_realtime_input = supports_realtime
        backend_summary = message.get("backend_summary")
        if isinstance(backend_summary, str) and backend_summary:
            self._last_known_backend_summary = backend_summary
        runtime_status = message.get("runtime_status")
        if isinstance(runtime_status, str) and runtime_status:
            self._last_known_runtime_status = runtime_status
        warm_state = self._decode_warm_state(message.get("warm_state"))
        if warm_state is not None:
            self._last_known_warm_state = warm_state

    def _decode_result(self, method: str, result: object) -> object | None:
        if method == "warm_state":
            warm_state = self._decode_warm_state(result)
            if warm_state is not None:
                return warm_state
        return result

    def _decode_warm_state(self, payload: object) -> BackendWarmState | None:
        if not isinstance(payload, dict):
            return None
        try:
            return BackendWarmState(
                resource_mode=str(payload["resource_mode"]),
                ready=bool(payload["ready"]),
                warmed=bool(payload["warmed"]),
                warmup_running=bool(payload["warmup_running"]),
                supports_keydown_warmup=bool(payload["supports_keydown_warmup"]),
                last_activity_at_monotonic=(
                    None
                    if payload.get("last_activity_at_monotonic") is None
                    else float(payload["last_activity_at_monotonic"])
                ),
            )
        except Exception:
            return None

    def _terminate_child_locked(self) -> None:
        conn = self._conn
        process = self._process
        self._conn = None
        self._process = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        else:
            process.join(timeout=0.1)


def _isolated_backend_child_main(
    conn: Connection,
    config_payload: dict[str, object],
    backend_factory_spec: str | None,
) -> None:
    backend: SpeechToTextBackend | None = None
    try:
        config = _load_config_payload(config_payload)
        backend = _build_child_backend(config, backend_factory_spec=backend_factory_spec)
        conn.send(
            {
                "type": "started",
                "ok": True,
                "supports_realtime_input": bool(backend.supports_realtime_input()),
                **_child_metadata(backend),
            }
        )
        while True:
            try:
                request = conn.recv()
            except EOFError:
                break
            if not isinstance(request, dict):
                continue
            request_id = int(request.get("request_id", -1))
            method = str(request.get("method", ""))
            payload = request.get("payload")
            try:
                result = _invoke_backend_method(backend, method, payload)
            except Exception as exc:
                conn.send(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                        "remote_traceback": traceback.format_exc(),
                    }
                )
                if method == "close":
                    break
                continue
            conn.send(
                {
                    "request_id": request_id,
                    "ok": True,
                    "result": result,
                    **_child_metadata(backend),
                }
            )
            if method == "close":
                break
    except Exception as exc:
        try:
            conn.send(
                {
                    "type": "started",
                    "ok": False,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                    "remote_traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass
        os._exit(0)


def _build_child_backend(
    config: AppConfig,
    *,
    backend_factory_spec: str | None,
) -> SpeechToTextBackend:
    if backend_factory_spec:
        module_name, attr_name = backend_factory_spec.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, attr_name)
        backend = factory(config)
        if backend is None:
            raise RuntimeError(f"Backend factory {backend_factory_spec} returned None")
        return backend
    return create_stt_backend(config, post_processor=NoopTextPostProcessor())


def _invoke_backend_method(
    backend: SpeechToTextBackend,
    method: str,
    payload: object,
) -> object | None:
    payload_dict = payload if isinstance(payload, dict) else {}
    if method == "preflight_model":
        return backend.preflight_model()
    if method == "transcribe":
        audio = np.asarray(payload_dict["audio"])
        sample_rate = int(payload_dict["sample_rate"])
        return backend.transcribe(audio, sample_rate)
    if method == "transcribe_stream":
        audio = np.asarray(payload_dict["audio"])
        sample_rate = int(payload_dict["sample_rate"])
        return list(backend.transcribe_stream(audio, sample_rate))
    if method == "warm_state":
        return _serialize_warm_state(backend.warm_state())
    if method == "warmup_for_low_latency":
        backend.warmup_for_low_latency()
        return None
    if method == "supports_realtime_input":
        return bool(backend.supports_realtime_input())
    if method == "maybe_release_idle_resources":
        backend.maybe_release_idle_resources()
        return None
    if method == "runtime_status":
        return backend.runtime_status()
    if method == "backend_summary":
        return backend.backend_summary()
    if method == "close":
        backend.close()
        return None
    raise RuntimeError(f"Unsupported isolated STT method: {method}")


def _child_metadata(backend: SpeechToTextBackend) -> dict[str, object]:
    return {
        "backend_summary": backend.backend_summary(),
        "runtime_status": backend.runtime_status(),
        "warm_state": _serialize_warm_state(backend.warm_state()),
    }


def _serialize_warm_state(state: BackendWarmState) -> dict[str, object]:
    return {
        "resource_mode": state.resource_mode,
        "ready": state.ready,
        "warmed": state.warmed,
        "warmup_running": state.warmup_running,
        "supports_keydown_warmup": state.supports_keydown_warmup,
        "last_activity_at_monotonic": state.last_activity_at_monotonic,
    }


def _dump_config_payload(config: AppConfig) -> dict[str, object]:
    dump = getattr(config, "model_dump", None)
    if callable(dump):
        return dump(mode="python")
    export = getattr(config, "dict", None)
    if callable(export):
        return export()
    raise TypeError("AppConfig does not support model export")


def _load_config_payload(payload: dict[str, object]) -> AppConfig:
    validate = getattr(AppConfig, "model_validate", None)
    if callable(validate):
        return validate(payload)
    parse_obj = getattr(AppConfig, "parse_obj", None)
    if callable(parse_obj):
        return parse_obj(payload)
    return AppConfig(**payload)
