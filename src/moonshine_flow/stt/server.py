"""Lifecycle manager for local vLLM server processes."""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import urlopen

LOGGER = logging.getLogger(__name__)


def _find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(slots=True)
class VLLMServerConfig:
    host: str = "127.0.0.1"
    startup_timeout_seconds: float = 120.0
    health_poll_interval_seconds: float = 0.5
    log_tail_lines: int = 80


class VLLMServerManager:
    """Start/stop a local vLLM server for one configured model."""

    def __init__(self, config: VLLMServerConfig | None = None) -> None:
        self._config = config or VLLMServerConfig()
        self._process: subprocess.Popen[str] | None = None
        self._model_id: str | None = None
        self._port: int | None = None

    @property
    def endpoint_url(self) -> str:
        if self._port is None:
            raise RuntimeError("vLLM server is not started")
        return f"http://{self._config.host}:{self._port}"

    @property
    def websocket_url(self) -> str:
        endpoint = self.endpoint_url
        return endpoint.replace("http://", "ws://", 1) + "/v1/realtime?intent=transcription"

    def ensure_started(self, model_id: str) -> str:
        if self._process is not None and self._process.poll() is None and self._model_id == model_id:
            return self.endpoint_url
        self.stop()
        self._start(model_id)
        return self.endpoint_url

    def _start(self, model_id: str) -> None:
        port = _find_open_port()
        command = self._build_command(model_id=model_id, port=port)
        LOGGER.info("Starting local vLLM server: %s", " ".join(command))
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._model_id = model_id
        self._port = port
        self._wait_until_ready()

    @staticmethod
    def _build_command(*, model_id: str, port: int) -> list[str]:
        vllm_bin = shutil.which("vllm")
        if vllm_bin:
            return [
                vllm_bin,
                "serve",
                model_id,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
        return [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

    def _wait_until_ready(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self._config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(self._startup_failure_message())
            if self._is_healthy():
                return
            time.sleep(self._config.health_poll_interval_seconds)
        raise RuntimeError("Timed out waiting for local vLLM server to become ready")

    def _is_healthy(self) -> bool:
        for path in ("/health", "/v1/models"):
            try:
                with urlopen(self.endpoint_url + path, timeout=1.0):
                    return True
            except URLError:
                continue
            except TimeoutError:
                continue
            except Exception:
                continue
        return False

    def _startup_failure_message(self) -> str:
        if self._process is None:
            return "vLLM server failed to start"
        stdout = ""
        stderr = ""
        try:
            if self._process.stdout is not None:
                stdout = self._tail_stream(self._process.stdout)
            if self._process.stderr is not None:
                stderr = self._tail_stream(self._process.stderr)
        except Exception:
            pass
        detail_parts = ["Local vLLM server exited before startup completed"]
        if stdout:
            detail_parts.append(f"stdout tail:\n{stdout}")
        if stderr:
            detail_parts.append(f"stderr tail:\n{stderr}")
        return "\n".join(detail_parts)

    def _tail_stream(self, stream) -> str:
        tail: deque[str] = deque(maxlen=self._config.log_tail_lines)
        for line in stream:
            tail.append(line.rstrip("\n"))
        return "\n".join(tail)

    def stop(self) -> None:
        process = self._process
        self._process = None
        self._model_id = None
        self._port = None
        if process is None:
            return
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)

