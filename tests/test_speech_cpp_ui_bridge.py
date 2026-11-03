"""Installer integration probes and bridge effects are entirely injected."""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

from ptarmigan_flow import macos_app
from ptarmigan_flow.config import AppConfig, write_config
from ptarmigan_flow.permissions import PermissionReport
from ptarmigan_flow.web_bridge import BridgeDependencies, WebBridgeDispatcher


def test_bridge_cached_runtime_is_separate_from_models_and_install_has_no_inputs(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path, AppConfig())
    before = path.read_bytes()
    status = {"status": "missing", "install_allowed": True, "operation_id": 3}
    calls = []
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: path,
            check_permissions=lambda: PermissionReport(True, True, True),
            available_model_entries=lambda: [],
            resolve_dictionary_path=lambda: tmp_path / "dictionary.toml",
            login_is_enabled=lambda: False,
            language_was_selected=lambda: True,
            hotkey_was_confirmed=lambda: True,
            speech_cpp_status=lambda: dict(status),
            start_speech_cpp_install=lambda: (
                calls.append("install")
                or {
                    "started": True,
                    "speech_cpp": dict(status, status="installing", operation_id=4),
                }
            ),
        )
    )
    snapshot = dispatcher.handle_action("getState", {})
    assert snapshot["speech_cpp"] == status
    assert snapshot["models"] == []
    assert dispatcher.handle_action("installSpeechCpp", {})["started"]
    for payload in ({"url": "https://invalid"}, {"path": "/tmp/elsewhere"}, {"model": "haas:x"}):
        with pytest.raises(ValueError, match="no arguments"):
            dispatcher.handle_action("installSpeechCpp", payload)
    assert calls == ["install"]
    assert path.read_bytes() == before


@pytest.fixture
def probe(monkeypatch):
    """Use the agreed installer API without depending on its implementation."""
    monkeypatch.delenv("PTARMIGAN_FLOW_SPEECH_CPP_PATH", raising=False)
    installer = types.ModuleType("ptarmigan_flow.stt.speech_cpp_installer")
    installer.managed_speech_cpp_path = lambda: Path("/managed/speech")
    installer.speech_cpp_install_supported = lambda: True
    monkeypatch.setitem(sys.modules, installer.__name__, installer)
    monkeypatch.setattr(macos_app.haas_runtime, "find_speech_executable", lambda: None)

    class Runtime:
        generation = 0
        closed = False
        version = (0, 10, 1)
        error = None

        def __init__(self, *, executable):
            self.executable = executable

        def run(self, args, *, timeout_seconds, generation, cancel_event):
            assert args == ["--version"] and generation == 0
            assert timeout_seconds == 5.0 and not cancel_event.is_set()
            if self.error:
                raise RuntimeError(self.error)
            return f"speech.cpp {'.'.join(map(str, self.version))}, C API 0.4"

        def close(self):
            Runtime.closed = True

    monkeypatch.setattr(macos_app.haas_runtime, "SpeechCppRuntime", Runtime)
    return installer, Runtime


def test_background_probe_missing_unsupported_and_available_sources(probe, monkeypatch):
    installer, runtime = probe
    cancel = threading.Event()
    assert macos_app._speech_cpp_runtime_status(cancel)["status"] == "missing"
    installer.speech_cpp_install_supported = lambda: False
    status = macos_app._speech_cpp_runtime_status(cancel)
    assert status["status"] == "unsupported" and not status["install_allowed"]
    for path, source in (("/external/speech", "external"), ("/managed/speech", "managed")):
        monkeypatch.setattr(
            macos_app.haas_runtime, "find_speech_executable", lambda path=path: Path(path)
        )
        status = macos_app._speech_cpp_runtime_status(cancel)
        assert status["status"] == "available" and status["source"] == source
        assert status["version"] == "0.10.1" and not status["install_allowed"]
    assert runtime.closed


@pytest.mark.parametrize(
    "error,expected", [("speech.cpp 0.9.3 is too old", "old"), ("bad binary", "failed")]
)
def test_background_probe_old_or_failed_external_runtime_can_be_replaced(
    probe,
    monkeypatch,
    error,
    expected,
):
    _, runtime = probe
    monkeypatch.setattr(macos_app.haas_runtime, "find_speech_executable", lambda: Path("/external"))
    runtime.error = error
    status = macos_app._speech_cpp_runtime_status(threading.Event())
    assert status["status"] == expected and status["install_allowed"]
    assert runtime.closed


def test_background_probe_override_never_claims_managed_runtime_is_active(probe, monkeypatch):
    _, runtime = probe
    monkeypatch.setenv("PTARMIGAN_FLOW_SPEECH_CPP_PATH", "/override")
    monkeypatch.setattr(macos_app.haas_runtime, "find_speech_executable", lambda: Path("/override"))
    status = macos_app._speech_cpp_runtime_status(threading.Event())
    assert status["source"] == "override" and not status["install_allowed"]
    runtime.error = "speech.cpp 0.9.3 is too old"
    status = macos_app._speech_cpp_runtime_status(threading.Event())
    assert status["status"] == "override_invalid" and not status["install_allowed"]

    def invalid():
        raise RuntimeError("Invalid PTARMIGAN_FLOW_SPEECH_CPP_PATH")

    monkeypatch.setattr(macos_app.haas_runtime, "find_speech_executable", invalid)
    status = macos_app._speech_cpp_runtime_status(threading.Event())
    assert status["status"] == "override_invalid" and not status["install_allowed"]


def test_cancelled_probe_does_not_execute_version_command(probe, monkeypatch):
    _, runtime = probe
    monkeypatch.setattr(macos_app.haas_runtime, "find_speech_executable", lambda: Path("/external"))
    runtime.error = "must not run"
    cancel = threading.Event()
    cancel.set()
    macos_app._speech_cpp_runtime_status(cancel)
    assert runtime.closed


def test_background_version_probe_receives_termination_cancellation(probe, monkeypatch):
    _, runtime = probe
    entered = threading.Event()
    cancel = threading.Event()

    def run(self, args, *, timeout_seconds, generation, cancel_event):
        assert self.executable == Path("/external")
        assert cancel_event is cancel
        entered.set()
        assert cancel_event.wait(2)
        raise RuntimeError("cancelled")

    monkeypatch.setattr(runtime, "run", run)
    monkeypatch.setattr(macos_app.haas_runtime, "find_speech_executable", lambda: Path("/external"))
    worker = threading.Thread(target=macos_app._speech_cpp_runtime_status, args=(cancel,))
    worker.start()
    assert entered.wait(2)
    cancel.set()
    worker.join(timeout=2)
    assert not worker.is_alive() and runtime.closed
