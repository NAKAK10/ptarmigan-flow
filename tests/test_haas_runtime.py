from __future__ import annotations

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ptarmigan_flow.stt import haas_runtime as runtime


def _executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not executable code; never run by these tests")
    path.chmod(0o700)
    return path


@pytest.fixture(autouse=True)
def no_real_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must never launch an installed speech executable")

    monkeypatch.setattr(runtime.subprocess, "Popen", forbidden)
    monkeypatch.delenv(runtime.SPEECH_CPP_PATH_ENV, raising=False)
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: None)
    monkeypatch.setattr(runtime, "_managed_speech_cpp_executable", lambda: None)


def test_discovery_override_is_one_path_and_has_priority(tmp_path, monkeypatch) -> None:
    override = _executable(tmp_path / "custom path;$(ignored)/speech")
    monkeypatch.setenv(runtime.SPEECH_CPP_PATH_ENV, str(override))
    monkeypatch.setattr(runtime.shutil, "which", lambda _: pytest.fail("PATH used before override"))
    assert runtime.find_speech_executable() == override


@pytest.mark.parametrize("value", ["", " ", "/missing/speech", "speech --version"])
def test_invalid_override_does_not_silently_fall_back(value, monkeypatch) -> None:
    monkeypatch.setenv(runtime.SPEECH_CPP_PATH_ENV, value)
    with pytest.raises(RuntimeError, match="Invalid PTARMIGAN_FLOW_SPEECH_CPP_PATH"):
        runtime.find_speech_executable()


def test_nonexecutable_override_rejected(tmp_path, monkeypatch) -> None:
    path = tmp_path / "speech"
    path.write_text("not executable")
    monkeypatch.setenv(runtime.SPEECH_CPP_PATH_ENV, str(path))
    with pytest.raises(RuntimeError, match="Invalid"):
        runtime.find_speech_executable()


def test_path_precedes_gui_fallback(tmp_path, monkeypatch) -> None:
    on_path = _executable(tmp_path / "bin/speech")
    _executable(tmp_path / ".local/bin/speech")
    monkeypatch.setattr(runtime.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: str(on_path))
    assert runtime.find_speech_executable() == on_path


@pytest.mark.parametrize("index", [0, 1, 2, None])
def test_gui_fallback_order_without_shell_path(index, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(runtime.Path, "home", lambda: tmp_path)
    candidates = [
        tmp_path / ".local/bin/speech",
        Path("/opt/homebrew/bin/speech"),
        Path("/usr/local/bin/speech"),
    ]
    usable = set(candidates[index:]) if index is not None else set()
    monkeypatch.setattr(runtime.Path, "is_file", lambda path: path in usable)
    monkeypatch.setattr(runtime.os, "access", lambda path, mode: path in usable)
    assert runtime.find_speech_executable() == (candidates[index] if index is not None else None)


@pytest.mark.parametrize("version", ["0.9.4", "0.9.10", "0.10.0", "1.0.0"])
def test_version_accepts_minimum_and_compares_numerically(version) -> None:
    assert runtime.parse_speech_version(f"speech.cpp {version}, C API 3.7\n") == tuple(
        int(part) for part in version.split(".")
    )


@pytest.mark.parametrize(
    "banner",
    [
        "speech.cpp 0.9.3, C API 3.7",
        "speech.cpp 0.8.99, C API 3.7",
        "speech 9.0.0",
        "0.9.4",
        "",
        "speech.cpp 0.9.4-dev, C API 3.7",
        "speech.cpp 0.9.4, C API 3.7\nextra output",
    ],
)
def test_version_rejects_old_or_unrecognized_banners(banner) -> None:
    with pytest.raises(RuntimeError, match="speech.cpp"):
        runtime.parse_speech_version(banner)


class FakeProcess:
    def __init__(self, argv, *, stdout, stderr, code=0, output=b"ok", diagnostic=b"", **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        stdout.write(output)
        stderr.write(diagnostic)
        self.returncode = None
        self.code = code
        self.killed = False
        self.waits = []

    def wait(self, *, timeout):
        self.waits.append(timeout)
        if self.returncode is None:
            self.returncode = self.code
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def _install_process(monkeypatch, **options):
    processes = []
    monkeypatch.setattr(runtime, "find_speech_executable", lambda: Path("/custom/speech"))

    def popen(argv, **kwargs):
        process = FakeProcess(argv, **kwargs, **options)
        processes.append(process)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    return processes


def test_run_uses_safe_argv_separate_logs_and_finite_wait(monkeypatch) -> None:
    processes = _install_process(monkeypatch, output="日本語".encode(), diagnostic=b"loading")
    runner = runtime.SpeechCppRuntime()
    assert (
        runner.run(["asr", "a path;$(literal).gguf"], timeout_seconds=20, generation=0) == "日本語"
    )
    process = processes[0]
    assert process.argv == ["/custom/speech", "asr", "a path;$(literal).gguf"]
    assert process.kwargs["shell"] is False
    assert process.kwargs["stdin"] == subprocess.DEVNULL
    assert process.kwargs["env"]["SPEECH_NO_UPDATE_CHECK"] == "1"
    assert process.waits[0] == 20
    assert runner._process is None


def test_version_probe_is_bounded(monkeypatch) -> None:
    processes = _install_process(monkeypatch, output=b"speech.cpp 0.9.4, C API 3.7\n")
    assert runtime.SpeechCppRuntime().check_version(generation=0) == (0, 9, 4)
    assert processes[0].argv == ["/custom/speech", "--version"]
    assert processes[0].waits[0] == 5


@pytest.mark.parametrize("code", [1, 2, 3, -6])
def test_failed_process_rejects_even_valid_partial_stdout(code, monkeypatch) -> None:
    _install_process(monkeypatch, code=code, output=b'{"text":"partial"}', diagnostic=b"failure")
    with pytest.raises(RuntimeError, match=f"status {code}: failure"):
        runtime.SpeechCppRuntime().run([], timeout_seconds=1, generation=0)


def test_error_diagnostics_are_bounded_and_decode_invalid_bytes(monkeypatch) -> None:
    _install_process(monkeypatch, code=1, diagnostic=b"x" * 5000 + b"\xfftail")
    with pytest.raises(RuntimeError) as error:
        runtime.SpeechCppRuntime().run([], timeout_seconds=1, generation=0)
    assert len(str(error.value)) < 2100
    assert str(error.value).endswith("�tail")


@pytest.mark.parametrize(
    "output, message",
    [
        (b"\xff", "invalid UTF-8"),
        (b"x" * (1024 * 1024 + 1), "size limit"),
    ],
)
def test_invalid_stdout_fails(output, message, monkeypatch) -> None:
    _install_process(monkeypatch, output=output)
    with pytest.raises(RuntimeError, match=message):
        runtime.SpeechCppRuntime().run([], timeout_seconds=1, generation=0)


def test_missing_and_unlaunchable_runtime(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "find_speech_executable", lambda: None)
    with pytest.raises(RuntimeError, match="executable not found"):
        runtime.SpeechCppRuntime().check_version(generation=0)
    monkeypatch.setattr(runtime, "find_speech_executable", lambda: Path("/bad/speech"))

    def broken(*args, **kwargs):
        raise OSError("incompatible architecture")

    monkeypatch.setattr(runtime.subprocess, "Popen", broken)
    with pytest.raises(RuntimeError, match="Cannot start.*incompatible architecture"):
        runtime.SpeechCppRuntime().check_version(generation=0)


def test_timeout_kills_and_reaps_process_and_allows_next_request(monkeypatch) -> None:
    processes = _install_process(monkeypatch)
    original = FakeProcess.wait

    def wait(self, *, timeout):
        if not self.killed:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return original(self, timeout=timeout)

    monkeypatch.setattr(FakeProcess, "wait", wait)
    runner = runtime.SpeechCppRuntime()
    with pytest.raises(RuntimeError, match="timed out"):
        runner.run([], timeout_seconds=0.01, generation=0)
    assert processes[0].killed
    assert processes[0].returncode == -9
    assert runner._process is None
    monkeypatch.setattr(FakeProcess, "wait", original)
    assert runner.run([], timeout_seconds=1, generation=runner.generation) == "ok"


@pytest.mark.parametrize("close", [False, True])
def test_cancellation_reaps_without_waiting_on_request_lock(close, monkeypatch) -> None:
    started = threading.Event()
    killed = threading.Event()
    processes = _install_process(monkeypatch)
    original_wait = FakeProcess.wait
    original_kill = FakeProcess.kill

    def wait(self, *, timeout):
        started.set()
        if not killed.wait(timeout=timeout):
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return original_wait(self, timeout=timeout)

    def kill(self):
        original_kill(self)
        killed.set()

    monkeypatch.setattr(FakeProcess, "wait", wait)
    monkeypatch.setattr(FakeProcess, "kill", kill)
    runner = runtime.SpeechCppRuntime()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(runner.run, ["asr"], timeout_seconds=2, generation=0)
        assert started.wait(timeout=1)
        (runner.close if close else runner.cancel)()
        with pytest.raises(runtime.SpeechCppCancelledError):
            pending.result(timeout=1)
    assert processes[0].killed
    assert runner._process is None
    with pytest.raises(runtime.SpeechCppCancelledError):
        runner.run([], timeout_seconds=1, generation=0)
    assert len(processes) == 1
    if close:
        runner.close()  # Idempotent, and new requests remain prohibited.
        with pytest.raises(runtime.SpeechCppCancelledError):
            _ = runner.generation
    else:
        assert runner.run([], timeout_seconds=1, generation=runner.generation) == "ok"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_never_starts_process(timeout) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        runtime.SpeechCppRuntime().run([], timeout_seconds=timeout, generation=0)


def test_unreaped_child_remains_owned_and_blocks_another_launch(monkeypatch) -> None:
    processes = _install_process(monkeypatch)
    original_wait = FakeProcess.wait
    original_kill = FakeProcess.kill

    def stuck_wait(self, *, timeout):
        raise subprocess.TimeoutExpired(self.argv, timeout)

    monkeypatch.setattr(FakeProcess, "wait", stuck_wait)
    monkeypatch.setattr(FakeProcess, "kill", lambda self: None)
    runner = runtime.SpeechCppRuntime()
    with pytest.raises(RuntimeError, match="did not exit after being killed"):
        runner.run([], timeout_seconds=1, generation=0)
    assert runner._process is processes[0]
    with pytest.raises(RuntimeError, match="Previous speech.cpp process has not exited"):
        runner.run([], timeout_seconds=1, generation=0)
    assert len(processes) == 1
    monkeypatch.setattr(FakeProcess, "wait", original_wait)
    monkeypatch.setattr(FakeProcess, "kill", original_kill)
    runner.close()
    assert runner._process is None
