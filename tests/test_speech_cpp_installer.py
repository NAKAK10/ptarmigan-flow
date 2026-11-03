from __future__ import annotations

import hashlib
import io
import multiprocessing
import ssl
import stat
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptarmigan_flow.stt import haas_runtime as runtime
from ptarmigan_flow.stt import speech_cpp_installer as installer


def macho():
    return struct.pack("<8I", 0xFEEDFACF, 0x0100000C, 0, 2, 0, 0, 0, 0) + b"x" * 96


def member(name, payload=b"ignored", mode=stat.S_IFREG | 0o7777):
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = mode << 16
    return info, payload


def archive_bytes(entries=None):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for info, payload in (
            entries
            if entries is not None
            else [
                member("speech", macho()),
                member("libspeech.4.dylib"),
                member("libspeech.dylib", b"libspeech.4.dylib", stat.S_IFLNK | 0o777),
                member("speech.h"),
            ]
        ):
            bundle.writestr(info, payload)
    return output.getvalue()


class Response(io.BytesIO):
    status = 200

    def __init__(self, payload, *, headers=None, url=installer._ASSET_URL):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))} if headers is None else headers
        self.url = url

    def geturl(self):
        return self.url


class Process:
    output = b"speech.cpp 0.10.1, C API 4.0\n"

    def __init__(self, argv, *, stdout, stderr, **kwargs):
        assert argv[1:] == ["--version"]
        path = Path(argv[0])
        assert path.parent.name.startswith(".extract-")
        assert path.read_bytes() == macho()
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
        assert list(path.parent.iterdir()) == [path]
        assert kwargs["shell"] is False
        assert kwargs["env"]["SPEECH_NO_UPDATE_CHECK"] == "1"
        stdout.write(self.output)
        self.returncode = None
        self.killed = False

    def wait(self, *, timeout):
        assert 0 < timeout <= 5
        self.returncode = 0 if not self.killed else -9
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    open_download = installer._open_download
    slot = threading.BoundedSemaphore(1)
    monkeypatch.setattr(installer, "_NETWORK_SLOT", slot)
    monkeypatch.setattr(installer.Path, "home", lambda: tmp_path)
    monkeypatch.delenv(runtime.SPEECH_CPP_PATH_ENV, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected real network or executable invocation")

    monkeypatch.setattr(installer, "_open_download", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    yield open_download
    assert slot.acquire(timeout=1), "Fake network worker did not stop"
    slot.release()


@pytest.fixture
def release(monkeypatch):
    monkeypatch.setattr(installer, "speech_cpp_install_supported", lambda: True)
    monkeypatch.setattr(installer, "_EXECUTABLE_BYTES", len(macho()))
    monkeypatch.setattr(subprocess, "Popen", Process)

    def set_archive(payload=None, **response_kwargs):
        payload = archive_bytes() if payload is None else payload
        monkeypatch.setattr(installer, "_ARCHIVE_BYTES", len(payload))
        monkeypatch.setattr(installer, "_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())

        def open_download(timeout):
            assert 0 < timeout <= installer._SOCKET_TIMEOUT_SECONDS
            return Response(payload, **response_kwargs)

        monkeypatch.setattr(installer, "_open_download", open_download)
        return payload

    set_archive()
    return set_archive


def old_install():
    path = installer.managed_speech_cpp_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"previous executable")
    path.chmod(0o700)
    return path


def assert_preserved(path):
    assert path.read_bytes() == b"previous executable"
    assert sorted(p.name for p in path.parent.iterdir()) == [".install.lock", "speech"]


def test_pinned_identity_and_path(tmp_path):
    assert installer._VERSION == "0.10.1"
    assert installer._ASSET_URL == (
        "https://github.com/nyosegawa/speech.cpp/releases/download/v0.10.1/"
        "speech-0.10.1-macos-arm64-metal.zip"
    )
    assert installer._ARCHIVE_SHA256 == (
        "ff24f6f58c9d2dedb01caa88ba043ae845c76eff228e4cbfae5de749174404c1"
    )
    assert installer._ARCHIVE_BYTES == 6087757
    assert installer._EXECUTABLE_BYTES == 11588712
    assert installer.managed_speech_cpp_path() == (
        tmp_path
        / "Library/Application Support/ptarmigan-flow/runtimes/speech.cpp"
        / "0.10.1/macos-arm64-metal/speech"
    )
    assert list(tmp_path.iterdir()) == []


def test_install_only_regular_speech_with_controlled_permissions_and_progress(release):
    progress = []
    threads = []

    def observe(value):
        progress.append(value)
        threads.append(threading.get_ident())

    path = installer.install_speech_cpp(observe)
    assert path == installer.managed_speech_cpp_path()
    assert path.read_bytes() == macho()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert sorted(p.name for p in path.parent.iterdir()) == [".install.lock", "speech"]
    assert list(dict.fromkeys(p["status"] for p in progress)) == [
        "downloading",
        "verifying",
        "installing",
        "complete",
    ]
    assert progress[0]["downloaded_bytes"] == 0
    assert all(p["total_bytes"] == installer._ARCHIVE_BYTES for p in progress)
    counts = [p["downloaded_bytes"] for p in progress]
    assert counts == sorted(counts)
    assert counts[-1] == installer._ARCHIVE_BYTES
    assert set(threads) == {threading.get_ident()}
    assert runtime.find_speech_executable() == path


def test_atomic_replace_and_old_copy_untouched_until_commit(release, monkeypatch):
    path = old_install()
    replace = installer.os.replace
    calls = []

    def atomic(source, destination, **kwargs):
        assert_preserved_before = path.read_bytes()
        assert assert_preserved_before == b"previous executable"
        assert Path(source).parent != path.parent
        assert destination == "speech"
        assert "dst_dir_fd" in kwargs
        calls.append(source)
        replace(source, destination, **kwargs)

    monkeypatch.setattr(installer.os, "replace", atomic)
    assert installer.install_speech_cpp() == path
    assert path.read_bytes() == macho()
    assert len(calls) == 1


def test_replace_failure_preserves_old_copy(release, monkeypatch):
    path = old_install()

    def fail(*args, **kwargs):
        raise PermissionError("replace denied")

    monkeypatch.setattr(installer.os, "replace", fail)
    with pytest.raises(installer.SpeechCppInstallError, match="replace denied"):
        installer.install_speech_cpp()
    assert_preserved(path)


def test_hash_verified_before_zip_or_validation(release, monkeypatch):
    path = old_install()
    release(b"not even a ZIP")
    monkeypatch.setattr(installer, "_ARCHIVE_SHA256", "0" * 64)
    monkeypatch.setattr(installer.zipfile, "ZipFile", lambda _: pytest.fail("Untrusted ZIP opened"))
    monkeypatch.setattr(installer, "_validate_macho", lambda _: pytest.fail("Untrusted validation"))
    with pytest.raises(installer.SpeechCppInstallError, match="SHA-256 mismatch"):
        installer.install_speech_cpp()
    assert_preserved(path)


@pytest.mark.parametrize("kind", ["truncated", "oversized", "wrong-header", "encoding", "status"])
def test_network_size_and_response_bounds(release, monkeypatch, kind):
    path = old_install()
    payload = release()
    if kind == "truncated":
        monkeypatch.setattr(installer, "_ARCHIVE_BYTES", len(payload) + 1)
        response = Response(payload, headers={})
        error = "truncated"
    elif kind == "oversized":
        monkeypatch.setattr(installer, "_ARCHIVE_BYTES", len(payload) - 1)
        response = Response(payload, headers={})
        error = "size limit"
    elif kind == "wrong-header":
        response = Response(payload, headers={"Content-Length": "999999"})
        error = "wrong size"
    elif kind == "encoding":
        response = Response(payload, headers={"Content-Encoding": "gzip"})
        error = "encoding"
    else:
        response = Response(payload)
        response.status = 206
        error = "HTTP 200"
    monkeypatch.setattr(installer, "_open_download", lambda _: response)
    with pytest.raises(installer.SpeechCppInstallError, match=error):
        installer.install_speech_cpp()
    assert_preserved(path)


def test_network_error_is_actionable_and_preserves_previous(release, monkeypatch):
    path = old_install()

    def fail(_):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(installer, "_open_download", fail)
    with pytest.raises(installer.SpeechCppInstallError, match="offline.*network.*retry"):
        installer.install_speech_cpp()
    assert_preserved(path)


@pytest.mark.parametrize(
    "path", ["../speech", "/speech", "a/../../speech", "a\\speech", "a//b", "a/./b", "C:speech"]
)
def test_reject_traversal_anywhere_even_ignored_members(release, path):
    previous = old_install()
    release(archive_bytes([member("speech", macho()), member(path)]))
    with pytest.raises(installer.SpeechCppInstallError, match="member path"):
        installer.install_speech_cpp()
    assert_preserved(previous)


def test_duplicate_zip_members_rejected(release):
    with pytest.warns(UserWarning, match="Duplicate name"):
        release(archive_bytes([member("speech", macho()), member("speech", macho())]))
    with pytest.raises(installer.SpeechCppInstallError, match="duplicate"):
        installer.install_speech_cpp()


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o700, 0o700])
def test_speech_must_be_explicit_regular_file(release, mode):
    release(archive_bytes([member("speech", macho(), mode)]))
    with pytest.raises(installer.SpeechCppInstallError, match="regular file"):
        installer.install_speech_cpp()


@pytest.mark.parametrize("kind", ["missing", "nested", "size", "count", "expanded", "badzip"])
def test_archive_bounds_and_layout(release, monkeypatch, kind):
    if kind == "missing":
        release(archive_bytes([member("speech.h")]))
        error = "missing"
    elif kind == "nested":
        release(archive_bytes([member("bin/speech", macho())]))
        error = "missing"
    elif kind == "size":
        release(archive_bytes([member("speech", b"short")]))
        error = "executable size"
    elif kind == "count":
        release(archive_bytes([member("speech", macho()), *[member(str(i)) for i in range(16)]]))
        error = "member count"
    elif kind == "expanded":
        monkeypatch.setattr(installer, "_MAX_EXPANDED_BYTES", 127)
        error = "expanded size"
    else:
        release(b"hash-trusted invalid ZIP")
        error = "not a zip file"
    path = old_install()
    with pytest.raises(installer.SpeechCppInstallError, match=error):
        installer.install_speech_cpp()
    assert_preserved(path)


@pytest.mark.parametrize(
    "header",
    [
        b"ELF" * 43,
        struct.pack("<8I", 0xFEEDFACF, 0x01000007, 0, 2, 0, 0, 0, 0) + b"x" * 96,
        struct.pack("<8I", 0xFEEDFACF, 0x0100000C, 0, 6, 0, 0, 0, 0) + b"x" * 96,
    ],
)
def test_macho_validation_before_execute(release, monkeypatch, header):
    release(archive_bytes([member("speech", header)]))
    monkeypatch.setattr(installer, "_EXECUTABLE_BYTES", len(header))
    monkeypatch.setattr(installer, "_probe", lambda *args: pytest.fail("Invalid binary executed"))
    path = old_install()
    with pytest.raises(installer.SpeechCppInstallError, match="Mach-O"):
        installer.install_speech_cpp()
    assert_preserved(path)


@pytest.mark.parametrize(
    "output",
    [b"speech.cpp 0.10.0, C API 4.0", b"speech.cpp 0.10.2, C API 4.0", b"other program", b"\xff"],
)
def test_probe_requires_exact_version(release, monkeypatch, output):
    monkeypatch.setattr(Process, "output", output)
    path = old_install()
    with pytest.raises(installer.SpeechCppInstallError):
        installer.install_speech_cpp()
    assert_preserved(path)


@pytest.mark.parametrize("status", ["before-start", "downloading", "verifying", "installing"])
def test_cancellation_preserves_old_executable(release, status):
    path = old_install()
    cancel = threading.Event()
    if status == "before-start":
        cancel.set()

    def progress(value):
        if value["status"] == status:
            cancel.set()

    with pytest.raises(installer.SpeechCppInstallCancelledError, match="cancelled"):
        installer.install_speech_cpp(progress, cancel)
    assert path.read_bytes() == b"previous executable"
    assert not list(path.parent.glob(".download-*"))
    assert not list(path.parent.glob(".extract-*"))


@pytest.mark.parametrize("cancelled", [False, True])
def test_stalled_network_caller_is_bounded_and_cancellable(release, monkeypatch, cancelled):
    started = threading.Event()
    unblock = threading.Event()
    finished = threading.Event()
    cancel = threading.Event()
    path = old_install()

    def stall(_):
        started.set()
        try:
            assert unblock.wait(3)
            return Response(archive_bytes())
        finally:
            finished.set()

    monkeypatch.setattr(installer, "_open_download", stall)
    monkeypatch.setattr(installer, "_INSTALL_TIMEOUT_SECONDS", 0.15 if not cancelled else 3)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(installer.install_speech_cpp, None, cancel)
            assert started.wait(1)
            if cancelled:
                cancel.set()
            error = (
                installer.SpeechCppInstallCancelledError
                if cancelled
                else installer.SpeechCppInstallError
            )
            with pytest.raises(error, match="cancelled" if cancelled else "timed out"):
                pending.result(timeout=1)
        assert_preserved(path)
    finally:
        unblock.set()
        assert finished.wait(1)


def test_cancellation_during_real_runtime_probe_kills_and_reaps(release, monkeypatch):
    started = threading.Event()
    cancel = threading.Event()
    processes = []

    class Hanging(Process):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            processes.append(self)

        def wait(self, *, timeout):
            if self.killed:
                return -9
            started.set()
            time.sleep(min(0.01, timeout))
            raise subprocess.TimeoutExpired("fake speech", timeout)

    monkeypatch.setattr(subprocess, "Popen", Hanging)
    path = old_install()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(installer.install_speech_cpp, None, cancel)
        assert started.wait(1)
        cancel.set()
        with pytest.raises(installer.SpeechCppInstallCancelledError):
            pending.result(timeout=1)
    assert len(processes) == 1
    assert processes[0].killed
    assert_preserved(path)


def test_probe_timeout_preserves_old_copy(release, monkeypatch):
    processes = []

    class Hanging(Process):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            processes.append(self)

        def wait(self, *, timeout):
            if self.killed:
                return -9
            time.sleep(min(0.01, timeout))
            raise subprocess.TimeoutExpired("fake speech", timeout)

    monkeypatch.setattr(subprocess, "Popen", Hanging)
    monkeypatch.setattr(installer, "_PROBE_TIMEOUT_SECONDS", 0.05)
    path = old_install()
    with pytest.raises(installer.SpeechCppInstallError, match="timed out"):
        installer.install_speech_cpp()
    assert processes[0].killed
    assert_preserved(path)


@pytest.mark.parametrize("target", ["speech", "macos-arm64-metal", "speech.cpp", ".install.lock"])
def test_reject_symlink_destination_components_and_lock(release, tmp_path, target):
    destination = installer.managed_speech_cpp_path()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim"
    victim.write_bytes(b"untouched")
    if target in {"speech", ".install.lock"}:
        destination.parent.mkdir(parents=True)
        (destination.parent / target).symlink_to(victim)
    else:
        component = next(p for p in destination.parents if p.name == target)
        component.parent.mkdir(parents=True)
        component.symlink_to(outside, target_is_directory=True)
    with pytest.raises(installer.SpeechCppInstallError):
        installer.install_speech_cpp()
    assert victim.read_bytes() == b"untouched"
    assert list(outside.iterdir()) == [victim]


def test_refuse_destination_changed_to_symlink_during_probe(release, monkeypatch, tmp_path):
    path = old_install()
    victim = tmp_path / "victim"
    victim.write_bytes(b"untouched")

    def change(*args):
        path.unlink()
        path.symlink_to(victim)

    monkeypatch.setattr(installer, "_probe", change)
    with pytest.raises(installer.SpeechCppInstallError, match="symlink"):
        installer.install_speech_cpp()
    assert victim.read_bytes() == b"untouched"


def hold_lock(parent, connection):
    with installer._directory(Path(parent), create=False) as descriptor:
        with installer._install_lock(descriptor):
            connection.send("locked")
            connection.recv()


def test_interprocess_lock_and_recovery_after_release(release):
    path = old_install()
    context = multiprocessing.get_context("spawn")
    connection, child = context.Pipe()
    process = context.Process(target=hold_lock, args=(str(path.parent), child))
    process.start()
    try:
        assert connection.poll(5)
        assert connection.recv() == "locked"
        with pytest.raises(installer.SpeechCppInstallError, match="Another app copy"):
            installer.install_speech_cpp()
        assert_preserved(path)
        connection.send("release")
        process.join(5)
        assert process.exitcode == 0
        assert installer.install_speech_cpp() == path
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        child.close()
        connection.close()


@pytest.mark.parametrize(
    "system,machine,physical,expected",
    [
        ("Darwin", "arm64", None, True),
        ("Darwin", "x86_64", b"1\n", True),
        ("Darwin", "x86_64", b"0\n", False),
        ("Darwin", "x86_64", b"", False),
        ("Darwin", "x86_64", "timeout", False),
        ("Darwin", "x86_64", "missing", False),
        ("Darwin", "ppc", None, False),
        ("Linux", "arm64", None, False),
    ],
)
def test_intel_and_rosetta_detection(monkeypatch, system, machine, physical, expected):
    monkeypatch.setattr(installer.platform, "system", lambda: system)
    monkeypatch.setattr(installer.platform, "machine", lambda: machine)
    calls = []

    def sysctl(argv, **kwargs):
        calls.append(argv)
        assert argv == ["/usr/sbin/sysctl", "-n", "hw.optional.arm64"]
        assert kwargs["timeout"] == 1
        assert kwargs["shell"] is False
        if physical == "timeout":
            raise subprocess.TimeoutExpired(argv, 1)
        if physical == "missing":
            raise FileNotFoundError("sysctl")
        return SimpleNamespace(returncode=0, stdout=physical)

    monkeypatch.setattr(subprocess, "run", sysctl)
    assert installer.speech_cpp_install_supported() is expected
    assert bool(calls) == (system == "Darwin" and machine == "x86_64")


def test_unsupported_has_actionable_intel_error_and_creates_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(installer, "speech_cpp_install_supported", lambda: False)
    with pytest.raises(installer.SpeechCppInstallError, match="No official Intel.*PTARMIGAN_FLOW"):
        installer.install_speech_cpp()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/asset",
        "https://evil.example/a",
        "https://github.com.evil.example/a",
        "https://user:pass@github.com/a",
        "https://github.com:123/a",
        "file:///tmp/a",
    ],
)
def test_unsafe_redirects_rejected(url):
    handler = installer._Redirects()
    request = urllib.request.Request(installer._ASSET_URL)
    with pytest.raises(installer.SpeechCppInstallError, match="HTTPS"):
        handler.redirect_request(request, None, 302, "Found", {}, url)


def test_redirect_limit_and_https_asset_allowed():
    handler = installer._Redirects()
    request = urllib.request.Request(installer._ASSET_URL)
    url = "https://release-assets.githubusercontent.com/signed-asset?signature=test"
    for _ in range(installer._MAX_REDIRECTS):
        assert handler.redirect_request(request, None, 302, "Found", {}, url).full_url == url
    with pytest.raises(installer.SpeechCppInstallError, match="Too many"):
        handler.redirect_request(request, None, 302, "Found", {}, url)


def test_frozen_ca_context_uses_certifi_and_has_stdlib_fallback(monkeypatch):
    import sys

    calls = []
    context = object()
    monkeypatch.setattr(
        ssl, "create_default_context", lambda **kwargs: calls.append(kwargs) or context
    )
    monkeypatch.setitem(sys.modules, "certifi", SimpleNamespace(where=lambda: "/frozen/cacert.pem"))
    assert installer._ssl_context() is context
    assert calls == [{"cafile": "/frozen/cacert.pem"}]
    monkeypatch.setitem(sys.modules, "certifi", None)
    assert installer._ssl_context() is context
    assert calls[-1] == {}


def test_callback_errors_cannot_fail_a_committed_install(release):
    statuses = []

    def broken(value):
        statuses.append(value["status"])
        raise ValueError("observer failed")

    assert installer.install_speech_cpp(broken).is_file()
    assert statuses[-1] == "complete"


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"external executable, never invoked")
    path.chmod(0o700)
    return path


def test_discovery_override_precedes_managed_and_external(tmp_path, monkeypatch):
    managed = old_install()
    override = executable(tmp_path / "custom/speech")
    external = executable(tmp_path / "bin/speech")
    monkeypatch.setenv(runtime.SPEECH_CPP_PATH_ENV, str(override))
    monkeypatch.setattr(runtime.shutil, "which", lambda _: str(external))
    assert runtime.find_speech_executable() == override
    assert managed.read_bytes() == b"previous executable"


@pytest.mark.parametrize("value", ["", " ", "/missing/speech", "speech --version"])
def test_invalid_override_stays_error_with_managed_install(value, monkeypatch):
    old_install()
    monkeypatch.setenv(runtime.SPEECH_CPP_PATH_ENV, value)
    with pytest.raises(RuntimeError, match="Invalid PTARMIGAN_FLOW_SPEECH_CPP_PATH"):
        runtime.find_speech_executable()


def test_explicit_install_activates_managed_without_modifying_external(
    release, tmp_path, monkeypatch
):
    external = executable(tmp_path / "bin/speech")
    original = external.read_bytes()
    monkeypatch.setattr(runtime.shutil, "which", lambda _: str(external))
    assert runtime.find_speech_executable() == external
    managed = installer.install_speech_cpp()
    assert runtime.find_speech_executable() == managed
    assert external.read_bytes() == original
    assert runtime.SPEECH_CPP_PATH_ENV not in runtime.os.environ


@pytest.mark.parametrize(
    "state", ["absent", "nonexecutable", "directory", "symlink", "symlink-parent"]
)
def test_unusable_managed_install_preserves_external_lookup(state, tmp_path, monkeypatch):
    managed = installer.managed_speech_cpp_path()
    external = executable(tmp_path / "bin/speech")
    monkeypatch.setattr(runtime.shutil, "which", lambda _: str(external))
    managed.parent.mkdir(parents=True)
    if state == "nonexecutable":
        managed.write_bytes(b"incomplete")
        managed.chmod(0o600)
    elif state == "directory":
        managed.mkdir()
    elif state == "symlink":
        managed.symlink_to(external)
    elif state == "symlink-parent":
        managed.parent.rmdir()
        managed.parent.symlink_to(external.parent, target_is_directory=True)
    assert runtime.find_speech_executable() == external


def test_discovery_does_not_probe_architecture_or_version(tmp_path, monkeypatch):
    managed = old_install()
    monkeypatch.setattr(
        installer, "speech_cpp_install_supported", lambda: pytest.fail("Discovery queried support")
    )
    monkeypatch.setattr(
        runtime.shutil, "which", lambda _: pytest.fail("Managed should precede PATH")
    )
    assert runtime.find_speech_executable() == managed


def test_mid_download_cancellation_stops_before_verification(release, monkeypatch):
    payload = release()
    cancel = threading.Event()
    path = old_install()

    class Chunked(Response):
        def read1(self, size):
            return super().read1(min(size, 16))

    monkeypatch.setattr(installer, "_open_download", lambda _: Chunked(payload))
    monkeypatch.setattr(installer, "_extract", lambda *args: pytest.fail("Cancelled ZIP opened"))
    progress = []

    def observe(value):
        progress.append(value)
        if value["downloaded_bytes"] > 0:
            cancel.set()

    with pytest.raises(installer.SpeechCppInstallCancelledError):
        installer.install_speech_cpp(observe, cancel)
    assert all(p["status"] == "downloading" for p in progress)
    assert_preserved(path)


def test_extraction_cancellation_prevents_probe(release, monkeypatch):
    cancel = threading.Event()
    path = old_install()
    original = zipfile.ZipExtFile.read

    def read(source, size=-1):
        result = original(source, size)
        cancel.set()
        return result

    monkeypatch.setattr(zipfile.ZipExtFile, "read", read)
    monkeypatch.setattr(installer, "_probe", lambda *args: pytest.fail("Cancelled binary executed"))
    with pytest.raises(installer.SpeechCppInstallCancelledError):
        installer.install_speech_cpp(cancel_event=cancel)
    assert_preserved(path)


def test_crc_failure_preserves_previous_install(release):
    # Build a stored member so payload corruption is deterministic; update only
    # the fixture's pinned digest, leaving the ZIP CRC inconsistent.
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as bundle:
        info, payload = member("speech", macho())
        bundle.writestr(info, payload)
    damaged = output.getvalue().replace(b"x" * 96, b"y" * 96)
    release(damaged)
    path = old_install()
    with pytest.raises(installer.SpeechCppInstallError, match="CRC"):
        installer.install_speech_cpp()
    assert_preserved(path)


def test_probe_failed_exit_even_with_correct_banner(release, monkeypatch):
    class Failed(Process):
        def wait(self, *, timeout):
            self.returncode = 1
            return 1

    monkeypatch.setattr(subprocess, "Popen", Failed)
    path = old_install()
    with pytest.raises(installer.SpeechCppInstallError, match="exited with status 1"):
        installer.install_speech_cpp()
    assert_preserved(path)


def test_parent_replacement_during_probe_rejected(release, monkeypatch):
    path = old_install()
    relocated = path.parent.with_name("original-directory")

    def change(*args):
        path.parent.rename(relocated)
        path.parent.mkdir()

    monkeypatch.setattr(installer, "_probe", change)
    with pytest.raises(installer.SpeechCppInstallError, match="directory changed"):
        installer.install_speech_cpp()
    assert (relocated / "speech").read_bytes() == b"previous executable"
    assert not path.exists()


def test_hardlinked_lock_is_rejected(release, tmp_path):
    path = old_install()
    victim = tmp_path / "victim"
    victim.write_bytes(b"untouched")
    (path.parent / ".install.lock").hardlink_to(victim)
    with pytest.raises(installer.SpeechCppInstallError, match="Unsafe.*lock"):
        installer.install_speech_cpp()
    assert victim.read_bytes() == b"untouched"


def test_real_network_opener_uses_tls_redirect_policy_and_bounded_timeout(monkeypatch, isolated):
    # Exercise construction without opening any real network connection.
    calls = []
    context = object()
    monkeypatch.setattr(installer, "_ssl_context", lambda: context)

    class HTTPS:
        def __init__(self, *, context):
            assert context is not None
            calls.append(context)

    class Opener:
        def open(self, request, *, timeout):
            assert request.full_url == installer._ASSET_URL
            assert request.get_header("Accept-encoding") == "identity"
            assert timeout == 2
            return "response"

    def build(*handlers):
        assert isinstance(handlers[0], installer._Redirects)
        assert isinstance(handlers[1], HTTPS)
        return Opener()

    monkeypatch.setattr(urllib.request, "HTTPSHandler", HTTPS)
    monkeypatch.setattr(urllib.request, "build_opener", build)
    assert isolated(2) == "response"
    assert calls == [context]


def test_cancelled_stalled_network_workers_cannot_accumulate(release, monkeypatch):
    started = threading.Event()
    unblock = threading.Event()
    cancel = threading.Event()
    slot = threading.BoundedSemaphore(1)
    monkeypatch.setattr(installer, "_NETWORK_SLOT", slot)
    path = old_install()

    def stall(_):
        started.set()
        assert unblock.wait(3)
        return Response(archive_bytes())

    monkeypatch.setattr(installer, "_open_download", stall)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(installer.install_speech_cpp, None, cancel)
            assert started.wait(1)
            cancel.set()
            with pytest.raises(installer.SpeechCppInstallCancelledError):
                pending.result(timeout=1)
        with pytest.raises(installer.SpeechCppInstallError, match="still stopping"):
            installer.install_speech_cpp()
        assert_preserved(path)
    finally:
        unblock.set()
        assert slot.acquire(timeout=1)
        slot.release()
    release()
    assert installer.install_speech_cpp() == path


@pytest.mark.parametrize("status", ["verifying", "installing"])
@pytest.mark.parametrize("replacement_kind", ["symlink", "directory"])
def test_parent_replacement_before_extraction_never_probes_substituted_zip(
    release, monkeypatch, tmp_path, status, replacement_kind
):
    path = old_install()
    relocated = path.parent.with_name("original-directory")
    replacement = tmp_path / "replacement"
    substituted = macho()[:-1] + b"!"
    alternate_zip = archive_bytes([member("speech", substituted)])
    assert substituted != macho()
    assert len(substituted) == installer._EXECUTABLE_BYTES
    assert hashlib.sha256(alternate_zip).hexdigest() != installer._ARCHIVE_SHA256
    temporary_directories = []
    temporary_directory = installer.tempfile.TemporaryDirectory
    probe = installer._probe
    probe_calls = []
    popen_calls = []
    attacks = []

    def track_directories(*args, **kwargs):
        result = temporary_directory(*args, **kwargs)
        temporary_directories.append(Path(result.name))
        return result

    def record_probe(executable, *args):
        probe_calls.append(executable)
        return probe(executable, *args)

    def record_popen(argv, **kwargs):
        popen_calls.append(Path(argv[0]).read_bytes())
        return Process(argv, **kwargs)

    def replace_parent(value):
        if value["status"] != status:
            return
        # Reproduce the review's attack: redirect the managed parent and mirror
        # both staging names, supplying a valid ZIP with an untrusted Mach-O.
        path.parent.rename(relocated)
        if replacement_kind == "symlink":
            replacement.mkdir()
            path.parent.symlink_to(replacement, target_is_directory=True)
        else:
            path.parent.mkdir()
        for directory in temporary_directories:
            if directory.name.startswith((".download-", ".extract-")):
                mirror = path.parent / directory.name
                mirror.mkdir()
                if directory.name.startswith(".download-"):
                    (mirror / "release.zip").write_bytes(alternate_zip)
        attacks.append(True)

    monkeypatch.setattr(installer.tempfile, "TemporaryDirectory", track_directories)
    monkeypatch.setattr(installer, "_probe", record_probe)
    monkeypatch.setattr(subprocess, "Popen", record_popen)
    with pytest.raises(installer.SpeechCppInstallError):
        installer.install_speech_cpp(replace_parent)
    assert attacks == [True], "Callback exceptions must not silently bypass the reproduction"
    assert probe_calls == []
    assert popen_calls == []
    assert (relocated / "speech").read_bytes() == b"previous executable"


def test_private_staging_extracts_the_same_open_archive_that_was_hashed(release, monkeypatch):
    path = old_install()
    download = installer._download
    extract = installer._extract
    archives = []
    workspaces = []

    def record_download(output, *args):
        assert not isinstance(output, Path)
        assert output.seekable()
        assert stat.S_ISREG(installer.os.fstat(output.fileno()).st_mode)
        archives.append(output)
        return download(output, *args)

    def record_extract(archive, staged, *args):
        assert archive is archives[0]
        assert not archive.closed
        assert hashlib.sha256(archive.read()).hexdigest() == installer._ARCHIVE_SHA256
        archive.seek(0)
        assert not staged.is_relative_to(path.parent)
        workspace = staged.parent.parent
        assert stat.S_IMODE(workspace.stat().st_mode) == 0o700
        assert len(list(workspace.glob(".download-*"))) == 1
        assert staged.parent.name.startswith(".extract-")
        workspaces.append(workspace)
        return extract(archive, staged, *args)

    monkeypatch.setattr(installer, "_download", record_download)
    monkeypatch.setattr(installer, "_extract", record_extract)
    assert installer.install_speech_cpp() == path
    assert archives[0].closed
    assert not workspaces[0].exists()


def test_parent_replaced_after_extraction_is_rejected_before_probe(release, monkeypatch):
    path = old_install()
    relocated = path.parent.with_name("original-directory")
    validate = installer._validate_macho
    changes = []
    probes = []

    def change_after_validation(staged):
        validate(staged)
        path.parent.rename(relocated)
        path.parent.mkdir()
        mirror = path.parent / staged.parent.name
        mirror.mkdir()
        (mirror / "speech").write_bytes(macho()[:-1] + b"!")
        changes.append(True)

    monkeypatch.setattr(installer, "_validate_macho", change_after_validation)
    monkeypatch.setattr(installer, "_probe", lambda *args: probes.append(args))
    with pytest.raises(installer.SpeechCppInstallError, match="directory changed"):
        installer.install_speech_cpp()
    assert changes == [True]
    assert probes == []
    assert_preserved(relocated / "speech")


def test_late_parent_redirection_cannot_redirect_private_probe(release, monkeypatch):
    path = old_install()
    relocated = path.parent.with_name("original-directory")
    probe = installer._probe
    launched = []

    def redirect_before_launch(staged, *args):
        # Model replacement after the pre-probe identity check. Managed paths
        # can change, but the probe's private executable path does not use them.
        path.parent.rename(relocated)
        path.parent.mkdir()
        mirror = path.parent / staged.parent.name
        mirror.mkdir()
        (mirror / "speech").write_bytes(macho()[:-1] + b"!")
        return probe(staged, *args)

    def record_popen(argv, **kwargs):
        launched.append(Path(argv[0]).read_bytes())
        return Process(argv, **kwargs)

    monkeypatch.setattr(installer, "_probe", redirect_before_launch)
    monkeypatch.setattr(subprocess, "Popen", record_popen)
    with pytest.raises(installer.SpeechCppInstallError, match="directory changed"):
        installer.install_speech_cpp()
    assert launched == [macho()]
    assert_preserved(relocated / "speech")


@pytest.mark.parametrize("failure", ["cancel", "fsync", "parent-replacement"])
def test_final_copy_failure_preserves_old_runtime_and_cleans_anchored_stage(
    release, monkeypatch, failure
):
    path = old_install()
    relocated = path.parent.with_name("original-directory")
    cancel = threading.Event()
    fsync = installer.os.fsync
    syncs = []

    def fail_final_sync(descriptor):
        syncs.append(descriptor)
        if len(syncs) == 2:  # Extraction sync, then final destination staging sync.
            if failure == "cancel":
                cancel.set()
            elif failure == "fsync":
                raise OSError("final copy fsync failed")
            else:
                path.parent.rename(relocated)
                path.parent.mkdir()
        return fsync(descriptor)

    monkeypatch.setattr(installer.os, "fsync", fail_final_sync)
    error = (
        installer.SpeechCppInstallCancelledError
        if failure == "cancel"
        else installer.SpeechCppInstallError
    )
    with pytest.raises(error):
        installer.install_speech_cpp(cancel_event=cancel)
    assert len(syncs) == 2
    assert_preserved(relocated / "speech" if failure == "parent-replacement" else path)


def test_private_workspace_cannot_be_redirected_into_managed_tree(release, monkeypatch):
    path = old_install()
    temporary_directory = installer.tempfile.TemporaryDirectory
    downloads = []

    def unsafe_temp_directory(*args, **kwargs):
        if kwargs.get("prefix") == "pflow-speech-":
            kwargs["dir"] = path.parent
        return temporary_directory(*args, **kwargs)

    monkeypatch.setattr(installer.tempfile, "TemporaryDirectory", unsafe_temp_directory)
    monkeypatch.setattr(installer, "_download", lambda *args: downloads.append(args))
    with pytest.raises(installer.SpeechCppInstallError, match="outside the managed app directory"):
        installer.install_speech_cpp()
    assert downloads == []
    assert_preserved(path)
