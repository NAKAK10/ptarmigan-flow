"""Pinned, app-owned speech.cpp installation; no shell or model dependencies.

Only the verified standalone executable is installed. Network work is confined
to a daemon thread and a bounded queue: DNS/TLS implementations that ignore
socket timeouts cannot delay cancellation or write into a discarded staging area.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import queue
import secrets
import ssl
import stat
import struct
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

_VERSION = "0.10.1"
_ASSET_URL = (
    "https://github.com/nyosegawa/speech.cpp/releases/download/v0.10.1/"
    "speech-0.10.1-macos-arm64-metal.zip"
)
_ARCHIVE_SHA256 = "ff24f6f58c9d2dedb01caa88ba043ae845c76eff228e4cbfae5de749174404c1"
_ARCHIVE_BYTES = 6087757
_EXECUTABLE_BYTES = 11588712
_MAX_MEMBERS = 16
_MAX_EXPANDED_BYTES = 64 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024
_SOCKET_TIMEOUT_SECONDS = 5.0
_INSTALL_TIMEOUT_SECONDS = 120.0
_PROBE_TIMEOUT_SECONDS = 5.0
_MAX_REDIRECTS = 5
_NETWORK_SLOT = threading.BoundedSemaphore(1)
_DOWNLOAD_HOSTS = frozenset(
    {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
)
ProgressCallback = Callable[[dict[str, str | int]], None]


class SpeechCppInstallError(RuntimeError):
    """Installation failed without replacing the previous managed executable."""


class SpeechCppInstallCancelledError(SpeechCppInstallError):
    """The caller cancelled installation before the atomic replacement."""


def managed_speech_cpp_path() -> Path:
    """Return the pinned managed path without creating anything."""
    return (
        Path.home()
        / "Library/Application Support/ptarmigan-flow/runtimes/speech.cpp"
        / _VERSION
        / "macos-arm64-metal/speech"
    )


def speech_cpp_install_supported() -> bool:
    """Support physical Apple Silicon macOS, including a process under Rosetta."""
    if platform.system() != "Darwin":
        return False
    if platform.machine().lower() in {"arm64", "aarch64"}:
        return True
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return False
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.optional.arm64"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
            check=False,
            shell=False,
        )
        return result.returncode == 0 and result.stdout.strip() == b"1"
    except (OSError, subprocess.TimeoutExpired):
        return False


def _check(cancel_event: threading.Event, deadline: float) -> None:
    if cancel_event.is_set():
        raise SpeechCppInstallCancelledError("speech.cpp installation cancelled; retry when ready.")
    if time.monotonic() >= deadline:
        raise SpeechCppInstallError(
            "speech.cpp installation timed out; check the network and retry."
        )


@contextlib.contextmanager
def _directory(path: Path, *, create: bool) -> Iterator[int]:
    """Walk using no-follow directory descriptors, including existing ancestors."""
    if not path.is_absolute() or ".." in path.parts:
        raise SpeechCppInstallError("Managed runtime directory must be an absolute, safe path.")
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _destination_stat(directory: int, name: str) -> os.stat_result | None:
    try:
        entry = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(entry.st_mode):
        raise SpeechCppInstallError(
            "Managed speech destination must be a regular file, never a symlink; "
            "remove the conflicting entry and retry."
        )
    return entry


def _check_directory_identity(path: Path, directory: int) -> None:
    """Reject redirected managed paths before any extraction, probe, or commit."""
    try:
        with _directory(path, create=False) as current:
            before, after = os.fstat(directory), os.fstat(current)
            if (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino):
                return
    except OSError as exc:
        raise SpeechCppInstallError("Managed directory changed; retry installation.") from exc
    raise SpeechCppInstallError("Managed directory changed; retry installation.")


def _managed_speech_cpp_executable() -> Path | None:
    """Cheap, read-only discovery; unsafe or incomplete managed installs are absent."""
    path = managed_speech_cpp_path()
    try:
        with _directory(path.parent, create=False) as directory:
            entry = _destination_stat(directory, path.name)
            if entry is not None and os.access(path, os.X_OK):
                return path
    except (OSError, SpeechCppInstallError):
        pass
    return None


@contextlib.contextmanager
def _install_lock(directory: int) -> Iterator[None]:
    # flock releases on process exit. Keep the inode so waiters cannot lock an
    # unlinked inode while a third app instance creates a different lock file.
    import fcntl

    descriptor = os.open(
        ".install.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
        0o600,
        dir_fd=directory,
    )
    try:
        entry = os.fstat(descriptor)
        if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
            raise SpeechCppInstallError("Unsafe speech.cpp installation lock; remove it and retry.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpeechCppInstallError(
                "Another app copy is installing speech.cpp; wait for it to finish and retry."
            ) from exc
        yield
    finally:
        os.close(descriptor)


def _ssl_context() -> ssl.SSLContext:
    # Match app_updater: frozen OpenSSL may lack system roots; certifi is bundled.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _DOWNLOAD_HOSTS
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise SpeechCppInstallError(
            "Unsafe speech.cpp download redirect; HTTPS GitHub assets only."
        )


class _Redirects(urllib.request.HTTPRedirectHandler):
    def __init__(self) -> None:
        self.count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.count += 1
        if self.count > _MAX_REDIRECTS:
            raise SpeechCppInstallError("Too many speech.cpp download redirects; retry later.")
        _validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_download(timeout: float) -> Any:
    _validate_url(_ASSET_URL)
    opener = urllib.request.build_opener(
        _Redirects(), urllib.request.HTTPSHandler(context=_ssl_context())
    )
    return opener.open(
        urllib.request.Request(
            _ASSET_URL,
            headers={"User-Agent": "ptarmigan-flow", "Accept-Encoding": "identity"},
        ),
        timeout=timeout,
    )


def _download(
    output: BinaryIO,
    progress: Callable[[str, int], None],
    cancel_event: threading.Event,
    deadline: float,
) -> str:
    messages: queue.Queue[bytes | Exception | None] = queue.Queue(maxsize=2)
    stopped = threading.Event()

    def publish(value: bytes | Exception | None) -> None:
        while not stopped.is_set():
            try:
                messages.put(value, timeout=0.05)
                return
            except queue.Full:
                pass

    def fetch() -> None:
        # No staging-file access or callbacks in this worker, even after timeout.
        result = None
        try:
            timeout = min(_SOCKET_TIMEOUT_SECONDS, max(0.001, deadline - time.monotonic()))
            with _open_download(timeout) as response:
                _validate_url(response.geturl())
                if response.status != 200:
                    raise SpeechCppInstallError("speech.cpp download did not return HTTP 200.")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != _ARCHIVE_BYTES:
                    raise SpeechCppInstallError("speech.cpp download has the wrong size; retry.")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise SpeechCppInstallError("Unexpected speech.cpp download encoding.")
                downloaded = 0
                while not stopped.is_set():
                    chunk = response.read1(_CHUNK_BYTES)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > _ARCHIVE_BYTES:
                        raise SpeechCppInstallError("speech.cpp download exceeds the size limit.")
                    publish(chunk)
        except Exception as exc:
            result = exc
        finally:
            network_slot.release()
        publish(result)

    _check(cancel_event, deadline)
    progress("downloading", 0)
    _check(cancel_event, deadline)
    # A pathological DNS/TLS call can outlive cancellation. Bound those workers
    # across retries as well as bounding each request's queue and caller wait.
    network_slot = _NETWORK_SLOT
    if not network_slot.acquire(blocking=False):
        raise SpeechCppInstallError(
            "The previous speech.cpp network request is still stopping; retry shortly."
        )
    try:
        threading.Thread(target=fetch, name="speech.cpp-download", daemon=True).start()
    except Exception:
        network_slot.release()
        raise
    downloaded = 0
    digest = hashlib.sha256()
    try:
        while True:
            _check(cancel_event, deadline)
            try:
                message = messages.get(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if isinstance(message, Exception):
                raise SpeechCppInstallError(
                    f"Cannot download speech.cpp: {message}. Check the network and retry."
                ) from message
            if message is None:
                break
            output.write(message)
            digest.update(message)
            downloaded += len(message)
            progress("downloading", downloaded)
        output.flush()
        _check(cancel_event, deadline)
        if downloaded != _ARCHIVE_BYTES:
            raise SpeechCppInstallError("speech.cpp download was truncated; retry.")
        return digest.hexdigest()
    finally:
        stopped.set()


def _extract(
    archive: BinaryIO, destination: Path, cancel_event: threading.Event, deadline: float
) -> None:
    """Only call after checking the pinned hash. Never apply ZIP permissions."""
    _check(cancel_event, deadline)

    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if not 1 <= len(members) <= _MAX_MEMBERS:
            raise SpeechCppInstallError("speech.cpp ZIP exceeds the member count limit.")
        names: set[str] = set()
        expanded = 0
        executable = None
        for member in members:
            _check(cancel_event, deadline)
            name = member.filename
            parts = name.rstrip("/").split("/")
            if (
                name != member.orig_filename
                or "\\" in name
                or ":" in name
                or any(part in {"", ".", ".."} for part in parts)
                or name.rstrip("/") in names
            ):
                raise SpeechCppInstallError("Unsafe or duplicate speech.cpp ZIP member path.")
            names.add(name.rstrip("/"))
            expanded += member.file_size
            if expanded > _MAX_EXPANDED_BYTES or member.file_size < 0:
                raise SpeechCppInstallError("speech.cpp ZIP exceeds the expanded size limit.")
            if member.flag_bits & 1:
                raise SpeechCppInstallError("Encrypted speech.cpp ZIP members are unsupported.")
            if name == "speech":
                kind = stat.S_IFMT(member.external_attr >> 16)
                if member.is_dir() or kind != stat.S_IFREG:
                    raise SpeechCppInstallError(
                        "ZIP speech must be a regular file, never a symlink."
                    )
                if member.file_size != _EXECUTABLE_BYTES:
                    raise SpeechCppInstallError("ZIP speech has the wrong executable size.")
                executable = member
        if executable is None:
            raise SpeechCppInstallError("speech.cpp ZIP is missing the root speech executable.")
        with bundle.open(executable) as source, destination.open("xb") as output:
            extracted = 0
            while chunk := source.read(_CHUNK_BYTES):
                _check(cancel_event, deadline)
                extracted += len(chunk)
                if extracted > _EXECUTABLE_BYTES:
                    raise SpeechCppInstallError("ZIP speech exceeds the executable size limit.")
                output.write(chunk)
            if extracted != _EXECUTABLE_BYTES:
                raise SpeechCppInstallError("ZIP speech executable was truncated.")
            output.flush()
            os.fsync(output.fileno())


def _validate_macho(executable: Path) -> None:
    with executable.open("rb") as source:
        header = source.read(32)
    # MH_MAGIC_64, CPU_TYPE_ARM64, MH_EXECUTE; this pinned release is thin arm64.
    if len(header) != 32 or struct.unpack("<III", header[:12])[:2] != (0xFEEDFACF, 0x0100000C):
        raise SpeechCppInstallError("Downloaded speech is not a 64-bit arm64 Mach-O executable.")
    if struct.unpack("<I", header[12:16])[0] != 2:
        raise SpeechCppInstallError("Downloaded speech Mach-O is not an executable.")


def _probe(executable: Path, cancel_event: threading.Event, deadline: float) -> None:
    # Local import avoids the discovery -> installer -> runtime import cycle.
    from ptarmigan_flow.stt.haas_runtime import SpeechCppRuntime, parse_speech_version

    _check(cancel_event, deadline)
    runner = SpeechCppRuntime(executable=executable)
    try:
        output = runner.run(
            ["--version"],
            generation=runner.generation,
            timeout_seconds=min(_PROBE_TIMEOUT_SECONDS, deadline - time.monotonic()),
            cancel_event=cancel_event,
        )
        if parse_speech_version(output) != (0, 10, 1):
            raise SpeechCppInstallError("Downloaded speech.cpp version is not exactly 0.10.1.")
    finally:
        runner.close()
    _check(cancel_event, deadline)


def _commit(
    staged: Path,
    destination: Path,
    directory: int,
    cancel_event: threading.Event,
    deadline: float,
) -> None:
    """Copy from private staging, then replace using the locked directory FD.

    The temporary workspace may be on another filesystem. Create the final
    staging file on the destination filesystem without reopening a managed path.
    """
    _check_directory_identity(destination.parent, directory)
    name = f".speech-{secrets.token_hex(16)}"
    descriptor = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
    )
    try:
        with os.fdopen(descriptor, "wb") as output, staged.open("rb") as source:
            copied = 0
            while chunk := source.read(_CHUNK_BYTES):
                _check(cancel_event, deadline)
                copied += len(chunk)
                if copied > _EXECUTABLE_BYTES:
                    raise SpeechCppInstallError("Staged speech executable exceeds the size limit.")
                output.write(chunk)
            if copied != _EXECUTABLE_BYTES:
                raise SpeechCppInstallError("Staged speech executable has the wrong size.")
            output.flush()
            os.fchmod(output.fileno(), 0o700)
            os.fsync(output.fileno())
        _destination_stat(directory, destination.name)
        _check_directory_identity(destination.parent, directory)
        _check(cancel_event, deadline)
        os.replace(name, destination.name, src_dir_fd=directory, dst_dir_fd=directory)
    finally:
        # If the managed parent was renamed, cleanup still addresses its inode.
        try:
            os.unlink(name, dir_fd=directory)
        except FileNotFoundError:
            pass


def install_speech_cpp(
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Install the pinned executable atomically, preserving the old copy on failure.

    The synchronous callback receives status and byte counts on the caller's
    thread. Cancellation is honored through the last check before os.replace;
    after that commit point the installation is complete.
    """
    cancel = cancel_event if cancel_event is not None else threading.Event()
    deadline = time.monotonic() + _INSTALL_TIMEOUT_SECONDS
    _check(cancel, deadline)
    supported = speech_cpp_install_supported()
    _check(cancel, deadline)
    if not supported:
        raise SpeechCppInstallError(
            "Automatic speech.cpp installation requires Apple Silicon macOS. "
            "No official Intel macOS binary is available; configure an external source-built "
            "speech executable using PTARMIGAN_FLOW_SPEECH_CPP_PATH."
        )
    destination = managed_speech_cpp_path()

    def progress(status: str, downloaded: int) -> None:
        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "status": status,
                        "downloaded_bytes": downloaded,
                        "total_bytes": _ARCHIVE_BYTES,
                    }
                )
            except Exception:
                # Observers must not invalidate an installation (especially
                # after the atomic commit). Cancellation has its own event.
                pass

    try:
        with _directory(destination.parent, create=True) as directory, _install_lock(directory):
            _destination_stat(directory, destination.name)
            _check(cancel, deadline)
            # Private OS temporary staging is independent of the managed tree.
            # This assumes the current user's private temporary files are trusted;
            # it does not defend against arbitrary same-user writes into them.
            with tempfile.TemporaryDirectory(
                prefix="pflow-speech-", ignore_cleanup_errors=True
            ) as workspace:
                workspace_path = Path(workspace).resolve(strict=True)
                if workspace_path.is_relative_to(
                    Path.home() / "Library/Application Support/ptarmigan-flow"
                ):
                    raise SpeechCppInstallError(
                        "Private speech.cpp staging must be outside the managed app directory; "
                        "check the system temporary directory and retry."
                    )
                with (
                    tempfile.TemporaryDirectory(
                        prefix=".download-", dir=workspace_path, ignore_cleanup_errors=True
                    ) as download,
                    tempfile.TemporaryDirectory(
                        prefix=".extract-", dir=workspace_path, ignore_cleanup_errors=True
                    ) as extraction,
                    tempfile.TemporaryFile(mode="w+b", dir=download) as archive,
                ):
                    staged = Path(extraction) / "speech"
                    # Retain the downloaded file object through verification and
                    # ZIP parsing; never reopen a downloaded archive by pathname.
                    digest = _download(archive, progress, cancel, deadline)
                    progress("verifying", _ARCHIVE_BYTES)
                    _check(cancel, deadline)
                    _check_directory_identity(destination.parent, directory)
                    if digest != _ARCHIVE_SHA256:
                        raise SpeechCppInstallError(
                            "speech.cpp SHA-256 mismatch; discard the download and retry."
                        )
                    progress("installing", _ARCHIVE_BYTES)
                    _check_directory_identity(destination.parent, directory)
                    archive.seek(0)
                    _extract(archive, staged, cancel, deadline)
                    _validate_macho(staged)
                    staged.chmod(0o700)
                    _check_directory_identity(destination.parent, directory)
                    _check(cancel, deadline)
                    _probe(staged, cancel, deadline)
                    _commit(staged, destination, directory, cancel, deadline)
            progress("complete", _ARCHIVE_BYTES)
        return destination
    except SpeechCppInstallError:
        raise
    except Exception as exc:
        _check(cancel, deadline)
        raise SpeechCppInstallError(
            f"Cannot install speech.cpp: {exc}. Check the managed directory permissions "
            "and network, then retry."
        ) from exc
