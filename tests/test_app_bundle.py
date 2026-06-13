from __future__ import annotations

import os
import plistlib
from pathlib import Path

from ptarmigan_flow import app_bundle


def _write_executable(path: Path, content: bytes = b"#!/bin/sh\nexit 0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755)


def test_launch_agent_prefix_from_env_returns_none_without_env(monkeypatch) -> None:
    for key in (
        app_bundle.ENV_BOOTSTRAP_SCRIPT,
        app_bundle.ENV_LIBEXEC,
        app_bundle.ENV_VAR_DIR,
        app_bundle.ENV_PYTHON,
        app_bundle.ENV_UV,
    ):
        monkeypatch.delenv(key, raising=False)

    prefix = app_bundle.launch_agent_prefix_from_env(executable_path=Path("/tmp/PtarmiganFlow"))
    assert prefix is None


def test_install_app_bundle_from_env_creates_bundle(tmp_path: Path, monkeypatch) -> None:
    bootstrap_script = tmp_path / "libexec" / "src" / "ptarmigan_flow" / "homebrew_bootstrap.py"
    bootstrap_script.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_script.write_text("print('ok')\n", encoding="utf-8")
    python_bin = tmp_path / "python3.11"
    uv_bin = tmp_path / "uv"
    _write_executable(python_bin)
    _write_executable(uv_bin)

    monkeypatch.setenv(app_bundle.ENV_BOOTSTRAP_SCRIPT, str(bootstrap_script))
    monkeypatch.setenv(app_bundle.ENV_LIBEXEC, str(tmp_path / "libexec"))
    monkeypatch.setenv(app_bundle.ENV_VAR_DIR, str(tmp_path / "var"))
    monkeypatch.setenv(app_bundle.ENV_PYTHON, str(python_bin))
    monkeypatch.setenv(app_bundle.ENV_UV, str(uv_bin))

    destination = tmp_path / "Applications" / app_bundle.APP_BUNDLE_NAME
    installed = app_bundle.install_app_bundle_from_env(destination)

    assert installed == destination
    executable = app_bundle.app_bundle_executable_path(destination)
    assert executable.exists()
    assert os.access(executable, os.X_OK)
    with (destination / "Contents" / "Info.plist").open("rb") as fp:
        payload = plistlib.load(fp)
    assert payload["CFBundleExecutable"] == app_bundle.APP_EXECUTABLE_NAME
    assert payload["CFBundleIdentifier"] == app_bundle.APP_BUNDLE_IDENTIFIER
    assert payload["CFBundleIconFile"] == app_bundle.APP_ICON_FILE
    assert payload["LSUIElement"] is True
    assert "NSMicrophoneUsageDescription" in payload
    assert (
        destination / "Contents" / "Resources" / app_bundle.APP_ICON_FILE
    ).read_bytes() == app_bundle.app_icon_bytes()
    assert (destination / "Contents" / "Resources" / "webui" / "index.html").is_file()
    assert (destination / "Contents" / "Resources" / "webui" / "app.css").is_file()
    assert (destination / "Contents" / "Resources" / "webui" / "app.js").is_file()


def test_resolve_launch_agent_app_command_uses_default_bundle_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bootstrap_script = tmp_path / "libexec" / "src" / "ptarmigan_flow" / "homebrew_bootstrap.py"
    bootstrap_script.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_script.write_text("print('ok')\n", encoding="utf-8")
    python_bin = tmp_path / "python3.11"
    uv_bin = tmp_path / "uv"
    _write_executable(python_bin)
    _write_executable(uv_bin)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(app_bundle.ENV_BOOTSTRAP_SCRIPT, str(bootstrap_script))
    monkeypatch.setenv(app_bundle.ENV_LIBEXEC, str(tmp_path / "libexec"))
    monkeypatch.setenv(app_bundle.ENV_VAR_DIR, str(tmp_path / "var"))
    monkeypatch.setenv(app_bundle.ENV_PYTHON, str(python_bin))
    monkeypatch.setenv(app_bundle.ENV_UV, str(uv_bin))

    command = app_bundle.resolve_launch_agent_app_command()

    assert command is not None
    assert command[0].endswith("/Applications/PtarmiganFlow.app/Contents/MacOS/PtarmiganFlow")
    assert command[1] == str(bootstrap_script)
    assert command[-1] == "--"


def test_resolve_real_python_binary_prefers_python_app(tmp_path: Path) -> None:
    base = tmp_path / "opt" / "homebrew" / "Cellar" / "python@3.11" / "3.11.14_3"
    launcher = base / "Frameworks" / "Python.framework" / "Versions" / "3.11" / "bin" / "python3.11"
    real_binary = (
        base
        / "Frameworks"
        / "Python.framework"
        / "Versions"
        / "3.11"
        / "Resources"
        / "Python.app"
        / "Contents"
        / "MacOS"
        / "Python"
    )
    _write_executable(launcher)
    _write_executable(real_binary)

    resolved = app_bundle._resolve_real_python_binary(launcher)
    assert resolved == real_binary


def test_resolve_real_python_binary_falls_back_when_python_app_missing(tmp_path: Path) -> None:
    launcher = tmp_path / "python3.11"
    _write_executable(launcher)

    resolved = app_bundle._resolve_real_python_binary(launcher)
    assert resolved == launcher


def _make_bundle_env(
    tmp_path: Path,
    monkeypatch,
    *,
    python_content: bytes = b"pythonbin-v1",
) -> Path:
    """Set up env vars and file fixtures for install_app_bundle_from_env tests."""
    bootstrap_script = tmp_path / "libexec" / "src" / "ptarmigan_flow" / "homebrew_bootstrap.py"
    bootstrap_script.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_script.write_text("print('ok')\n", encoding="utf-8")
    python_bin = tmp_path / "python3.11"
    uv_bin = tmp_path / "uv"
    _write_executable(python_bin, python_content)
    _write_executable(uv_bin)

    monkeypatch.setenv(app_bundle.ENV_BOOTSTRAP_SCRIPT, str(bootstrap_script))
    monkeypatch.setenv(app_bundle.ENV_LIBEXEC, str(tmp_path / "libexec"))
    monkeypatch.setenv(app_bundle.ENV_VAR_DIR, str(tmp_path / "var"))
    monkeypatch.setenv(app_bundle.ENV_PYTHON, str(python_bin))
    monkeypatch.setenv(app_bundle.ENV_UV, str(uv_bin))

    return tmp_path / "Applications" / app_bundle.APP_BUNDLE_NAME


def test_install_app_bundle_skips_resign_when_executable_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    """When source and dest executables are identical, _resign_app_bundle must not be called."""
    destination = _make_bundle_env(tmp_path, monkeypatch)

    resign_calls: list[Path] = []
    monkeypatch.setattr(app_bundle, "_resign_app_bundle", lambda p: resign_calls.append(p))

    # First install: executable does not exist yet, so it will copy + sign.
    app_bundle.install_app_bundle_from_env(destination)
    assert len(resign_calls) == 1, "expected one sign on first install"
    resign_calls.clear()

    # Second install with identical source: nothing should change → no re-sign.
    app_bundle.install_app_bundle_from_env(destination)
    assert resign_calls == [], "re-sign should be skipped when nothing changed"


def test_install_app_bundle_resigns_when_executable_changed(
    tmp_path: Path, monkeypatch
) -> None:
    """When source binary differs from dest, _resign_app_bundle must be called."""
    destination = _make_bundle_env(tmp_path, monkeypatch, python_content=b"pythonbin-v1")
    python_bin = tmp_path / "python3.11"

    resign_calls: list[Path] = []
    monkeypatch.setattr(app_bundle, "_resign_app_bundle", lambda p: resign_calls.append(p))

    # First install.
    app_bundle.install_app_bundle_from_env(destination)
    resign_calls.clear()

    # Replace source binary with different content.
    _write_executable(python_bin, b"pythonbin-v2")

    # Second install: executable changed → should re-sign.
    app_bundle.install_app_bundle_from_env(destination)
    assert len(resign_calls) == 1, "re-sign should run after executable update"
