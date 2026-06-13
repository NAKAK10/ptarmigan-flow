"""macOS permission detection utilities."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PermissionReport:
    """Permission states used by CLI and daemon startup."""

    microphone: bool
    accessibility: bool
    input_monitoring: bool

    @property
    def all_granted(self) -> bool:
        return self.microphone and self.accessibility and self.input_monitoring

    @property
    def missing(self) -> list[str]:
        missing_permissions: list[str] = []
        if not self.accessibility:
            missing_permissions.append("Accessibility")
        if not self.input_monitoring:
            missing_permissions.append("Input Monitoring")
        if not self.microphone:
            missing_permissions.append("Microphone")
        return missing_permissions


@dataclass(slots=True)
class LaunchdPermissionProbe:
    """Result of permission probing through launchd context."""

    ok: bool
    command: list[str] | None = None
    report: PermissionReport | None = None
    error: str | None = None
    stdout: str | None = None
    stderr: str | None = None


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _parse_permission_report_from_text(text: str) -> PermissionReport | None:
    values: dict[str, bool] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        normalized_key = key.strip().lower()
        normalized_value = value.strip().upper()
        if normalized_key == "microphone":
            values["microphone"] = normalized_value == "OK"
        elif normalized_key == "accessibility":
            values["accessibility"] = normalized_value == "OK"
        elif normalized_key == "input monitoring":
            values["input_monitoring"] = normalized_value == "OK"

    expected = {"microphone", "accessibility", "input_monitoring"}
    if set(values) != expected:
        return None
    return PermissionReport(
        microphone=values["microphone"],
        accessibility=values["accessibility"],
        input_monitoring=values["input_monitoring"],
    )


def check_all_permissions_subprocess(
    *,
    runner=subprocess.run,
    executable: str | os.PathLike[str] | None = None,
) -> PermissionReport | None:
    """Check permissions in a fresh process to avoid TCC cache in the GUI process."""
    command = [
        str(executable or sys.executable),
        "-m",
        "ptarmigan_flow.cli",
        "check-permissions",
    ]
    try:
        process = runner(command, capture_output=True, text=True)
    except Exception as exc:
        LOGGER.debug("Could not run subprocess permission check: %s", exc)
        return None

    returncode = getattr(process, "returncode", None)
    if returncode not in {0, 2}:
        return None

    stdout = getattr(process, "stdout", "") or ""
    stderr = getattr(process, "stderr", "") or ""
    parse_source = "\n".join(part for part in (stdout, stderr) if part)
    report = _parse_permission_report_from_text(parse_source)
    if report is None:
        LOGGER.debug("Could not parse subprocess permission check output")
    return report


def check_permissions_in_launchd_context(
    *,
    uid: int | None = None,
    command: Sequence[str] | None = None,
) -> LaunchdPermissionProbe:
    """Run permission check through launchctl asuser to compare launchd context."""
    command_parts = [str(part) for part in (command or ("pflow", "check-permissions"))]
    if not _is_macos():
        report = check_all_permissions()
        return LaunchdPermissionProbe(ok=True, command=command_parts, report=report)

    resolved_uid = uid if uid is not None else os.getuid()
    launchctl_command = ["launchctl", "asuser", str(resolved_uid), *command_parts]

    try:
        process = subprocess.run(
            launchctl_command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return LaunchdPermissionProbe(
            ok=False,
            command=command_parts,
            error=f"Failed to run launchctl: {exc}",
        )

    stdout_text = process.stdout.strip() or None
    stderr_text = process.stderr.strip() or None
    parse_source = "\n".join(part for part in (stdout_text, stderr_text) if part)
    report = _parse_permission_report_from_text(parse_source)
    if report is None:
        detail = (
            f"Could not parse permission status from launchd check output "
            f"(exit={process.returncode}, command={' '.join(command_parts)})"
        )
        return LaunchdPermissionProbe(
            ok=False,
            command=command_parts,
            error=detail,
            stdout=stdout_text,
            stderr=stderr_text,
        )

    return LaunchdPermissionProbe(
        ok=process.returncode in {0, 2},
        command=command_parts,
        report=report,
        stdout=stdout_text,
        stderr=stderr_text,
    )


def check_microphone_permission() -> bool:
    """Best-effort microphone permission preflight."""
    if not _is_macos():
        return True

    try:
        from AVFoundation import AVAuthorizationStatusAuthorized, AVCaptureDevice, AVMediaTypeAudio

        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
        return status == AVAuthorizationStatusAuthorized
    except Exception as exc:
        LOGGER.debug("Could not preflight microphone permission via AVFoundation: %s", exc)

    try:
        import sounddevice as sd

        with sd.InputStream(channels=1, samplerate=16000, blocksize=64):
            pass
        return True
    except Exception as exc:
        LOGGER.debug("Could not probe microphone stream: %s", exc)
        return False


def check_accessibility_permission() -> bool:
    """Check macOS Accessibility permission."""
    if not _is_macos():
        return True

    try:
        import ApplicationServices as app_services

        return bool(app_services.AXIsProcessTrusted())
    except Exception as exc:
        LOGGER.debug(
            "Could not preflight accessibility permission via ApplicationServices: %s",
            exc,
        )

    try:
        import Quartz

        trusted = getattr(Quartz, "AXIsProcessTrusted", None)
        if trusted is None:
            return False
        return bool(trusted())
    except Exception as exc:
        LOGGER.debug("Could not preflight accessibility permission via Quartz: %s", exc)
        return False


def check_input_monitoring_permission() -> bool:
    """Check macOS Input Monitoring permission."""
    if not _is_macos():
        return True

    try:
        import Quartz

        preflight = getattr(Quartz, "CGPreflightListenEventAccess", None)
        if preflight is None:
            return True
        return bool(preflight())
    except Exception as exc:
        LOGGER.debug("Could not preflight input monitoring permission: %s", exc)
        return False


def check_all_permissions() -> PermissionReport:
    """Run all permission probes."""
    return PermissionReport(
        microphone=check_microphone_permission(),
        accessibility=check_accessibility_permission(),
        input_monitoring=check_input_monitoring_permission(),
    )


def request_microphone_permission(timeout_seconds: float = 10.0) -> bool:
    """Trigger microphone permission request if possible."""
    if not _is_macos():
        return True

    if check_microphone_permission():
        return True

    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio

        done = Event()

        def completion(_granted: bool) -> None:
            done.set()

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeAudio,
            completion,
        )
        done.wait(timeout_seconds)
    except Exception as exc:
        LOGGER.debug("Could not request microphone permission: %s", exc)

    return check_microphone_permission()


def request_accessibility_permission() -> bool:
    """Trigger Accessibility prompt if possible."""
    if not _is_macos():
        return True

    if check_accessibility_permission():
        return True

    try:
        import ApplicationServices as app_services

        options = {app_services.kAXTrustedCheckOptionPrompt: True}
        app_services.AXIsProcessTrustedWithOptions(options)
    except Exception as exc:
        LOGGER.debug(
            "Could not request accessibility permission via ApplicationServices: %s",
            exc,
        )
        try:
            import Quartz

            trusted_with_options = getattr(Quartz, "AXIsProcessTrustedWithOptions", None)
            prompt_key = getattr(Quartz, "kAXTrustedCheckOptionPrompt", None)
            if trusted_with_options is not None and prompt_key is not None:
                trusted_with_options({prompt_key: True})
        except Exception as fallback_exc:
            LOGGER.debug(
                "Could not request accessibility permission via Quartz: %s",
                fallback_exc,
            )

    return check_accessibility_permission()


def request_input_monitoring_permission() -> bool:
    """Trigger Input Monitoring prompt if possible."""
    if not _is_macos():
        return True

    if check_input_monitoring_permission():
        return True

    try:
        import Quartz

        request = getattr(Quartz, "CGRequestListenEventAccess", None)
        if request is not None:
            request()
    except Exception as exc:
        LOGGER.debug("Could not request input monitoring permission: %s", exc)

    return check_input_monitoring_permission()


def request_all_permissions() -> PermissionReport:
    """Request any missing permissions and return the final state."""
    before = check_all_permissions()
    if before.all_granted:
        return before

    if not before.accessibility:
        request_accessibility_permission()
    if not before.input_monitoring:
        request_input_monitoring_permission()
    if not before.microphone:
        request_microphone_permission()

    return check_all_permissions()


def _resolve_python_app_from_executable(executable: Path) -> Path | None:
    resolved = executable.resolve(strict=False)
    marker = "/Frameworks/Python.framework/Versions/"
    text = str(resolved)
    if marker not in text:
        return None

    prefix, suffix = text.split(marker, 1)
    version = suffix.split("/", 1)[0].strip()
    if not version:
        return None

    return (
        Path(prefix)
        / "Frameworks"
        / "Python.framework"
        / "Versions"
        / version
        / "Resources"
        / "Python.app"
    )


def _prefer_homebrew_opt_path(path: Path) -> Path:
    text = str(path)
    marker = "/Cellar/python@3.11/"
    if marker not in text:
        return path

    prefix, suffix = text.split(marker, 1)
    remainder = suffix.split("/", 1)
    if len(remainder) < 2:
        return path

    return Path(prefix) / "opt" / "python@3.11" / remainder[1]


def current_permission_executable() -> Path:
    return Path(sys.executable).resolve(strict=False)


def recommended_permission_target(executable: Path | None = None) -> Path:
    resolved_executable = (executable or current_permission_executable()).resolve(strict=False)
    python_app = _resolve_python_app_from_executable(resolved_executable)
    if python_app is not None:
        return _prefer_homebrew_opt_path(python_app)
    return _prefer_homebrew_opt_path(resolved_executable)


def reset_app_bundle_tcc(bundle_identifier: str) -> bool:
    """Reset TCC entries for *bundle_identifier* so macOS prompts for re-grant.

    Resets both Accessibility and Input Monitoring (ListenEvent) via ``tccutil``.
    Returns True when at least one reset succeeded, False otherwise.
    Best-effort: never raises.
    """
    tccutil = shutil.which("tccutil")
    if not tccutil:
        return False

    _TCC_SERVICES = ("Accessibility", "ListenEvent")
    any_ok = False
    for service in _TCC_SERVICES:
        try:
            result = subprocess.run(
                [tccutil, "reset", service, bundle_identifier],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                any_ok = True
                LOGGER.debug("TCC reset OK: service=%s bundle=%s", service, bundle_identifier)
            else:
                LOGGER.debug(
                    "TCC reset failed: service=%s bundle=%s stderr=%s",
                    service,
                    bundle_identifier,
                    result.stderr.strip(),
                )
        except OSError as exc:
            LOGGER.debug("tccutil OSError: %s", exc)
    return any_ok


def format_permission_guidance(report: PermissionReport) -> str:
    """Build user-facing setup instructions for missing permissions."""
    if report.all_granted:
        return "All required permissions are granted."

    executable = current_permission_executable()
    preferred_target = recommended_permission_target(executable)

    lines = [
        "Missing macOS permissions detected:",
        *[f"- {item}" for item in report.missing],
        "",
        "Open: System Settings -> Privacy & Security",
        "Then enable this terminal/app in:",
        "- Accessibility",
        "- Input Monitoring",
        "- Microphone",
        "",
        f"Current executable: {executable}",
        f"Preferred permission target: {preferred_target}",
    ]

    if preferred_target != executable:
        lines.append(
            "If daemon and terminal permission states differ, enable the preferred target above."
        )

    if os.environ.get("XPC_SERVICE_NAME") == "com.ptarmiganflow.daemon":
        lines.append("Detected launchd context: com.ptarmiganflow.daemon")

    lines.extend(
        [
        "",
        "After changing permissions, restart ptarmigan-flow.",
        ]
    )
    return "\n".join(lines)
