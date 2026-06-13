import sys
from types import SimpleNamespace

from ptarmigan_flow.permissions import (
    LaunchdPermissionProbe,
    PermissionReport,
    _parse_permission_report_from_text,
    check_accessibility_permission,
    check_all_permissions_subprocess,
    check_permissions_in_launchd_context,
    format_permission_guidance,
    recommended_permission_target,
    request_accessibility_permission,
    request_all_permissions,
    reset_app_bundle_tcc,
)


def test_request_all_permissions_noop_when_already_granted(monkeypatch) -> None:
    granted = PermissionReport(microphone=True, accessibility=True, input_monitoring=True)
    monkeypatch.setattr("ptarmigan_flow.permissions.check_all_permissions", lambda: granted)

    result = request_all_permissions()
    assert result.all_granted


def test_request_all_permissions_requests_only_missing(monkeypatch) -> None:
    states = [
        PermissionReport(microphone=False, accessibility=True, input_monitoring=False),
        PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
    ]
    calls = {"mic": 0, "ax": 0, "im": 0}

    def fake_check_all_permissions() -> PermissionReport:
        return states.pop(0)

    def fake_request_mic() -> bool:
        calls["mic"] += 1
        return True

    def fake_request_ax() -> bool:
        calls["ax"] += 1
        return True

    def fake_request_im() -> bool:
        calls["im"] += 1
        return True

    monkeypatch.setattr(
        "ptarmigan_flow.permissions.check_all_permissions",
        fake_check_all_permissions,
    )
    monkeypatch.setattr(
        "ptarmigan_flow.permissions.request_microphone_permission",
        fake_request_mic,
    )
    monkeypatch.setattr(
        "ptarmigan_flow.permissions.request_accessibility_permission",
        fake_request_ax,
    )
    monkeypatch.setattr(
        "ptarmigan_flow.permissions.request_input_monitoring_permission",
        fake_request_im,
    )

    result = request_all_permissions()

    assert result.all_granted
    assert calls == {"mic": 1, "ax": 0, "im": 1}


def test_request_all_permissions_requests_accessibility_before_microphone(monkeypatch) -> None:
    states = [
        PermissionReport(microphone=False, accessibility=False, input_monitoring=False),
        PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
    ]
    order: list[str] = []

    def fake_check_all_permissions() -> PermissionReport:
        return states.pop(0)

    monkeypatch.setattr(
        "ptarmigan_flow.permissions.check_all_permissions",
        fake_check_all_permissions,
    )
    monkeypatch.setattr(
        "ptarmigan_flow.permissions.request_microphone_permission",
        lambda: order.append("mic") or True,
    )
    monkeypatch.setattr(
        "ptarmigan_flow.permissions.request_accessibility_permission",
        lambda: order.append("ax") or True,
    )
    monkeypatch.setattr(
        "ptarmigan_flow.permissions.request_input_monitoring_permission",
        lambda: order.append("im") or True,
    )

    result = request_all_permissions()

    assert result.all_granted
    assert order == ["ax", "im", "mic"]


def test_check_accessibility_permission_uses_application_services(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.permissions._is_macos", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "ApplicationServices",
        SimpleNamespace(AXIsProcessTrusted=lambda: True),
    )

    assert check_accessibility_permission()


def test_request_accessibility_permission_prompts_via_application_services(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.permissions._is_macos", lambda: True)
    calls = {"count": 0}

    def fake_trusted() -> bool:
        return calls["count"] > 0

    def fake_prompt(_options: dict[object, object]) -> bool:
        calls["count"] += 1
        return True

    monkeypatch.setitem(
        sys.modules,
        "ApplicationServices",
        SimpleNamespace(
            AXIsProcessTrusted=fake_trusted,
            AXIsProcessTrustedWithOptions=fake_prompt,
            kAXTrustedCheckOptionPrompt=object(),
        ),
    )

    assert request_accessibility_permission()
    assert calls["count"] == 1


def test_format_permission_guidance_includes_current_executable(monkeypatch) -> None:
    monkeypatch.setattr(sys, "executable", "/tmp/python3.11")
    report = PermissionReport(microphone=False, accessibility=True, input_monitoring=False)

    guidance = format_permission_guidance(report)

    assert "Missing macOS permissions detected:" in guidance
    assert "Current executable:" in guidance
    assert "Preferred permission target:" in guidance
    assert guidance.count("python3.11") >= 1


def test_format_permission_guidance_includes_preferred_target_and_launchd_hint(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "executable",
        "/opt/homebrew/Cellar/python@3.11/3.11.14_3/Frameworks/Python.framework/Versions/3.11/bin/python3.11",
    )
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.ptarmiganflow.daemon")
    report = PermissionReport(microphone=False, accessibility=False, input_monitoring=False)

    guidance = format_permission_guidance(report)

    assert "Preferred permission target:" in guidance
    assert (
        "/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/Resources/Python.app"
        in guidance
    )
    assert "enable the preferred target above" in guidance
    assert "Detected launchd context: com.ptarmiganflow.daemon" in guidance


def test_recommended_permission_target_falls_back_to_executable(monkeypatch) -> None:
    monkeypatch.setattr(sys, "executable", "/tmp/custom/bin/python")

    resolved = recommended_permission_target()
    assert str(resolved).endswith("/tmp/custom/bin/python")


def test_parse_permission_report_from_text_parses_expected_lines() -> None:
    text = "\n".join(
        [
            "Microphone: OK",
            "Accessibility: MISSING",
            "Input Monitoring: OK",
        ]
    )

    report = _parse_permission_report_from_text(text)

    assert report == PermissionReport(microphone=True, accessibility=False, input_monitoring=True)


def test_parse_permission_report_from_text_returns_none_when_incomplete() -> None:
    text = "\n".join(
        [
            "Microphone: OK",
            "Accessibility: OK",
        ]
    )

    report = _parse_permission_report_from_text(text)
    assert report is None


def test_check_all_permissions_subprocess_parses_stdout_and_uses_module_command() -> None:
    called: dict[str, object] = {}

    def fake_run(command, *, capture_output, text):
        called["command"] = command
        called["capture_output"] = capture_output
        called["text"] = text
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    "Microphone: OK",
                    "Accessibility: OK",
                    "Input Monitoring: OK",
                ]
            ),
            stderr="",
        )

    report = check_all_permissions_subprocess(runner=fake_run, executable="/tmp/PtarmiganFlow")

    assert report == PermissionReport(
        microphone=True,
        accessibility=True,
        input_monitoring=True,
    )
    assert called == {
        "command": [
            "/tmp/PtarmiganFlow",
            "-m",
            "ptarmigan_flow.cli",
            "check-permissions",
        ],
        "capture_output": True,
        "text": True,
    }


def test_check_all_permissions_subprocess_accepts_cli_incomplete_status_exit() -> None:
    def fake_run(_command, *, capture_output, text):
        assert capture_output is True
        assert text is True
        return SimpleNamespace(
            returncode=2,
            stdout="\n".join(
                [
                    "Microphone: OK",
                    "Accessibility: OK",
                    "Input Monitoring: MISSING",
                ]
            ),
            stderr="",
        )

    report = check_all_permissions_subprocess(runner=fake_run, executable="/tmp/PtarmiganFlow")

    assert report == PermissionReport(
        microphone=True,
        accessibility=True,
        input_monitoring=False,
    )


def test_check_all_permissions_subprocess_returns_none_for_failed_exit_or_empty_output() -> None:
    failed = check_all_permissions_subprocess(
        runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="Microphone: OK\nAccessibility: OK\nInput Monitoring: OK",
            stderr="",
        ),
        executable="/tmp/PtarmiganFlow",
    )
    empty = check_all_permissions_subprocess(
        runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
        executable="/tmp/PtarmiganFlow",
    )

    assert failed is None
    assert empty is None


def test_check_all_permissions_subprocess_uses_stderr_when_stdout_is_unparseable() -> None:
    def fake_run(_command, *, capture_output, text):
        assert capture_output is True
        assert text is True
        return SimpleNamespace(
            returncode=0,
            stdout="debug noise",
            stderr="\n".join(
                [
                    "Microphone: MISSING",
                    "Accessibility: OK",
                    "Input Monitoring: OK",
                ]
            ),
        )

    report = check_all_permissions_subprocess(runner=fake_run, executable="/tmp/PtarmiganFlow")

    assert report == PermissionReport(
        microphone=False,
        accessibility=True,
        input_monitoring=True,
    )


def test_check_all_permissions_subprocess_returns_none_when_runner_raises() -> None:
    def fake_run(_command, *, capture_output, text):
        raise OSError("boom")

    assert (
        check_all_permissions_subprocess(runner=fake_run, executable="/tmp/PtarmiganFlow")
        is None
    )


def test_check_permissions_in_launchd_context_short_circuits_on_non_macos(monkeypatch) -> None:
    expected = PermissionReport(microphone=True, accessibility=True, input_monitoring=True)
    monkeypatch.setattr("ptarmigan_flow.permissions._is_macos", lambda: False)
    monkeypatch.setattr("ptarmigan_flow.permissions.check_all_permissions", lambda: expected)
    monkeypatch.setattr(
        "ptarmigan_flow.permissions.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess should not run")),
    )

    probe = check_permissions_in_launchd_context()

    assert probe == LaunchdPermissionProbe(
        ok=True,
        command=["pflow", "check-permissions"],
        report=expected,
    )


def test_check_permissions_in_launchd_context_parses_permission_output(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.permissions._is_macos", lambda: True)
    monkeypatch.setattr("ptarmigan_flow.permissions.os.getuid", lambda: 501)

    called: dict[str, object] = {}

    def fake_run(command, *, check, capture_output, text):
        called["command"] = command
        assert check is False
        assert capture_output is True
        assert text is True
        return SimpleNamespace(
            returncode=2,
            stdout="\n".join(
                [
                    "Microphone: MISSING",
                    "Accessibility: OK",
                    "Input Monitoring: OK",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr("ptarmigan_flow.permissions.subprocess.run", fake_run)

    probe = check_permissions_in_launchd_context(command=["pflow", "check-permissions"])

    assert called["command"] == ["launchctl", "asuser", "501", "pflow", "check-permissions"]
    assert probe.ok is True
    assert probe.command == ["pflow", "check-permissions"]
    assert probe.report == PermissionReport(
        microphone=False,
        accessibility=True,
        input_monitoring=True,
    )


def test_check_permissions_in_launchd_context_reports_parse_error(monkeypatch) -> None:
    monkeypatch.setattr("ptarmigan_flow.permissions._is_macos", lambda: True)
    monkeypatch.setattr("ptarmigan_flow.permissions.os.getuid", lambda: 501)
    monkeypatch.setattr(
        "ptarmigan_flow.permissions.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="unexpected output",
            stderr="trace",
        ),
    )

    probe = check_permissions_in_launchd_context()

    assert probe.ok is False
    assert probe.report is None
    assert probe.error is not None
    assert "Could not parse permission status" in probe.error
    assert "command=pflow check-permissions" in probe.error
    assert probe.command == ["pflow", "check-permissions"]
    assert probe.stdout == "unexpected output"
    assert probe.stderr == "trace"


def test_reset_app_bundle_tcc_calls_tccutil_for_both_services(monkeypatch) -> None:
    """reset_app_bundle_tcc must invoke tccutil for Accessibility and ListenEvent."""
    import ptarmigan_flow.permissions as perms_mod

    monkeypatch.setattr(perms_mod.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    calls: list[list[str]] = []

    def fake_run(cmd, *, check, capture_output, text):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(perms_mod.subprocess, "run", fake_run)

    result = reset_app_bundle_tcc("com.example.app")

    assert result is True
    services = [c[2] for c in calls]  # tccutil reset <SERVICE> <bundle>
    assert "Accessibility" in services
    assert "ListenEvent" in services
    assert all(c[3] == "com.example.app" for c in calls)


def test_reset_app_bundle_tcc_returns_false_when_tccutil_missing(monkeypatch) -> None:
    """reset_app_bundle_tcc returns False when tccutil is not on PATH."""
    import ptarmigan_flow.permissions as perms_mod

    monkeypatch.setattr(perms_mod.shutil, "which", lambda _cmd: None)

    result = reset_app_bundle_tcc("com.example.app")

    assert result is False


def test_reset_app_bundle_tcc_returns_false_when_all_calls_fail(monkeypatch) -> None:
    """reset_app_bundle_tcc returns False when every tccutil call exits non-zero."""
    import ptarmigan_flow.permissions as perms_mod

    monkeypatch.setattr(perms_mod.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(
        perms_mod.subprocess,
        "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=1, stderr="error"),
    )

    result = reset_app_bundle_tcc("com.example.app")

    assert result is False
