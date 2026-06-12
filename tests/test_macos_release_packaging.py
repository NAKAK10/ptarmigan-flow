from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_includes_pyinstaller_for_release_builds() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dev_dependencies = payload["project"]["optional-dependencies"]["dev"]

    assert any(dep.startswith("pyinstaller") for dep in dev_dependencies)


def test_macos_app_entrypoint_contains_onboarding_controls() -> None:
    source = (ROOT / "src/ptarmigan_flow/macos_app.py").read_text(encoding="utf-8")

    assert "def main(" in source
    assert "check_all_permissions" in source
    assert "request_microphone_permission" in source
    assert "request_accessibility_permission" in source
    assert "request_input_monitoring_permission" in source
    assert "install_launch_agent" in source
    assert "open_config" in source
    assert "Microphone" in source
    assert "Accessibility" in source
    assert "Input Monitoring" in source


def test_pyinstaller_spec_builds_ptarmiganflow_app() -> None:
    spec = (ROOT / "packaging/macos/PtarmiganFlow.spec").read_text(encoding="utf-8")

    assert "src/ptarmigan_flow/macos_app.py" in spec
    assert "ROOT = Path(SPECPATH).parents[1]" in spec
    assert "str(ROOT / \"src/ptarmigan_flow/macos_app.py\")" in spec
    assert "name='PtarmiganFlow'" in spec
    assert "bundle_identifier='com.ptarmiganflow.app'" in spec
    assert "NSMicrophoneUsageDescription" in spec
    assert "LSMinimumSystemVersion" in spec
    assert "target_arch='arm64'" in spec


def test_release_workflow_builds_notarizes_and_uploads_draft_release() -> None:
    workflow = (ROOT / ".github/workflows/release-macos-app.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in workflow
    assert "tag:" in workflow
    assert "macos-14" in workflow
    assert "pyinstaller" in workflow.lower()
    assert "Validate Apple release secrets" in workflow
    assert "Missing GitHub secret:" in workflow
    assert "Developer ID Application" in workflow
    assert "codesign" in workflow
    assert "notarytool submit" in workflow
    assert "stapler staple" in workflow
    assert "PtarmiganFlow-macos-arm64.zip" in workflow
    assert "softprops/action-gh-release@v2" in workflow
    assert "draft: true" in workflow
    assert "APPLE_CERTIFICATE_BASE64" in workflow
    assert "APPLE_CERTIFICATE_PASSWORD" in workflow
    assert "APPLE_TEAM_ID" in workflow
    assert "APPLE_ID" in workflow
    assert "APPLE_APP_SPECIFIC_PASSWORD" in workflow
