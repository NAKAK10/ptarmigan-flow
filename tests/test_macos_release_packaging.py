from __future__ import annotations

import json
import os
import subprocess
import sys
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


def test_pyinstaller_spec_collects_only_release_backend_packages() -> None:
    spec = (ROOT / "packaging/macos/PtarmiganFlow.spec").read_text(encoding="utf-8")

    assert "RELEASE_BACKEND_PACKAGES" in spec
    release_packages_block = spec.split("RELEASE_BACKEND_PACKAGES", 1)[1].split(")", 1)[0]
    assert '"moonshine_voice"' in spec
    assert '"mlx_audio"' not in release_packages_block
    assert '"mlx_whisper"' not in release_packages_block
    assert '"voxmlx"' not in release_packages_block
    assert '"mistral_common"' not in release_packages_block
    assert '"torch"' in spec
    assert '"transformers"' in spec
    assert "OPTIONAL_BACKEND_MODULE_PREFIXES" in spec
    assert "ptarmigan_flow.stt.granite_mlx" in spec
    assert "ptarmigan_flow.stt.granite_transformers" in spec
    assert "MOONSHINE_EXCLUDED_HIDDENIMPORTS" in spec
    assert "moonshine_voice.libmoonshine" in spec


def test_stt_factory_import_does_not_import_optional_backend_modules() -> None:
    code = """
import json
import sys
import ptarmigan_flow.stt.factory  # noqa: F401

optional_modules = [
    "ptarmigan_flow.stt.granite_mlx",
    "ptarmigan_flow.stt.granite_transformers",
    "ptarmigan_flow.stt.mlx_whisper",
    "ptarmigan_flow.stt.voxtral_mlx",
    "ptarmigan_flow.stt.voxtral_transformers",
]
loaded = [name for name in optional_modules if name in sys.modules]
print(json.dumps(loaded))
raise SystemExit(1 if loaded else 0)
""".strip()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    loaded = json.loads(result.stdout.strip() or "[]")
    assert result.returncode == 0, f"optional backend modules were imported: {loaded}"


def test_release_workflow_builds_notarizes_and_uploads_draft_release() -> None:
    workflow = (ROOT / ".github/workflows/release-macos-app.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in workflow
    assert "tag:" in workflow
    assert "macos-15" in workflow
    assert "pyinstaller" in workflow.lower()
    assert "uv sync --extra dev" not in workflow
    assert "packaging/macos/requirements-release.txt" in workflow
    assert ".release-venv" in workflow
    assert "uv venv .release-venv --python 3.11 --managed-python" in workflow
    assert "uv pip install --python .release-venv" in workflow
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
