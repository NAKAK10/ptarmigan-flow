from __future__ import annotations

import json
import os
import re
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
    assert "login_item" in source
    assert "open_config" in source
    assert "Microphone" in source
    assert "Accessibility" in source
    assert "Input Monitoring" in source
    assert "WebUIController" in source
    assert "WebBridgeDispatcher" in source


def test_pyinstaller_spec_builds_ptarmiganflow_app() -> None:
    spec = (ROOT / "packaging/macos/PtarmiganFlow.spec").read_text(encoding="utf-8")

    assert "src/ptarmigan_flow/macos_app.py" in spec
    assert "ROOT = Path(SPECPATH).parents[1]" in spec
    assert 'APP_ICON = ROOT / "assets/icon/PtarmiganFlow.icns"' in spec
    assert 'WEBUI_SOURCE = ROOT / "src/ptarmigan_flow/webui"' in spec
    assert '(str(WEBUI_SOURCE), "ptarmigan_flow/webui")' in spec
    assert "str(ROOT / \"src/ptarmigan_flow/macos_app.py\")" in spec
    assert "name='PtarmiganFlow'" in spec
    assert "icon=str(APP_ICON)" in spec
    assert "bundle_identifier='com.ptarmiganflow.app'" in spec
    assert '"CFBundleIconFile": "PtarmiganFlow.icns"' in spec
    assert "NSMicrophoneUsageDescription" in spec
    assert "LSMinimumSystemVersion" in spec
    assert '"LSUIElement": True' in spec
    assert 'PYINSTALLER_TARGET_ARCH' in spec
    assert "target_arch=TARGET_ARCH" in spec


def test_macos_release_icon_asset_exists() -> None:
    assert (ROOT / "assets/icon/ptarmigan-logo.png").is_file()
    assert (ROOT / "assets/icon/PtarmiganFlow.icns").is_file()
    assert (ROOT / "src/ptarmigan_flow/resources/PtarmiganFlow.icns").is_file()


def test_pyproject_packages_icon_resource_in_wheel() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    wheel_target = payload["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert "src/ptarmigan_flow" in wheel_target["packages"]


def test_macos_app_pyobjc_helpers_are_python_only() -> None:
    source = (ROOT / "src/ptarmigan_flow/web_ui.py").read_text(encoding="utf-8")

    for method_name in ["dispatch_to_javascript", "push_event", "show"]:
        assert re.search(
            rf"@objc\.python_method\n\s+def {method_name}\(",
            source,
        ), f"{method_name} should not be exposed as an Objective-C selector"


def test_webui_window_is_not_released_when_closed() -> None:
    source = (ROOT / "src/ptarmigan_flow/web_ui.py").read_text(encoding="utf-8")

    assert "setReleasedWhenClosed_(False)" in source


def test_macos_app_handles_reopen_for_accessory_app() -> None:
    source = (ROOT / "src/ptarmigan_flow/macos_app.py").read_text(encoding="utf-8")

    assert "applicationShouldHandleReopen_hasVisibleWindows_" in source


def test_webui_assets_exist_and_do_not_hardcode_english_ui_copy() -> None:
    webui = ROOT / "src/ptarmigan_flow/webui"

    assert (webui / "index.html").is_file()
    assert (webui / "app.css").is_file()
    assert (webui / "app.js").is_file()
    script = (webui / "app.js").read_text(encoding="utf-8")
    assert "window.webkit.messageHandlers.bridge.postMessage" in script
    assert "window.app.dispatch" in script
    assert "strings." in script
    assert "data-select-model" in script
    assert "settingsDraft.model = button.dataset.selectModel" in script
    assert "data-download-model" in script
    assert 'data-action="restart"' in script
    for literal in (
        "Start Voice Input",
        "Settings",
        "Dictionary Editor",
        "Allow",
        "Open System Settings",
        "Download",
    ):
        assert f'"{literal}"' not in script
        assert f"`{literal}`" not in script


def test_pyinstaller_spec_bundles_mlx_backends() -> None:
    spec = (ROOT / "packaging/macos/PtarmiganFlow.spec").read_text(encoding="utf-8")

    assert "RELEASE_BACKEND_PACKAGES" in spec
    release_packages_block = spec.split("RELEASE_BACKEND_PACKAGES", 1)[1].split(")", 1)[0]
    # Moonshine plus the MLX inference engines are bundled so any supported MLX
    # model works locally; weights download on demand at runtime.
    assert '"moonshine_voice"' in release_packages_block
    assert '"mlx"' in release_packages_block
    assert '"mlx_audio"' in release_packages_block
    assert '"mlx_whisper"' in release_packages_block
    assert '"voxmlx"' in release_packages_block
    assert '"mistral_common"' in release_packages_block
    # torch stays excluded; transformers (Granite tokenizer) and numba/scipy
    # (imported by mlx_whisper at import time) must be bundled for MLX presets.
    package_excludes_block = spec.split("OPTIONAL_BACKEND_PACKAGE_EXCLUDES = (", 1)[1].split(
        ")", 1
    )[0]
    assert '"torch"' in package_excludes_block
    for package in ("transformers", "numba", "llvmlite", "scipy"):
        assert f'"{package}"' not in package_excludes_block
    # MLX backend modules are now shipped; only the Transformers fallbacks and
    # the test helper remain excluded.
    module_excludes_block = spec.split("OPTIONAL_BACKEND_MODULE_PREFIXES", 1)[1].split(")", 1)[0]
    assert "ptarmigan_flow.stt.granite_mlx" not in module_excludes_block
    assert "ptarmigan_flow.stt.granite_transformers" in module_excludes_block
    assert "ptarmigan_flow.stt.voxtral_transformers" in module_excludes_block
    assert "MOONSHINE_EXCLUDED_HIDDENIMPORTS" in spec
    assert "moonshine_voice.libmoonshine" in spec


def test_pyinstaller_spec_uses_moonshine_only_backend_packages_on_x86_64() -> None:
    spec = (ROOT / "packaging/macos/PtarmiganFlow.spec").read_text(encoding="utf-8")

    assert "X86_64_BACKEND_PACKAGES" in spec
    x86_packages_block = spec.split("X86_64_BACKEND_PACKAGES", 1)[1].split(")", 1)[0]
    assert '"moonshine_voice"' in x86_packages_block
    assert '"mlx"' not in x86_packages_block
    assert '"mlx_audio"' not in x86_packages_block
    assert '"mlx_whisper"' not in x86_packages_block
    assert '"voxmlx"' not in x86_packages_block
    assert '"mistral_common"' not in x86_packages_block
    assert "def _release_backend_packages(arch: str)" in spec
    assert 'if arch == "x86_64":' in spec
    assert "return X86_64_BACKEND_PACKAGES" in spec
    assert "return RELEASE_BACKEND_PACKAGES" in spec
    assert "_backend_packages = _release_backend_packages(TARGET_ARCH)" in spec
    assert "for package in _backend_packages:" in spec


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
    assert "strategy:" in workflow
    assert "matrix:" in workflow
    assert "macos-15" in workflow
    assert "macos-15-intel" in workflow
    assert "arch: arm64" in workflow
    assert "arch: x86_64" in workflow
    assert "moonshine_source: pypi" in workflow
    assert "moonshine_source: source" in workflow
    assert "PYINSTALLER_TARGET_ARCH: ${{ matrix.arch }}" in workflow
    assert "pyinstaller" in workflow.lower()
    assert "uv sync --extra dev" not in workflow
    assert "packaging/macos/requirements-release.txt" in workflow
    assert ".release-venv" in workflow
    assert "uv venv .release-venv --python 3.11 --managed-python" in workflow
    assert "uv pip install --python .release-venv" in workflow
    assert "moonshine-ai/moonshine" in workflow
    # Pinned to the original v0.0.49 commit because upstream moved the tag
    # to a rewritten commit that no longer builds from source.
    assert "ref: 15c8bb9412c3db3b78c56e897e6574dbee51a0d2" in workflow
    assert "lfs: true" in workflow
    assert "onnxruntime-osx-x86_64" in workflow
    assert "libonnxruntime.*.dylib" in workflow
    assert "cmake -B build-moonshine -S moonshine-src/core" in workflow
    assert "cmake --build build-moonshine --target moonshine -j 4" in workflow
    assert "MACOSX_DEPLOYMENT_TARGET: \"13.0\"" in workflow
    assert "Validate Apple release secrets" in workflow
    assert "Missing GitHub secret:" in workflow
    assert "Developer ID Application" in workflow
    assert "codesign" in workflow
    assert "notarytool submit" in workflow
    assert "for action in staple validate; do" in workflow
    assert 'xcrun stapler "$action" dist/PtarmiganFlow.app' in workflow
    assert "PtarmiganFlow-macos-arm64.zip" in workflow
    assert "PtarmiganFlow-macos-x86_64.zip" in workflow
    assert "actions/checkout@v7" in workflow
    assert "astral-sh/setup-uv@v8.2.0" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "actions/download-artifact@v8" in workflow
    assert "if: ${{ !cancelled() }}" in workflow
    assert "Mandatory arm64 release artifact is missing" in workflow
    assert "::warning::Optional x86_64 release artifact is missing" in workflow
    assert "fail_on_unmatched_files" not in workflow
    assert "Validate Moonshine native libraries" in workflow
    assert "softprops/action-gh-release@v3" in workflow
    assert "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24" not in workflow
    assert "draft: true" in workflow
    assert "APPLE_CERTIFICATE_BASE64" in workflow
    assert "APPLE_CERTIFICATE_PASSWORD" in workflow
    assert "APPLE_TEAM_ID" in workflow
    assert "APPLE_ID" in workflow
    assert "APPLE_APP_SPECIFIC_PASSWORD" in workflow


def test_pyproject_includes_webkit_pyobjc_for_wkwebview_release_builds() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependencies_block = pyproject.split("dependencies = [", 1)[1].split("\n]", 1)[0]

    assert re.search(
        r"^\s+\"pyobjc-framework-WebKit>=11\.0; platform_system == 'Darwin'\",\s*$",
        dependencies_block,
        re.MULTILINE,
    )


def test_release_requirements_include_pyproject_pyobjc_dependencies() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pyobjc_dependencies = {
        dep.split(";", 1)[0].strip()
        for dep in payload["project"]["dependencies"]
        if dep.startswith("pyobjc-")
    }
    release_requirements = {
        line.strip()
        for line in (ROOT / "packaging/macos/requirements-release.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert "pyobjc-framework-WebKit>=11.0" in pyobjc_dependencies
    assert pyobjc_dependencies <= release_requirements


def test_pyinstaller_spec_fails_when_runtime_modules_are_missing() -> None:
    spec = (ROOT / "packaging/macos/PtarmiganFlow.spec").read_text(encoding="utf-8")
    required_block = spec.split("REQUIRED_RUNTIME_MODULES = (", 1)[1].split(")", 1)[0]

    for module in ("objc", "AppKit", "Foundation", "WebKit", "Quartz", "AVFoundation"):
        assert f'"{module}"' in required_block
    assert "find_spec(name) is None" in spec
    assert "raise SystemExit(" in spec


ARM64_MARKER = "platform_system == 'Darwin' and platform_machine == 'arm64'"


def _release_requirement_lines() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / "packaging/macos/requirements-release.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    ]


def test_release_requirements_include_arm64_mlx_backends() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pyproject_specs = {
        dep.split(";", 1)[0].strip()
        for dep in payload["project"]["dependencies"]
        if dep.startswith(("mlx-", "voxmlx", "mistral-common"))
    }
    release_lines = _release_requirement_lines()
    arm64_release_specs = {
        line.split(";", 1)[0].strip()
        for line in release_lines
        if ";" in line and line.split(";", 1)[1].strip() == ARM64_MARKER
    }

    assert {
        "mlx-audio>=0.4.1",
        "mlx-whisper>=0.4.2",
        "voxmlx>=0.0.2",
        "mistral-common[audio]>=1.5.0",
    } <= pyproject_specs
    # Same version floors as pyproject.toml, but arm64-only so the x86_64 app
    # build does not try to install (or bundle) MLX.
    assert pyproject_specs <= arm64_release_specs
    assert any(spec.startswith("huggingface-hub") for spec in arm64_release_specs)
    for line in release_lines:
        name = re.split(r"[<>=\[;]", line, maxsplit=1)[0].strip().lower()
        if name in {"mlx-audio", "mlx-whisper", "voxmlx", "mistral-common", "huggingface-hub"}:
            assert line.endswith(f"; {ARM64_MARKER}"), line


def test_pyinstaller_spec_fails_arm64_build_without_mlx_runtime_modules() -> None:
    spec = (ROOT / "packaging/macos/PtarmiganFlow.spec").read_text(encoding="utf-8")
    arm64_block = spec.split("REQUIRED_ARM64_RUNTIME_MODULES = (", 1)[1].split(")", 1)[0]

    for module in (
        "mlx",
        "mlx_audio",
        "mlx_whisper",
        "voxmlx",
        "mistral_common",
        "huggingface_hub",
        "transformers",
        "numba",
        "scipy",
    ):
        assert f'"{module}"' in arm64_block
    assert "def _required_runtime_modules(arch: str)" in spec
    assert "return REQUIRED_RUNTIME_MODULES + REQUIRED_ARM64_RUNTIME_MODULES" in spec
    assert "for name in _required_runtime_modules(TARGET_ARCH) if find_spec(name) is None" in spec
