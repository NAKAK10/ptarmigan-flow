"""Run onboarding listener regressions with Node's built-in test runner."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_webui_onboarding_language_validation() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend listener tests")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("webui_onboarding.test.cjs"))],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
