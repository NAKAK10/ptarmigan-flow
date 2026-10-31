"""Run focused frontend lifecycle regressions with Node's built-in test runner."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_webui_download_lifecycle() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend lifecycle tests")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("webui_download.test.cjs"))],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
