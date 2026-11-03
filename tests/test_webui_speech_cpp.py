"""Run the real speech.cpp frontend lifecycle in Node's VM."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_webui_speech_cpp_lifecycle():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend lifecycle tests")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("webui_speech_cpp.test.cjs"))],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
