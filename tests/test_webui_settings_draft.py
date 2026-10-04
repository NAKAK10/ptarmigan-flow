"""Run the dependency-free app.js draft regression tests when Node is available."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_settings_draft_javascript() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the web UI regression tests")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            node,
            "--test",
            str(root / "tests/webui_settings_draft.test.cjs"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
