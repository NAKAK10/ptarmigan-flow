"""Execute frontend save handlers without native macOS effects."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ptarmigan_flow.onboarding_strings import strings_for

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("language", ["en", "ja", "zh"])
def test_save_feedback_and_retry(language: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend handler regression tests")
    result = subprocess.run(
        [
            node,
            str(ROOT / "tests/webui_save_feedback.cjs"),
            str(ROOT / "src/ptarmigan_flow/webui/app.js"),
        ],
        input=json.dumps(strings_for(language)),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "save feedback regression checks passed" in result.stdout
