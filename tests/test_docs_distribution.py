from __future__ import annotations

from pathlib import Path


def test_readmes_document_language_and_model_selection() -> None:
    for path in [Path("README.md"), Path("README.ja.md"), Path("README.llm.md")]:
        text = path.read_text(encoding="utf-8")
        assert "pflow config language" in text
        assert "pflow init" in text
        assert "en" in text
        assert "ja" in text
        assert "zh" in text
        assert "pflow list model --hub-search" in text
        assert "unverified" in text.lower() or "未検証" in text


def test_readmes_document_distribution_paths() -> None:
    for path in [Path("README.md"), Path("README.ja.md"), Path("README.llm.md")]:
        text = path.read_text(encoding="utf-8")
        assert "PtarmiganFlow-macos-arm64.zip" in text
        assert "Homebrew" in text
        assert "Microphone" in text or "マイク" in text
        assert "Accessibility" in text or "アクセシビリティ" in text
        assert "Input Monitoring" in text or "入力監視" in text
