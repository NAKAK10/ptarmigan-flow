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
        assert "docs/release-prep.md" in text


def test_release_prep_doc_lists_required_secrets_and_commands() -> None:
    text = Path("docs/release-prep.md").read_text(encoding="utf-8")

    for secret in [
        "APPLE_CERTIFICATE_BASE64",
        "APPLE_CERTIFICATE_PASSWORD",
        "APPLE_TEAM_ID",
        "APPLE_ID",
        "APPLE_APP_SPECIFIC_PASSWORD",
    ]:
        assert secret in text

    assert "gh secret set" in text
    assert "base64" in text
    assert "release-macos-app.yml" in text
    assert "pages.yml" in text
