from __future__ import annotations

from ptarmigan_flow.text_processing.default_corrections import build_default_ruleset


def test_build_default_ruleset_has_no_disabled_regex() -> None:
    ruleset = build_default_ruleset()

    assert ruleset.regex_count > 0
    # Every canonical in the default table must have produced at least one rule,
    # i.e. nothing was silently disabled by the builder.
    canonicals = {rule.canonical for rule in ruleset.regex_rules}
    for expected in ("Claude", "ChatGPT", "OpenAI", "Gemini", "Copilot", "Anthropic"):
        assert expected in canonicals


def test_default_ruleset_corrects_claude_mid_sentence() -> None:
    ruleset = build_default_ruleset()

    assert ruleset.apply("クロードで実装して") == "Claudeで実装して"


def test_default_ruleset_corrects_chatgpt() -> None:
    ruleset = build_default_ruleset()

    assert ruleset.apply("チャットGPT") == "ChatGPT"
    assert ruleset.apply("チャットジーピーティー") == "ChatGPT"
