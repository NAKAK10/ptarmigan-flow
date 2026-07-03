from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from ptarmigan_flow.text_processing.corrections import (
    CompiledRegexRule,
    CorrectionRuleSet,
    merge_rulesets,
)
from ptarmigan_flow.text_processing.service import CorrectionService
from ptarmigan_flow.transcription_corrections import (
    CorrectionDictionaryError,
    default_dictionary_path,
    load_corrections_dictionary,
    resolve_dictionary_path,
)


@dataclass
class _StubTextConfig:
    dictionary_path: str | None


@dataclass
class _StubConfig:
    text: _StubTextConfig


def test_load_missing_default_dictionary_is_disabled_without_warning(tmp_path: Path) -> None:
    path = tmp_path / "missing.toml"

    result = load_corrections_dictionary(path, explicitly_configured=False)

    assert result.loaded is False
    assert result.rules.exact_count == 0
    assert result.rules.regex_count == 0
    assert result.warnings == []


def test_load_missing_explicit_dictionary_warns_and_continues(tmp_path: Path) -> None:
    path = tmp_path / "missing.toml"

    result = load_corrections_dictionary(path, explicitly_configured=True)

    assert result.loaded is False
    assert len(result.warnings) == 1
    assert "not found" in result.warnings[0].message


def test_load_invalid_toml_reports_line_and_column(tmp_path: Path) -> None:
    path = tmp_path / "dictionary.toml"
    path.write_text('[exact]\n"Ptarmigan Flow" = ["a"\n', encoding="utf-8")

    with pytest.raises(CorrectionDictionaryError) as exc_info:
        load_corrections_dictionary(path, explicitly_configured=False)

    message = str(exc_info.value)
    assert str(path) in message
    assert "line" in message or "at" in message


def test_apply_exact_and_regex_rules(tmp_path: Path) -> None:
    path = tmp_path / "dictionary.toml"
    path.write_text(
        """
[exact]
"Ptarmigan Flow" = ["ぷたーみがんふろー", "ぷたーみがんふ"]

[regex]
"Ptarmigan Flow" = ["ぷたーみがんふ(ろー)?"]
"GPT" = ["(?i)じーぴーてぃー"]
""".strip(),
        encoding="utf-8",
    )

    result = load_corrections_dictionary(path, explicitly_configured=False)

    assert result.loaded is True
    assert result.rules.apply("ぷたーみがんふ") == "Ptarmigan Flow"
    assert result.rules.apply("これは ぷたーみがんふろー です") == "これはPtarmigan Flowです"
    assert result.rules.apply("じーぴーてぃー") == "GPT"


def test_invalid_regex_is_disabled_with_warning(tmp_path: Path) -> None:
    path = tmp_path / "dictionary.toml"
    path.write_text(
        """
[regex]
"Ptarmigan Flow" = ["(invalid"]
""".strip(),
        encoding="utf-8",
    )

    result = load_corrections_dictionary(path, explicitly_configured=False)

    assert result.loaded is True
    assert result.rules.regex_count == 0
    assert result.disabled_regex_count == 1
    assert len(result.warnings) == 1


def test_resolve_dictionary_path_defaults_and_explicit(tmp_path: Path) -> None:
    default_path, explicit_default = resolve_dictionary_path(None)
    assert default_path == default_dictionary_path()
    assert explicit_default is False

    explicit_path, explicit = resolve_dictionary_path(
        "dictionary.toml",
        config_path=tmp_path / "config.toml",
    )
    assert explicit is True
    assert explicit_path == (tmp_path / "dictionary.toml").resolve()


def test_load_for_config_merges_defaults_without_dictionary_file(tmp_path: Path) -> None:
    # Point at a non-existent file so there are no user rules, only defaults.
    missing = tmp_path / "missing.toml"
    config = _StubConfig(text=_StubTextConfig(dictionary_path=str(missing)))
    service = CorrectionService.create_default()

    result = service.load_for_config(config=config, config_path=tmp_path / "config.toml")

    assert result.loaded is False
    assert result.rules.apply("クロードで実装して") == "Claudeで実装して"


def test_load_for_config_user_rule_wins_over_default(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.toml"
    dictionary.write_text(
        """
[regex]
"CLAUDE_CUSTOM" = ["クロード"]
""".strip(),
        encoding="utf-8",
    )
    config = _StubConfig(text=_StubTextConfig(dictionary_path=str(dictionary)))
    service = CorrectionService.create_default()

    result = service.load_for_config(config=config, config_path=tmp_path / "config.toml")

    assert result.loaded is True
    # The user's mapping (クロード -> CLAUDE_CUSTOM) wins over the default Claude rule.
    assert result.rules.apply("クロードで実装して") == "CLAUDE_CUSTOMで実装して"


def _regex_rule(canonical: str, pattern: str, order: int) -> CompiledRegexRule:
    import re

    return CompiledRegexRule(
        canonical=canonical,
        pattern=pattern,
        compiled=re.compile(pattern),
        order=order,
    )


def test_merge_rulesets_override_wins_on_exact_and_regex_order() -> None:
    base = CorrectionRuleSet(
        exact_lookup={"a": "BASE_A", "b": "BASE_B"},
        regex_rules=[_regex_rule("BASE", "x", 0)],
    )
    override = CorrectionRuleSet(
        exact_lookup={"a": "OVERRIDE_A"},
        regex_rules=[_regex_rule("OVERRIDE", "x", 0)],
    )

    merged = merge_rulesets(base, override)

    # Exact: override wins on shared key, base-only key preserved.
    assert merged.exact_lookup == {"a": "OVERRIDE_A", "b": "BASE_B"}
    # Regex: override placed first with renumbered orders 0..n.
    assert [(r.canonical, r.order) for r in merged.regex_rules] == [
        ("OVERRIDE", 0),
        ("BASE", 1),
    ]
    # And the override (smaller order) wins the tie during application.
    assert merged.apply("x") == "OVERRIDE"
