from __future__ import annotations

import argparse

import pytest

import ptarmigan_flow.presentation.cli.commands as commands
from ptarmigan_flow.config import AppConfig, load_config, write_config
from ptarmigan_flow.presentation.cli.parser import build_parser


def _write_glossary(cfg_path, terms: list[str]) -> None:
    config = AppConfig()
    config.text.llm_correction.glossary = list(terms)
    write_config(cfg_path, config)


def test_build_parser_registers_glossary_commands() -> None:
    parser = build_parser()

    root_args = parser.parse_args(["glossary"])
    assert root_args.func is commands.cmd_glossary

    list_args = parser.parse_args(["glossary", "list", "--config", "/tmp/custom.toml"])
    assert list_args.func is commands.cmd_glossary_list
    assert list_args.config == "/tmp/custom.toml"

    add_args = parser.parse_args(["glossary", "add", "issue", "PR", "--config", "/tmp/c.toml"])
    assert add_args.func is commands.cmd_glossary_add
    assert add_args.terms == ["issue", "PR"]
    assert add_args.config == "/tmp/c.toml"

    remove_args = parser.parse_args(["glossary", "remove", "issue"])
    assert remove_args.func is commands.cmd_glossary_remove
    assert remove_args.terms == ["issue"]
    assert remove_args.config is None


def test_help_lists_glossary_command(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    assert "glossary" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        parser.parse_args(["glossary", "--help"])
    out = capsys.readouterr().out
    assert "list" in out
    assert "add" in out
    assert "remove" in out


def test_cmd_glossary_without_target_prints_usage(capsys) -> None:
    result = commands.cmd_glossary(argparse.Namespace())

    assert result == 0
    assert "pflow glossary add" in capsys.readouterr().out


def test_cmd_glossary_list_empty_prints_hint(tmp_path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"

    result = commands.cmd_glossary_list(argparse.Namespace(config=str(cfg_path)))

    assert result == 0
    assert "pflow glossary add" in capsys.readouterr().out


def test_cmd_glossary_list_prints_terms_in_order(tmp_path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    _write_glossary(cfg_path, ["issue", "PR", "Ptarmigan"])

    result = commands.cmd_glossary_list(argparse.Namespace(config=str(cfg_path)))

    assert result == 0
    assert capsys.readouterr().out.splitlines() == ["issue", "PR", "Ptarmigan"]


def test_cmd_glossary_add_strips_and_skips_duplicates(tmp_path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    _write_glossary(cfg_path, ["issue"])

    result = commands.cmd_glossary_add(
        argparse.Namespace(config=str(cfg_path), terms=["  PR ", "issue", "PR", "Ptarmigan"])
    )

    assert result == 0
    captured = capsys.readouterr()
    assert "Already registered, skipped: issue" in captured.out
    assert "Already registered, skipped: PR" in captured.out
    assert "Added: PR" in captured.out
    assert "Added: Ptarmigan" in captured.out
    loaded = load_config(cfg_path)
    assert loaded.text.llm_correction.glossary == ["issue", "PR", "Ptarmigan"]


def test_cmd_glossary_add_rejects_empty_terms(tmp_path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    _write_glossary(cfg_path, ["issue"])

    result = commands.cmd_glossary_add(argparse.Namespace(config=str(cfg_path), terms=["", "   "]))

    assert result != 0
    assert "Error" in capsys.readouterr().err
    assert load_config(cfg_path).text.llm_correction.glossary == ["issue"]


def test_cmd_glossary_add_mixed_empty_and_valid_terms(tmp_path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"

    result = commands.cmd_glossary_add(
        argparse.Namespace(config=str(cfg_path), terms=[" ", "issue"])
    )

    assert result == 0
    captured = capsys.readouterr()
    assert "empty" in captured.err
    assert "Added: issue" in captured.out
    assert load_config(cfg_path).text.llm_correction.glossary == ["issue"]


def test_cmd_glossary_add_only_duplicates_does_not_write(tmp_path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    _write_glossary(cfg_path, ["issue"])
    before = cfg_path.read_text(encoding="utf-8")

    result = commands.cmd_glossary_add(argparse.Namespace(config=str(cfg_path), terms=["issue"]))

    assert result == 0
    assert "Already registered, skipped: issue" in capsys.readouterr().out
    assert cfg_path.read_text(encoding="utf-8") == before


def test_cmd_glossary_remove_removes_and_notifies_unknown(tmp_path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    _write_glossary(cfg_path, ["issue", "PR", "Ptarmigan"])

    result = commands.cmd_glossary_remove(
        argparse.Namespace(config=str(cfg_path), terms=[" PR ", "unknown"])
    )

    assert result == 0
    captured = capsys.readouterr()
    assert "Removed: PR" in captured.out
    assert "Not registered, skipped: unknown" in captured.out
    assert load_config(cfg_path).text.llm_correction.glossary == ["issue", "Ptarmigan"]


def test_cmd_glossary_remove_nothing_removed_exits_non_zero(tmp_path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    _write_glossary(cfg_path, ["issue"])
    before = cfg_path.read_text(encoding="utf-8")

    result = commands.cmd_glossary_remove(
        argparse.Namespace(config=str(cfg_path), terms=["unknown"])
    )

    assert result != 0
    captured = capsys.readouterr()
    assert "Not registered, skipped: unknown" in captured.out
    assert "Error" in captured.err
    assert cfg_path.read_text(encoding="utf-8") == before


def test_cmd_glossary_add_preserves_other_config_values(tmp_path) -> None:
    cfg_path = tmp_path / "config.toml"
    config = AppConfig()
    config.language = "ja"
    config.stt.model = "moonshine:base"
    config.audio.sample_rate = 22050
    config.text.llm_correction.model = "custom-model"
    config.text.llm_correction.timeout_seconds = 3.0
    config.text.llm_correction.glossary = ["issue"]
    write_config(cfg_path, config)

    assert commands.cmd_glossary_add(argparse.Namespace(config=str(cfg_path), terms=["PR"])) == 0
    assert (
        commands.cmd_glossary_remove(argparse.Namespace(config=str(cfg_path), terms=["issue"])) == 0
    )

    loaded = load_config(cfg_path)
    assert loaded.text.llm_correction.glossary == ["PR"]
    assert loaded.language == "ja"
    assert loaded.stt.model == "moonshine:base"
    assert loaded.audio.sample_rate == 22050
    assert loaded.text.llm_correction.model == "custom-model"
    assert loaded.text.llm_correction.timeout_seconds == 3.0
