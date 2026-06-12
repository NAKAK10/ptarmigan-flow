from __future__ import annotations

import argparse

import ptarmigan_flow.presentation.cli.commands as commands
from ptarmigan_flow.config import VLLMStartupPreset, load_config, write_config
from ptarmigan_flow.presentation.cli.parser import build_parser


def _install_input_sequence(monkeypatch, responses: list[str]) -> None:
    iterator = iter(responses)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(iterator))


def test_build_parser_registers_config_command_and_sections() -> None:
    parser = build_parser()

    root_args = parser.parse_args(["config"])
    assert root_args.func is commands.cmd_config
    assert root_args.config_target is None

    section_args = parser.parse_args(["config", "audio", "--config", "/tmp/custom.toml"])
    assert section_args.func is commands.cmd_config
    assert section_args.config_target == "audio"
    assert section_args.config == "/tmp/custom.toml"


def test_cmd_config_audio_updates_section(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.toml"

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    _install_input_sequence(
        monkeypatch,
        ["22050", "2", "", "60", "", "", "", "0.5", "-", "2"],
    )

    result = commands.cmd_config(argparse.Namespace(config=str(cfg_path), config_target="audio"))

    assert result == 0
    loaded = load_config(cfg_path)
    assert loaded.audio.sample_rate == 22050
    assert loaded.audio.channels == 2
    assert loaded.audio.max_record_seconds == 60
    assert loaded.audio.trailing_silence_seconds == 0.5
    assert loaded.audio.input_device is None
    assert loaded.audio.input_device_policy.value == "system_default"


def test_cmd_config_stt_always_prompts_vllm_preset_for_non_vllm_model(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    cfg_path = tmp_path / "config.toml"

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    _install_input_sequence(monkeypatch, ["moonshine:base", "", ""])

    result = commands.cmd_config(argparse.Namespace(config=str(cfg_path), config_target="stt"))

    assert result == 0
    captured = capsys.readouterr()
    loaded = load_config(cfg_path)
    assert loaded.stt.model == "moonshine:base"
    assert loaded.stt.vllm.startup_preset == VLLMStartupPreset.OFF
    assert "Used only when stt.model is vllm:..." in captured.out


def test_cmd_config_stt_prompts_vllm_preset_when_needed(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.toml"

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    _install_input_sequence(
        monkeypatch,
        ["6", "vllm:mistralai/Voxtral-Mini-4B-Realtime-2602", "45", "2"],
    )

    result = commands.cmd_config(argparse.Namespace(config=str(cfg_path), config_target="stt"))

    assert result == 0
    loaded = load_config(cfg_path)
    assert loaded.stt.model == "vllm:mistralai/Voxtral-Mini-4B-Realtime-2602"
    assert loaded.stt.idle_shutdown_seconds == 45.0
    assert loaded.stt.vllm.startup_preset == VLLMStartupPreset.BALANCED


def test_cmd_config_text_updates_text_settings(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.toml"

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    _install_input_sequence(
        monkeypatch,
        ["/tmp/dict.toml", "1", "", "http://localhost:8080", "qwen-test", "4.5", "700", "sk-test", "y"],
    )

    result = commands.cmd_config(argparse.Namespace(config=str(cfg_path), config_target="text"))

    assert result == 0
    loaded = load_config(cfg_path)
    assert loaded.text.dictionary_path == "/tmp/dict.toml"
    assert loaded.text.llm_correction.mode.value == "always"
    assert loaded.text.llm_correction.provider == "ollama"
    assert loaded.text.llm_correction.base_url == "http://localhost:8080"
    assert loaded.text.llm_correction.model == "qwen-test"
    assert loaded.text.llm_correction.timeout_seconds == 4.5
    assert loaded.text.llm_correction.max_input_chars == 700
    assert loaded.text.llm_correction.api_key == "sk-test"
    assert loaded.text.llm_correction.enabled_tools is True


def test_cmd_config_without_section_uses_root_picker(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.toml"

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    _install_input_sequence(monkeypatch, ["output", "2", "cmd+shift+v"])

    result = commands.cmd_config(argparse.Namespace(config=str(cfg_path), config_target=None))

    assert result == 0
    loaded = load_config(cfg_path)
    assert loaded.output.mode.value == "clipboard_paste"
    assert loaded.output.paste_shortcut == "cmd+shift+v"


def test_cmd_config_returns_2_when_noninteractive(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(commands, "_is_interactive_session", lambda: False)

    result = commands.cmd_config(argparse.Namespace(config=str(cfg_path), config_target="audio"))

    assert result == 2


def test_cmd_config_cancel_does_not_write_changes(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.toml"
    config = load_config(cfg_path)
    config.language = "en"
    write_config(cfg_path, config)

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(KeyboardInterrupt()))

    result = commands.cmd_config(argparse.Namespace(config=str(cfg_path), config_target="language"))

    assert result == 130
    loaded = load_config(cfg_path)
    assert loaded.language == "en"


def test_cmd_config_language_saves_described_builtin_choice(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    cfg_path = tmp_path / "config.toml"

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    _install_input_sequence(monkeypatch, ["2"])

    result = commands.cmd_config(argparse.Namespace(config=str(cfg_path), config_target="language"))

    assert result == 0
    loaded = load_config(cfg_path)
    assert loaded.language == "ja"
    captured = capsys.readouterr()
    assert "en" in captured.out
    assert "English transcription/correction" in captured.out
    assert "ja" in captured.out
    assert "Japanese transcription/correction" in captured.out
    assert "zh" in captured.out
    assert "Chinese transcription/correction" in captured.out


def test_cmd_config_language_saves_custom_choice(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.toml"

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    _install_input_sequence(monkeypatch, ["4", "fr"])

    result = commands.cmd_config(argparse.Namespace(config=str(cfg_path), config_target="language"))

    assert result == 0
    loaded = load_config(cfg_path)
    assert loaded.language == "fr"


def test_cmd_init_uses_shared_full_editor(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.toml"

    def _fake_edit_all(config) -> None:
        config.language = "ja"

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(commands, "_edit_all_config_sections", _fake_edit_all)

    result = commands.cmd_init(argparse.Namespace(config=str(cfg_path)))

    assert result == 0
    loaded = load_config(cfg_path)
    assert loaded.language == "ja"
