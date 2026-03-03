import argparse
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from moonshine_flow import cli
from moonshine_flow.permissions import LaunchdPermissionProbe, PermissionReport
from moonshine_flow.text_processing.repository import TomlCorrectionRepository


def test_cmd_run_requests_missing_permissions_in_launchd_context(monkeypatch) -> None:
    fake_daemon_mod = ModuleType("moonshine_flow.daemon")
    calls = {"mic": 0, "ax": 0, "im": 0, "stop": 0}
    permission_states = [
        PermissionReport(microphone=False, accessibility=False, input_monitoring=False),
        PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
    ]

    class FakeDaemon:
        def __init__(self, _config, post_processor=None) -> None:
            del post_processor
            self.transcriber = SimpleNamespace(preflight_model=lambda: "moonshine-voice")

        def run_forever(self) -> None:
            raise KeyboardInterrupt

        def stop(self) -> None:
            calls["stop"] += 1

    fake_daemon_mod.MoonshineFlowDaemon = FakeDaemon
    monkeypatch.setitem(sys.modules, "moonshine_flow.daemon", fake_daemon_mod)
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: SimpleNamespace(runtime=SimpleNamespace(log_level="INFO")),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda _level: None)
    monkeypatch.setattr(cli, "_has_moonshine_backend", lambda: True)
    monkeypatch.setattr(cli, "consume_restart_permission_suppression", lambda: False)
    monkeypatch.setattr(cli, "check_all_permissions", lambda: permission_states.pop(0))
    monkeypatch.setattr(
        cli,
        "request_microphone_permission",
        lambda: calls.__setitem__("mic", calls["mic"] + 1) or True,
    )
    monkeypatch.setattr(
        cli,
        "request_accessibility_permission",
        lambda: calls.__setitem__("ax", calls["ax"] + 1) or True,
    )
    monkeypatch.setattr(
        cli,
        "request_input_monitoring_permission",
        lambda: calls.__setitem__("im", calls["im"] + 1) or True,
    )
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.moonshineflow.daemon")

    exit_code = cli.cmd_run(argparse.Namespace(config=None))

    assert exit_code == 0
    assert calls["mic"] == 1
    assert calls["ax"] == 1
    assert calls["im"] == 1
    assert calls["stop"] == 1


def test_load_corrections_warns_when_explicit_path_missing(tmp_path: Path) -> None:
    config = SimpleNamespace(text=SimpleNamespace(dictionary_path="missing-dict.toml"))

    result, error = cli._load_corrections_with_diagnostics(
        config,
        config_path=tmp_path / "config.toml",
    )

    assert error is None
    assert result is not None
    assert result.loaded is False
    assert len(result.warnings) == 1


def test_load_corrections_disables_when_default_path_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        TomlCorrectionRepository,
        "default_dictionary_path",
        staticmethod(lambda: tmp_path / "transcription_corrections.toml"),
    )
    config = SimpleNamespace(text=SimpleNamespace(dictionary_path=None))

    result, error = cli._load_corrections_with_diagnostics(
        config,
        config_path=tmp_path / "config.toml",
    )

    assert error is None
    assert result is not None
    assert result.loaded is False


def test_build_runtime_post_processor_returns_base_when_llm_disabled() -> None:
    class _BaseProcessor:
        def apply(self, text: str) -> str:
            return text + "-base"

    base = _BaseProcessor()
    config = SimpleNamespace(
        text=SimpleNamespace(
            llm_correction=SimpleNamespace(
                mode="never",
            )
        )
    )

    result = cli._build_runtime_post_processor(config, base_processor=base)

    assert result is base


def test_build_runtime_post_processor_chains_base_and_llm(monkeypatch) -> None:
    class _BaseProcessor:
        def apply(self, text: str) -> str:
            return text + "-base"

    calls: dict[str, object] = {}

    class _FakeLLMProcessor:
        def __init__(self, settings) -> None:
            calls["settings"] = settings

        def preflight(self) -> None:
            calls["preflight"] = True

        def apply(self, text: str) -> str:
            return text + "-llm"

    monkeypatch.setattr(cli, "LLMPostProcessor", _FakeLLMProcessor)

    config = SimpleNamespace(
        text=SimpleNamespace(
            llm_correction=SimpleNamespace(
                mode="always",
                provider="ollama",
                base_url="http://localhost:11434",
                model="qwen2.5:7b-instruct",
                timeout_seconds=2.5,
                max_input_chars=500,
                api_key=None,
                enabled_tools=False,
            )
        )
    )

    result = cli._build_runtime_post_processor(config, base_processor=_BaseProcessor())

    assert result.apply("text") == "text-base-llm"
    assert calls["preflight"] is True
    settings = calls["settings"]
    assert settings.provider == "ollama"
    assert settings.base_url == "http://localhost:11434"
    assert settings.enabled_tools is False


def test_build_runtime_post_processor_ignores_invalid_llm_config() -> None:
    class _BaseProcessor:
        def apply(self, text: str) -> str:
            return text

    base = _BaseProcessor()
    config = SimpleNamespace(
        text=SimpleNamespace(
            llm_correction=SimpleNamespace(
                mode="always",
                provider="ollama",
                base_url="",
                model="",
            )
        )
    )

    result = cli._build_runtime_post_processor(config, base_processor=base)

    assert result is base


def test_should_enable_llm_correction_for_this_run_mode_always() -> None:
    llm_cfg = SimpleNamespace(mode="always")
    assert cli._should_enable_llm_correction_for_this_run(llm_cfg) is True


def test_should_enable_llm_correction_for_this_run_mode_never() -> None:
    llm_cfg = SimpleNamespace(mode="never")
    assert cli._should_enable_llm_correction_for_this_run(llm_cfg) is False


def test_should_enable_llm_correction_for_this_run_mode_ask_interactive_yes(monkeypatch) -> None:
    llm_cfg = SimpleNamespace(mode="ask")
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_prompt_llm_correction_for_this_run", lambda: True)
    assert cli._should_enable_llm_correction_for_this_run(llm_cfg) is True


def test_should_enable_llm_correction_for_this_run_mode_ask_interactive_no(monkeypatch) -> None:
    llm_cfg = SimpleNamespace(mode="ask")
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_prompt_llm_correction_for_this_run", lambda: False)
    assert cli._should_enable_llm_correction_for_this_run(llm_cfg) is False


def test_should_enable_llm_correction_for_this_run_mode_ask_non_interactive(monkeypatch) -> None:
    llm_cfg = SimpleNamespace(mode="ask")
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: False)
    assert cli._should_enable_llm_correction_for_this_run(llm_cfg) is False


def test_cmd_run_skips_permission_requests_outside_launchd(monkeypatch) -> None:
    fake_daemon_mod = ModuleType("moonshine_flow.daemon")
    calls = {"requests": 0, "stop": 0}

    class FakeDaemon:
        def __init__(self, _config, post_processor=None) -> None:
            del post_processor
            self.transcriber = SimpleNamespace(preflight_model=lambda: "moonshine-voice")

        def run_forever(self) -> None:
            raise KeyboardInterrupt

        def stop(self) -> None:
            calls["stop"] += 1

    fake_daemon_mod.MoonshineFlowDaemon = FakeDaemon
    monkeypatch.setitem(sys.modules, "moonshine_flow.daemon", fake_daemon_mod)
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: SimpleNamespace(runtime=SimpleNamespace(log_level="INFO")),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda _level: None)
    monkeypatch.setattr(cli, "_has_moonshine_backend", lambda: True)
    monkeypatch.setattr(cli, "consume_restart_permission_suppression", lambda: False)
    monkeypatch.setattr(
        cli,
        "check_all_permissions",
        lambda: PermissionReport(microphone=False, accessibility=False, input_monitoring=False),
    )
    monkeypatch.setattr(
        cli,
        "request_microphone_permission",
        lambda: calls.__setitem__("requests", calls["requests"] + 1) or True,
    )
    monkeypatch.setattr(
        cli,
        "request_accessibility_permission",
        lambda: calls.__setitem__("requests", calls["requests"] + 1) or True,
    )
    monkeypatch.setattr(
        cli,
        "request_input_monitoring_permission",
        lambda: calls.__setitem__("requests", calls["requests"] + 1) or True,
    )
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)

    exit_code = cli.cmd_run(argparse.Namespace(config=None))

    assert exit_code == 0
    assert calls["requests"] == 0
    assert calls["stop"] == 1


def test_cmd_run_skips_permission_requests_once_after_restart_marker(monkeypatch) -> None:
    fake_daemon_mod = ModuleType("moonshine_flow.daemon")
    calls = {"requests": 0, "stop": 0}

    class FakeDaemon:
        def __init__(self, _config, post_processor=None) -> None:
            del post_processor
            self.transcriber = SimpleNamespace(preflight_model=lambda: "moonshine-voice")

        def run_forever(self) -> None:
            raise KeyboardInterrupt

        def stop(self) -> None:
            calls["stop"] += 1

    fake_daemon_mod.MoonshineFlowDaemon = FakeDaemon
    monkeypatch.setitem(sys.modules, "moonshine_flow.daemon", fake_daemon_mod)
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: SimpleNamespace(runtime=SimpleNamespace(log_level="INFO")),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda _level: None)
    monkeypatch.setattr(cli, "_has_moonshine_backend", lambda: True)
    monkeypatch.setattr(cli, "consume_restart_permission_suppression", lambda: True)
    monkeypatch.setattr(
        cli,
        "check_all_permissions",
        lambda: PermissionReport(microphone=False, accessibility=False, input_monitoring=False),
    )
    monkeypatch.setattr(
        cli,
        "request_microphone_permission",
        lambda: calls.__setitem__("requests", calls["requests"] + 1) or True,
    )
    monkeypatch.setattr(
        cli,
        "request_accessibility_permission",
        lambda: calls.__setitem__("requests", calls["requests"] + 1) or True,
    )
    monkeypatch.setattr(
        cli,
        "request_input_monitoring_permission",
        lambda: calls.__setitem__("requests", calls["requests"] + 1) or True,
    )
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.moonshineflow.daemon")

    exit_code = cli.cmd_run(argparse.Namespace(config=None))

    assert exit_code == 0
    assert calls["requests"] == 0
    assert calls["stop"] == 1


def test_has_moonshine_backend_true(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: object() if name == "moonshine_voice" else None,
    )
    assert cli._has_moonshine_backend()


def test_has_moonshine_backend_false(monkeypatch) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None)
    assert not cli._has_moonshine_backend()


def test_backend_guidance_has_actionable_text() -> None:
    guidance = cli._backend_guidance()
    assert "uv sync" in guidance
    assert "Moonshine backend package is missing" in guidance


def test_check_permissions_parser_has_request_flag() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["check-permissions", "--request"])
    assert args.request is True


def test_doctor_parser_has_launchd_check_flag() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["doctor", "--launchd-check"])
    assert args.launchd_check is True


def test_init_parser_has_config_option() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["init", "--config", "/tmp/config.toml"])
    assert args.command == "init"
    assert args.config == "/tmp/config.toml"


def test_cmd_init_requires_interactive_terminal(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: False)
    exit_code = cli.cmd_init(argparse.Namespace(config=None))
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "interactive terminal" in captured.err


def test_dim_plain_when_ansi_disabled(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: False)
    assert cli._dim("x") == "x"


def test_dim_wrapped_when_ansi_enabled(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: True)
    assert cli._dim("x") == "\x1b[2mx\x1b[0m"


def test_format_prompt_includes_current(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: False)
    prompt = cli._format_prompt("hotkey.key", "right_shift", current_display="right_shift")
    assert "(current: right_shift)" in prompt


def test_prompt_text_shows_keep_line(monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: False)
    assert cli._prompt_text("hotkey.key", "right_shift") == "right_shift"
    captured = capsys.readouterr()
    assert "keep: right_shift" in captured.out


def test_prompt_choice_accepts_number(monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: False)
    value = cli._prompt_choice(
        "model.size",
        "tiny",
        ["tiny", "base"],
    )
    captured = capsys.readouterr()
    assert value == "base"
    assert "1. tiny" in captured.out
    assert "2. base" in captured.out


def test_cmd_init_updates_values_and_keeps_others(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: False)

    answers = iter(
        [
            "right_shift",  # hotkey.key
            "",  # audio.sample_rate
            "",  # audio.channels
            "",  # audio.dtype
            "",  # audio.max_record_seconds
            "",  # audio.release_tail_seconds
            "",  # audio.trailing_silence_seconds
            "",  # audio.input_device
            "",  # model.size
            "ja",  # model.language
            "",  # model.device
            "",  # output.mode
            "",  # output.paste_shortcut
            "",  # runtime.log_level
            "",  # runtime.notify_on_error
            "",  # text.dictionary_path
            "ask",  # text.llm_correction.mode
            "",  # text.llm_correction.provider
            "",  # text.llm_correction.base_url
            "llama3.2:latest",  # text.llm_correction.model
            "",  # text.llm_correction.timeout_seconds
            "",  # text.llm_correction.max_input_chars
            "",  # text.llm_correction.api_key
            "",  # text.llm_correction.enabled_tools
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    exit_code = cli.cmd_init(argparse.Namespace(config=None))

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "keep: 16000" in captured.out
    updated = cli.load_config(cfg_path)
    assert updated.hotkey.key == "right_shift"
    assert updated.model.language == "ja"
    assert updated.text.llm_correction.mode.value == "ask"
    assert updated.text.llm_correction.model == "llama3.2:latest"


def test_cmd_init_accepts_provider_other(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: False)

    answers = iter(
        [
            "",  # hotkey.key
            "",  # audio.sample_rate
            "",  # audio.channels
            "",  # audio.dtype
            "",  # audio.max_record_seconds
            "",  # audio.release_tail_seconds
            "",  # audio.trailing_silence_seconds
            "",  # audio.input_device
            "",  # model.size
            "",  # model.language
            "",  # model.device
            "",  # output.mode
            "",  # output.paste_shortcut
            "",  # runtime.log_level
            "",  # runtime.notify_on_error
            "",  # text.dictionary_path
            "",  # text.llm_correction.mode
            "3",  # text.llm_correction.provider -> other
            "my-local-provider",  # text.llm_correction.provider_other
            "",  # text.llm_correction.base_url
            "",  # text.llm_correction.model
            "",  # text.llm_correction.timeout_seconds
            "",  # text.llm_correction.max_input_chars
            "",  # text.llm_correction.api_key
            "",  # text.llm_correction.enabled_tools
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    exit_code = cli.cmd_init(argparse.Namespace(config=None))

    assert exit_code == 0
    _ = capsys.readouterr()
    updated = cli.load_config(cfg_path)
    assert updated.text.llm_correction.provider == "my-local-provider"


def test_list_devices_parser_has_config_option() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["list", "devices", "--config", "/tmp/config.toml"])
    assert args.command == "list"
    assert args.config == "/tmp/config.toml"
    assert args.list_target == "devices"


def test_list_parser_without_target_shows_help_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["list"])
    assert args.command == "list"
    assert args.list_target is None
    assert args.func == cli.cmd_list


def test_list_parser_alias_accepts_devices_target() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["list", "devices"])
    assert args.command == "list"
    assert args.list_target == "devices"
    assert args.func == cli.cmd_list_devices


def test_list_parser_accepts_ollama_target() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["list", "ollama", "--config", "/tmp/config.toml"])
    assert args.command == "list"
    assert args.list_target == "ollama"
    assert args.func == cli.cmd_list_ollama
    assert args.config == "/tmp/config.toml"


def test_list_parser_accepts_lmstudio_target() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["list", "lmstudio", "--config", "/tmp/config.toml"])
    assert args.command == "list"
    assert args.list_target == "lmstudio"
    assert args.func == cli.cmd_list_lmstudio
    assert args.config == "/tmp/config.toml"


def test_cmd_list_shows_available_commands(capsys) -> None:
    exit_code = cli.cmd_list(argparse.Namespace())
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Available list commands" in captured.out
    assert "mflow list devices" in captured.out
    assert "mflow list ollama" in captured.out
    assert "mflow list lmstudio" in captured.out


def test_cmd_list_ollama_requires_interactive_terminal(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: False)

    exit_code = cli.cmd_list_ollama(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "interactive terminal" in captured.err


def test_cmd_list_ollama_selects_model_and_updates_config(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: False)
    monkeypatch.setattr(
        cli,
        "_query_ollama_model_names",
        lambda **_kwargs: ["llama3.2:latest", "qwen2.5:7b-instruct"],
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    exit_code = cli.cmd_list_ollama(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Current text.llm_correction.model" in captured.out
    assert "text.llm_correction.model = qwen2.5:7b-instruct" in captured.out
    updated = cli.load_config(cfg_path)
    assert updated.text.llm_correction.model == "qwen2.5:7b-instruct"


def test_cmd_list_ollama_enter_keeps_current_model(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg = cli.load_config(cfg_path)
    cfg.text.llm_correction.model = "llama3.2:latest"
    cli.write_config(cfg_path, cfg)

    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: False)
    monkeypatch.setattr(
        cli,
        "_query_ollama_model_names",
        lambda **_kwargs: ["llama3.2:latest", "qwen2.5:7b-instruct"],
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    exit_code = cli.cmd_list_ollama(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "keep: 1" in captured.out
    updated = cli.load_config(cfg_path)
    assert updated.text.llm_correction.model == "llama3.2:latest"


def test_cmd_list_ollama_errors_when_provider_is_not_ollama(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg = cli.load_config(cfg_path)
    cfg.text.llm_correction.provider = "lmstudio"
    cli.write_config(cfg_path, cfg)

    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(
        cli,
        "_query_ollama_model_names",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not call ollama API")),
    )

    exit_code = cli.cmd_list_ollama(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "requires text.llm_correction.provider = \"ollama\"" in captured.err


def test_cmd_list_ollama_reports_query_failure(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(
        cli,
        "_query_ollama_model_names",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    exit_code = cli.cmd_list_ollama(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "boom" in captured.err


def test_cmd_list_lmstudio_requires_interactive_terminal(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: False)

    exit_code = cli.cmd_list_lmstudio(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "interactive terminal" in captured.err


def test_cmd_list_lmstudio_selects_model_and_updates_config(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg = cli.load_config(cfg_path)
    cfg.text.llm_correction.provider = "lmstudio"
    cfg.text.llm_correction.model = "local-model-a"
    cli.write_config(cfg_path, cfg)

    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: False)
    monkeypatch.setattr(
        cli,
        "_query_lmstudio_model_names",
        lambda **_kwargs: ["local-model-a", "local-model-b"],
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    exit_code = cli.cmd_list_lmstudio(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "text.llm_correction.model = local-model-b" in captured.out
    updated = cli.load_config(cfg_path)
    assert updated.text.llm_correction.model == "local-model-b"


def test_cmd_list_lmstudio_errors_when_provider_is_not_lmstudio(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(
        cli,
        "_query_lmstudio_model_names",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not call lmstudio API")),
    )

    exit_code = cli.cmd_list_lmstudio(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "requires text.llm_correction.provider = \"lmstudio\"" in captured.err


def test_cmd_list_lmstudio_reports_query_failure(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg = cli.load_config(cfg_path)
    cfg.text.llm_correction.provider = "lmstudio"
    cli.write_config(cfg_path, cfg)

    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(
        cli,
        "_query_lmstudio_model_names",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom-lmstudio")),
    )

    exit_code = cli.cmd_list_lmstudio(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "boom-lmstudio" in captured.err


def test_cmd_list_devices_requires_interactive_terminal(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: False)
    exit_code = cli.cmd_list_devices(argparse.Namespace(config=None))
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "interactive terminal" in captured.err


def test_cmd_list_devices_selects_device_and_updates_config(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: False)
    monkeypatch.setattr(
        cli,
        "_query_input_devices",
        lambda: (
            [
                {"index": 0, "name": "MacBook Microphone", "max_input_channels": 1},
                {"index": 3, "name": "USB Microphone", "max_input_channels": 1},
            ],
            0,
        ),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    exit_code = cli.cmd_list_devices(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "audio.input_device = 3" in captured.out
    updated = cli.load_config(cfg_path)
    assert updated.audio.input_device == 3


def test_cmd_list_devices_selects_unset_and_updates_config(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg = cli.load_config(cfg_path)
    cfg.audio.input_device = 7
    cli.write_config(cfg_path, cfg)

    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_supports_ansi_styles", lambda: False)
    monkeypatch.setattr(
        cli,
        "_query_input_devices",
        lambda: ([{"index": 1, "name": "Mic", "max_input_channels": 1}], 1),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "0")

    exit_code = cli.cmd_list_devices(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "audio.input_device = <unset>" in captured.out
    updated = cli.load_config(cfg_path)
    assert updated.audio.input_device is None


def test_cmd_list_devices_reports_query_failure(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: cfg_path)
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_query_input_devices", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    exit_code = cli.cmd_list_devices(argparse.Namespace(config=None))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "boom" in captured.err


def test_install_app_bundle_parser_has_path() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["install-app-bundle", "--path", "/tmp/MoonshineFlow.app"])
    assert args.path == "/tmp/MoonshineFlow.app"


def test_install_launch_agent_parser_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["install-launch-agent"])
    assert args.request_permissions is True
    assert args.allow_missing_permissions is False
    assert args.verbose_bootstrap is False
    assert args.install_app_bundle is True


def test_install_launch_agent_parser_allows_no_request_permissions() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["install-launch-agent", "--no-request-permissions"])
    assert args.request_permissions is False


def test_install_launch_agent_parser_allows_no_install_app_bundle() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["install-launch-agent", "--no-install-app-bundle"])
    assert args.install_app_bundle is False


def test_restart_launch_agent_parser() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["restart-launch-agent"])
    assert args.command == "restart-launch-agent"


def test_cmd_install_launch_agent_aborts_when_permissions_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, "read_launch_agent_plist", lambda: None)
    monkeypatch.setattr(cli, "install_app_bundle_from_env", lambda _path=None: None)
    monkeypatch.setattr(cli, "resolve_launch_agent_program_prefix", lambda: ["/tmp/mflow"])
    called: dict[str, list[str]] = {}

    def fake_launchd_check(*, command: list[str]) -> LaunchdPermissionProbe:
        called["command"] = command
        return LaunchdPermissionProbe(
            ok=True,
            command=command,
            report=PermissionReport(microphone=False, accessibility=True, input_monitoring=True),
        )

    monkeypatch.setattr(cli, "check_permissions_in_launchd_context", fake_launchd_check)
    monkeypatch.setattr(
        cli,
        "install_launch_agent",
        lambda _: (_ for _ in ()).throw(AssertionError("install should not run")),
    )
    args = argparse.Namespace(
        config=None,
        request_permissions=True,
        allow_missing_permissions=False,
        verbose_bootstrap=False,
        install_app_bundle=True,
    )

    exit_code = cli.cmd_install_launch_agent(args)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert called["command"] == ["/tmp/mflow", "check-permissions", "--request"]
    assert "Launchd permission check command: /tmp/mflow check-permissions --request" in captured.out
    assert "Launch agent installation was aborted" in captured.err
    assert "allow-missing-permissions" in captured.err


def test_cmd_install_launch_agent_allows_missing_permissions(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, "read_launch_agent_plist", lambda: None)
    monkeypatch.setattr(cli, "install_app_bundle_from_env", lambda _path=None: None)
    monkeypatch.setattr(cli, "resolve_launch_agent_program_prefix", lambda: ["/tmp/mflow"])
    called: dict[str, list[str]] = {}

    def fake_launchd_check(*, command: list[str]) -> LaunchdPermissionProbe:
        called["command"] = command
        return LaunchdPermissionProbe(
            ok=True,
            command=command,
            report=PermissionReport(microphone=False, accessibility=True, input_monitoring=True),
        )

    monkeypatch.setattr(cli, "check_permissions_in_launchd_context", fake_launchd_check)
    monkeypatch.setattr(cli, "install_launch_agent", lambda _: Path("/tmp/agent.plist"))
    args = argparse.Namespace(
        config=None,
        request_permissions=True,
        allow_missing_permissions=True,
        verbose_bootstrap=False,
        install_app_bundle=True,
    )

    exit_code = cli.cmd_install_launch_agent(args)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert called["command"] == ["/tmp/mflow", "check-permissions", "--request"]
    assert "continuing with missing permissions" in captured.err
    assert "Launchd target executable: /tmp/mflow" in captured.err
    assert "Installed launch agent: /tmp/agent.plist" in captured.out


def test_cmd_install_launch_agent_uses_check_permissions_when_request_disabled(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, "read_launch_agent_plist", lambda: None)
    monkeypatch.setattr(cli, "install_app_bundle_from_env", lambda _path=None: None)
    monkeypatch.setattr(cli, "resolve_launch_agent_program_prefix", lambda: ["/tmp/mflow"])
    called: dict[str, list[str]] = {}

    def fake_launchd_check(*, command: list[str]) -> LaunchdPermissionProbe:
        called["command"] = command
        return LaunchdPermissionProbe(
            ok=True,
            command=command,
            report=PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
        )

    monkeypatch.setattr(cli, "check_permissions_in_launchd_context", fake_launchd_check)
    monkeypatch.setattr(cli, "install_launch_agent", lambda _: Path("/tmp/agent.plist"))
    args = argparse.Namespace(
        config=None,
        request_permissions=False,
        allow_missing_permissions=False,
        verbose_bootstrap=False,
        install_app_bundle=True,
    )

    exit_code = cli.cmd_install_launch_agent(args)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert called["command"] == ["/tmp/mflow", "check-permissions"]
    assert "Installed launch agent: /tmp/agent.plist" in captured.out


def test_cmd_install_launch_agent_aborts_when_launchd_check_parse_fails(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, "read_launch_agent_plist", lambda: None)
    monkeypatch.setattr(cli, "install_app_bundle_from_env", lambda _path=None: None)
    monkeypatch.setattr(cli, "resolve_launch_agent_program_prefix", lambda: ["/tmp/mflow"])
    monkeypatch.setattr(
        cli,
        "check_permissions_in_launchd_context",
        lambda **_kwargs: LaunchdPermissionProbe(
            ok=False,
            command=["/tmp/mflow", "check-permissions", "--request"],
            error="Could not parse permission status from launchd check output",
            stdout="some output",
            stderr="traceback",
        ),
    )
    monkeypatch.setattr(
        cli,
        "install_launch_agent",
        lambda _: (_ for _ in ()).throw(AssertionError("install should not run")),
    )
    args = argparse.Namespace(
        config=None,
        request_permissions=True,
        allow_missing_permissions=False,
        verbose_bootstrap=False,
        install_app_bundle=True,
    )

    exit_code = cli.cmd_install_launch_agent(args)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Could not verify launchd permission state" in captured.err
    assert "Could not parse permission status from launchd check output" in captured.err
    assert "Launchd check stdout:" in captured.err
    assert "Launchd check stderr:" in captured.err


def test_cmd_doctor_prints_launch_agent_and_log_paths(monkeypatch, capsys) -> None:
    fake_transcriber_mod = ModuleType("moonshine_flow.transcriber")

    class FakeTranscriber:
        def __init__(self, model_size: str, language: str, device: str) -> None:
            self._summary = f"{model_size}:{language}:{device}"

        def backend_summary(self) -> str:
            return self._summary

    fake_transcriber_mod.MoonshineTranscriber = FakeTranscriber
    monkeypatch.setitem(sys.modules, "moonshine_flow.transcriber", fake_transcriber_mod)
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: SimpleNamespace(
            model=SimpleNamespace(size=SimpleNamespace(value="base"), language="ja", device="mps")
        ),
    )
    monkeypatch.setattr(cli, "find_spec", lambda _: object())
    monkeypatch.setattr(
        cli,
        "check_all_permissions",
        lambda: PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
    )
    monkeypatch.setattr(
        cli,
        "read_launch_agent_plist",
        lambda: {
            "Label": "com.moonshineflow.daemon",
            "ProgramArguments": ["/usr/bin/python3", "-m", "moonshine_flow.cli", "run"],
        },
    )
    monkeypatch.setattr(cli, "launch_agent_path", lambda: Path("/tmp/com.moonshineflow.daemon.plist"))
    monkeypatch.setattr(
        cli,
        "launch_agent_log_paths",
        lambda: (Path("/tmp/daemon.out.log"), Path("/tmp/daemon.err.log")),
    )
    monkeypatch.setattr(cli, "recommended_permission_target", lambda: Path("/tmp/target.app"))

    exit_code = cli.cmd_doctor(argparse.Namespace(config=None, launchd_check=False))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "LaunchAgent plist: FOUND (/tmp/com.moonshineflow.daemon.plist)" in captured.out
    assert "LaunchAgent program: /usr/bin/python3 -m moonshine_flow.cli run" in captured.out
    assert "Permission target (recommended): /tmp/target.app" in captured.out
    assert "Daemon stdout log: /tmp/daemon.out.log" in captured.out
    assert "Daemon stderr log: /tmp/daemon.err.log" in captured.out


def test_cmd_install_app_bundle_succeeds(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "install_app_bundle_from_env", lambda _path: Path("/tmp/MoonshineFlow.app"))

    exit_code = cli.cmd_install_app_bundle(argparse.Namespace(path="/tmp/MoonshineFlow.app"))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Installed app bundle: /tmp/MoonshineFlow.app" in captured.out


def test_cmd_install_app_bundle_reports_unavailable_context(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "install_app_bundle_from_env", lambda _path: None)

    exit_code = cli.cmd_install_app_bundle(argparse.Namespace(path=None))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "App bundle install is unavailable in this context" in captured.err


def test_cmd_restart_launch_agent_succeeds(monkeypatch, capsys, tmp_path: Path) -> None:
    plist_path = tmp_path / "com.moonshineflow.daemon.plist"
    plist_path.write_text("plist", encoding="utf-8")
    monkeypatch.setattr(cli, "restart_launch_agent", lambda: True)
    monkeypatch.setattr(cli, "launch_agent_path", lambda: plist_path)
    monkeypatch.setattr(cli, "read_launch_agent_plist", lambda: {})
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: False)

    exit_code = cli.cmd_restart_launch_agent(argparse.Namespace())

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Restarted launch agent: {plist_path}" in captured.out


def test_cmd_restart_launch_agent_reports_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "restart_launch_agent", lambda: False)
    monkeypatch.setattr(cli, "launch_agent_path", lambda: Path("/tmp/missing.plist"))

    exit_code = cli.cmd_restart_launch_agent(argparse.Namespace())

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Launch agent is not installed." in captured.out


def test_cmd_restart_launch_agent_reports_failure(monkeypatch, capsys, tmp_path: Path) -> None:
    plist_path = tmp_path / "com.moonshineflow.daemon.plist"
    plist_path.write_text("plist", encoding="utf-8")

    def raise_error() -> bool:
        raise RuntimeError("launchctl restart failed: denied")

    monkeypatch.setattr(cli, "restart_launch_agent", raise_error)
    monkeypatch.setattr(cli, "launch_agent_path", lambda: plist_path)
    monkeypatch.setattr(cli, "read_launch_agent_plist", lambda: {})
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: False)

    exit_code = cli.cmd_restart_launch_agent(argparse.Namespace())

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "launchctl restart failed: denied" in captured.err


def test_should_enable_llm_correction_for_this_run_uses_launchd_env_override(monkeypatch) -> None:
    llm_cfg = SimpleNamespace(mode="never")
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.moonshineflow.daemon")
    monkeypatch.setenv("MFLOW_LLM_ENABLED", "1")
    assert cli._should_enable_llm_correction_for_this_run(llm_cfg) is True


def test_should_enable_llm_correction_for_this_run_launchd_env_invalid_falls_back(monkeypatch) -> None:
    llm_cfg = SimpleNamespace(mode="never")
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.moonshineflow.daemon")
    monkeypatch.setenv("MFLOW_LLM_ENABLED", "oops")
    assert cli._should_enable_llm_correction_for_this_run(llm_cfg) is False


def test_cmd_install_launch_agent_interactive_yes_preflight_failure_sets_false(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_prompt_launchd_llm_enabled", lambda _current: True)
    monkeypatch.setattr(cli, "_preflight_llm_for_launchd", lambda _config: (False, "boom"))
    monkeypatch.setattr(cli, "read_launch_agent_plist", lambda: {})
    monkeypatch.setattr(cli, "install_app_bundle_from_env", lambda _path=None: None)
    monkeypatch.setattr(cli, "resolve_launch_agent_program_prefix", lambda: ["/tmp/mflow"])
    monkeypatch.setattr(
        cli,
        "check_permissions_in_launchd_context",
        lambda **_kwargs: LaunchdPermissionProbe(
            ok=True,
            command=["/tmp/mflow", "check-permissions", "--request"],
            report=PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
        ),
    )
    called: dict[str, object] = {}

    def fake_install(config_path: Path, **kwargs) -> Path:
        called["config_path"] = config_path
        called["kwargs"] = kwargs
        return Path("/tmp/agent.plist")

    monkeypatch.setattr(cli, "install_launch_agent", fake_install)
    args = argparse.Namespace(
        config=None,
        request_permissions=True,
        allow_missing_permissions=False,
        verbose_bootstrap=False,
        install_app_bundle=True,
    )

    exit_code = cli.cmd_install_launch_agent(args)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert called["kwargs"] == {"llm_enabled_override": False}
    assert "preflight failed" in captured.err.lower()
    assert "selected yes" in captured.err.lower()
    assert "switched to no" in captured.err.lower()
    assert "Launchd LLM enabled override: false" in captured.out


def test_cmd_restart_launch_agent_interactive_yes_preflight_failure_sets_false(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    plist_path = tmp_path / "com.moonshineflow.daemon.plist"
    plist_path.write_text("plist", encoding="utf-8")
    monkeypatch.setattr(cli, "launch_agent_path", lambda: plist_path)
    monkeypatch.setattr(cli, "read_launch_agent_plist", lambda: {})
    monkeypatch.setattr(cli, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(cli, "_prompt_launchd_llm_enabled", lambda _current: True)
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _value=None: Path("/tmp/config.toml"))
    monkeypatch.setattr(cli, "load_config", lambda _path: object())
    monkeypatch.setattr(cli, "_preflight_llm_for_launchd", lambda _config: (False, "boom"))
    called: dict[str, object] = {}

    def fake_restart(**kwargs) -> bool:
        called["kwargs"] = kwargs
        return True

    monkeypatch.setattr(cli, "restart_launch_agent", fake_restart)

    exit_code = cli.cmd_restart_launch_agent(argparse.Namespace())

    captured = capsys.readouterr()
    assert exit_code == 0
    assert called["kwargs"] == {"llm_enabled_override": False}
    assert "preflight failed" in captured.err.lower()
    assert "selected yes" in captured.err.lower()
    assert "switched to no" in captured.err.lower()
    assert "Launchd LLM enabled override: false" in captured.out


def test_latest_launchd_runtime_warning_detects_not_trusted(tmp_path: Path) -> None:
    err_log = tmp_path / "daemon.err.log"
    err_log.write_text(
        "\n".join(
            [
                "2026-02-27 10:00:00,000 INFO Moonshine Flow daemon starting",
                "2026-02-27 10:00:00,100 WARNING [pynput.keyboard.Listener] This process is not trusted!",
            ]
        ),
        encoding="utf-8",
    )

    warning = cli._latest_launchd_runtime_warning(err_log)
    assert warning is not None
    assert "not trusted" in warning


def test_cmd_doctor_prints_runtime_warning_from_daemon_log(monkeypatch, capsys, tmp_path: Path) -> None:
    fake_transcriber_mod = ModuleType("moonshine_flow.transcriber")

    class FakeTranscriber:
        def __init__(self, model_size: str, language: str, device: str) -> None:
            self._summary = f"{model_size}:{language}:{device}"

        def backend_summary(self) -> str:
            return self._summary

    fake_transcriber_mod.MoonshineTranscriber = FakeTranscriber
    monkeypatch.setitem(sys.modules, "moonshine_flow.transcriber", fake_transcriber_mod)
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: SimpleNamespace(
            model=SimpleNamespace(size=SimpleNamespace(value="base"), language="ja", device="mps")
        ),
    )
    monkeypatch.setattr(cli, "find_spec", lambda _: object())
    monkeypatch.setattr(
        cli,
        "check_all_permissions",
        lambda: PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
    )
    monkeypatch.setattr(
        cli,
        "read_launch_agent_plist",
        lambda: {
            "Label": "com.moonshineflow.daemon",
            "ProgramArguments": ["/usr/bin/python3", "-m", "moonshine_flow.cli", "run"],
        },
    )
    monkeypatch.setattr(cli, "launch_agent_path", lambda: Path("/tmp/com.moonshineflow.daemon.plist"))
    out_log = tmp_path / "daemon.out.log"
    err_log = tmp_path / "daemon.err.log"
    err_log.write_text(
        "\n".join(
            [
                "2026-02-27 10:00:00,000 INFO [moonshine_flow.daemon] Moonshine Flow daemon starting",
                "2026-02-27 10:00:00,100 WARNING [pynput.keyboard.Listener] This process is not trusted!",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "launch_agent_log_paths", lambda: (out_log, err_log))
    monkeypatch.setattr(cli, "recommended_permission_target", lambda: Path("/tmp/target.app"))

    exit_code = cli.cmd_doctor(argparse.Namespace(config=None, launchd_check=False))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Launchd runtime status: WARNING" in captured.out
    assert "not trusted" in captured.out


def test_cmd_doctor_marks_permissions_incomplete_when_runtime_warning_present(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    fake_transcriber_mod = ModuleType("moonshine_flow.transcriber")

    class FakeTranscriber:
        def __init__(self, model_size: str, language: str, device: str) -> None:
            self._summary = f"{model_size}:{language}:{device}"

        def backend_summary(self) -> str:
            return self._summary

    fake_transcriber_mod.MoonshineTranscriber = FakeTranscriber
    monkeypatch.setitem(sys.modules, "moonshine_flow.transcriber", fake_transcriber_mod)
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: SimpleNamespace(
            model=SimpleNamespace(size=SimpleNamespace(value="base"), language="ja", device="mps")
        ),
    )
    monkeypatch.setattr(cli, "find_spec", lambda _: object())
    monkeypatch.setattr(
        cli,
        "check_all_permissions",
        lambda: PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
    )
    monkeypatch.setattr(
        cli,
        "read_launch_agent_plist",
        lambda: {
            "Label": "com.moonshineflow.daemon",
            "ProgramArguments": ["/usr/bin/python3", "-m", "moonshine_flow.cli", "run"],
        },
    )
    monkeypatch.setattr(cli, "launch_agent_path", lambda: Path("/tmp/com.moonshineflow.daemon.plist"))
    out_log = tmp_path / "daemon.out.log"
    err_log = tmp_path / "daemon.err.log"
    err_log.write_text(
        "\n".join(
            [
                "2026-02-27 10:00:00,000 INFO [moonshine_flow.daemon] Moonshine Flow daemon starting",
                "2026-02-27 10:00:00,100 WARNING [pynput.keyboard.Listener] This process is not trusted!",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "launch_agent_log_paths", lambda: (out_log, err_log))
    monkeypatch.setattr(cli, "recommended_permission_target", lambda: Path("/tmp/target.app"))
    monkeypatch.setattr(
        cli,
        "check_permissions_in_launchd_context",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("launchd check should not run")),
    )

    exit_code = cli.cmd_doctor(argparse.Namespace(config=None, launchd_check=False))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Terminal permissions: OK" in captured.out
    assert "Permissions: INCOMPLETE" in captured.out
    assert "Launchd runtime log indicates trust failure" in captured.out


def test_cmd_doctor_prints_install_hint_when_launch_agent_missing(monkeypatch, capsys) -> None:
    fake_transcriber_mod = ModuleType("moonshine_flow.transcriber")

    class FakeTranscriber:
        def __init__(self, model_size: str, language: str, device: str) -> None:
            self._summary = f"{model_size}:{language}:{device}"

        def backend_summary(self) -> str:
            return self._summary

    fake_transcriber_mod.MoonshineTranscriber = FakeTranscriber
    monkeypatch.setitem(sys.modules, "moonshine_flow.transcriber", fake_transcriber_mod)
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: SimpleNamespace(
            model=SimpleNamespace(size=SimpleNamespace(value="base"), language="ja", device="mps")
        ),
    )
    monkeypatch.setattr(cli, "find_spec", lambda _: object())
    monkeypatch.setattr(
        cli,
        "check_all_permissions",
        lambda: PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
    )
    monkeypatch.setattr(cli, "read_launch_agent_plist", lambda: None)
    monkeypatch.setattr(cli, "launch_agent_path", lambda: Path("/tmp/com.moonshineflow.daemon.plist"))
    monkeypatch.setattr(
        cli,
        "launch_agent_log_paths",
        lambda: (Path("/tmp/daemon.out.log"), Path("/tmp/daemon.err.log")),
    )
    monkeypatch.setattr(cli, "recommended_permission_target", lambda: Path("/tmp/target.app"))

    exit_code = cli.cmd_doctor(argparse.Namespace(config=None, launchd_check=False))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "LaunchAgent plist: MISSING (/tmp/com.moonshineflow.daemon.plist)" in captured.out
    assert "Install LaunchAgent: mflow install-launch-agent" in captured.out


def test_cmd_doctor_compares_launchd_permissions_when_enabled(monkeypatch, capsys) -> None:
    fake_transcriber_mod = ModuleType("moonshine_flow.transcriber")

    class FakeTranscriber:
        def __init__(self, model_size: str, language: str, device: str) -> None:
            self._summary = f"{model_size}:{language}:{device}"

        def backend_summary(self) -> str:
            return self._summary

    fake_transcriber_mod.MoonshineTranscriber = FakeTranscriber
    monkeypatch.setitem(sys.modules, "moonshine_flow.transcriber", fake_transcriber_mod)
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: SimpleNamespace(
            model=SimpleNamespace(size=SimpleNamespace(value="base"), language="ja", device="mps")
        ),
    )
    monkeypatch.setattr(cli, "find_spec", lambda _: object())
    monkeypatch.setattr(
        cli,
        "check_all_permissions",
        lambda: PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
    )
    monkeypatch.setattr(
        cli,
        "read_launch_agent_plist",
        lambda: {
            "Label": "com.moonshineflow.daemon",
            "ProgramArguments": ["/usr/bin/python3", "-m", "moonshine_flow.cli", "run"],
        },
    )
    monkeypatch.setattr(cli, "launch_agent_path", lambda: Path("/tmp/com.moonshineflow.daemon.plist"))
    monkeypatch.setattr(
        cli,
        "launch_agent_log_paths",
        lambda: (Path("/tmp/daemon.out.log"), Path("/tmp/daemon.err.log")),
    )
    called: dict[str, list[str]] = {}

    def fake_launchd_check(*, command: list[str]) -> LaunchdPermissionProbe:
        called["command"] = command
        return LaunchdPermissionProbe(
            ok=True,
            report=PermissionReport(microphone=False, accessibility=True, input_monitoring=True),
        )

    monkeypatch.setattr(cli, "check_permissions_in_launchd_context", fake_launchd_check)

    exit_code = cli.cmd_doctor(argparse.Namespace(config=None, launchd_check=True))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert called["command"] == ["/usr/bin/python3", "-m", "moonshine_flow.cli", "check-permissions"]
    assert "Launchd permissions: INCOMPLETE" in captured.out
    assert "Launchd missing permissions: Microphone" in captured.out
    assert "Permission mismatch detected between terminal and launchd contexts" in captured.out


def test_cmd_doctor_reports_launchd_check_error(monkeypatch, capsys) -> None:
    fake_transcriber_mod = ModuleType("moonshine_flow.transcriber")

    class FakeTranscriber:
        def __init__(self, model_size: str, language: str, device: str) -> None:
            self._summary = f"{model_size}:{language}:{device}"

        def backend_summary(self) -> str:
            return self._summary

    fake_transcriber_mod.MoonshineTranscriber = FakeTranscriber
    monkeypatch.setitem(sys.modules, "moonshine_flow.transcriber", fake_transcriber_mod)
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: SimpleNamespace(
            model=SimpleNamespace(size=SimpleNamespace(value="base"), language="ja", device="mps")
        ),
    )
    monkeypatch.setattr(cli, "find_spec", lambda _: object())
    monkeypatch.setattr(
        cli,
        "check_all_permissions",
        lambda: PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
    )
    monkeypatch.setattr(
        cli,
        "read_launch_agent_plist",
        lambda: {
            "Label": "com.moonshineflow.daemon",
            "ProgramArguments": ["/usr/local/bin/mflow", "run", "--config", "/tmp/config.toml"],
        },
    )
    monkeypatch.setattr(cli, "launch_agent_path", lambda: Path("/tmp/com.moonshineflow.daemon.plist"))
    monkeypatch.setattr(
        cli,
        "launch_agent_log_paths",
        lambda: (Path("/tmp/daemon.out.log"), Path("/tmp/daemon.err.log")),
    )
    called: dict[str, list[str]] = {}

    def fake_launchd_check(*, command: list[str]) -> LaunchdPermissionProbe:
        called["command"] = command
        return LaunchdPermissionProbe(
            ok=False,
            error="Could not parse permission status from launchd check output (exit=1)",
            stderr="launchctl failed",
        )

    monkeypatch.setattr(cli, "check_permissions_in_launchd_context", fake_launchd_check)

    exit_code = cli.cmd_doctor(argparse.Namespace(config=None, launchd_check=True))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert called["command"] == ["/usr/local/bin/mflow", "check-permissions"]
    assert "Launchd permissions: ERROR" in captured.out
    assert "Launchd check error:" in captured.out
    assert "Launchd check stderr: launchctl failed" in captured.out


def test_parser_version_long_flag_outputs_version(capsys) -> None:
    version_value = "9.9.9"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "package_version", lambda name: version_value)
    try:
        parser = cli.build_parser()
        parser.prog = "moonshine-flow"
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
    finally:
        monkeypatch.undo()

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert captured.out.strip() == f"moonshine-flow {version_value}"


def test_parser_version_short_flag_outputs_version(capsys) -> None:
    version_value = "9.9.10"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "package_version", lambda name: version_value)
    try:
        parser = cli.build_parser()
        parser.prog = "moonshine-flow"
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["-v"])
    finally:
        monkeypatch.undo()

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert captured.out.strip() == f"moonshine-flow {version_value}"


def test_parser_version_falls_back_when_package_metadata_missing(capsys) -> None:
    def raise_not_found(name: str) -> str:
        raise cli.PackageNotFoundError(name)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "package_version", raise_not_found)
    try:
        parser = cli.build_parser()
        parser.prog = "moonshine-flow"
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
    finally:
        monkeypatch.undo()

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert captured.out.strip() == "moonshine-flow 0.0.0.dev0"


def test_resolve_app_version_reads_installed_metadata() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "package_version", lambda name: "1.2.3")
    try:
        assert cli._resolve_app_version() == "1.2.3"
    finally:
        monkeypatch.undo()


def test_resolve_app_version_fallback_when_metadata_missing() -> None:
    def raise_not_found(name: str) -> str:
        raise cli.PackageNotFoundError(name)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "package_version", raise_not_found)
    try:
        assert cli._resolve_app_version() == "0.0.0.dev0"
    finally:
        monkeypatch.undo()


def _make_doctor_monkeypatches(monkeypatch, *, out_log: Path, err_log: Path) -> None:
    """Apply common monkeypatches for cmd_doctor tests."""
    from types import ModuleType, SimpleNamespace

    fake_transcriber_mod = ModuleType("moonshine_flow.transcriber")

    class FakeTranscriber:
        def __init__(self, model_size: str, language: str, device: str) -> None:
            self._summary = f"{model_size}:{language}:{device}"

        def backend_summary(self) -> str:
            return self._summary

    fake_transcriber_mod.MoonshineTranscriber = FakeTranscriber
    monkeypatch.setitem(sys.modules, "moonshine_flow.transcriber", fake_transcriber_mod)
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: SimpleNamespace(
            model=SimpleNamespace(size=SimpleNamespace(value="base"), language="ja", device="mps")
        ),
    )
    monkeypatch.setattr(cli, "find_spec", lambda _: object())
    monkeypatch.setattr(
        cli,
        "check_all_permissions",
        lambda: PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
    )
    monkeypatch.setattr(
        cli,
        "read_launch_agent_plist",
        lambda: {
            "Label": "com.moonshineflow.daemon",
            "ProgramArguments": ["/usr/local/bin/mflow", "run", "--config", "/tmp/config.toml"],
        },
    )
    monkeypatch.setattr(cli, "launch_agent_path", lambda: Path("/tmp/com.moonshineflow.daemon.plist"))
    monkeypatch.setattr(cli, "launch_agent_log_paths", lambda: (out_log, err_log))
    monkeypatch.setattr(cli, "recommended_permission_target", lambda: Path("/tmp/target.app"))


def test_cmd_doctor_shows_warn_when_launchd_check_ok_but_runtime_not_trusted(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """WARN state: launchd check says all OK, but daemon log shows 'not trusted'."""
    out_log = tmp_path / "daemon.out.log"
    err_log = tmp_path / "daemon.err.log"
    err_log.write_text(
        "\n".join(
            [
                "2026-02-27 10:00:00,000 INFO [moonshine_flow.daemon] Moonshine Flow daemon starting",
                "2026-02-27 10:00:00,100 WARNING [pynput.keyboard.Listener] This process is not trusted!",
            ]
        ),
        encoding="utf-8",
    )
    _make_doctor_monkeypatches(monkeypatch, out_log=out_log, err_log=err_log)

    monkeypatch.setattr(
        cli,
        "check_permissions_in_launchd_context",
        lambda **_: LaunchdPermissionProbe(
            ok=True,
            command=["/usr/local/bin/mflow", "check-permissions"],
            report=PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
        ),
    )
    # Suppress codesign subprocess call (no real .app in test env)
    monkeypatch.setattr(cli, "get_app_bundle_codesign_info", lambda _: None)
    monkeypatch.setattr(cli, "app_bundle_executable_path", lambda _: Path("/nonexistent/exec"))

    exit_code = cli.cmd_doctor(argparse.Namespace(config=None, launchd_check=True))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Launchd runtime status: WARNING" in captured.out
    assert "Launchd permissions: OK" in captured.out
    assert "Permissions: WARN (launchd check OK but runtime not trusted)" in captured.out
    assert "TCC lost the binding" in captured.out


def test_cmd_doctor_launchd_check_shows_codesign_info(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """--launchd-check prints CDHash and mtime from get_app_bundle_codesign_info."""
    out_log = tmp_path / "daemon.out.log"
    err_log = tmp_path / "daemon.err.log"
    # No runtime warning in this test
    _make_doctor_monkeypatches(monkeypatch, out_log=out_log, err_log=err_log)

    monkeypatch.setattr(
        cli,
        "check_permissions_in_launchd_context",
        lambda **_: LaunchdPermissionProbe(
            ok=True,
            command=["/usr/local/bin/mflow", "check-permissions"],
            report=PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
        ),
    )
    # Fake codesign info
    monkeypatch.setattr(
        cli,
        "get_app_bundle_codesign_info",
        lambda _: {"CDHash": "abc123def456", "Identifier": "com.moonshineflow.app"},
    )
    # Fake exec path that exists in tmp_path
    fake_exec = tmp_path / "MoonshineFlow"
    fake_exec.write_bytes(b"fake")
    monkeypatch.setattr(cli, "app_bundle_executable_path", lambda _: fake_exec)
    monkeypatch.setattr(cli, "default_app_bundle_path", lambda: tmp_path / "MoonshineFlow.app")

    exit_code = cli.cmd_doctor(argparse.Namespace(config=None, launchd_check=True))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "App bundle CDHash: abc123def456" in captured.out
    assert "App bundle Identifier: com.moonshineflow.app" in captured.out
    assert "App bundle executable mtime:" in captured.out


def test_cmd_install_launch_agent_resets_tcc_when_bundle_installed(
    monkeypatch, capsys
) -> None:
    """install-launch-agent calls reset_app_bundle_tcc when app bundle is installed."""
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, "read_launch_agent_plist", lambda: None)
    monkeypatch.setattr(
        cli,
        "install_app_bundle_from_env",
        lambda _path=None: Path("/tmp/MoonshineFlow.app"),
    )
    monkeypatch.setattr(cli, "resolve_launch_agent_program_prefix", lambda: ["/tmp/mflow"])
    monkeypatch.setattr(
        cli,
        "check_permissions_in_launchd_context",
        lambda *, command: LaunchdPermissionProbe(
            ok=True,
            command=command,
            report=PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
        ),
    )
    monkeypatch.setattr(cli, "install_launch_agent", lambda _: Path("/tmp/agent.plist"))

    tcc_reset_calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "reset_app_bundle_tcc",
        lambda bundle_id: tcc_reset_calls.append(bundle_id) or True,
    )

    args = argparse.Namespace(
        config=None,
        request_permissions=True,
        allow_missing_permissions=False,
        verbose_bootstrap=False,
        install_app_bundle=True,
    )
    exit_code = cli.cmd_install_launch_agent(args)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert tcc_reset_calls == ["com.moonshineflow.app"]
    assert "TCC permissions reset" in captured.out
    assert "Re-grant Accessibility and Input Monitoring" in captured.out


def test_cmd_install_launch_agent_warns_when_tcc_reset_fails(
    monkeypatch, capsys
) -> None:
    """install-launch-agent prints a warning to stderr when TCC reset fails, but continues."""
    monkeypatch.setattr(cli, "_resolve_config_path", lambda _: Path("/tmp/config.toml"))
    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, "read_launch_agent_plist", lambda: None)
    monkeypatch.setattr(
        cli,
        "install_app_bundle_from_env",
        lambda _path=None: Path("/tmp/MoonshineFlow.app"),
    )
    monkeypatch.setattr(cli, "resolve_launch_agent_program_prefix", lambda: ["/tmp/mflow"])
    monkeypatch.setattr(
        cli,
        "check_permissions_in_launchd_context",
        lambda *, command: LaunchdPermissionProbe(
            ok=True,
            command=command,
            report=PermissionReport(microphone=True, accessibility=True, input_monitoring=True),
        ),
    )
    monkeypatch.setattr(cli, "install_launch_agent", lambda _: Path("/tmp/agent.plist"))
    monkeypatch.setattr(cli, "reset_app_bundle_tcc", lambda _bundle_id: False)

    args = argparse.Namespace(
        config=None,
        request_permissions=True,
        allow_missing_permissions=False,
        verbose_bootstrap=False,
        install_app_bundle=True,
    )
    exit_code = cli.cmd_install_launch_agent(args)

    captured = capsys.readouterr()
    assert exit_code == 0  # install should not be aborted
    assert "could not reset TCC permissions" in captured.err
