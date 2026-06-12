from __future__ import annotations

import argparse
import logging

import ptarmigan_flow.presentation.cli.commands as commands
from ptarmigan_flow.config import load_config
from ptarmigan_flow.presentation.cli.parser import build_parser
from ptarmigan_flow.stt.model_catalog import CatalogEntry
from ptarmigan_flow.stt.model_families import (
    GRANITE_HF_MODEL_ID,
    GRANITE_MLX_MODEL_ID,
    WHISPER_HF_MODEL_ID,
    WHISPER_MLX_MODEL_ID,
)


def _install_input_sequence(monkeypatch, responses: list[str]) -> None:
    iterator = iter(responses)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(iterator))


def _hub_entry(repo_id: str, *, backend: str = "mlx") -> CatalogEntry:
    return CatalogEntry(
        token=f"{backend}:{repo_id}",
        backend=backend,
        label=repo_id,
        description="unverified Hub result; 42 downloads",
        verified=False,
        source="hub",
    )


def test_stt_model_presets_include_granite() -> None:
    assert f"granite:{GRANITE_HF_MODEL_ID}" in commands._stt_model_presets()


def test_stt_model_presets_include_whisper() -> None:
    assert f"mlx:{WHISPER_HF_MODEL_ID}" in commands._stt_model_presets()


def test_stt_model_downloaded_display_resolves_granite_mlx_variant(monkeypatch) -> None:
    seen: list[str] = []

    monkeypatch.setattr(commands, "_is_macos_arm64", lambda: True)
    monkeypatch.setattr(
        commands,
        "_is_huggingface_model_downloaded",
        lambda model_id: seen.append(model_id) or True,
    )

    assert commands._stt_model_downloaded_display(f"granite:{GRANITE_HF_MODEL_ID}") == "yes"
    assert seen == [GRANITE_MLX_MODEL_ID]


def test_stt_model_downloaded_display_resolves_whisper_mlx_variant(monkeypatch) -> None:
    seen: list[str] = []

    monkeypatch.setattr(commands, "_is_macos_arm64", lambda: True)
    monkeypatch.setattr(
        commands,
        "_is_huggingface_model_downloaded",
        lambda model_id: seen.append(model_id) or True,
    )

    assert commands._stt_model_downloaded_display(f"mlx:{WHISPER_HF_MODEL_ID}") == "yes"
    assert seen == [WHISPER_MLX_MODEL_ID]


def test_stt_model_requires_startup_download_for_missing_huggingface_model(monkeypatch) -> None:
    monkeypatch.setattr(commands, "_stt_model_downloaded_display", lambda _token: "no")
    assert commands._stt_model_requires_startup_download(f"mlx:{WHISPER_HF_MODEL_ID}") is True


def test_log_stt_startup_download_if_needed_logs_backend_name(monkeypatch, caplog) -> None:
    monkeypatch.setattr(commands, "_stt_model_requires_startup_download", lambda _token: True)

    with caplog.at_level(logging.INFO, logger=commands.__name__):
        commands._log_stt_startup_download_if_needed(f"mlx:{WHISPER_HF_MODEL_ID}")

    assert (
        "Selected MLX model is not downloaded yet; startup preflight will download it now"
        in caplog.text
    )


def test_granite_backend_guidance_mentions_expected_dependency(monkeypatch) -> None:
    monkeypatch.setattr(commands, "_is_macos_arm64", lambda: True)
    assert "mlx-audio" in commands._granite_backend_guidance()

    monkeypatch.setattr(commands, "_is_macos_arm64", lambda: False)
    assert "transformers torch" in commands._granite_backend_guidance()


def test_build_parser_accepts_list_model_hub_search_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["list", "model", "--hub-search", "whisper", "--backend", "mlx", "--limit", "5"]
    )

    assert args.func is commands.cmd_list_model
    assert args.hub_search == "whisper"
    assert args.backend == "mlx"
    assert args.limit == 5


def test_cmd_list_model_hub_search_requires_backend(tmp_path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"

    result = commands.cmd_list_model(
        argparse.Namespace(config=str(cfg_path), hub_search="whisper", backend=None, limit=20)
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "--hub-search requires --backend (granite|mlx|voxtral|vllm)." in captured.err


def test_cmd_list_model_hub_search_warns_and_returns_2_on_runtime_error(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    cfg_path = tmp_path / "config.toml"

    def _raise_runtime_error(query: str, *, backend: str, limit: int = 20) -> list[CatalogEntry]:
        raise RuntimeError("Hugging Face Hub request failed (HTTP 500)")

    monkeypatch.setattr(commands, "search_hub_models", _raise_runtime_error)

    result = commands.cmd_list_model(
        argparse.Namespace(config=str(cfg_path), hub_search="whisper", backend="mlx", limit=20)
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "Warning: Hugging Face Hub request failed (HTTP 500)" in captured.err


def test_cmd_list_model_hub_search_interactive_saves_selected_entry(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    cfg_path = tmp_path / "config.toml"
    entries = [
        _hub_entry("openai/whisper-large-v3-turbo"),
        _hub_entry("vendor/open-model"),
    ]

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(commands, "_stt_model_downloaded_display", lambda _token: "no")
    monkeypatch.setattr(
        commands,
        "search_hub_models",
        lambda query, *, backend, limit=20: entries,
    )
    _install_input_sequence(monkeypatch, ["1"])

    result = commands.cmd_list_model(
        argparse.Namespace(config=str(cfg_path), hub_search="whisper", backend="mlx", limit=20)
    )

    assert result == 0
    loaded = load_config(cfg_path)
    assert loaded.stt.model == "mlx:openai/whisper-large-v3-turbo"
    captured = capsys.readouterr()
    assert "unverified Hub result" in captured.out
    assert f"Updated config: {cfg_path}" in captured.out


def test_cmd_list_model_hub_search_noninteractive_lists_results(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    cfg_path = tmp_path / "config.toml"
    entries = [_hub_entry("openai/whisper-large-v3-turbo")]

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: False)
    monkeypatch.setattr(commands, "_stt_model_downloaded_display", lambda _token: "no")
    monkeypatch.setattr(
        commands,
        "search_hub_models",
        lambda query, *, backend, limit=20: entries,
    )

    result = commands.cmd_list_model(
        argparse.Namespace(config=str(cfg_path), hub_search="whisper", backend="mlx", limit=20)
    )

    assert result == 0
    assert not cfg_path.exists()
    captured = capsys.readouterr()
    assert "1. mlx:openai/whisper-large-v3-turbo [unverified Hub result]" in captured.out


def test_cmd_list_model_hub_search_empty_results_returns_0(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    cfg_path = tmp_path / "config.toml"

    monkeypatch.setattr(
        commands,
        "search_hub_models",
        lambda query, *, backend, limit=20: [],
    )

    result = commands.cmd_list_model(
        argparse.Namespace(
            config=str(cfg_path), hub_search="no-such-model", backend="mlx", limit=20
        )
    )

    assert result == 0
    captured = capsys.readouterr()
    assert "No public Hugging Face Hub models matched query: no-such-model" in captured.out


def test_cmd_list_model_default_interactive_saves_preset_and_shows_verified(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    cfg_path = tmp_path / "config.toml"

    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(commands, "_stt_model_downloaded_display", lambda _token: "no")
    _install_input_sequence(monkeypatch, ["3"])

    result = commands.cmd_list_model(
        argparse.Namespace(config=str(cfg_path), hub_search=None, backend=None, limit=20)
    )

    assert result == 0
    loaded = load_config(cfg_path)
    assert loaded.stt.model == f"granite:{GRANITE_HF_MODEL_ID}"
    captured = capsys.readouterr()
    assert "[verified]" in captured.out
    assert "--hub-search" in captured.out
