from __future__ import annotations

import argparse
import json

import ptarmigan_flow.presentation.cli.commands as commands
from ptarmigan_flow.config import AppConfig, write_config
from ptarmigan_flow.presentation.cli.parser import build_parser


def _json_lines(output: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in output.splitlines()]


def test_build_parser_accepts_download_model_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["download-model", "--model", "mlx:vendor/model"])

    assert args.func is commands.cmd_download_model
    assert args.model == "mlx:vendor/model"


def test_cmd_download_model_streams_progress_and_done_json(
    monkeypatch,
    capsys,
) -> None:
    seen: list[str] = []

    def fake_download_model(token: str, *, progress_callback, snapshot_download=None) -> None:
        del snapshot_download
        seen.append(token)
        progress_callback(None, "preparing")
        progress_callback(0.5, "downloading")

    monkeypatch.setattr(commands.model_download, "is_model_downloaded", lambda _token: False)
    monkeypatch.setattr(commands.model_download, "download_model", fake_download_model)

    result = commands.cmd_download_model(
        argparse.Namespace(model="mlx:vendor/model", config=None)
    )

    assert result == 0
    assert seen == ["mlx:vendor/model"]
    assert _json_lines(capsys.readouterr().out) == [
        {"type": "progress", "fraction": None, "message": "preparing"},
        {"type": "progress", "fraction": 0.5, "message": "downloading"},
        {"type": "done"},
    ]


def test_cmd_download_model_uses_config_model_when_model_flag_is_omitted(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "config.toml"
    config = AppConfig()
    config.stt.model = "granite:vendor/config-model"
    write_config(config_path, config)
    seen: list[str] = []

    monkeypatch.setattr(commands.model_download, "is_model_downloaded", lambda _token: True)
    monkeypatch.setattr(
        commands.model_download,
        "download_model",
        lambda token, **_kwargs: seen.append(token),
    )

    result = commands.cmd_download_model(
        argparse.Namespace(model=None, config=str(config_path))
    )

    assert result == 0
    assert seen == []
    assert _json_lines(capsys.readouterr().out) == [{"type": "done"}]


def test_cmd_download_model_returns_error_json_on_failure(
    monkeypatch,
    capsys,
) -> None:
    def fake_download_model(token: str, *, progress_callback, snapshot_download=None) -> None:
        del token, progress_callback, snapshot_download
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(commands.model_download, "is_model_downloaded", lambda _token: False)
    monkeypatch.setattr(commands.model_download, "download_model", fake_download_model)

    result = commands.cmd_download_model(
        argparse.Namespace(model="mlx:vendor/model", config=None)
    )

    assert result == 1
    assert _json_lines(capsys.readouterr().out) == [
        {"type": "error", "message": "network unavailable"}
    ]
