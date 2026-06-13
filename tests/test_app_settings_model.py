from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ptarmigan_flow import app_settings_model
from ptarmigan_flow.app_settings_model import AppSettingsModel
from ptarmigan_flow.config import AppConfig, OutputMode, load_config, write_config


@dataclass(frozen=True)
class _Entry:
    token: str
    backend: str
    label: str = "Model"
    description: str = ""
    verified: bool = True
    source: str = "preset"


def _write_config(path: Path) -> AppConfig:
    config = AppConfig()
    config.stt.model = "moonshine:tiny"
    config.language = "ja"
    config.hotkey.key = "right_cmd"
    config.output.mode = OutputMode.DIRECT_TYPING
    config.audio.max_record_seconds = 45
    write_config(path, config)
    return config


def test_load_reads_editable_fields_from_existing_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)

    model = AppSettingsModel.load(config_path)

    assert model.model == "moonshine:tiny"
    assert model.language == "ja"
    assert model.hotkey == "right_cmd"
    assert model.output_mode == "direct_typing"


def test_validate_accepts_available_models_and_supported_choices(monkeypatch) -> None:
    monkeypatch.setattr(
        app_settings_model,
        "available_model_entries",
        lambda: [_Entry(token="moonshine:tiny", backend="moonshine")],
    )
    model = AppSettingsModel(
        model="moonshine:tiny",
        language="zh",
        hotkey="left_shift",
        output_mode="clipboard_paste",
    )

    assert model.validate() == []


def test_validate_reports_invalid_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        app_settings_model,
        "available_model_entries",
        lambda: [_Entry(token="moonshine:tiny", backend="moonshine")],
    )
    model = AppSettingsModel(
        model="granite:ibm-granite/granite-4.0-1b-speech",
        language="de",
        hotkey="space",
        output_mode="file",
    )

    assert model.validate() == ["model", "language", "hotkey", "output_mode"]


def test_save_updates_only_editable_fields_and_round_trips(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    original = _write_config(config_path)
    monkeypatch.setattr(
        app_settings_model,
        "available_model_entries",
        lambda: [_Entry(token="moonshine:base", backend="moonshine")],
    )
    model = AppSettingsModel(
        model="moonshine:base",
        language="en",
        hotkey="left_alt",
        output_mode="clipboard_paste",
    )

    model.save(config_path)

    loaded = load_config(config_path)
    assert loaded.stt.model == "moonshine:base"
    assert loaded.language == "en"
    assert loaded.hotkey.key == "left_alt"
    assert loaded.output.mode.value == "clipboard_paste"
    assert loaded.audio.max_record_seconds == original.audio.max_record_seconds
