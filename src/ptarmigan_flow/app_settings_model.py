"""PyObjC-free settings form model for the native macOS app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptarmigan_flow.config import OutputMode, ensure_config_exists, load_config, write_config
from ptarmigan_flow.stt.availability import available_model_entries

SUPPORTED_LANGUAGES = ("en", "ja", "zh")
SUPPORTED_HOTKEYS = (
    "right_cmd",
    "left_cmd",
    "right_shift",
    "left_shift",
    "right_alt",
    "left_alt",
    "right_ctrl",
    "left_ctrl",
)
SUPPORTED_OUTPUT_MODES = ("direct_typing", "clipboard_paste")


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


@dataclass(slots=True)
class AppSettingsModel:
    """Editable app settings shown in the macOS settings form."""

    model: str
    language: str
    hotkey: str
    output_mode: str

    @classmethod
    def load(cls, config_path: Path) -> AppSettingsModel:
        """Load editable fields from the existing app config."""
        ensure_config_exists(config_path)
        config = load_config(config_path)
        return cls(
            model=str(config.stt.model),
            language=str(config.language),
            hotkey=str(config.hotkey.key),
            output_mode=_value(config.output.mode),
        )

    def validate(self) -> list[str]:
        """Return invalid field names, or an empty list when the form can be saved."""
        available_models = {entry.token for entry in available_model_entries()}
        errors: list[str] = []
        if self.model not in available_models:
            errors.append("model")
        if self.language not in SUPPORTED_LANGUAGES:
            errors.append("language")
        if self.hotkey not in SUPPORTED_HOTKEYS:
            errors.append("hotkey")
        if self.output_mode not in SUPPORTED_OUTPUT_MODES:
            errors.append("output_mode")
        return errors

    def save(self, config_path: Path) -> None:
        """Validate and save editable fields back into the full config."""
        errors = self.validate()
        if errors:
            raise ValueError(", ".join(errors))

        ensure_config_exists(config_path)
        config = load_config(config_path)
        config.stt.model = self.model
        config.language = self.language
        config.hotkey.key = self.hotkey
        config.output.mode = OutputMode(self.output_mode)
        write_config(config_path, config)
