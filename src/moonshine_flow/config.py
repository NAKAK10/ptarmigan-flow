"""Configuration loading and validation."""

from __future__ import annotations

import logging
import shutil
import tomllib
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)


class ModelSize(StrEnum):
    """Supported Moonshine model sizes."""

    TINY = "tiny"
    BASE = "base"


class OutputMode(StrEnum):
    """Supported output modes."""

    CLIPBOARD_PASTE = "clipboard_paste"


class LLMProvider(StrEnum):
    """Supported providers for LLM text correction."""

    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"


class LLMCorrectionMode(StrEnum):
    """Runtime activation mode for LLM correction."""

    ALWAYS = "always"
    NEVER = "never"
    ASK = "ask"


class HotkeyConfig(BaseModel):
    """Hotkey configuration."""

    key: str = "right_cmd"


class AudioConfig(BaseModel):
    """Audio capture configuration."""

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    max_record_seconds: int = 30
    release_tail_seconds: float = 0.25
    trailing_silence_seconds: float = 0.5
    input_device: int | str | None = None


class ModelConfig(BaseModel):
    """Model configuration."""

    size: ModelSize = ModelSize.BASE
    language: str = "auto"
    device: str = "mps"


class OutputConfig(BaseModel):
    """Output injection configuration."""

    mode: OutputMode = OutputMode.CLIPBOARD_PASTE
    paste_shortcut: str = "cmd+v"


class RuntimeConfig(BaseModel):
    """Runtime configuration."""

    log_level: str = "INFO"
    notify_on_error: bool = True


class LLMCorrectionConfig(BaseModel):
    """Optional LLM correction settings."""

    mode: LLMCorrectionMode = LLMCorrectionMode.NEVER
    provider: str = LLMProvider.OLLAMA.value
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    timeout_seconds: float = 5.0
    max_input_chars: int = 500
    api_key: str | None = None
    enabled_tools: bool = False


class TextConfig(BaseModel):
    """Transcript text post-processing configuration."""

    dictionary_path: str | None = None
    llm_correction: LLMCorrectionConfig = Field(default_factory=LLMCorrectionConfig)


class AppConfig(BaseModel):
    """Top-level app configuration."""

    hotkey: HotkeyConfig = Field(default_factory=HotkeyConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    text: TextConfig = Field(default_factory=TextConfig)


def default_config_path() -> Path:
    """Return default user config path."""
    return Path("~/.config/moonshine-flow/config.toml").expanduser()


def _clamp_audio_seconds(value: float, *, field_name: str) -> float:
    if value < 0.0:
        LOGGER.warning("audio.%s=%s is below 0.0; using 0.0", field_name, value)
        return 0.0
    if value > 1.0:
        LOGGER.warning("audio.%s=%s exceeds 1.0; using 1.0", field_name, value)
        return 1.0
    return value


def _clamp_llm_timeout_seconds(value: float) -> float:
    if value < 0.5:
        LOGGER.warning("text.llm_correction.timeout_seconds=%s is below 0.5; using 0.5", value)
        return 0.5
    if value > 5.0:
        LOGGER.warning("text.llm_correction.timeout_seconds=%s exceeds 5.0; using 5.0", value)
        return 5.0
    return value


def _clamp_llm_max_input_chars(value: int) -> int:
    if value < 50:
        LOGGER.warning("text.llm_correction.max_input_chars=%s is below 50; using 50", value)
        return 50
    if value > 5000:
        LOGGER.warning("text.llm_correction.max_input_chars=%s exceeds 5000; using 5000", value)
        return 5000
    return value


def _dump_toml(data: dict[str, Any]) -> str:
    """Serialize TOML without requiring optional dependencies."""
    try:
        import tomli_w

        return tomli_w.dumps(data)
    except Exception:
        input_device = data["audio"].get("input_device")
        if input_device is None:
            input_device_line = ""
        elif isinstance(input_device, str):
            input_device_line = f"input_device = \"{input_device}\"\n"
        else:
            input_device_line = f"input_device = {input_device}\n"

        dictionary_path = data["text"].get("dictionary_path")
        if dictionary_path is None:
            dictionary_path_line = ""
        else:
            dictionary_path_line = f"dictionary_path = \"{dictionary_path}\"\n"
        llm_correction = data["text"].get("llm_correction", {})
        llm_api_key = llm_correction.get("api_key")
        if llm_api_key is None:
            llm_api_key_line = ""
        else:
            llm_api_key_line = f"api_key = \"{llm_api_key}\"\n"

        return (
            "[hotkey]\n"
            f"key = \"{data['hotkey']['key']}\"\n\n"
            "[audio]\n"
            f"sample_rate = {data['audio']['sample_rate']}\n"
            f"channels = {data['audio']['channels']}\n"
            f"dtype = \"{data['audio']['dtype']}\"\n"
            f"max_record_seconds = {data['audio']['max_record_seconds']}\n"
            f"release_tail_seconds = {data['audio']['release_tail_seconds']}\n"
            f"trailing_silence_seconds = {data['audio']['trailing_silence_seconds']}\n"
            f"{input_device_line}\n"
            "\n"
            "[model]\n"
            f"size = \"{data['model']['size']}\"\n"
            f"language = \"{data['model']['language']}\"\n"
            f"device = \"{data['model']['device']}\"\n\n"
            "[output]\n"
            f"mode = \"{data['output']['mode']}\"\n"
            f"paste_shortcut = \"{data['output']['paste_shortcut']}\"\n\n"
            "[runtime]\n"
            f"log_level = \"{data['runtime']['log_level']}\"\n"
            f"notify_on_error = {str(data['runtime']['notify_on_error']).lower()}\n\n"
            "[text]\n"
            f"{dictionary_path_line}\n"
            "[text.llm_correction]\n"
            f"mode = \"{llm_correction.get('mode', 'never')}\"\n"
            f"provider = \"{llm_correction.get('provider', 'ollama')}\"\n"
            f"base_url = \"{llm_correction.get('base_url', 'http://localhost:11434')}\"\n"
            f"model = \"{llm_correction.get('model', 'qwen2.5:7b-instruct')}\"\n"
            f"timeout_seconds = {llm_correction.get('timeout_seconds', 5.0)}\n"
            f"max_input_chars = {llm_correction.get('max_input_chars', 500)}\n"
            f"enabled_tools = {str(llm_correction.get('enabled_tools', False)).lower()}\n"
            f"{llm_api_key_line}"
        )


def _migrate_legacy_llm_correction(raw: dict[str, Any]) -> None:
    text_cfg = raw.get("text")
    if not isinstance(text_cfg, dict):
        return
    llm_cfg = text_cfg.get("llm_correction")
    if not isinstance(llm_cfg, dict):
        return
    if "mode" not in llm_cfg:
        enabled = llm_cfg.get("enabled")
        if isinstance(enabled, bool):
            llm_cfg["mode"] = "always" if enabled else "never"
            LOGGER.warning(
                "text.llm_correction.enabled is deprecated; "
                "use text.llm_correction.mode = \"always|never|ask\" instead"
            )
    if "enabled_tools" in llm_cfg:
        return
    disable_tools = llm_cfg.get("disable_tools")
    if isinstance(disable_tools, bool):
        llm_cfg["enabled_tools"] = not disable_tools
        LOGGER.warning(
            "text.llm_correction.disable_tools is deprecated; "
            "use text.llm_correction.enabled_tools instead"
        )


def _to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _to_primitive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_primitive(item) for item in value]
    return value


def write_example_config(path: Path) -> None:
    """Write an example config file."""
    default_cfg = AppConfig()
    write_config(path, default_cfg)


def write_config(path: Path, config: AppConfig) -> None:
    """Write a concrete app config file."""
    if hasattr(config, "model_dump"):
        cfg = config.model_dump(mode="json")
    else:
        cfg = config.dict()
    cfg = _to_primitive(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_toml(cfg), encoding="utf-8")


def ensure_config_exists(path: Path) -> None:
    """Ensure config file exists at the path."""
    if path.exists():
        return

    bundled = Path(__file__).resolve().parents[2] / "config.example.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if bundled.exists():
        shutil.copyfile(bundled, path)
        return

    write_example_config(path)


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from TOML."""
    config_path = path or default_config_path()
    ensure_config_exists(config_path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    _migrate_legacy_llm_correction(raw)
    if hasattr(AppConfig, "model_validate"):
        config = AppConfig.model_validate(raw)
    else:
        config = AppConfig.parse_obj(raw)

    config.audio.release_tail_seconds = _clamp_audio_seconds(
        float(config.audio.release_tail_seconds),
        field_name="release_tail_seconds",
    )
    config.audio.trailing_silence_seconds = _clamp_audio_seconds(
        float(config.audio.trailing_silence_seconds),
        field_name="trailing_silence_seconds",
    )
    config.text.llm_correction.timeout_seconds = _clamp_llm_timeout_seconds(
        float(config.text.llm_correction.timeout_seconds),
    )
    config.text.llm_correction.max_input_chars = _clamp_llm_max_input_chars(
        int(config.text.llm_correction.max_input_chars),
    )
    return config
