"""CLI entrypoint for moonshine-flow."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from moonshine_flow.app_bundle import (
    APP_BUNDLE_IDENTIFIER,
    app_bundle_executable_path,
    default_app_bundle_path,
    get_app_bundle_codesign_info,
    install_app_bundle_from_env,
)
from moonshine_flow.config import default_config_path, load_config, write_config
from moonshine_flow.launchd import (
    LAUNCHD_LLM_ENABLED_ENV,
    LAUNCH_AGENT_LABEL,
    consume_restart_permission_suppression,
    install_launch_agent,
    launch_agent_log_paths,
    launch_agent_path,
    read_launch_agent_plist,
    resolve_launch_agent_program_prefix,
    restart_launch_agent,
    uninstall_launch_agent,
)
from moonshine_flow.logging_setup import configure_logging
from moonshine_flow.permissions import (
    PermissionReport,
    check_all_permissions,
    check_permissions_in_launchd_context,
    format_permission_guidance,
    recommended_permission_target,
    request_accessibility_permission,
    request_all_permissions,
    request_input_monitoring_permission,
    request_microphone_permission,
    reset_app_bundle_tcc,
)
from moonshine_flow.stt.factory import create_stt_backend, parse_stt_model
from moonshine_flow.text_processing.interfaces import ChainedTextPostProcessor, TextPostProcessor
from moonshine_flow.text_processing.llm import (
    LLMClientError,
    LLMCorrectionSettings,
    LLMPostProcessor,
)
from moonshine_flow.text_processing.repository import (
    CorrectionDictionaryError,
    CorrectionDictionaryLoadResult,
)
from moonshine_flow.text_processing.service import CorrectionService

LOGGER = logging.getLogger(__name__)
_ANSI_YELLOW = "\x1b[33m"
_ANSI_GREEN = "\x1b[32m"
_ANSI_RESET = "\x1b[0m"


def _resolve_app_version() -> str:
    try:
        return package_version("moonshine-flow")
    except PackageNotFoundError:
        return "0.0.0.dev0"


def _resolve_config_path(path_value: str | None) -> Path:
    if path_value:
        return Path(path_value).expanduser()
    return default_config_path()


def _load_corrections_with_diagnostics(
    config: object,
    *,
    config_path: Path,
) -> tuple[CorrectionDictionaryLoadResult | None, str | None]:
    service = CorrectionService.create_default()
    try:
        result = service.load_for_config(
            config=config,
            config_path=config_path,
        )
    except CorrectionDictionaryError as exc:
        return None, str(exc)
    return result, None


def _normalize_optional_secret(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _parse_bool_token(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return None


def _launchd_llm_enabled_override_from_payload(payload: object) -> bool | None:
    if not isinstance(payload, dict):
        return None
    environment = payload.get("EnvironmentVariables")
    if not isinstance(environment, dict):
        return None
    return _parse_bool_token(environment.get(LAUNCHD_LLM_ENABLED_ENV))


def _launchd_llm_enabled_override_from_env() -> bool | None:
    if LAUNCHD_LLM_ENABLED_ENV not in os.environ:
        return None
    raw = os.environ.get(LAUNCHD_LLM_ENABLED_ENV)
    parsed = _parse_bool_token(raw)
    if parsed is None:
        LOGGER.warning(
            "Ignoring invalid %s=%r (expected 1/0/true/false)",
            LAUNCHD_LLM_ENABLED_ENV,
            raw,
        )
    return parsed


def _format_optional_bool(value: bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "<unset>"


def _runtime_language_from_config(config: object) -> str:
    language = getattr(config, "language", "en")
    if not isinstance(language, str):
        return "en"
    normalized = language.strip()
    if not normalized or normalized.lower() == "auto":
        return "en"
    return normalized


def _build_llm_settings_from_config(
    config: object,
    llm_cfg: object,
) -> LLMCorrectionSettings | None:
    provider = str(getattr(llm_cfg, "provider", "")).strip().lower()
    base_url = str(getattr(llm_cfg, "base_url", "")).strip()
    model = str(getattr(llm_cfg, "model", "")).strip()
    if not base_url or not model:
        return None
    return LLMCorrectionSettings(
        provider=provider,
        base_url=base_url,
        model=model,
        timeout_seconds=float(getattr(llm_cfg, "timeout_seconds", 5.0)),
        max_input_chars=int(getattr(llm_cfg, "max_input_chars", 500)),
        api_key=_normalize_optional_secret(getattr(llm_cfg, "api_key", None)),
        enabled_tools=bool(getattr(llm_cfg, "enabled_tools", False)),
        language=_runtime_language_from_config(config),
    )


def _stt_model_from_config(config: object) -> str:
    stt_cfg = getattr(config, "stt", None)
    token = str(getattr(stt_cfg, "model", "")).strip()
    if not token:
        raise ValueError("stt.model is empty")
    return token


def _is_moonshine_stt_model(config: object) -> bool:
    prefix, _model_id = parse_stt_model(_stt_model_from_config(config))
    return prefix == "moonshine"


def _is_vllm_stt_model(config: object) -> bool:
    prefix, _model_id = parse_stt_model(_stt_model_from_config(config))
    return prefix == "vllm"


def _is_mlx_stt_model(config: object) -> bool:
    prefix, _model_id = parse_stt_model(_stt_model_from_config(config))
    return prefix == "mlx"


def _is_voxtral_stt_model(config: object) -> bool:
    prefix, _model_id = parse_stt_model(_stt_model_from_config(config))
    return prefix == "voxtral"


def _supports_ansi_styles() -> bool:
    return _supports_ansi_for_stream(sys.stdout)


def _supports_ansi_styles_stderr() -> bool:
    return _supports_ansi_for_stream(sys.stderr)


def _supports_ansi_for_stream(stream: object) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty) or not isatty():
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    term = str(os.environ.get("TERM", "")).strip().lower()
    if term in {"", "dumb"}:
        return False
    return True


def _dim(text: str) -> str:
    if not _supports_ansi_styles():
        return text
    return f"\x1b[2m{text}\x1b[0m"


def _yellow(text: str, *, stderr: bool = False) -> str:
    if stderr:
        if not _supports_ansi_styles_stderr():
            return text
    else:
        if not _supports_ansi_styles():
            return text
    return f"{_ANSI_YELLOW}{text}{_ANSI_RESET}"


def _green(text: str, *, stderr: bool = False) -> str:
    if stderr:
        if not _supports_ansi_styles_stderr():
            return text
    else:
        if not _supports_ansi_styles():
            return text
    return f"{_ANSI_GREEN}{text}{_ANSI_RESET}"


def _display_value(value: object) -> str:
    if value is None:
        return "<unset>"
    text = str(value)
    if text == "":
        return "<empty>"
    return text


def _display_secret(value: str | None) -> str:
    if value:
        return "<SET>"
    return "<unset>"


def _format_prompt(
    label: str,
    default_display: str,
    *,
    current_display: str | None = None,
    suffix: str | None = None,
) -> str:
    current = current_display if current_display is not None else default_display
    prompt = f"{label} [{default_display}] {_dim(f'(current: {current})')}"
    if suffix:
        prompt += f" {suffix}"
    return f"{prompt}: "


def _print_keep(value_display: str) -> None:
    print(_dim(f"keep: {value_display}"))


def _prompt_text(label: str, default: str) -> str:
    raw = input(
        _format_prompt(
            label,
            default,
            current_display=_display_value(default),
        )
    ).strip()
    if raw == "":
        _print_keep(_display_value(default))
        return default
    return raw


def _prompt_optional_text(label: str, default: str | None) -> str | None:
    default_display = "" if default is None else default
    raw = input(
        _format_prompt(
            label,
            default_display,
            current_display=_display_value(default),
            suffix="(Enter to keep, '-' to unset)",
        )
    ).strip()
    if raw == "":
        _print_keep(_display_value(default))
        return default
    if raw == "-":
        return None
    return raw


def _prompt_optional_secret(label: str, default: str | None) -> str | None:
    default_display = "<SET>" if default else ""
    raw = input(
        _format_prompt(
            label,
            default_display,
            current_display=_display_secret(default),
            suffix="(Enter to keep, '-' to unset)",
        )
    ).strip()
    if raw == "":
        _print_keep(_display_secret(default))
        return default
    if raw == "-":
        return None
    return raw


def _prompt_bool(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(
            _format_prompt(
                label,
                suffix,
                current_display=str(default).lower(),
            )
        ).strip().lower()
        if raw == "":
            _print_keep(str(default).lower())
            return default
        if raw in {"y", "yes", "true", "1", "on"}:
            return True
        if raw in {"n", "no", "false", "0", "off"}:
            return False
        print("Please enter y or n.")


def _prompt_choice(label: str, default: str, choices: list[str]) -> str:
    if not choices:
        raise ValueError("choices must not be empty")

    allowed = {choice.lower(): choice for choice in choices}
    default_index = 1
    for idx, choice in enumerate(choices, start=1):
        if choice == default:
            default_index = idx
            break

    while True:
        print(f"{label} {_dim(f'(current: {default})')}")
        for idx, choice in enumerate(choices, start=1):
            marker = _dim(" (current)") if choice == default else ""
            print(f"  {idx}. {choice}{marker}")

        raw = input(f"Select number [{default_index}]: ").strip()
        if raw == "":
            _print_keep(default)
            return default
        if raw.isdigit():
            selected = int(raw)
            if 1 <= selected <= len(choices):
                return choices[selected - 1]
            print(f"Please choose a number between 1 and {len(choices)}.")
            continue
        choice = allowed.get(raw.lower())
        if choice is not None:
            return choice
        print(f"Please choose a number between 1 and {len(choices)}.")


def _prompt_int(
    label: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    while True:
        raw = input(
            _format_prompt(
                label,
                str(default),
                current_display=str(default),
            )
        ).strip()
        if raw == "":
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Please enter a value <= {maximum}.")
            continue
        if raw == "":
            _print_keep(str(default))
        return value


def _prompt_float(
    label: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    while True:
        raw = input(
            _format_prompt(
                label,
                str(default),
                current_display=str(default),
            )
        ).strip()
        if raw == "":
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue
        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Please enter a value <= {maximum}.")
            continue
        if raw == "":
            _print_keep(str(default))
        return value


def _prompt_input_device(default: int | str | None) -> int | str | None:
    default_display = "" if default is None else str(default)
    raw = input(
        _format_prompt(
            "audio.input_device",
            default_display,
            current_display=_display_value(default),
            suffix="(name/index, Enter to keep, '-' to unset)",
        )
    ).strip()
    if raw == "":
        _print_keep(_display_value(default))
        return default
    if raw == "-":
        return None
    if raw.lstrip("-").isdigit():
        return int(raw)
    return raw


def _query_input_devices() -> tuple[list[dict[str, Any]], int | None]:
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError(f"sounddevice is unavailable: {exc}") from exc

    try:
        raw_devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Failed to query audio devices: {exc}") from exc

    devices: list[dict[str, Any]] = []
    for index, device in enumerate(raw_devices):
        get_field = getattr(device, "get", None)
        if not callable(get_field):
            continue
        max_input_channels = int(get_field("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        name = str(get_field("name", f"Device {index}")).strip() or f"Device {index}"
        devices.append(
            {
                "index": index,
                "name": name,
                "max_input_channels": max_input_channels,
            }
        )

    default_input_index: int | None = None
    try:
        default_device = getattr(sd, "default", None)
        default_pair = getattr(default_device, "device", None)
        if isinstance(default_pair, (list, tuple)) and default_pair:
            default_candidate = default_pair[0]
            default_index = int(default_candidate)
            if default_index >= 0:
                default_input_index = default_index
    except Exception:
        default_input_index = None

    return devices, default_input_index


def _matches_configured_input_device(configured: int | str | None, *, index: int, name: str) -> bool:
    if configured is None:
        return False
    if isinstance(configured, int):
        return configured == index
    return configured == name


def _query_ollama_model_names(
    *,
    base_url: str,
    timeout_seconds: float,
    api_key: str | None,
) -> list[str]:
    base = base_url.rstrip("/") + "/"
    url = urljoin(base, "api/tags")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, method="GET", headers=headers)

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"Ollama request failed (HTTP {exc.code})") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Ollama connection failed: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Ollama request timed out") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama endpoint returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Ollama endpoint returned unexpected payload")

    models = payload.get("models")
    if not isinstance(models, list):
        raise RuntimeError("Ollama response is missing 'models'")

    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        names.append(name)
    return names


def _query_lmstudio_model_names(
    *,
    base_url: str,
    timeout_seconds: float,
    api_key: str | None,
) -> list[str]:
    base = base_url.rstrip("/") + "/"
    url = urljoin(base, "v1/models")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, method="GET", headers=headers)

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"LM Studio request failed (HTTP {exc.code})") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"LM Studio connection failed: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("LM Studio request timed out") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LM Studio endpoint returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("LM Studio endpoint returned unexpected payload")

    models = payload.get("data")
    if not isinstance(models, list):
        raise RuntimeError("LM Studio response is missing 'data'")

    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("id", "")).strip()
        if not name:
            continue
        names.append(name)
    return names


def _select_llm_model_and_save(
    *,
    config_path: Path,
    config: object,
    base_url: str,
    model_names: list[str],
) -> int:
    if not model_names:
        print(_yellow("Warning: No models were found.", stderr=True), file=sys.stderr)
        return 0

    llm_cfg = getattr(getattr(config, "text", None), "llm_correction", None)
    current_model = str(getattr(llm_cfg, "model", "")).strip()
    default_choice: int | None = None

    print(f"Config: {config_path}")
    print(f"Endpoint: {base_url}")
    print(f"Current text.llm_correction.model: {_display_value(current_model)}")
    print("Select model to save into config:")
    for menu_index, model_name in enumerate(model_names, start=1):
        marker = ""
        if model_name == current_model:
            marker = " (current)"
            default_choice = menu_index
        print(f"  {menu_index}. {model_name}{marker}")

    if default_choice is None:
        print(_yellow("Warning: current model is not in the available model list."))

    try:
        while True:
            prompt_default = "" if default_choice is None else str(default_choice)
            raw = input(f"Select number [{prompt_default}]: ").strip()
            if raw == "":
                if default_choice is None:
                    print("Please choose a number from the list.")
                    continue
                selected = default_choice
                _print_keep(str(default_choice))
            elif raw.isdigit():
                selected = int(raw)
            else:
                print(f"Please choose a number between 1 and {len(model_names)}.")
                continue

            if 1 <= selected <= len(model_names):
                break
            print(f"Please choose a number between 1 and {len(model_names)}.")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 130

    config.text.llm_correction.model = model_names[selected - 1]
    write_config(config_path, config)
    print(_green(f"Updated config: {config_path}"))
    print(f"text.llm_correction.model = {config.text.llm_correction.model}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    del args
    print("Available list commands:")
    print("  devices    List audio input devices and save selected device to config")
    print("  model      List STT model presets and save selected model to config")
    print("  typing     List output typing modes and save selected mode to config")
    print("  ollama     List downloaded Ollama models")
    print("  lmstudio   List loaded LM Studio models")
    print("")
    print("Usage:")
    print("  mflow list devices")
    print("  mflow list model")
    print("  mflow list typing")
    print("  mflow list ollama")
    print("  mflow list lmstudio")
    return 0


def _huggingface_cache_hub_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_huggingface_model_downloaded(model_id: str) -> bool:
    snapshots_dir = _huggingface_cache_hub_dir() / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    try:
        for snapshot in snapshots_dir.iterdir():
            if not snapshot.is_dir():
                continue
            try:
                next(snapshot.iterdir())
                return True
            except StopIteration:
                continue
    except OSError:
        return False
    return False


def _stt_model_downloaded_display(model_token: str) -> str:
    try:
        prefix, model_id = parse_stt_model(model_token)
    except ValueError:
        return "unknown"
    if prefix in {"vllm", "mlx", "voxtral"}:
        return "yes" if _is_huggingface_model_downloaded(model_id) else "no"
    # moonshine-voice manages model cache internally.
    return "unknown"


def _stt_model_presets() -> list[str]:
    return [
        "moonshine:tiny",
        "moonshine:base",
        "voxtral:mistralai/Voxtral-Mini-4B-Realtime-2602",
    ]


def cmd_list_model(args: argparse.Namespace) -> int:
    if not _is_interactive_session():
        print("`mflow list model` requires an interactive terminal.", file=sys.stderr)
        return 2

    config_path = _resolve_config_path(args.config)
    config = load_config(config_path)
    current_model = _stt_model_from_config(config)

    model_choices = [*_stt_model_presets(), "custom"]
    default_choice: int | None = None

    print(f"Config: {config_path}")
    print(f"Current stt.model: {current_model}")
    print(f"Current downloaded: {_stt_model_downloaded_display(current_model)}")
    print("Select STT model to save into config:")
    for menu_index, model_name in enumerate(model_choices, start=1):
        marker = ""
        if model_name == current_model:
            marker = " (current)"
            default_choice = menu_index
        downloaded_text = ""
        if model_name != "custom":
            downloaded_text = f" (downloaded: {_stt_model_downloaded_display(model_name)})"
        print(f"  {menu_index}. {model_name}{downloaded_text}{marker}")

    if default_choice is None:
        print(_yellow("Warning: current stt.model is not in the preset list."))

    try:
        while True:
            prompt_default = "" if default_choice is None else str(default_choice)
            raw = input(f"Select number [{prompt_default}]: ").strip()
            if raw == "":
                if default_choice is None:
                    print("Please choose a number from the list.")
                    continue
                selected = default_choice
                _print_keep(str(default_choice))
            elif raw.isdigit():
                selected = int(raw)
            else:
                print(f"Please choose a number between 1 and {len(model_choices)}.")
                continue

            if 1 <= selected <= len(model_choices):
                break
            print(f"Please choose a number between 1 and {len(model_choices)}.")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 130

    selected_model = model_choices[selected - 1]
    if selected_model == "custom":
        while True:
            token = _prompt_text(
                "stt.model.custom",
                current_model if default_choice is None else "",
            ).strip()
            try:
                parse_stt_model(token)
            except ValueError as exc:
                print(str(exc))
                continue
            selected_model = token
            break

    config.stt.model = selected_model
    write_config(config_path, config)
    print(_green(f"Updated config: {config_path}"))
    print(f"stt.model = {config.stt.model}")
    return 0


def cmd_list_typing(args: argparse.Namespace) -> int:
    if not _is_interactive_session():
        print("`mflow list typing` requires an interactive terminal.", file=sys.stderr)
        return 2

    config_path = _resolve_config_path(args.config)
    config = load_config(config_path)
    output_cfg = getattr(config, "output", None)
    current_mode = str(getattr(output_cfg, "mode", "direct_typing")).strip().lower()
    mode_type = type(getattr(output_cfg, "mode", "direct_typing"))
    if hasattr(mode_type, "__members__"):
        mode_choices = [member.value for member in mode_type]
    else:
        mode_choices = ["direct_typing", "clipboard_paste"]

    default_choice: int | None = None
    print(f"Config: {config_path}")
    print(f"Current output.mode: {current_mode}")
    print("Select output mode to save into config:")
    for menu_index, mode_name in enumerate(mode_choices, start=1):
        marker = " (current)" if mode_name == current_mode else ""
        description = ""
        if mode_name == "direct_typing":
            description = " - type directly without clipboard"
        elif mode_name == "clipboard_paste":
            description = " - copy+paste via clipboard"
        print(f"  {menu_index}. {mode_name}{description}{marker}")
        if mode_name == current_mode:
            default_choice = menu_index

    if default_choice is None:
        print(_yellow("Warning: current output.mode is not in the mode list."))

    try:
        while True:
            prompt_default = "" if default_choice is None else str(default_choice)
            raw = input(f"Select number [{prompt_default}]: ").strip()
            if raw == "":
                if default_choice is None:
                    print("Please choose a number from the list.")
                    continue
                selected = default_choice
                _print_keep(str(default_choice))
            elif raw.isdigit():
                selected = int(raw)
            else:
                print(f"Please choose a number between 1 and {len(mode_choices)}.")
                continue

            if 1 <= selected <= len(mode_choices):
                break
            print(f"Please choose a number between 1 and {len(mode_choices)}.")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 130

    selected_mode = mode_choices[selected - 1]
    try:
        output_cfg.mode = mode_type(selected_mode)
    except Exception:
        output_cfg.mode = selected_mode
    write_config(config_path, config)
    print(_green(f"Updated config: {config_path}"))
    print(f"output.mode = {output_cfg.mode}")
    return 0


def cmd_list_ollama(args: argparse.Namespace) -> int:
    if not _is_interactive_session():
        print("`mflow list ollama` requires an interactive terminal.", file=sys.stderr)
        return 2

    config_path = _resolve_config_path(args.config)
    config = load_config(config_path)

    llm_cfg = getattr(getattr(config, "text", None), "llm_correction", None)
    if llm_cfg is None:
        print("Error: text.llm_correction is missing in config.", file=sys.stderr)
        return 2

    provider = str(getattr(llm_cfg, "provider", "")).strip().lower()
    if provider != "ollama":
        print(
            f"Error: `mflow list ollama` requires text.llm_correction.provider = \"ollama\" "
            f"(current: {provider or '<unset>'}).",
            file=sys.stderr,
        )
        return 2

    base_url = str(getattr(llm_cfg, "base_url", "")).strip()
    if not base_url:
        print("Error: text.llm_correction.base_url is empty.", file=sys.stderr)
        return 2
    timeout_seconds = float(getattr(llm_cfg, "timeout_seconds", 5.0))
    api_key = _normalize_optional_secret(getattr(llm_cfg, "api_key", None))

    try:
        model_names = _query_ollama_model_names(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )
    except Exception as exc:
        print(_yellow(f"Warning: {exc}", stderr=True), file=sys.stderr)
        return 2

    return _select_llm_model_and_save(
        config_path=config_path,
        config=config,
        base_url=base_url,
        model_names=model_names,
    )


def cmd_list_lmstudio(args: argparse.Namespace) -> int:
    if not _is_interactive_session():
        print("`mflow list lmstudio` requires an interactive terminal.", file=sys.stderr)
        return 2

    config_path = _resolve_config_path(args.config)
    config = load_config(config_path)

    llm_cfg = getattr(getattr(config, "text", None), "llm_correction", None)
    if llm_cfg is None:
        print("Error: text.llm_correction is missing in config.", file=sys.stderr)
        return 2

    provider = str(getattr(llm_cfg, "provider", "")).strip().lower()
    if provider != "lmstudio":
        print(
            f"Error: `mflow list lmstudio` requires text.llm_correction.provider = \"lmstudio\" "
            f"(current: {provider or '<unset>'}).",
            file=sys.stderr,
        )
        return 2

    base_url = str(getattr(llm_cfg, "base_url", "")).strip()
    if not base_url:
        print("Error: text.llm_correction.base_url is empty.", file=sys.stderr)
        return 2
    timeout_seconds = float(getattr(llm_cfg, "timeout_seconds", 5.0))
    api_key = _normalize_optional_secret(getattr(llm_cfg, "api_key", None))

    try:
        model_names = _query_lmstudio_model_names(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )
    except Exception as exc:
        print(_yellow(f"Warning: {exc}", stderr=True), file=sys.stderr)
        return 2

    return _select_llm_model_and_save(
        config_path=config_path,
        config=config,
        base_url=base_url,
        model_names=model_names,
    )


def cmd_list_devices(args: argparse.Namespace) -> int:
    if not _is_interactive_session():
        print("`mflow list devices` requires an interactive terminal.", file=sys.stderr)
        return 2

    config_path = _resolve_config_path(args.config)
    config = load_config(config_path)

    try:
        devices, default_input_index = _query_input_devices()
    except Exception as exc:
        print(_yellow(f"Warning: {exc}", stderr=True), file=sys.stderr)
        return 2

    current_input_device = getattr(getattr(config, "audio", None), "input_device", None)
    default_choice: int | None = 0 if current_input_device is None else None

    print(f"Config: {config_path}")
    print(f"Current audio.input_device: {_display_value(current_input_device)}")
    print("Select input device to save into config:")
    current_marker = " (current)" if current_input_device is None else ""
    print(f"  0. system default (unset){current_marker}")
    for menu_index, device in enumerate(devices, start=1):
        index = int(device["index"])
        name = str(device["name"])
        max_input_channels = int(device["max_input_channels"])
        markers: list[str] = []
        if index == default_input_index:
            markers.append("default")
        if _matches_configured_input_device(current_input_device, index=index, name=name):
            markers.append("current")
            default_choice = menu_index
        marker_text = f" ({', '.join(markers)})" if markers else ""
        print(f"  {menu_index}. [{index}] {name} (inputs={max_input_channels}){marker_text}")

    if default_choice is None:
        print(_yellow("Warning: current audio.input_device is not in the detected input device list."))

    try:
        while True:
            prompt_default = "" if default_choice is None else str(default_choice)
            raw = input(f"Select number [{prompt_default}]: ").strip()
            if raw == "":
                if default_choice is None:
                    print("Please choose a number from the list.")
                    continue
                selected = default_choice
                _print_keep(str(default_choice))
            elif raw.isdigit():
                selected = int(raw)
            else:
                print(f"Please choose a number between 0 and {len(devices)}.")
                continue

            if 0 <= selected <= len(devices):
                break
            print(f"Please choose a number between 0 and {len(devices)}.")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 130

    if selected == 0:
        config.audio.input_device = None
    else:
        config.audio.input_device = int(devices[selected - 1]["index"])

    write_config(config_path, config)
    print(_green(f"Updated config: {config_path}"))
    print(f"audio.input_device = {_display_value(config.audio.input_device)}")
    return 0


def _is_interactive_session() -> bool:
    stdin = getattr(sys.stdin, "isatty", None)
    stdout = getattr(sys.stdout, "isatty", None)
    return bool(callable(stdin) and stdin() and callable(stdout) and stdout())


def _prompt_llm_correction_for_this_run() -> bool:
    try:
        answer = input("Enable LLM post-correction for this run? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


def _prompt_launchd_llm_enabled(current: bool | None) -> bool:
    default = current if current is not None else False
    return _prompt_bool("Enable LLM correction for launchd runs", default)


def _preflight_llm_for_launchd(config: object) -> tuple[bool, str | None]:
    llm_cfg = getattr(getattr(config, "text", None), "llm_correction", None)
    if llm_cfg is None:
        return False, "text.llm_correction is missing"
    settings = _build_llm_settings_from_config(config, llm_cfg)
    if settings is None:
        return False, "base_url/model is missing"
    try:
        processor = LLMPostProcessor(settings)
        processor.preflight()
    except Exception as exc:
        return False, str(exc)
    return True, None


def _resolve_launchd_llm_enabled_override_for_command(
    *,
    current_override: bool | None,
    preflight_func: Callable[[], tuple[bool, str | None]],
) -> bool | None:
    if not _is_interactive_session():
        print(
            f"No TTY; keeping existing launchd LLM override: "
            f"{_format_optional_bool(current_override)}"
        )
        return current_override

    selected = _prompt_launchd_llm_enabled(current_override)
    if not selected:
        return False

    ok, reason = preflight_func()
    if ok:
        print(
            _green(
                "LLM preflight succeeded during install/restart; "
                "launchd LLM override set to true."
            )
        )
        return True
    print(
        _yellow(
            "Warning: LLM preflight failed during install/restart; "
            "launchd LLM override set to false.",
            stderr=True,
        ),
        file=sys.stderr,
    )
    print(
        _yellow(
            "Warning: You selected YES, but it was switched to NO "
            "because the LLM endpoint was unreachable or invalid.",
            stderr=True,
        ),
        file=sys.stderr,
    )
    if reason:
        print(_yellow(f"Preflight detail: {reason}", stderr=True), file=sys.stderr)
    return False


def _should_enable_llm_correction_for_this_run(llm_cfg: object) -> bool:
    in_launchd_context = os.environ.get("XPC_SERVICE_NAME") == LAUNCH_AGENT_LABEL
    if in_launchd_context:
        launchd_override = _launchd_llm_enabled_override_from_env()
        if launchd_override is not None:
            return launchd_override

    mode = str(getattr(llm_cfg, "mode", "never")).strip().lower()
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode == "ask":
        if not _is_interactive_session():
            LOGGER.info(
                "LLM correction mode=ask in non-interactive session; "
                "disabling LLM correction for this run"
            )
            return False
        return _prompt_llm_correction_for_this_run()
    LOGGER.warning("Unknown LLM correction mode '%s'; disabling LLM correction", mode)
    return False


def _llm_enabled_for_this_run(config: object) -> bool:
    llm_cfg = getattr(getattr(config, "text", None), "llm_correction", None)
    if llm_cfg is None:
        return False
    return _should_enable_llm_correction_for_this_run(llm_cfg)


def _streaming_supported_by_output_mode(config: object) -> bool:
    del config
    return True


def _build_runtime_post_processor(
    config: object,
    *,
    base_processor: TextPostProcessor,
    llm_enabled_override: bool | None = None,
) -> TextPostProcessor:
    text_cfg = getattr(config, "text", None)
    llm_cfg = getattr(text_cfg, "llm_correction", None)
    llm_enabled = llm_enabled_override
    if llm_enabled is None:
        llm_enabled = llm_cfg is not None and _should_enable_llm_correction_for_this_run(llm_cfg)
    if llm_cfg is None or not llm_enabled:
        return base_processor

    settings = _build_llm_settings_from_config(config, llm_cfg)
    if settings is None:
        LOGGER.warning(
            "LLM correction is enabled but base_url/model is missing; "
            "continuing without LLM correction"
        )
        return base_processor

    try:
        llm_processor = LLMPostProcessor(settings)
    except Exception as exc:
        LOGGER.warning("Failed to initialize LLM correction; continuing without it (%s)", exc)
        return base_processor

    try:
        llm_processor.preflight()
    except LLMClientError as exc:
        LOGGER.warning("LLM correction preflight warning: %s", exc)
    except Exception as exc:
        LOGGER.warning("Unexpected LLM preflight failure: %s", exc)

    LOGGER.info(
        _green(
            "LLM correction enabled: provider=%s base_url=%s model=%s "
            "timeout=%.2fs max_input_chars=%d",
            stderr=True,
        ),
        settings.provider,
        settings.base_url,
        settings.model,
        settings.timeout_seconds,
        settings.max_input_chars,
    )
    return ChainedTextPostProcessor([base_processor, llm_processor])


def _format_secret_state(secret: str | None) -> str:
    if not secret:
        return "UNSET"
    return "SET"


def _prompt_stt_model(current: str) -> str:
    catalog = [*_stt_model_presets(), "custom"]
    default = current if current in catalog[:-1] else "custom"
    selected = _prompt_choice("stt.model", default, catalog)
    if selected != "custom":
        return selected
    while True:
        token = _prompt_text("stt.model.custom", current if default == "custom" else "").strip()
        try:
            parse_stt_model(token)
        except ValueError as exc:
            print(str(exc))
            continue
        return token


def cmd_init(args: argparse.Namespace) -> int:
    if not _is_interactive_session():
        print("`mflow init` requires an interactive terminal.", file=sys.stderr)
        return 2

    config_path = _resolve_config_path(args.config)
    config = load_config(config_path, allow_legacy_model_size=True)

    print(f"Config: {config_path}")
    print("Press Enter to keep current values. Use '-' to unset optional fields.")
    print(_dim("(current: ...) and keep lines are shown in dim text."))
    print("")
    try:
        config.hotkey.key = _prompt_text("hotkey.key", config.hotkey.key)

        config.audio.sample_rate = _prompt_int("audio.sample_rate", config.audio.sample_rate, minimum=1)
        config.audio.channels = _prompt_int("audio.channels", config.audio.channels, minimum=1)
        config.audio.dtype = _prompt_text("audio.dtype", config.audio.dtype)
        config.audio.max_record_seconds = _prompt_int(
            "audio.max_record_seconds",
            config.audio.max_record_seconds,
            minimum=1,
        )
        config.audio.release_tail_seconds = _prompt_float(
            "audio.release_tail_seconds",
            float(config.audio.release_tail_seconds),
            minimum=0.0,
            maximum=1.0,
        )
        config.audio.trailing_silence_seconds = _prompt_float(
            "audio.trailing_silence_seconds",
            float(config.audio.trailing_silence_seconds),
            minimum=0.0,
            maximum=1.0,
        )
        config.audio.input_device = _prompt_input_device(config.audio.input_device)

        config.stt.model = _prompt_stt_model(config.stt.model)
        config.language = _prompt_text("language", config.language)
        config.model.device = _prompt_text("model.device", config.model.device)

        output_mode_choices = [mode.value for mode in type(config.output.mode)]
        selected_output_mode = _prompt_choice(
            "output.mode",
            config.output.mode.value,
            output_mode_choices,
        )
        config.output.mode = type(config.output.mode)(selected_output_mode)
        config.output.paste_shortcut = _prompt_text("output.paste_shortcut", config.output.paste_shortcut)

        config.runtime.log_level = _prompt_text("runtime.log_level", config.runtime.log_level)
        config.runtime.notify_on_error = _prompt_bool(
            "runtime.notify_on_error",
            bool(config.runtime.notify_on_error),
        )

        config.text.dictionary_path = _prompt_optional_text(
            "text.dictionary_path",
            config.text.dictionary_path,
        )

        llm_mode_choices = [mode.value for mode in type(config.text.llm_correction.mode)]
        selected_llm_mode = _prompt_choice(
            "text.llm_correction.mode",
            config.text.llm_correction.mode.value,
            llm_mode_choices,
        )
        config.text.llm_correction.mode = type(config.text.llm_correction.mode)(selected_llm_mode)

        known_llm_providers = ["ollama", "lmstudio"]
        current_llm_provider = str(config.text.llm_correction.provider).strip().lower()
        provider_default = (
            current_llm_provider if current_llm_provider in known_llm_providers else "other"
        )
        selected_llm_provider = _prompt_choice(
            "text.llm_correction.provider",
            provider_default,
            known_llm_providers + ["other"],
        )
        if selected_llm_provider == "other":
            other_default = (
                current_llm_provider
                if current_llm_provider and current_llm_provider not in known_llm_providers
                else ""
            )
            while True:
                provider_other = _prompt_text(
                    "text.llm_correction.provider_other",
                    other_default,
                ).strip()
                if provider_other:
                    config.text.llm_correction.provider = provider_other
                    break
                print("Provider cannot be empty.")
        else:
            config.text.llm_correction.provider = selected_llm_provider
        config.text.llm_correction.base_url = _prompt_text(
            "text.llm_correction.base_url",
            config.text.llm_correction.base_url,
        )
        config.text.llm_correction.model = _prompt_text(
            "text.llm_correction.model",
            config.text.llm_correction.model,
        )
        config.text.llm_correction.timeout_seconds = _prompt_float(
            "text.llm_correction.timeout_seconds",
            float(config.text.llm_correction.timeout_seconds),
            minimum=0.5,
            maximum=5.0,
        )
        config.text.llm_correction.max_input_chars = _prompt_int(
            "text.llm_correction.max_input_chars",
            int(config.text.llm_correction.max_input_chars),
            minimum=50,
            maximum=5000,
        )
        config.text.llm_correction.api_key = _prompt_optional_secret(
            "text.llm_correction.api_key",
            _normalize_optional_secret(config.text.llm_correction.api_key),
        )
        config.text.llm_correction.enabled_tools = _prompt_bool(
            "text.llm_correction.enabled_tools",
            bool(config.text.llm_correction.enabled_tools),
        )
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 130

    write_config(config_path, config)
    print(f"Updated config: {config_path}")
    return 0


def _has_moonshine_backend() -> bool:
    return bool(find_spec("moonshine_voice"))


def _has_vllm_backend() -> bool:
    return bool(find_spec("vllm"))


def _has_websockets_backend() -> bool:
    return bool(find_spec("websockets"))


def _has_mlx_backend() -> bool:
    return bool(find_spec("mlx_whisper"))


def _is_macos_arm64() -> bool:
    machine = platform.machine().strip().lower()
    return sys.platform == "darwin" and machine in {"arm64", "aarch64"}


def _has_voxtral_mlx_backend() -> bool:
    return _is_macos_arm64() and bool(find_spec("voxmlx"))


def _has_voxtral_transformers_backend() -> bool:
    return bool(find_spec("transformers")) and bool(find_spec("mistral_common"))


def _has_voxtral_backend() -> bool:
    if _is_macos_arm64():
        return _has_voxtral_mlx_backend()
    return _has_voxtral_transformers_backend()


def _backend_guidance() -> str:
    return (
        "Moonshine backend package is missing. "
        "Install dependencies and run `uv sync` again."
    )


def _vllm_backend_guidance(missing: list[str]) -> str:
    missing_text = ", ".join(sorted(missing))
    machine = platform.machine().strip().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return (
            f"vLLM backend dependencies are missing ({missing_text}). "
            "Local vLLM is not currently available on macOS arm64 in this environment. "
            "Use stt.model=mlx:mlx-community/whisper-large-v3-turbo "
            "(for example: `mflow list model`), "
            "or use stt.model=moonshine:base, or run vLLM on a Linux host."
        )
    return (
        f"vLLM backend dependencies are missing ({missing_text}). "
        "Install them (example: `uv add vllm websockets`) or switch stt.model to moonshine:base."
    )


def _mlx_backend_guidance() -> str:
    return (
        "MLX backend dependency is missing (mlx-whisper). "
        "Install it (example: `uv add mlx-whisper`) or switch stt.model to moonshine:base."
    )


def _voxtral_backend_guidance() -> str:
    if _is_macos_arm64():
        return (
            "Voxtral backend dependencies are missing. "
            "On macOS arm64, install voxmlx (example: `uv add voxmlx`) "
            "or switch stt.model to moonshine:base."
        )
    return (
        "Voxtral backend dependencies are missing. "
        "Install them (example: `uv add --upgrade transformers \"mistral-common[audio]\"`) "
        "or switch stt.model to moonshine:base."
    )


def _format_command(command: list[str]) -> str:
    return " ".join(command)


def _format_launchd_permission_guidance(
    report: PermissionReport,
    *,
    target_executable: str | None,
) -> str:
    lines = [
        "Missing macOS permissions detected for launchd runtime:",
        *[f"- {item}" for item in report.missing],
        "",
        "Open: System Settings -> Privacy & Security",
        "Then enable this app in:",
        "- Accessibility",
        "- Input Monitoring",
        "- Microphone",
    ]
    if target_executable:
        lines.extend(
            [
                "",
                f"Launchd target executable: {target_executable}",
            ]
        )
    lines.extend(
        [
            "",
            "If the app does not appear in Input Monitoring, rerun "
            "`moonshine-flow install-launch-agent --request-permissions`.",
        ]
    )
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> int:
    from moonshine_flow.daemon import MoonshineFlowDaemon

    config_path = _resolve_config_path(args.config)
    config = load_config(config_path)
    configure_logging(config.runtime.log_level)

    correction_result, correction_error = _load_corrections_with_diagnostics(
        config,
        config_path=config_path,
    )
    if correction_error is not None:
        LOGGER.error("Failed to load transcription correction dictionary: %s", correction_error)
        return 5
    assert correction_result is not None
    for warning in correction_result.warnings:
        LOGGER.warning(warning.message)
    if correction_result.loaded:
        LOGGER.info(
            _green(
                "Transcription correction dictionary loaded: "
                "path=%s exact=%d regex=%d disabled_regex=%d",
                stderr=True,
            ),
            correction_result.path,
            correction_result.rules.exact_count,
            correction_result.rules.regex_count,
            correction_result.disabled_regex_count,
        )

    try:
        if _is_moonshine_stt_model(config) and not _has_moonshine_backend():
            LOGGER.error(_backend_guidance())
            return 3
        if _is_vllm_stt_model(config):
            LOGGER.info(
                "Selected vLLM model downloaded: %s",
                _stt_model_downloaded_display(_stt_model_from_config(config)),
            )
            missing: list[str] = []
            if not _has_vllm_backend():
                missing.append("vllm")
            if not _has_websockets_backend():
                missing.append("websockets")
            if missing:
                _, model_id = parse_stt_model(_stt_model_from_config(config))
                if (
                    model_id == "mistralai/Voxtral-Mini-4B-Realtime-2602"
                    and _has_voxtral_backend()
                ):
                    LOGGER.warning(_vllm_backend_guidance(missing))
                    LOGGER.warning(
                        "Using local Voxtral backend for this run instead of vLLM"
                    )
                    config.stt.model = f"voxtral:{model_id}"
                else:
                    LOGGER.error(_vllm_backend_guidance(missing))
                    return 3
        if _is_voxtral_stt_model(config):
            LOGGER.info(
                "Selected Voxtral model downloaded: %s",
                _stt_model_downloaded_display(_stt_model_from_config(config)),
            )
            if not _has_voxtral_backend():
                LOGGER.error(_voxtral_backend_guidance())
                return 3
        if _is_mlx_stt_model(config):
            LOGGER.info(
                "Selected MLX model downloaded: %s",
                _stt_model_downloaded_display(_stt_model_from_config(config)),
            )
            if not _has_mlx_backend():
                LOGGER.error(_mlx_backend_guidance())
                return 3
    except Exception as exc:
        LOGGER.error("Invalid stt.model configuration: %s", exc)
        return 2

    report = check_all_permissions()
    in_launchd_context = os.environ.get("XPC_SERVICE_NAME") == LAUNCH_AGENT_LABEL
    suppressed_after_restart = in_launchd_context and consume_restart_permission_suppression()
    if in_launchd_context and not report.all_granted:
        if suppressed_after_restart:
            LOGGER.info("Skipping permission request once after restart-launch-agent")
        else:
            # Trigger prompts from daemon context so launchd-triggered runs can obtain trust.
            if not report.accessibility:
                request_accessibility_permission()
            if not report.input_monitoring:
                request_input_monitoring_permission()
            if not report.microphone:
                request_microphone_permission()
            report = check_all_permissions()
    if not report.all_granted:
        LOGGER.warning(format_permission_guidance(report))

    llm_enabled = _llm_enabled_for_this_run(config)
    post_processor = _build_runtime_post_processor(
        config,
        base_processor=correction_result.rules,
        llm_enabled_override=llm_enabled,
    )
    output_cfg = getattr(config, "output", None)
    output_mode = str(getattr(output_cfg, "mode", "clipboard_paste")).strip().lower()
    output_supports_streaming = _streaming_supported_by_output_mode(config)
    enable_streaming = (not llm_enabled) and output_supports_streaming
    if llm_enabled:
        LOGGER.info("LLM correction is enabled; STT streaming output is disabled for this run")
    elif not output_supports_streaming:
        LOGGER.info("Output mode disables STT streaming for this run (%s)", output_mode)
    daemon = MoonshineFlowDaemon(
        config,
        post_processor=post_processor,
        enable_streaming=enable_streaming,
    )
    try:
        backend = daemon.transcriber.preflight_model()
        LOGGER.info(_green("Model preflight OK (%s)", stderr=True), backend)
    except Exception as exc:
        LOGGER.error("Model preflight failed: %s", exc)
        if "incompatible architecture" in str(exc).lower():
            LOGGER.error(
                "Detected architecture mismatch between Python runtime and Moonshine binaries. "
                "Run `moonshine-flow doctor` and ensure arm64 python@3.11 + uv are available on "
                "Apple Silicon (typically under /opt/homebrew)."
            )
        return 4

    try:
        daemon.run_forever()
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        daemon.stop()
    return 0


def cmd_check_permissions(args: argparse.Namespace) -> int:
    report = request_all_permissions() if args.request else check_all_permissions()
    print("Microphone:", "OK" if report.microphone else "MISSING")
    print("Accessibility:", "OK" if report.accessibility else "MISSING")
    print("Input Monitoring:", "OK" if report.input_monitoring else "MISSING")

    if report.all_granted:
        print(_green("\nAll required permissions are granted."))
        return 0

    print("\n" + format_permission_guidance(report))
    if not args.request:
        print(
            "\nTip: run `uv run moonshine-flow check-permissions --request` "
            "once to trigger prompts."
        )
    return 2


def cmd_install_launch_agent(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args.config)
    config = load_config(config_path)
    current_launchd_payload = read_launch_agent_plist()
    current_launchd_llm_override = _launchd_llm_enabled_override_from_payload(current_launchd_payload)
    desired_launchd_llm_override = _resolve_launchd_llm_enabled_override_for_command(
        current_override=current_launchd_llm_override,
        preflight_func=lambda: _preflight_llm_for_launchd(config),
    )
    if getattr(args, "install_app_bundle", True):
        bundle_path = install_app_bundle_from_env()
        if bundle_path is not None:
            print(_green(f"Installed app bundle: {bundle_path}"))
            tcc_reset_ok = reset_app_bundle_tcc(APP_BUNDLE_IDENTIFIER)
            if tcc_reset_ok:
                print(
                    _green(
                        "TCC permissions reset for MoonshineFlow.app. "
                        "Re-grant Accessibility and Input Monitoring in "
                        "System Settings -> Privacy & Security."
                    )
                )
            else:
                print(
                    _yellow(
                        "Warning: could not reset TCC permissions automatically. "
                        "If Accessibility or Input Monitoring appear stale, "
                        "remove and re-add MoonshineFlow manually in System Settings.",
                        stderr=True,
                    ),
                    file=sys.stderr,
                )

    permission_check_command = [*resolve_launch_agent_program_prefix(), "check-permissions"]
    if args.request_permissions:
        permission_check_command.append("--request")
    print(f"Launchd permission check command: {_format_command(permission_check_command)}")

    probe = check_permissions_in_launchd_context(command=permission_check_command)
    if probe.report is None:
        print(
            "Could not verify launchd permission state before installing launch agent.",
            file=sys.stderr,
        )
        if probe.error:
            print(probe.error, file=sys.stderr)
        if probe.stdout:
            print(f"Launchd check stdout:\n{probe.stdout}", file=sys.stderr)
        if probe.stderr:
            print(f"Launchd check stderr:\n{probe.stderr}", file=sys.stderr)
        if not args.allow_missing_permissions:
            print(
                "\nLaunch agent installation was aborted because launchd permission state "
                "could not be verified.",
                file=sys.stderr,
            )
            print(
                "Retry after fixing permission checks, or run with "
                "`--allow-missing-permissions` to install anyway.",
                file=sys.stderr,
            )
            return 2

        print(
            _yellow(
                "Warning: continuing with unverified permissions because "
                "`--allow-missing-permissions` was specified.",
                stderr=True,
            ),
            file=sys.stderr,
        )
    elif not probe.report.all_granted:
        guidance = _format_launchd_permission_guidance(
            probe.report,
            target_executable=permission_check_command[0] if permission_check_command else None,
        )
        if not args.allow_missing_permissions:
            print(guidance, file=sys.stderr)
            print(
                "\nLaunch agent installation was aborted because missing launchd permissions can "
                "prevent hotkey detection and paste output.",
                file=sys.stderr,
            )
            print(
                "Retry after granting permissions, or run with "
                "`--allow-missing-permissions` to install anyway.",
                file=sys.stderr,
            )
            return 2

        print(
            _yellow(
                "Warning: continuing with missing permissions because "
                "`--allow-missing-permissions` was specified.",
                stderr=True,
            ),
            file=sys.stderr,
        )
        print(guidance, file=sys.stderr)

    if desired_launchd_llm_override is None:
        plist_path = install_launch_agent(config_path)
    else:
        plist_path = install_launch_agent(
            config_path,
            llm_enabled_override=desired_launchd_llm_override,
        )
    print(_green(f"Installed launch agent: {plist_path}"))
    launchd_override_text = _format_optional_bool(desired_launchd_llm_override)
    if desired_launchd_llm_override is True:
        print(_green(f"Launchd LLM enabled override: {launchd_override_text}"))
    else:
        print(f"Launchd LLM enabled override: {launchd_override_text}")
    return 0


def cmd_uninstall_launch_agent(args: argparse.Namespace) -> int:
    del args
    removed = uninstall_launch_agent()
    if removed:
        print(_green(f"Removed launch agent: {launch_agent_path()}"))
    else:
        print("Launch agent is not installed.")
    return 0


def cmd_restart_launch_agent(args: argparse.Namespace) -> int:
    del args
    if not launch_agent_path().exists():
        print("Launch agent is not installed.")
        return 2

    current_launchd_payload = read_launch_agent_plist()
    current_launchd_llm_override = _launchd_llm_enabled_override_from_payload(current_launchd_payload)

    def _preflight_for_restart() -> tuple[bool, str | None]:
        config = load_config(_resolve_config_path(None))
        return _preflight_llm_for_launchd(config)

    desired_launchd_llm_override = _resolve_launchd_llm_enabled_override_for_command(
        current_override=current_launchd_llm_override,
        preflight_func=_preflight_for_restart,
    )

    try:
        if desired_launchd_llm_override is None:
            restarted = restart_launch_agent()
        else:
            restarted = restart_launch_agent(llm_enabled_override=desired_launchd_llm_override)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not restarted:
        print("Launch agent is not installed.")
        return 2
    print(_green(f"Restarted launch agent: {launch_agent_path()}"))
    launchd_override_text = _format_optional_bool(desired_launchd_llm_override)
    if desired_launchd_llm_override is True:
        print(_green(f"Launchd LLM enabled override: {launchd_override_text}"))
    else:
        print(f"Launchd LLM enabled override: {launchd_override_text}")
    return 0


def cmd_install_app_bundle(args: argparse.Namespace) -> int:
    app_path = Path(args.path).expanduser() if args.path else default_app_bundle_path()
    installed = install_app_bundle_from_env(app_path)
    if installed is None:
        print(
            "App bundle install is unavailable in this context. "
            "Run this via Homebrew-installed `mflow`.",
            file=sys.stderr,
        )
        return 2
    print(_green(f"Installed app bundle: {installed}"))
    return 0


def _derive_launchd_permission_check_command(
    launchd_payload: dict[str, object] | None,
) -> list[str]:
    """Resolve a check-permissions command that matches launchd runtime context."""
    default_command = ["mflow", "check-permissions"]
    if not isinstance(launchd_payload, dict):
        return default_command

    program_args = launchd_payload.get("ProgramArguments")
    if not isinstance(program_args, list) or not program_args:
        return default_command

    resolved_parts = [str(part) for part in program_args]
    if "run" in resolved_parts:
        run_index = resolved_parts.index("run")
        prefix = resolved_parts[:run_index]
        if prefix:
            return [*prefix, "check-permissions"]

    if resolved_parts:
        return [resolved_parts[0], "check-permissions"]
    return default_command


def _derive_launchd_permission_target(launchd_payload: dict[str, object] | None) -> str | None:
    """Resolve permission target path used by launchd daemon process."""
    if not isinstance(launchd_payload, dict):
        return None
    program_args = launchd_payload.get("ProgramArguments")
    if not isinstance(program_args, list) or not program_args:
        return None
    target = str(program_args[0]).strip()
    return target or None


def _latest_launchd_runtime_warning(err_log_path: Path) -> str | None:
    """Return latest launchd runtime warning text from daemon stderr log when present."""
    result = _latest_launchd_runtime_warning_with_timestamp(err_log_path)
    if result is None:
        return None
    return result[0]


def _latest_launchd_runtime_warning_with_timestamp(
    err_log_path: Path,
) -> tuple[str, str | None] | None:
    """Return (warning_message, detected_timestamp) from daemon stderr log, or None.

    *detected_timestamp* is the raw log line prefix of the first matching line,
    or None when no timestamp could be extracted.
    """
    if not err_log_path.exists():
        return None
    try:
        lines = err_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if not lines:
        return None

    latest_start = 0
    for idx, line in enumerate(lines):
        if "Moonshine Flow daemon starting" in line:
            latest_start = idx

    recent = lines[latest_start:]

    def _extract_timestamp(line: str) -> str | None:
        """Return leading timestamp portion of a log line if recognisable."""
        # Common format: "2026-02-27 10:00:00,100 ..."
        parts = line.split(" ")
        if len(parts) >= 2 and len(parts[0]) == 10 and parts[0].count("-") == 2:
            return f"{parts[0]} {parts[1]}"
        return None

    for line in recent:
        if "This process is not trusted!" in line:
            return "pynput listener is not trusted in daemon runtime context", _extract_timestamp(
                line
            )
    for line in recent:
        if "Missing macOS permissions detected:" in line:
            return "daemon runtime detected missing macOS permissions", _extract_timestamp(line)
    return None


def _print_codesign_info(target_path: str) -> None:
    """Print codesign metadata for the app bundle derived from *target_path*."""
    # Determine the bundle path: if target is already an .app, use it directly;
    # otherwise try the default MoonshineFlow.app and resolve the executable mtime.
    candidate_bundle = default_app_bundle_path()
    exec_path = app_bundle_executable_path(candidate_bundle)

    if exec_path.exists():
        try:
            mtime = exec_path.stat().st_mtime
            mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"App bundle executable mtime: {mtime_str} ({exec_path})")
        except OSError:
            pass

    codesign_info = get_app_bundle_codesign_info(candidate_bundle)
    if codesign_info:
        for key in ("CDHash", "Identifier", "TeamIdentifier", "Signature Type"):
            if key in codesign_info:
                print(f"App bundle {key}: {codesign_info[key]}")
    else:
        print(f"App bundle codesign info: unavailable ({candidate_bundle})")


def cmd_doctor(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args.config)
    config = load_config(config_path)
    correction_result, correction_error = _load_corrections_with_diagnostics(
        config,
        config_path=config_path,
    )

    os_machine = os.uname().machine if hasattr(os, "uname") else "unknown"
    py_machine = platform.machine()

    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS machine: {os_machine}")
    print(f"Python machine: {py_machine}")
    print(f"Config: {config_path}")
    if correction_error is not None:
        print(f"Correction dictionary: ERROR ({correction_error})")
    else:
        assert correction_result is not None
        print(f"Correction dictionary path: {correction_result.path}")
        if not correction_result.loaded:
            print("Correction dictionary: DISABLED (file not found)")
        elif correction_result.warnings:
            print(
                "Correction dictionary: WARN "
                f"(exact={correction_result.rules.exact_count}, "
                f"regex={correction_result.rules.regex_count}, "
                f"disabled_regex={correction_result.disabled_regex_count})"
            )
            for warning in correction_result.warnings:
                print(_yellow(f"Correction dictionary warning: {warning.message}"))
        else:
            print(
                _green(
                    "Correction dictionary: OK "
                    f"(exact={correction_result.rules.exact_count}, "
                    f"regex={correction_result.rules.regex_count})"
                )
            )
    llm_cfg = getattr(getattr(config, "text", None), "llm_correction", None)
    if llm_cfg is None:
        print("LLM correction: DISABLED")
    else:
        mode = str(getattr(llm_cfg, "mode", "never"))
        provider = str(getattr(llm_cfg, "provider", "ollama"))
        base_url = str(getattr(llm_cfg, "base_url", ""))
        model = str(getattr(llm_cfg, "model", ""))
        timeout_seconds = float(getattr(llm_cfg, "timeout_seconds", 5.0))
        max_input_chars = int(getattr(llm_cfg, "max_input_chars", 500))
        enabled_tools = bool(getattr(llm_cfg, "enabled_tools", False))
        api_key = _normalize_optional_secret(getattr(llm_cfg, "api_key", None))
        print(f"LLM correction mode: {mode}")
        print(
            "LLM correction config: "
            f"provider={provider} base_url={base_url} model={model} "
            f"timeout_seconds={timeout_seconds:.2f} max_input_chars={max_input_chars} "
            f"enabled_tools={enabled_tools} api_key={_format_secret_state(api_key)}"
        )
    print(f"Permission target (recommended): {recommended_permission_target()}")
    launchd_payload = read_launch_agent_plist()
    out_log_path, err_log_path = launch_agent_log_paths()
    if launchd_payload is None:
        print(f"LaunchAgent plist: MISSING ({launch_agent_path()})")
        print("Install LaunchAgent: mflow install-launch-agent")
        launchd_permission_target = None
        print("Launchd LLM enabled override: <unset>")
    else:
        print(f"LaunchAgent plist: FOUND ({launch_agent_path()})")
        print(f"LaunchAgent label: {launchd_payload.get('Label', 'UNKNOWN')}")
        program_args = launchd_payload.get("ProgramArguments")
        if isinstance(program_args, list) and program_args:
            print(f"LaunchAgent program: {' '.join(str(part) for part in program_args)}")
        else:
            print("LaunchAgent program: UNKNOWN")
        launchd_llm_override = _launchd_llm_enabled_override_from_payload(launchd_payload)
        launchd_llm_override_text = _format_optional_bool(launchd_llm_override)
        if launchd_llm_override is True:
            print(_green(f"Launchd LLM enabled override: {launchd_llm_override_text}"))
        else:
            print(f"Launchd LLM enabled override: {launchd_llm_override_text}")
        launchd_permission_target = _derive_launchd_permission_target(launchd_payload)
        if launchd_permission_target:
            print(f"Launchd permission target (recommended): {launchd_permission_target}")
    print(f"Daemon stdout log: {out_log_path}")
    print(f"Daemon stderr log: {err_log_path}")
    runtime_warning_result = _latest_launchd_runtime_warning_with_timestamp(err_log_path)
    runtime_warning: str | None = None
    runtime_warning_timestamp: str | None = None
    if runtime_warning_result is not None:
        runtime_warning, runtime_warning_timestamp = runtime_warning_result
        timestamp_suffix = f" at {runtime_warning_timestamp}" if runtime_warning_timestamp else ""
        print(_yellow(f"Launchd runtime status: WARNING ({runtime_warning}{timestamp_suffix})"))

    for pkg in (
        "moonshine_voice",
        "sounddevice",
        "pynput",
        "websockets",
        "vllm",
        "mlx_whisper",
        "transformers",
        "mistral_common",
    ):
        print(f"Package {pkg}:", "FOUND" if find_spec(pkg) else "MISSING")

    requires_moonshine_backend = False
    try:
        requires_moonshine_backend = _is_moonshine_stt_model(config)
    except ValueError as exc:
        print(f"stt.model: ERROR ({exc})")

    if requires_moonshine_backend and not _has_moonshine_backend():
        print(_backend_guidance())

    try:
        stt_backend = create_stt_backend(config)
        print("Transcriber:", stt_backend.backend_summary())
    except Exception as exc:
        print(f"Transcriber: ERROR ({exc})")

    report = check_all_permissions()
    terminal_status = "OK" if report.all_granted else "INCOMPLETE"
    if report.all_granted:
        print(_green(f"Terminal permissions: {terminal_status}"))
    else:
        print(f"Terminal permissions: {terminal_status}")

    launchd_report = None
    probe_error = False
    should_check_launchd = bool(getattr(args, "launchd_check", False))
    if should_check_launchd:
        launchd_command = _derive_launchd_permission_check_command(launchd_payload)
        probe = check_permissions_in_launchd_context(command=launchd_command)
        if probe.command:
            print(f"Launchd check command: {_format_command(probe.command)}")
        else:
            print(f"Launchd check command: {_format_command(launchd_command)}")

        # Show app bundle codesign info alongside the launchd permission check
        if launchd_permission_target:
            _print_codesign_info(launchd_permission_target)

        if probe.report is not None:
            launchd_report = probe.report
            launchd_status = "OK" if launchd_report.all_granted else "INCOMPLETE"
            if launchd_report.all_granted:
                print(_green(f"Launchd permissions: {launchd_status}"))
            else:
                print(f"Launchd permissions: {launchd_status}")
            if not launchd_report.all_granted:
                print(f"Launchd missing permissions: {', '.join(launchd_report.missing)}")
            if set(launchd_report.missing) != set(report.missing):
                print(
                    "Permission mismatch detected between terminal and launchd contexts. "
                    "Grant permissions for the launchd target shown above."
                )
            if runtime_warning:
                print(
                    "Launchd runtime log indicates trust failure despite check output. "
                    "Restart the launch agent after granting permissions."
                )
        else:
            probe_error = True
            print("Launchd permissions: ERROR")
            if probe.error:
                print(f"Launchd check error: {probe.error}")
            if probe.stdout:
                print(f"Launchd check stdout: {probe.stdout}")
            if probe.stderr:
                print(f"Launchd check stderr: {probe.stderr}")

    # Determine overall permissions status:
    # - INCOMPLETE: any definitive permission missing
    # - WARN: all checks pass but runtime log shows trust failure (TCC instability)
    # - OK: everything is granted and no runtime warning
    effective_incomplete = not report.all_granted
    if launchd_report is not None:
        effective_incomplete = effective_incomplete or not launchd_report.all_granted
    if probe_error:
        effective_incomplete = True

    # WARN = launchd check reports OK, but runtime log shows "not trusted"
    # This indicates TCC registered the permission but the signing identity may have drifted.
    effective_warn = (
        not effective_incomplete
        and runtime_warning is not None
        and launchd_report is not None
        and launchd_report.all_granted
    )
    # When launchd check was not run, runtime_warning alone causes INCOMPLETE (existing behaviour)
    if runtime_warning and not should_check_launchd:
        effective_incomplete = True

    if effective_incomplete:
        print("Permissions: INCOMPLETE")
    elif effective_warn:
        print(_yellow("Permissions: WARN (launchd check OK but runtime not trusted)"))
    else:
        print(_green("Permissions: OK"))

    if not report.all_granted:
        print(format_permission_guidance(report))
    elif launchd_report is not None and not launchd_report.all_granted:
        if launchd_permission_target:
            print(
                "Grant permissions for this launchd target and restart the launch agent: "
                f"{launchd_permission_target}"
            )
        else:
            print(
                "Grant permissions for the launchd target shown above "
                "and restart the launch agent."
            )
    elif probe_error:
        print("Could not verify launchd permission state from launchctl output.")
    elif effective_warn:
        target = launchd_permission_target or str(recommended_permission_target())
        print(
            "Launchd check reports OK but runtime log shows trust failure. "
            "This typically means the app bundle was re-signed and TCC lost the binding. "
            "Re-grant Accessibility/Input Monitoring for this target and restart: "
            f"{target}"
        )
    elif runtime_warning:
        target = launchd_permission_target or str(recommended_permission_target())
        print(
            "Launchd runtime log indicates trust failure. "
            "Re-grant Accessibility/Input Monitoring for this target and restart: "
            f"{target}"
        )

    if platform.system() == "Darwin" and os_machine == "arm64" and py_machine != "arm64":
        print(
            _yellow(
                "\nWarning: Apple Silicon macOS is running an x86_64 Python environment "
                "(likely Rosetta). Moonshine packages may be unavailable. "
                "Use an arm64 Python interpreter."
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="moonshine-flow")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {_resolve_app_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run background daemon")
    run_parser.add_argument("--config", default=None, help="Path to config TOML")
    run_parser.set_defaults(func=cmd_run)

    init_parser = subparsers.add_parser("init", help="Interactively edit config")
    init_parser.add_argument("--config", default=None, help="Path to config TOML")
    init_parser.set_defaults(func=cmd_init)

    list_parser = subparsers.add_parser(
        "list",
        help="List available resources",
    )
    list_parser.set_defaults(func=cmd_list)
    list_subparsers = list_parser.add_subparsers(dest="list_target")
    list_devices_parser = list_subparsers.add_parser(
        "devices",
        help="List audio input devices and save selected device to config",
    )
    list_devices_parser.add_argument("--config", default=None, help="Path to config TOML")
    list_devices_parser.set_defaults(func=cmd_list_devices)

    list_model_parser = list_subparsers.add_parser(
        "model",
        help="List STT model presets and save selected model to config",
    )
    list_model_parser.add_argument("--config", default=None, help="Path to config TOML")
    list_model_parser.set_defaults(func=cmd_list_model)

    list_typing_parser = list_subparsers.add_parser(
        "typing",
        help="List output typing modes and save selected mode to config",
    )
    list_typing_parser.add_argument("--config", default=None, help="Path to config TOML")
    list_typing_parser.set_defaults(func=cmd_list_typing)

    list_ollama_parser = list_subparsers.add_parser(
        "ollama",
        help="List downloaded Ollama models",
    )
    list_ollama_parser.add_argument("--config", default=None, help="Path to config TOML")
    list_ollama_parser.set_defaults(func=cmd_list_ollama)

    list_lmstudio_parser = list_subparsers.add_parser(
        "lmstudio",
        help="List loaded LM Studio models",
    )
    list_lmstudio_parser.add_argument("--config", default=None, help="Path to config TOML")
    list_lmstudio_parser.set_defaults(func=cmd_list_lmstudio)

    check_parser = subparsers.add_parser("check-permissions", help="Check macOS permissions")
    check_parser.add_argument(
        "--request",
        action="store_true",
        help="Request missing macOS permissions (shows system prompts when possible)",
    )
    check_parser.set_defaults(func=cmd_check_permissions)

    doctor_parser = subparsers.add_parser("doctor", help="Show runtime diagnostics")
    doctor_parser.add_argument("--config", default=None, help="Path to config TOML")
    doctor_parser.add_argument(
        "--launchd-check",
        action="store_true",
        help="Compare permission status in launchd context via launchctl asuser",
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    install_parser = subparsers.add_parser("install-launch-agent", help="Install launchd agent")
    install_parser.add_argument("--config", default=None, help="Path to config TOML")
    install_parser.add_argument(
        "--request-permissions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request missing macOS permissions before installing launchd agent",
    )
    install_parser.add_argument(
        "--allow-missing-permissions",
        action="store_true",
        help="Install launchd agent even when required macOS permissions are missing",
    )
    install_parser.add_argument(
        "--verbose-bootstrap",
        action="store_true",
        help="Show detailed runtime bootstrap logs when recovery runs",
    )
    install_parser.add_argument(
        "--install-app-bundle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create/update ~/Applications/MoonshineFlow.app before installing launchd agent",
    )
    install_parser.set_defaults(func=cmd_install_launch_agent)

    uninstall_parser = subparsers.add_parser("uninstall-launch-agent", help="Remove launchd agent")
    uninstall_parser.set_defaults(func=cmd_uninstall_launch_agent)

    restart_parser = subparsers.add_parser("restart-launch-agent", help="Restart launchd agent")
    restart_parser.set_defaults(func=cmd_restart_launch_agent)

    app_bundle_parser = subparsers.add_parser(
        "install-app-bundle",
        help="Create or update ~/Applications/MoonshineFlow.app from current runtime",
    )
    app_bundle_parser.add_argument(
        "--path",
        default=None,
        help="Custom .app destination path (default: ~/Applications/MoonshineFlow.app)",
    )
    app_bundle_parser.set_defaults(func=cmd_install_app_bundle)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
