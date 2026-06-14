"""PyObjC-free Web UI action dispatcher."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ptarmigan_flow import app_relaunch, login_item, onboarding_strings
from ptarmigan_flow import onboarding_flow as onboarding_flow_module
from ptarmigan_flow.app_settings_model import (
    SUPPORTED_HOTKEYS,
    SUPPORTED_LANGUAGES,
    SUPPORTED_OUTPUT_MODES,
    AppSettingsModel,
)
from ptarmigan_flow.config import (
    LLMCorrectionMode,
    OutputMode,
    default_config_path,
    ensure_config_exists,
    load_config,
    write_config,
)
from ptarmigan_flow.corrections_editor_model import (
    CorrectionsEditorModel,
    CorrectionValidationError,
)
from ptarmigan_flow.onboarding_flow import OnboardingFlow
from ptarmigan_flow.permissions import (
    PermissionReport,
    check_all_permissions,
    request_accessibility_permission,
    request_input_monitoring_permission,
    request_microphone_permission,
)
from ptarmigan_flow.stt import availability, model_download
from ptarmigan_flow.stt.model_catalog import CatalogEntry
from ptarmigan_flow.transcription_corrections import resolve_dictionary_path

PermissionKind = str

_SYSTEM_SETTINGS_URLS = {
    "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    "accessibility": (
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    ),
    "input_monitoring": (
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
    ),
}


def _default_dictionary_path() -> Path:
    path, _explicit = resolve_dictionary_path(None)
    return path


def _default_request_permission(kind: PermissionKind) -> None:
    if kind == "microphone":
        request_microphone_permission()
        return
    if kind == "accessibility":
        request_accessibility_permission()
        return
    if kind == "input_monitoring":
        request_input_monitoring_permission()
        return
    raise ValueError(f"Unsupported permission kind: {kind}")


def _default_open_system_settings(kind: PermissionKind) -> None:
    url = _SYSTEM_SETTINGS_URLS.get(kind)
    if url is None:
        raise ValueError(f"Unsupported permission kind: {kind}")
    if sys.platform == "darwin":
        subprocess.run(["open", url], check=False)


def _default_open_config_file() -> None:
    path = default_config_path()
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)


def _noop() -> None:
    return


@dataclass(slots=True)
class BridgeDependencies:
    """External effects used by the Web UI bridge."""

    config_path: Callable[[], Path] = default_config_path
    check_permissions: Callable[[], PermissionReport] = check_all_permissions
    available_model_entries: Callable[[], list[CatalogEntry]] = availability.available_model_entries
    is_model_downloaded: Callable[[str], bool] = model_download.is_model_downloaded
    resolve_dictionary_path: Callable[[], Path] = _default_dictionary_path
    login_is_enabled: Callable[[], bool] = login_item.is_enabled
    login_register: Callable[[], bool] = login_item.register
    login_unregister: Callable[[], bool] = login_item.unregister
    request_permission: Callable[[PermissionKind], None] = _default_request_permission
    open_system_settings: Callable[[PermissionKind], None] = _default_open_system_settings
    open_config_file: Callable[[], None] = _default_open_config_file
    start_dictation: Callable[[], object | None] = _noop
    stop_dictation: Callable[[], None] = _noop
    daemon_is_running: Callable[[], bool] = lambda: False
    restart_app: Callable[[], bool] = app_relaunch.relaunch_app
    mark_language_selected: Callable[[], None] = onboarding_flow_module.mark_language_selected


@dataclass(slots=True)
class WebBridgeDispatcher:
    """Dispatch JS bridge actions to existing app logic."""

    deps: BridgeDependencies = field(default_factory=BridgeDependencies)
    onboarding_flow: OnboardingFlow = field(default_factory=OnboardingFlow)

    def set_onboarding_flow(self, onboarding_flow: OnboardingFlow) -> None:
        self.onboarding_flow = onboarding_flow

    def handle_action(self, action: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        body = payload or {}
        handlers = {
            "getState": self._get_state,
            "chooseLanguage": self._choose_language,
            "requestPermission": self._request_permission,
            "openSystemSettings": self._open_system_settings,
            "openConfigFile": self._open_config_file,
            "startDictation": self._start_dictation,
            "stopDictation": self._stop_dictation,
            "saveSettings": self._save_settings,
            "listModels": self._list_models,
            "loadDictionary": self._load_dictionary,
            "saveDictionary": self._save_dictionary,
            "toggleLogin": self._toggle_login,
            "restartApp": self._restart_app,
        }
        handler = handlers.get(action)
        if handler is None:
            raise ValueError(f"Unsupported bridge action: {action}")
        return handler(body)

    def _config_path(self) -> Path:
        return self.deps.config_path()

    def _load_config(self):
        config_path = self._config_path()
        ensure_config_exists(config_path)
        return load_config(config_path)

    def _write_config(self, config) -> None:
        write_config(self._config_path(), config)

    def _language(self) -> str:
        try:
            return str(self._load_config().language)
        except Exception:
            return "en"

    def _get_state(self, _payload: dict[str, Any]) -> dict[str, Any]:
        config = self._load_config()
        language = str(config.language)
        return {
            "language": language,
            "onboarding_step": self.onboarding_flow.current_step,
            "permissions": _permission_payload(self.deps.check_permissions()),
            "model": str(config.stt.model),
            "settings": _settings_payload(config),
            "models": self._model_entries_payload(),
            "dictionary": self._load_dictionary({}),
            "login_enabled": bool(self.deps.login_is_enabled()),
            "daemon_running": bool(self.deps.daemon_is_running()),
            "strings": onboarding_strings.strings_for(language),
        }

    def _choose_language(self, payload: dict[str, Any]) -> dict[str, Any]:
        code = str(payload.get("code", "")).strip().lower()
        config = self._load_config()
        config.language = code
        self._write_config(config)
        self.onboarding_flow.choose_language(code)
        self.deps.mark_language_selected()
        return {
            "language": code,
            "onboarding_step": self.onboarding_flow.current_step,
            "strings": onboarding_strings.strings_for(code),
        }

    def _request_permission(self, payload: dict[str, Any]) -> dict[str, str]:
        kind = _permission_kind(payload)
        self.deps.request_permission(kind)
        return {"requested": kind}

    def _open_system_settings(self, payload: dict[str, Any]) -> dict[str, str]:
        kind = _permission_kind(payload)
        self.deps.open_system_settings(kind)
        return {"opened": kind}

    def _open_config_file(self, _payload: dict[str, Any]) -> dict[str, bool]:
        self.deps.open_config_file()
        return {"opened": True}

    def _start_dictation(self, _payload: dict[str, Any]) -> dict[str, Any]:
        start_result = self.deps.start_dictation()
        result: dict[str, Any] = {"daemon_running": bool(self.deps.daemon_is_running())}
        if isinstance(start_result, dict):
            result.update(start_result)
        elif start_result is not None:
            result["error_message"] = str(start_result)
        return result

    def _stop_dictation(self, _payload: dict[str, Any]) -> dict[str, bool]:
        self.deps.stop_dictation()
        return {"daemon_running": bool(self.deps.daemon_is_running())}

    def _save_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = str(payload.get("model", "")).strip()
        language = str(payload.get("language", "")).strip().lower()
        hotkey = str(payload.get("hotkey", "")).strip()
        output_mode = str(payload.get("output_mode", "")).strip()
        llm_payload = _payload_dict(payload.get("llm_correction"))

        settings = AppSettingsModel(
            model=model,
            language=language,
            hotkey=hotkey,
            output_mode=output_mode,
        )
        errors = self._validate_settings(settings)
        errors.extend(_validate_llm_payload(llm_payload))
        if errors:
            return {"saved": False, "errors": errors}

        config = self._load_config()
        config.stt.model = settings.model
        config.language = settings.language
        config.hotkey.key = settings.hotkey
        config.output.mode = OutputMode(settings.output_mode)
        config.text.llm_correction.mode = LLMCorrectionMode(str(llm_payload["mode"]).strip())
        config.text.llm_correction.provider = str(llm_payload["provider"]).strip()
        config.text.llm_correction.model = str(llm_payload["model"]).strip()
        config.text.llm_correction.base_url = str(llm_payload["base_url"]).strip()
        self._write_config(config)
        return {"saved": True, "settings": _settings_payload(config)}

    def _validate_settings(self, settings: AppSettingsModel) -> list[str]:
        available_models = {entry.token for entry in self.deps.available_model_entries()}
        errors: list[str] = []
        if settings.model not in available_models:
            errors.append("model")
        if settings.language not in SUPPORTED_LANGUAGES:
            errors.append("language")
        if settings.hotkey not in SUPPORTED_HOTKEYS:
            errors.append("hotkey")
        if settings.output_mode not in SUPPORTED_OUTPUT_MODES:
            errors.append("output_mode")
        return errors

    def _list_models(self, _payload: dict[str, Any]) -> dict[str, Any]:
        return {"models": self._model_entries_payload()}

    def _model_entries_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "token": entry.token,
                "backend": entry.backend,
                "label": entry.label,
                "description": entry.description,
                "downloaded": bool(self.deps.is_model_downloaded(entry.token)),
            }
            for entry in self.deps.available_model_entries()
        ]

    def _load_dictionary(self, _payload: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
        model = CorrectionsEditorModel.load(self.deps.resolve_dictionary_path())
        return {"exact": model.exact, "regex": model.regex}

    def _save_dictionary(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = CorrectionsEditorModel(
            exact=_string_list_table(payload.get("exact")),
            regex=_string_list_table(payload.get("regex")),
        )
        errors = model.validate()
        if errors:
            return {
                "saved": False,
                "errors": [_validation_error_payload(error) for error in errors],
            }
        model.save(self.deps.resolve_dictionary_path())
        return {"saved": True}

    def _toggle_login(self, payload: dict[str, Any]) -> dict[str, bool]:
        current = bool(self.deps.login_is_enabled())
        requested = payload.get("enabled")
        target = not current if requested is None else bool(requested)
        if target and not current:
            self.deps.login_register()
        elif not target and current:
            self.deps.login_unregister()
        return {"login_enabled": bool(self.deps.login_is_enabled())}

    def _restart_app(self, _payload: dict[str, Any]) -> dict[str, bool]:
        return {"restarted": bool(self.deps.restart_app())}


_DEFAULT_DISPATCHER = WebBridgeDispatcher()


def handle_action(action: str, payload: dict) -> dict[str, Any]:
    """Dispatch one bridge action through the default dispatcher."""
    return _DEFAULT_DISPATCHER.handle_action(action, payload)


def _permission_payload(report: PermissionReport) -> dict[str, bool]:
    return {
        "microphone": bool(report.microphone),
        "accessibility": bool(report.accessibility),
        "input_monitoring": bool(report.input_monitoring),
    }


def _settings_payload(config) -> dict[str, Any]:
    llm = config.text.llm_correction
    return {
        "model": str(config.stt.model),
        "language": str(config.language),
        "hotkey": str(config.hotkey.key),
        "output_mode": str(getattr(config.output.mode, "value", config.output.mode)),
        "llm_correction": {
            "mode": str(getattr(llm.mode, "value", llm.mode)),
            "provider": str(llm.provider),
            "model": str(llm.model),
            "base_url": str(llm.base_url),
        },
    }


def _permission_kind(payload: dict[str, Any]) -> str:
    kind = str(payload.get("kind", "")).strip()
    if kind not in _SYSTEM_SETTINGS_URLS:
        raise ValueError(f"Unsupported permission kind: {kind}")
    return kind


def _payload_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _validate_llm_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    mode = str(payload.get("mode", "")).strip()
    if mode not in {item.value for item in LLMCorrectionMode}:
        errors.append("llm_correction.mode")
    for field_name in ("provider", "model", "base_url"):
        if not str(payload.get(field_name, "")).strip():
            errors.append(f"llm_correction.{field_name}")
    return errors


def _string_list_table(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    table: dict[str, list[str]] = {}
    for key, raw_values in value.items():
        if isinstance(raw_values, list):
            table[str(key)] = [str(item) for item in raw_values]
        else:
            table[str(key)] = [str(raw_values)]
    return table


def _validation_error_payload(error: CorrectionValidationError) -> dict[str, Any]:
    return {
        "section": error.section,
        "key": error.key,
        "index": error.index,
        "pattern": error.pattern,
        "message": error.message,
    }
