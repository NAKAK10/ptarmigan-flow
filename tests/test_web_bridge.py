from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from ptarmigan_flow.config import (
    AppConfig,
    LLMCorrectionMode,
    OutputMode,
    load_config,
    write_config,
)
from ptarmigan_flow.corrections_editor_model import CorrectionsEditorModel
from ptarmigan_flow.onboarding_flow import OnboardingFlow
from ptarmigan_flow.permissions import PermissionReport
from ptarmigan_flow.stt.model_catalog import CatalogEntry
from ptarmigan_flow.web_bridge import BridgeDependencies, WebBridgeDispatcher, handle_action


def test_ns_to_py_converts_foundation_collections_recursively() -> None:
    Foundation = pytest.importorskip("Foundation")
    from ptarmigan_flow import web_ui

    assert hasattr(web_ui, "_ns_to_py")
    body = Foundation.NSDictionary.dictionaryWithDictionary_(
        {
            "id": "message-1",
            "payload": Foundation.NSDictionary.dictionaryWithDictionary_(
                {
                    "items": Foundation.NSArray.arrayWithArray_(
                        [
                            Foundation.NSDictionary.dictionaryWithDictionary_({"name": "tiny"}),
                            "plain",
                        ]
                    )
                }
            ),
        }
    )

    converted = web_ui._ns_to_py(body)

    assert converted == {
        "id": "message-1",
        "payload": {"items": [{"name": "tiny"}, "plain"]},
    }
    assert type(converted) is dict
    assert type(converted["payload"]) is dict
    assert type(converted["payload"]["items"]) is list
    assert type(converted["payload"]["items"][0]) is dict


@dataclass(frozen=True)
class _DaemonState:
    running: bool = False


def _entry(token: str = "moonshine:tiny") -> CatalogEntry:
    backend, _, _model_id = token.partition(":")
    return CatalogEntry(
        token=token,
        backend=backend,
        label=f"Label {token}",
        description=f"Description {token}",
        verified=True,
        source="preset",
    )


def _write_config(path: Path, *, language: str = "ja", model: str = "moonshine:tiny") -> None:
    config = AppConfig()
    config.language = language
    config.stt.model = model
    config.hotkey.key = "left_cmd"
    config.output.mode = OutputMode.CLIPBOARD_PASTE
    config.text.llm_correction.mode = LLMCorrectionMode.ALWAYS
    config.text.llm_correction.provider = "ollama"
    config.text.llm_correction.model = "qwen-test"
    config.text.llm_correction.base_url = "http://localhost:11434"
    write_config(path, config)


def _dispatcher(
    tmp_path: Path,
    *,
    permissions: PermissionReport | None = None,
    daemon: _DaemonState = _DaemonState(),
    models: list[CatalogEntry] | None = None,
    downloaded: set[str] | None = None,
) -> WebBridgeDispatcher:
    config_path = tmp_path / "config.toml"
    if not config_path.exists():
        _write_config(config_path)
    dictionary_path = tmp_path / "dictionary.toml"
    if not dictionary_path.exists():
        CorrectionsEditorModel(
            exact={"Ptarmigan": ["ptarmigan"]},
            regex={"email": [r"e-?mail"]},
        ).save(dictionary_path)
    model_entries = models if models is not None else [_entry("moonshine:tiny")]
    downloaded_tokens = downloaded if downloaded is not None else {"moonshine:tiny"}

    deps = BridgeDependencies(
        config_path=lambda: config_path,
        check_permissions=lambda: permissions
        or PermissionReport(microphone=True, accessibility=False, input_monitoring=False),
        available_model_entries=lambda: model_entries,
        is_model_downloaded=lambda token: token in downloaded_tokens,
        resolve_dictionary_path=lambda: dictionary_path,
        login_is_enabled=lambda: True,
        daemon_is_running=lambda: daemon.running,
    )
    flow = OnboardingFlow()
    flow.choose_language("ja")
    return WebBridgeDispatcher(deps=deps, onboarding_flow=flow)


def test_get_state_returns_web_ui_snapshot_with_localized_strings_and_download_state(
    tmp_path: Path,
) -> None:
    dispatcher = _dispatcher(
        tmp_path,
        permissions=PermissionReport(
            microphone=True,
            accessibility=False,
            input_monitoring=True,
        ),
        daemon=_DaemonState(running=True),
        models=[_entry("moonshine:tiny"), _entry("granite:ibm/test")],
        downloaded={"moonshine:tiny"},
    )

    state = dispatcher.handle_action("getState", {})

    assert state["language"] == "ja"
    assert state["onboarding_step"] == "hotkey"
    assert state["permissions"] == {
        "microphone": True,
        "accessibility": False,
        "input_monitoring": True,
    }
    assert state["model"] == "moonshine:tiny"
    assert state["settings"]["language"] == "ja"
    assert state["settings"]["hotkey"] == "left_cmd"
    assert state["settings"]["output_mode"] == "clipboard_paste"
    assert state["settings"]["llm_correction"] == {
        "mode": "always",
        "provider": "ollama",
        "model": "qwen-test",
        "base_url": "http://localhost:11434",
    }
    assert state["models"] == [
        {
            "token": "moonshine:tiny",
            "backend": "moonshine",
            "label": "Label moonshine:tiny",
            "description": "Description moonshine:tiny",
            "downloaded": True,
        },
        {
            "token": "granite:ibm/test",
            "backend": "granite",
            "label": "Label granite:ibm/test",
            "description": "Description granite:ibm/test",
            "downloaded": False,
        },
    ]
    assert state["dictionary"] == {
        "exact": {"Ptarmigan": ["ptarmigan"]},
        "regex": {"email": [r"e-?mail"]},
    }
    assert state["login_enabled"] is True
    assert state["daemon_running"] is True
    assert state["strings"]["settings_window_title"] == "設定"


def test_get_state_setup_required_true_when_onboarding_incomplete(tmp_path: Path) -> None:
    dispatcher = _dispatcher(tmp_path)
    dispatcher.deps.language_was_selected = lambda: True
    dispatcher.deps.hotkey_was_confirmed = lambda: False

    state = dispatcher.handle_action("getState", {})

    assert state["setup_required"] is True


def test_get_state_setup_required_false_when_onboarding_complete(tmp_path: Path) -> None:
    dispatcher = _dispatcher(tmp_path)
    dispatcher.deps.language_was_selected = lambda: True
    dispatcher.deps.hotkey_was_confirmed = lambda: True

    state = dispatcher.handle_action("getState", {})

    assert state["setup_required"] is False


def test_get_state_setup_required_true_when_state_file_missing(tmp_path: Path) -> None:
    # Neither flag was ever recorded (state file absent), which must also
    # count as "setup required" -- not just an explicit False on one flag.
    dispatcher = _dispatcher(tmp_path)
    dispatcher.deps.language_was_selected = lambda: False
    dispatcher.deps.hotkey_was_confirmed = lambda: False

    state = dispatcher.handle_action("getState", {})

    assert state["setup_required"] is True


def test_begin_hotkey_capture_calls_deps_callback(tmp_path: Path) -> None:
    calls: list[bool] = []
    dispatcher = _dispatcher(tmp_path)
    dispatcher.deps.set_hotkey_capture_active = lambda active: calls.append(active)

    result = dispatcher.handle_action("beginHotkeyCapture", {})

    assert result == {"capturing": True}
    assert calls == [True]


def test_end_hotkey_capture_calls_deps_callback(tmp_path: Path) -> None:
    calls: list[bool] = []
    dispatcher = _dispatcher(tmp_path)
    dispatcher.deps.set_hotkey_capture_active = lambda active: calls.append(active)

    result = dispatcher.handle_action("endHotkeyCapture", {})

    assert result == {"capturing": False}
    assert calls == [False]


def test_choose_language_writes_config_marks_persistence_and_advances_flow(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, language="en")
    marks: list[str] = []
    deps = BridgeDependencies(
        config_path=lambda: config_path,
        mark_language_selected=lambda: marks.append("selected"),
        available_model_entries=lambda: [_entry()],
    )
    dispatcher = WebBridgeDispatcher(deps=deps, onboarding_flow=OnboardingFlow())

    result = dispatcher.handle_action("chooseLanguage", {"code": "zh"})

    assert result["language"] == "zh"
    assert result["onboarding_step"] == "hotkey"
    assert marks == ["selected"]
    assert load_config(config_path).language == "zh"
    assert result["strings"]["settings_window_title"] == "设置"


def test_confirm_hotkey_writes_config_marks_persistence_and_advances_flow(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, language="ja")
    marks: list[str] = []
    flow = OnboardingFlow()
    flow.choose_language("ja")
    deps = BridgeDependencies(
        config_path=lambda: config_path,
        mark_hotkey_confirmed=lambda: marks.append("confirmed"),
        available_model_entries=lambda: [_entry()],
    )
    dispatcher = WebBridgeDispatcher(deps=deps, onboarding_flow=flow)

    result = dispatcher.handle_action("confirmHotkey", {"hotkey": "right_shift"})

    assert result == {"hotkey": "right_shift", "onboarding_step": "microphone"}
    assert marks == ["confirmed"]
    assert load_config(config_path).hotkey.key == "right_shift"
    assert flow.selected_hotkey == "right_shift"


def test_confirm_hotkey_rejects_invalid_key_without_writing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, language="ja")
    marks: list[str] = []
    flow = OnboardingFlow()
    flow.choose_language("ja")
    deps = BridgeDependencies(
        config_path=lambda: config_path,
        mark_hotkey_confirmed=lambda: marks.append("confirmed"),
        available_model_entries=lambda: [_entry()],
    )
    dispatcher = WebBridgeDispatcher(deps=deps, onboarding_flow=flow)

    result = dispatcher.handle_action("confirmHotkey", {"hotkey": "shift_right"})

    assert result == {"saved": False, "errors": ["hotkey"]}
    assert marks == []
    assert load_config(config_path).hotkey.key == "left_cmd"
    assert flow.current_step == "hotkey"


def test_save_settings_updates_core_fields_and_llm_correction(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, model="moonshine:tiny")
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: config_path,
            available_model_entries=lambda: [_entry("granite:ibm/test")],
        ),
    )

    result = dispatcher.handle_action(
        "saveSettings",
        {
            "model": "granite:ibm/test",
            "language": "en",
            "hotkey": "right_shift",
            "output_mode": "direct_typing",
            "llm_correction": {
                "mode": "ask",
                "provider": "lmstudio",
                "model": "qwen3",
                "base_url": "http://localhost:1234",
            },
        },
    )

    config = load_config(config_path)
    assert result["saved"] is True
    assert config.stt.model == "granite:ibm/test"
    assert config.language == "en"
    assert config.hotkey.key == "right_shift"
    assert config.output.mode == OutputMode.DIRECT_TYPING
    assert config.text.llm_correction.mode == LLMCorrectionMode.ASK
    assert config.text.llm_correction.provider == "lmstudio"
    assert config.text.llm_correction.model == "qwen3"
    assert config.text.llm_correction.base_url == "http://localhost:1234"


def test_save_settings_returns_validation_errors_without_writing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, language="ja", model="moonshine:tiny")
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: config_path,
            available_model_entries=lambda: [_entry("moonshine:tiny")],
        ),
    )

    result = dispatcher.handle_action(
        "saveSettings",
        {
            "model": "missing:model",
            "language": "de",
            "hotkey": "bad_hotkey",
            "output_mode": "bad_output",
            "llm_correction": {
                "mode": "sometimes",
                "provider": "",
                "model": "",
                "base_url": "",
            },
        },
    )

    assert result == {
        "saved": False,
        "errors": [
            "model",
            "language",
            "hotkey",
            "output_mode",
            "llm_correction.mode",
            "llm_correction.provider",
            "llm_correction.model",
            "llm_correction.base_url",
        ],
    }
    config = load_config(config_path)
    assert config.language == "ja"
    assert config.stt.model == "moonshine:tiny"


def test_dictionary_actions_load_save_and_return_regex_validation_errors(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "dictionary.toml"
    CorrectionsEditorModel(exact={"AI": ["a i"]}, regex={}).save(dictionary_path)
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: tmp_path / "config.toml",
            resolve_dictionary_path=lambda: dictionary_path,
            available_model_entries=lambda: [_entry()],
        ),
    )

    loaded = dispatcher.handle_action("loadDictionary", {})
    invalid = dispatcher.handle_action(
        "saveDictionary",
        {
            "exact": {"AI": ["A.I."]},
            "regex": {"broken": ["("]},
        },
    )
    saved = dispatcher.handle_action(
        "saveDictionary",
        {
            "exact": {"AI": ["A.I."]},
            "regex": {"email": [r"e-?mail"]},
        },
    )

    assert loaded == {"exact": {"AI": ["a i"]}, "regex": {}}
    assert invalid["saved"] is False
    assert set(invalid["errors"][0]) == {"section", "key", "index", "pattern", "message"}
    assert invalid["errors"][0]["section"] == "regex"
    assert invalid["errors"][0]["key"] == "broken"
    assert invalid["errors"][0]["index"] == 0
    assert invalid["errors"][0]["pattern"] == "("
    assert isinstance(invalid["errors"][0]["message"], str)
    assert invalid["errors"][0]["message"]
    assert saved == {"saved": True}
    assert CorrectionsEditorModel.load(dictionary_path).regex == {"email": [r"e-?mail"]}


def test_side_effect_actions_delegate_to_injected_functions(tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []
    login_enabled = {"value": False}
    daemon_running = {"value": False}

    def register_login() -> bool:
        login_enabled["value"] = True
        calls.append(("login", "register"))
        return True

    def unregister_login() -> bool:
        login_enabled["value"] = False
        calls.append(("login", "unregister"))
        return True

    def start_dictation() -> None:
        daemon_running["value"] = True
        calls.append(("daemon", "start"))

    def stop_dictation() -> None:
        daemon_running["value"] = False
        calls.append(("daemon", "stop"))

    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: tmp_path / "config.toml",
            available_model_entries=lambda: [_entry()],
            login_is_enabled=lambda: login_enabled["value"],
            login_register=register_login,
            login_unregister=unregister_login,
            request_permission=lambda kind: calls.append(("permission", kind)),
            open_system_settings=lambda kind: calls.append(("settings", kind)),
            start_dictation=start_dictation,
            stop_dictation=stop_dictation,
            daemon_is_running=lambda: daemon_running["value"],
            restart_app=lambda: calls.append(("restart", None)) or True,
        ),
    )

    assert dispatcher.handle_action("requestPermission", {"kind": "microphone"}) == {
        "requested": "microphone"
    }
    assert dispatcher.handle_action("openSystemSettings", {"kind": "accessibility"}) == {
        "opened": "accessibility"
    }
    assert dispatcher.handle_action("startDictation", {}) == {"daemon_running": True}
    assert dispatcher.handle_action("stopDictation", {}) == {"daemon_running": False}
    assert dispatcher.handle_action("toggleLogin", {"enabled": True}) == {"login_enabled": True}
    assert dispatcher.handle_action("toggleLogin", {"enabled": False}) == {
        "login_enabled": False
    }
    assert dispatcher.handle_action("restartApp", {}) == {"restarted": True}
    assert calls == [
        ("permission", "microphone"),
        ("settings", "accessibility"),
        ("daemon", "start"),
        ("daemon", "stop"),
        ("login", "register"),
        ("login", "unregister"),
        ("restart", None),
    ]


def test_open_config_file_action_delegates_to_injected_function(tmp_path: Path) -> None:
    calls: list[str] = []
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: tmp_path / "config.toml",
            available_model_entries=lambda: [_entry()],
            open_config_file=lambda: calls.append("open"),
        ),
    )

    assert dispatcher.handle_action("openConfigFile", {}) == {"opened": True}
    assert calls == ["open"]


def test_top_level_handle_action_uses_default_dispatcher() -> None:
    result = handle_action("listModels", {})

    assert isinstance(result["models"], list)


def test_download_model_starts_download_for_valid_not_downloaded_token(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: tmp_path / "config.toml",
            available_model_entries=lambda: [_entry("granite:ibm/test")],
            is_model_downloaded=lambda _token: False,
            start_model_download=lambda token: calls.append(token) or True,
        ),
    )

    result = dispatcher.handle_action("downloadModel", {"model": "granite:ibm/test"})

    assert result == {"started": True, "model": "granite:ibm/test"}
    assert calls == ["granite:ibm/test"]


def test_download_model_reports_busy_when_dependency_reports_not_started(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: tmp_path / "config.toml",
            available_model_entries=lambda: [_entry("granite:ibm/test")],
            is_model_downloaded=lambda _token: False,
            start_model_download=lambda token: calls.append(token) or False,
        ),
    )

    result = dispatcher.handle_action("downloadModel", {"model": "granite:ibm/test"})

    assert result == {"started": False, "errors": ["busy"]}
    assert calls == ["granite:ibm/test"]


def test_download_model_rejects_unknown_token_without_calling_dependency(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: tmp_path / "config.toml",
            available_model_entries=lambda: [_entry("granite:ibm/test")],
            is_model_downloaded=lambda _token: False,
            start_model_download=lambda token: calls.append(token),
        ),
    )

    result = dispatcher.handle_action("downloadModel", {"model": "unknown:model"})

    assert result == {"started": False, "errors": ["model"]}
    assert calls == []


def test_download_model_reports_already_downloaded_without_calling_dependency(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: tmp_path / "config.toml",
            available_model_entries=lambda: [_entry("granite:ibm/test")],
            is_model_downloaded=lambda _token: True,
            start_model_download=lambda token: calls.append(token),
        ),
    )

    result = dispatcher.handle_action("downloadModel", {"model": "granite:ibm/test"})

    assert result == {"started": False, "already_downloaded": True}
    assert calls == []
