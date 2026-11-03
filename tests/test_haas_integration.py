"""Haas wiring tests; all downloads and runtime probes are synthetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptarmigan_flow import app_settings_model
from ptarmigan_flow import config as config_module
from ptarmigan_flow.app_settings_model import AppSettingsModel
from ptarmigan_flow.application.use_cases.llm_runtime import (
    build_llm_settings_from_config,
    runtime_language_from_config,
)
from ptarmigan_flow.config import AppConfig, load_config, write_config
from ptarmigan_flow.permissions import PermissionReport
from ptarmigan_flow.presentation.cli import commands
from ptarmigan_flow.stt import availability, model_download
from ptarmigan_flow.stt.factory import create_stt_backend, parse_stt_model
from ptarmigan_flow.stt.haas import HaasSTTBackend
from ptarmigan_flow.stt.haas_artifact import HAAS_MODEL_ID, HAAS_MODEL_TOKEN, HAAS_Q4_K_ARTIFACT
from ptarmigan_flow.stt.model_catalog import (
    HUB_SEARCH_BACKENDS,
    search_hub_models,
    verified_model_entries,
)
from ptarmigan_flow.stt.model_families import resolve_runtime_model_id
from ptarmigan_flow.stt.runtime_backend import create_runtime_stt_backend
from ptarmigan_flow.text_processing import llm as llm_module
from ptarmigan_flow.web_bridge import BridgeDependencies, WebBridgeDispatcher


@pytest.fixture(autouse=True)
def forbid_runtime(monkeypatch):
    from ptarmigan_flow.stt import haas_runtime

    def forbidden(*args, **kwargs):
        pytest.fail("Integration tests must not launch speech or download real models")

    monkeypatch.setattr(haas_runtime.subprocess, "Popen", forbidden)


def _config(path, *, language="ja", model="moonshine:tiny"):
    config = AppConfig()
    config.language = language
    config.stt.model = model
    write_config(path, config)
    return config


@pytest.mark.parametrize("language", ["ja", "auto"])
@pytest.mark.parametrize(
    "device, resolved",
    [
        ("cpu", "cpu"),
        ("auto", "auto"),
        ("mps", "auto"),
        ("gpu", "gpu"),
    ],
)
def test_factory_preserves_processor_and_keeps_haas_outside_python_isolation(
    language,
    device,
    resolved,
) -> None:
    config = AppConfig(language=language, stt={"model": HAAS_MODEL_TOKEN})
    processor = object()
    config.audio.trailing_silence_seconds = 0.25
    config.model.device = device
    backend = create_runtime_stt_backend(config, post_processor=processor)
    assert isinstance(backend, HaasSTTBackend)
    assert backend._post_processor is processor
    assert backend._trailing_silence == 0.25
    assert backend._device == resolved
    assert config.model.device == device
    assert parse_stt_model(HAAS_MODEL_TOKEN) == ("haas", HAAS_MODEL_ID)
    for arm64 in (True, False):
        assert (
            resolve_runtime_model_id(
                prefix="haas",
                model_id=HAAS_MODEL_ID,
                macos_arm64=arm64,
            )
            == HAAS_MODEL_ID
        )
    backend.close()


def test_factory_rejects_unsupported_device_without_changing_config() -> None:
    config = AppConfig(language="ja", stt={"model": HAAS_MODEL_TOKEN})
    config.model.device = "cuda"
    with pytest.raises(ValueError, match="Haas model.device must be"):
        create_stt_backend(config)
    assert config.model.device == "cuda"


def test_parser_and_factory_reject_arbitrary_gguf_and_non_japanese() -> None:
    with pytest.raises(ValueError, match="supports only"):
        parse_stt_model("haas:other/repo")
    config = AppConfig()
    config.stt.model = HAAS_MODEL_TOKEN
    with pytest.raises(ValueError, match="pflow config language"):
        create_stt_backend(config)
    assert config.language == "en"  # No hidden language mutation.


@pytest.mark.parametrize("language", ["ja", "auto"])
def test_config_accepts_haas_language_in_pydantic_and_loader(language, tmp_path) -> None:
    config = AppConfig.model_validate({"language": language, "stt": {"model": HAAS_MODEL_TOKEN}})
    path = tmp_path / "config.toml"
    write_config(path, config)
    assert load_config(path).language == language
    assert config_module._normalize_top_level_language(language, model=HAAS_MODEL_TOKEN) == language


@pytest.mark.parametrize("language", ["en", "zh", "de", ""])
def test_both_config_validators_reject_explicit_non_japanese(language, tmp_path) -> None:
    with pytest.raises(ValueError, match="Japanese-only"):
        AppConfig.model_validate({"language": language, "stt": {"model": HAAS_MODEL_TOKEN}})
    with pytest.raises(ValueError, match="language"):
        config_module._normalize_top_level_language(language, model=HAAS_MODEL_TOKEN)
    path = tmp_path / "config.toml"
    _config(path, language=language, model=HAAS_MODEL_TOKEN)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Japanese-only"):
        load_config(path)
    assert path.read_bytes() == before


def test_other_models_keep_defaults_and_reject_auto(tmp_path) -> None:
    path = tmp_path / "config.toml"
    assert load_config(path).stt.model == "moonshine:tiny"
    assert load_config(path).language == "en"
    _config(path, language="auto")
    with pytest.raises(ValueError, match="auto.*no longer supported"):
        load_config(path)


@pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
def test_haas_auto_corrects_in_japanese_without_changing_speech_or_persisted_language(
    provider,
    tmp_path,
) -> None:
    path = tmp_path / "config.toml"
    config = _config(path, language="auto", model=HAAS_MODEL_TOKEN)
    config.text.llm_correction.provider = provider
    write_config(path, config)
    before = path.read_bytes()
    assert runtime_language_from_config(config) == "ja"
    settings = build_llm_settings_from_config(config, config.text.llm_correction)
    assert settings is not None and settings.language == "ja"
    assert settings.provider == provider
    assert settings.model == config.text.llm_correction.model
    assert settings.base_url == config.text.llm_correction.base_url
    prompt = llm_module._build_system_instruction(settings.language)
    assert prompt == llm_module._JA_SYSTEM_INSTRUCTION
    assert "Output language must be 'en'" not in prompt
    backend = create_stt_backend(config)
    assert backend._language == "auto"
    backend.close()
    assert config.language == "auto"
    assert load_config(path).language == "auto"
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "model, language, expected",
    [
        ("moonshine:tiny", "auto", "en"),
        ("mlx:example/model", "fr", "fr"),
        ("haas:sakasegawa/Haas-ja-v1.0-GGUF", "ja", "ja"),
        ("moonshine:tiny", None, "en"),
    ],
)
def test_runtime_language_fallbacks_remain_unchanged(model, language, expected):
    config = SimpleNamespace(language=language, stt=SimpleNamespace(model=model))
    assert runtime_language_from_config(config) == expected
    assert config.language == language


def test_catalog_fixed_preset_does_not_claim_generic_hub_support() -> None:
    entry = next(entry for entry in verified_model_entries() if entry.token == HAAS_MODEL_TOKEN)
    assert entry.backend == "haas" and entry.verified
    assert "Q4_K" in entry.label and "Japanese" in entry.label
    assert "speech.cpp >=0.9.4" in entry.description
    assert "haas" not in HUB_SEARCH_BACKENDS
    with pytest.raises(ValueError, match="arbitrary Hub GGUFs"):
        search_hub_models("anything", backend="haas")


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    data = b"synthetic downloaded model"
    artifact = replace(
        HAAS_Q4_K_ARTIFACT,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(model_download, "HAAS_Q4_K_ARTIFACT", artifact)
    return artifact, data


def _download_fixture(artifact, data, calls):
    def download(**kwargs):
        calls.append(kwargs)
        progress = kwargs["tqdm_class"](total=len(data), desc="Q4_K bytes", mininterval=0)
        progress.update(len(data) // 2)
        progress.update(len(data) - len(data) // 2)
        progress.close()
        path = artifact.cached_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    return download


def test_exact_file_download_progress_and_offline_status(artifact) -> None:
    spec, data = artifact
    events, calls = [], []
    assert model_download.resolve_download_repo_id(HAAS_MODEL_TOKEN) == spec.repo_id
    assert not model_download.is_model_downloaded(HAAS_MODEL_TOKEN)
    model_download.download_model(
        HAAS_MODEL_TOKEN,
        progress_callback=lambda *event: events.append(event),
        hf_hub_download=_download_fixture(spec, data, calls),
        snapshot_download=lambda *args, **kwargs: pytest.fail("Must fetch one file only"),
    )
    assert {key: value for key, value in calls[0].items() if key != "tqdm_class"} == {
        "repo_id": HAAS_MODEL_ID,
        "filename": spec.filename,
        "revision": spec.revision,
        "force_download": False,
    }
    assert any(fraction is not None and 0 < fraction < 1 for fraction, _ in events)
    assert (None, "verifying") in events
    assert events[-1] == (1.0, "ready")
    assert model_download.is_model_downloaded(HAAS_MODEL_TOKEN)
    assert commands._stt_model_downloaded_display(HAAS_MODEL_TOKEN) == "yes"
    model_download.download_model(
        HAAS_MODEL_TOKEN,
        progress_callback=lambda *args: None,
        hf_hub_download=lambda **kwargs: pytest.fail("Valid pinned cache should work offline"),
    )


def test_cache_status_ignores_other_files_and_download_repairs_corrupt_bytes(artifact) -> None:
    spec, data = artifact
    path = spec.cached_path()
    path.parent.mkdir(parents=True)
    (path.parent / "README.md").write_text("present")
    (path.parent / "Haas-616M-ja-v1.0-Q8_0.gguf").write_bytes(data)
    assert not model_download.is_model_downloaded(HAAS_MODEL_TOKEN)
    path.write_bytes(b"partial")
    assert not model_download.is_model_downloaded(HAAS_MODEL_TOKEN)
    path.write_bytes(b"x" * len(data))
    calls = []
    model_download.download_model(
        HAAS_MODEL_TOKEN,
        progress_callback=lambda *args: None,
        hf_hub_download=_download_fixture(spec, data, calls),
    )
    assert calls[0]["force_download"] is True
    assert spec.verify_file(path) == path


def test_corrupt_download_never_reports_ready(artifact) -> None:
    spec, data = artifact
    events = []
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        model_download.download_model(
            HAAS_MODEL_TOKEN,
            progress_callback=lambda *event: events.append(event),
            hf_hub_download=_download_fixture(spec, b"x" * len(data), []),
        )
    assert events[-1] == (None, "verifying")
    assert (1.0, "ready") not in events


def test_cli_explicit_download_verifies_even_when_cache_is_present(
    artifact,
    monkeypatch,
    capsys,
) -> None:
    spec, data = artifact
    calls = []
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download_fixture(spec, data, calls))
    args = argparse.Namespace(model=HAAS_MODEL_TOKEN, config=None)
    assert commands.cmd_download_model(args) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1] == {"type": "done"}
    assert any(event.get("message") == "verifying" for event in events)
    spec.cached_path().write_bytes(b"x" * len(data))
    assert commands.cmd_download_model(args) == 0
    assert calls[-1]["force_download"] is True


def test_gui_download_uses_pinned_status_without_selecting_model(artifact, tmp_path) -> None:
    spec, data = artifact
    path = tmp_path / "config.toml"
    _config(path, language="en")
    before = path.read_bytes()
    started = []
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: path,
            available_model_entries=lambda: [
                entry for entry in verified_model_entries() if entry.backend == "haas"
            ],
            is_model_downloaded=model_download.is_model_downloaded,
            start_model_download=lambda token: started.append(token) or True,
        )
    )
    result = dispatcher.handle_action("downloadModel", {"model": HAAS_MODEL_TOKEN})
    assert result == {"started": True, "model": HAAS_MODEL_TOKEN}
    assert started == [HAAS_MODEL_TOKEN]
    spec.cached_path().parent.mkdir(parents=True)
    spec.cached_path().write_bytes(data)
    result = dispatcher.handle_action("downloadModel", {"model": HAAS_MODEL_TOKEN})
    assert result == {"started": True, "model": HAAS_MODEL_TOKEN}
    assert started == [HAAS_MODEL_TOKEN, HAAS_MODEL_TOKEN]
    assert path.read_bytes() == before


def test_gui_explicit_retry_repairs_same_size_corruption_via_cli_downloader(
    artifact,
    monkeypatch,
    capsys,
) -> None:
    import huggingface_hub

    spec, data = artifact
    spec.cached_path().parent.mkdir(parents=True)
    spec.cached_path().write_bytes(b"x" * len(data))
    # Cheap status legitimately says present; the explicit action must still hash it.
    assert model_download.is_model_downloaded(HAAS_MODEL_TOKEN)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        spec.verify_file(spec.cached_path())
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download_fixture(spec, data, calls))

    def start_download(token):
        assert commands.cmd_download_model(argparse.Namespace(model=token, config=None)) == 0
        return True

    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            available_model_entries=lambda: [
                entry for entry in verified_model_entries() if entry.backend == "haas"
            ],
            is_model_downloaded=model_download.is_model_downloaded,
            start_model_download=start_download,
        )
    )
    result = dispatcher.handle_action("downloadModel", {"model": HAAS_MODEL_TOKEN})
    assert result == {"started": True, "model": HAAS_MODEL_TOKEN}
    assert len(calls) == 1 and calls[0]["force_download"] is True
    assert spec.verify_file(spec.cached_path()) == spec.cached_path()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(event.get("message") == "verifying" for event in events)
    assert events[-1] == {"type": "done"}


@pytest.mark.parametrize("language", ["en", "zh"])
def test_onboarding_rejects_incompatible_haas_language_before_writing(
    language,
    tmp_path,
) -> None:
    path = tmp_path / "config.toml"
    _config(path, language="ja", model=HAAS_MODEL_TOKEN)
    before = path.read_bytes()
    marks = []
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: path,
            check_permissions=lambda: PermissionReport(True, True, True),
            available_model_entries=lambda: [],
            resolve_dictionary_path=lambda: tmp_path / "dictionary.toml",
            login_is_enabled=lambda: False,
            language_was_selected=lambda: False,
            hotkey_was_confirmed=lambda: False,
            mark_language_selected=lambda: marks.append("selected"),
        )
    )
    result = dispatcher.handle_action("chooseLanguage", {"code": language})
    assert result["saved"] is False
    assert result["errors"] == ["language"]
    assert "Japanese-only" in result["message"]
    assert path.read_bytes() == before
    assert marks == []
    assert dispatcher.onboarding_flow.current_step == "language"
    assert dispatcher.handle_action("getState", {})["language"] == "ja"

    result = dispatcher.handle_action("chooseLanguage", {"code": "ja"})
    assert result["onboarding_step"] == "hotkey"
    assert marks == ["selected"]
    assert load_config(path).stt.model == HAAS_MODEL_TOKEN


@pytest.mark.parametrize("language", ["auto", "de", ""])
def test_onboarding_rejects_values_outside_its_choices_before_writing(language, tmp_path):
    path = tmp_path / "config.toml"
    _config(path, language="ja", model=HAAS_MODEL_TOKEN)
    before = path.read_bytes()
    dispatcher = WebBridgeDispatcher(deps=BridgeDependencies(config_path=lambda: path))
    result = dispatcher.handle_action("chooseLanguage", {"code": language})
    assert result["saved"] is False
    assert path.read_bytes() == before
    assert dispatcher.onboarding_flow.current_step == "language"


@pytest.mark.parametrize("language, expected", [("en", 2), ("ja", 0), ("auto", 0)])
def test_cli_model_selection_rejects_conflicts_without_mutation(
    language,
    expected,
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    path = tmp_path / "config.toml"
    _config(path, language="en" if language == "auto" else language)
    if language == "auto":
        _config(path, language="auto", model=HAAS_MODEL_TOKEN)
    before = path.read_bytes()
    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    index = commands._stt_model_presets().index(HAAS_MODEL_TOKEN) + 1
    monkeypatch.setattr("builtins.input", lambda _prompt: str(index))
    monkeypatch.setattr(commands, "_stt_model_downloaded_display", lambda _token: "no")
    assert (
        commands.cmd_list_model(argparse.Namespace(config=str(path), hub_search=None)) == expected
    )
    if expected:
        assert path.read_bytes() == before
        assert "pflow config language" in capsys.readouterr().err
    else:
        assert load_config(path).stt.model == HAAS_MODEL_TOKEN
        assert load_config(path).language == language


def test_cli_config_editor_rejects_invalid_pair_without_saving(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.toml"
    _config(path, language="en")
    before = path.read_bytes()
    monkeypatch.setattr(
        commands,
        "_resolve_config_section_editor",
        lambda _section: lambda config: setattr(config.stt, "model", HAAS_MODEL_TOKEN),
    )
    assert commands._run_config_editor(config_path=path, section_name="stt") == 2
    assert path.read_bytes() == before
    assert "Japanese-only" in capsys.readouterr().err


def test_cli_leaving_haas_auto_requires_explicit_language_choice(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.toml"
    _config(path, language="auto", model=HAAS_MODEL_TOKEN)
    before = path.read_bytes()
    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")
    monkeypatch.setattr(commands, "_stt_model_downloaded_display", lambda _: "no")
    assert commands.cmd_list_model(argparse.Namespace(config=str(path), hub_search=None)) == 2
    assert path.read_bytes() == before
    assert "pflow config language" in capsys.readouterr().err


def test_cli_doctor_reports_path_version_cache_and_failure(artifact, monkeypatch) -> None:
    monkeypatch.setattr(commands, "find_speech_executable", lambda: Path("/custom/speech"))
    monkeypatch.setattr(
        commands.SpeechCppRuntime, "check_version", lambda *args, **kwargs: (0, 9, 4)
    )
    lines = commands._haas_diagnostics(HAAS_MODEL_TOKEN)
    assert "Haas Q4_K downloaded: no" in lines
    assert "speech.cpp executable: /custom/speech" in lines
    assert "speech.cpp version: 0.9.4 (>=0.9.4 required)" in lines

    def fail(*args, **kwargs):
        raise RuntimeError("speech.cpp 0.9.3 is too old; install >=0.9.4")

    monkeypatch.setattr(commands.SpeechCppRuntime, "check_version", fail)
    assert "ERROR" in commands._haas_diagnostics(HAAS_MODEL_TOKEN)[-1]


def test_cli_doctor_uses_haas_diagnostics_without_model_inference(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    path = tmp_path / "config.toml"
    _config(path, model=HAAS_MODEL_TOKEN)
    monkeypatch.setattr(commands, "_haas_diagnostics", lambda _: ["speech.cpp version: fixture"])
    monkeypatch.setattr(
        commands, "check_all_permissions", lambda: PermissionReport(True, True, True)
    )
    monkeypatch.setattr(commands, "read_launch_agent_plist", lambda: None)
    monkeypatch.setattr(commands, "_latest_launchd_runtime_warning_with_timestamp", lambda _: None)
    monkeypatch.setattr(commands, "recommended_permission_target", lambda: "/fixture/python")
    monkeypatch.setattr(commands, "find_spec", lambda _: None)
    monkeypatch.setattr(commands.platform, "platform", lambda: "fixture macOS")
    monkeypatch.setattr(
        commands, "_load_corrections_with_diagnostics", lambda *a, **k: (None, "off")
    )
    assert commands.cmd_doctor(argparse.Namespace(config=str(path), launchd_check=False)) == 0
    output = capsys.readouterr().out
    assert "speech.cpp version: fixture" in output
    assert "backend=speech.cpp-haas" in output


def test_haas_available_for_selection_without_external_runtime(monkeypatch) -> None:
    monkeypatch.setattr(availability.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(availability, "find_speech_executable", lambda: None)
    assert availability.is_backend_available("haas")
    assert HAAS_MODEL_TOKEN in [entry.token for entry in availability.available_model_entries()]
    assert "PTARMIGAN_FLOW_SPEECH_CPP_PATH" in availability.backend_runtime_guidance("haas")
    monkeypatch.setattr(availability, "find_speech_executable", lambda: Path("/custom/speech"))
    assert availability.backend_runtime_guidance("haas") is None
    assert availability.backend_runtime_guidance("moonshine") is None


@pytest.mark.parametrize("language", ["en", "zh", "ja", "auto"])
def test_both_gui_validators_and_save_paths(language, tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    _config(path, language="en")
    before = path.read_bytes()
    entries = [entry for entry in verified_model_entries() if entry.backend == "haas"]
    monkeypatch.setattr(app_settings_model, "available_model_entries", lambda: entries)
    dispatcher = WebBridgeDispatcher(
        deps=BridgeDependencies(
            config_path=lambda: path,
            available_model_entries=lambda: entries,
            backend_runtime_guidance=lambda _backend: "Install speech.cpp >=0.9.4",
            is_model_downloaded=lambda _token: False,
        )
    )
    settings = AppSettingsModel(HAAS_MODEL_TOKEN, language, "right_cmd", "direct_typing")
    invalid = language not in {"ja", "auto"}
    assert settings.validate() == (["language"] if invalid else [])
    assert dispatcher._validate_settings(settings) == settings.validate()
    result = dispatcher.handle_action(
        "saveSettings",
        {
            "model": settings.model,
            "language": language,
            "hotkey": settings.hotkey,
            "output_mode": settings.output_mode,
            "llm_correction": {
                "mode": "never",
                "provider": "ollama",
                "model": "fixture",
                "base_url": "http://localhost:11434",
            },
        },
    )
    assert result["saved"] is not invalid
    if invalid:
        assert "Japanese-only" in result["message"]
        with pytest.raises(ValueError, match="Japanese-only"):
            settings.save(path)
        assert path.read_bytes() == before
    else:
        settings.save(path)
        assert load_config(path).language == language
    card = dispatcher.handle_action("listModels", {})["models"][0]
    assert "Install speech.cpp >=0.9.4" in card["description"]
    assert card["token"] == HAAS_MODEL_TOKEN and not card["downloaded"]


def test_packaging_uses_shared_download_dependencies_without_bundling_speech() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    release = Path("packaging/macos/requirements-release.txt").read_text().splitlines()
    spec = Path("packaging/macos/PtarmiganFlow.spec").read_text()
    lock = tomllib.loads(Path("uv.lock").read_text())
    root = next(package for package in lock["package"] if package["name"] == "ptarmigan-flow")
    for requirement in ("huggingface-hub>=1.0", "tqdm>=4.67.0"):
        assert requirement in pyproject["project"]["dependencies"] and requirement in release
        name = requirement.split(">=", 1)[0]
        assert {"name": name} in root["dependencies"]
    for module in ("haas", "haas_artifact", "haas_runtime"):
        assert f'"ptarmigan_flow.stt.{module}"' in spec
    common = spec.split("REQUIRED_RUNTIME_MODULES = (", 1)[1].split(")", 1)[0]
    assert '"huggingface_hub"' in common and '"tqdm"' in common
    assert "binaries = []" in spec and "speech.cpp" not in spec


def test_documentation_covers_explicit_selection_runtime_and_limits() -> None:
    text = Path("docs/haas.md").read_text()
    for required in (
        HAAS_MODEL_TOKEN,
        HAAS_Q4_K_ARTIFACT.revision,
        HAAS_Q4_K_ARTIFACT.sha256,
        "PTARMIGAN_FLOW_SPEECH_CPP_PATH",
        "~/.local/bin/speech",
        "0.9.4",
        "every utterance",
        "VAD",
        "Japanese only",
        "pflow config language",
    ):
        assert required in text
    defaults = tomllib.loads(Path("config.example.toml").read_text())
    assert defaults["language"] == "en" and defaults["stt"]["model"] == "moonshine:tiny"
