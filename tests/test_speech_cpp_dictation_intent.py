"""Exercise user Start/Stop against synthetic runtime completions via the real bridge."""

import pytest
from test_macos_app import _LLMRecordingDaemonController, _speech_cpp_controller

from ptarmigan_flow import macos_app
from ptarmigan_flow.config import AppConfig, LLMCorrectionMode, write_config


@pytest.fixture
def intent_controller(monkeypatch, tmp_path):
    controller, _, events, _ = _speech_cpp_controller(monkeypatch)
    path = tmp_path / "config.toml"
    config = AppConfig()
    config.language = "ja"
    config.stt.model = "haas:sakasegawa/Haas-ja-v1.0-GGUF"
    write_config(path, config)
    controller.bridge.deps.config_path = lambda: path
    controller.daemon_controller = _LLMRecordingDaemonController(path)
    controller._last_permission_report = macos_app.PermissionReport(True, True, True)
    controller._push_daemon_state = lambda: None
    controller._install_hotkey_event_monitor = lambda: None
    controller._start_daemon_status_timer = lambda: None
    controller._llm_enabled_override_for_start = lambda: None
    monkeypatch.setattr(macos_app.availability, "is_backend_available", lambda _: True)
    monkeypatch.setattr(macos_app.availability, "backend_runtime_guidance", lambda _: None)
    monkeypatch.setattr(macos_app.model_download, "is_model_downloaded", lambda _: True)

    def begin(kind):
        if kind == "install":
            assert controller.bridge.handle_action("installSpeechCpp", {})["started"]
        else:
            controller._speech_cpp_checked_at = 0
            assert not controller.bridge.handle_action("startDictation", {})["daemon_running"]
            assert controller._speech_cpp_start_when_ready
        assert controller._speech_cpp_busy
        assert not controller.daemon_controller.is_running

    def complete(kind):
        controller.applySpeechCppStatus_(
            {
                "operation_id": controller._speech_cpp_operation_id,
                "finished": True,
                "recover": kind == "install",
                "status": {"status": "available", "version": "0.10.1", "install_allowed": False},
            }
        )
        assert events[-1][1]["status"] == "available"
        assert not controller._speech_cpp_busy

    def switch_model():
        config.stt.model = "moonshine:tiny"
        write_config(path, config)

    return controller, begin, complete, switch_model


@pytest.mark.parametrize("kind", ["install", "probe"])
def test_explicit_stop_blocks_runtime_completion_restart(intent_controller, kind):
    controller, begin, complete, _ = intent_controller
    begin(kind)
    assert controller.bridge.handle_action("stopDictation", {}) == {"daemon_running": False}
    assert controller._speech_cpp_busy  # Dictation Stop leaves runtime work intact.
    complete(kind)
    assert not controller.daemon_controller.is_running
    assert controller.daemon_controller.overrides == []
    # A later permission/model callback cannot restore the revoked startup intent either.
    controller._start_daemon_if_ready()
    assert not controller.daemon_controller.is_running
    assert controller.bridge.handle_action("startDictation", {})["daemon_running"]
    assert controller.daemon_controller.overrides == [None]


@pytest.mark.parametrize("kind", ["install", "probe"])
def test_explicit_start_again_waits_for_current_runtime_completion(intent_controller, kind):
    controller, begin, complete, _ = intent_controller
    begin(kind)
    controller.bridge.handle_action("stopDictation", {})
    assert not controller.bridge.handle_action("startDictation", {})["daemon_running"]
    complete(kind)
    assert controller.daemon_controller.is_running
    assert controller.daemon_controller.overrides == [None]


@pytest.mark.parametrize("kind", ["install", "probe"])
def test_internal_stop_preserves_automatic_haas_recovery(intent_controller, kind):
    controller, begin, complete, _ = intent_controller
    begin(kind)
    controller._stop_daemon()  # Internal restart/config-reload cleanup, not user Stop.
    complete(kind)
    assert controller.daemon_controller.is_running
    assert controller.daemon_controller.overrides == [None]


@pytest.mark.parametrize("kind", ["install", "probe"])
@pytest.mark.parametrize("start_again", [False, True])
def test_runtime_completion_respects_current_unrelated_model_and_user_intent(
    intent_controller,
    kind,
    start_again,
):
    controller, begin, complete, switch_model = intent_controller
    begin(kind)
    controller.bridge.handle_action("stopDictation", {})
    switch_model()
    if start_again:
        assert controller.bridge.handle_action("startDictation", {})["daemon_running"]
    complete(kind)
    assert controller.daemon_controller.is_running is start_again
    assert controller.daemon_controller.overrides == ([None] if start_again else [])


@pytest.mark.parametrize("kind", ["install", "probe"])
def test_second_explicit_stop_revokes_reenabled_waiting(intent_controller, kind):
    controller, begin, complete, _ = intent_controller
    begin(kind)
    controller.bridge.handle_action("stopDictation", {})
    controller.bridge.handle_action("startDictation", {})
    controller.bridge.handle_action("stopDictation", {})
    complete(kind)
    assert not controller.daemon_controller.is_running
    assert controller.daemon_controller.overrides == []


@pytest.fixture
def ask_intent_controller(intent_controller):
    controller, begin, complete, switch_model = intent_controller
    config = controller.bridge._load_config()
    config.text.llm_correction.mode = LLMCorrectionMode.ASK
    write_config(controller.bridge.deps.config_path(), config)
    # Restore the actual ask path instead of bypassing its reentrant callback.
    del controller._llm_enabled_override_for_start
    return controller, begin, complete, switch_model


@pytest.mark.parametrize("kind", ["install", "probe"])
def test_stop_inside_ask_callback_blocks_runtime_recovery(ask_intent_controller, kind):
    controller, begin, complete, _ = ask_intent_controller
    prompts = []

    def ask(*_args):
        assert controller._llm_prompt_in_progress
        prompts.append("ask")
        assert controller.bridge.handle_action("stopDictation", {}) == {"daemon_running": False}
        return 0

    controller._run_alert = ask
    begin(kind)
    complete(kind)
    assert prompts == ["ask"]
    assert not controller.daemon_controller.is_running
    assert controller.daemon_controller.overrides == []
    assert not controller._llm_prompt_in_progress
    # A later explicit Start still works, using the normal remembered ask answer.
    assert controller.bridge.handle_action("startDictation", {})["daemon_running"]
    assert controller.daemon_controller.overrides == [True]


@pytest.mark.parametrize("kind", ["install", "probe"])
def test_quit_inside_ask_callback_blocks_runtime_recovery(ask_intent_controller, kind):
    controller, begin, complete, _ = ask_intent_controller
    prompts = []

    def ask(*_args):
        assert controller._llm_prompt_in_progress
        prompts.append("ask")
        controller.applicationWillTerminate_(None)
        return 0

    controller._run_alert = ask
    begin(kind)
    complete(kind)
    assert prompts == ["ask"]
    assert controller._speech_cpp_cancel_event.is_set()
    assert not controller.daemon_controller.is_running
    assert controller.daemon_controller.overrides == []
    assert not controller.bridge.handle_action("startDictation", {})["daemon_running"]


@pytest.mark.parametrize("kind", ["install", "probe"])
def test_model_switch_inside_ask_callback_blocks_stale_runtime_recovery(
    ask_intent_controller, kind
):
    controller, begin, complete, switch_model = ask_intent_controller
    prompts = []

    def ask(*_args):
        assert controller._llm_prompt_in_progress
        prompts.append("ask")
        switch_model()
        return 0

    controller._run_alert = ask
    begin(kind)
    complete(kind)
    assert prompts == ["ask"]
    assert controller.bridge._load_config().stt.model == "moonshine:tiny"
    assert not controller.daemon_controller.is_running
    assert controller.daemon_controller.overrides == []
    assert controller.bridge.handle_action("startDictation", {})["daemon_running"]
    assert len(controller.daemon_controller.overrides) == 1


@pytest.mark.parametrize("kind", ["install", "probe"])
def test_new_start_inside_ask_callback_runs_fresh_preflight(ask_intent_controller, kind):
    controller, begin, complete, switch_model = ask_intent_controller
    started_models = []
    original_start = controller.daemon_controller.start

    def start(**kwargs):
        started_models.append(controller.bridge._load_config().stt.model)
        original_start(**kwargs)

    controller.daemon_controller.start = start

    def ask(*_args):
        controller.bridge.handle_action("stopDictation", {})
        switch_model()
        assert not controller.bridge.handle_action("startDictation", {})["daemon_running"]
        return 0

    controller._run_alert = ask
    begin(kind)
    complete(kind)
    assert controller.daemon_controller.is_running
    assert started_models == ["moonshine:tiny"]
    assert controller.daemon_controller.overrides == [None]  # current config uses always, not ask


@pytest.mark.parametrize("kind", ["install", "probe"])
def test_new_runtime_generation_inside_ask_callback_blocks_stale_start(ask_intent_controller, kind):
    controller, begin, _, _ = ask_intent_controller
    prompts = []

    def ask(*_args):
        assert controller._llm_prompt_in_progress
        prompts.append("ask")
        controller._begin_speech_cpp_work(install=False)
        return 0

    controller._run_alert = ask
    begin(kind)
    controller.applySpeechCppStatus_(
        {
            "operation_id": controller._speech_cpp_operation_id,
            "finished": True,
            "recover": kind == "install",
            "status": {"status": "available", "install_allowed": False},
        }
    )
    assert prompts == ["ask"]
    assert controller._speech_cpp_status["status"] == "checking"
    assert not controller.daemon_controller.is_running
    assert controller.daemon_controller.overrides == []


@pytest.mark.parametrize("kind", ["install", "probe"])
def test_llm_mode_change_inside_ask_callback_blocks_stale_override(ask_intent_controller, kind):
    controller, begin, complete, _ = ask_intent_controller

    def ask(*_args):
        config = controller.bridge._load_config()
        config.text.llm_correction.mode = LLMCorrectionMode.NEVER
        write_config(controller.bridge.deps.config_path(), config)
        return 0

    controller._run_alert = ask
    begin(kind)
    complete(kind)
    assert not controller.daemon_controller.is_running
    assert controller.daemon_controller.overrides == []
    assert controller.bridge.handle_action("startDictation", {})["daemon_running"]
    assert controller.daemon_controller.overrides == [None]


@pytest.mark.parametrize("kind", ["install", "probe"])
def test_new_start_inside_ask_callback_rechecks_pending_runtime(ask_intent_controller, kind):
    controller, begin, _, _ = ask_intent_controller

    def ask(*_args):
        controller.bridge.handle_action("stopDictation", {})
        controller._begin_speech_cpp_work(install=False)
        assert not controller.bridge.handle_action("startDictation", {})["daemon_running"]
        return 0

    controller._run_alert = ask
    begin(kind)
    controller.applySpeechCppStatus_(
        {
            "operation_id": controller._speech_cpp_operation_id,
            "finished": True,
            "recover": kind == "install",
            "status": {"status": "available", "install_allowed": False},
        }
    )
    assert not controller.daemon_controller.is_running
    assert controller._speech_cpp_start_when_ready
    controller.applySpeechCppStatus_(
        {
            "operation_id": controller._speech_cpp_operation_id,
            "finished": True,
            "status": {"status": "available", "install_allowed": False},
        }
    )
    assert controller.daemon_controller.is_running
    assert controller.daemon_controller.overrides == [True]
