from __future__ import annotations

import json

import pytest

from ptarmigan_flow import onboarding_flow as onboarding_flow_module
from ptarmigan_flow.onboarding_flow import OnboardingFlow
from ptarmigan_flow.permissions import PermissionReport


def _report(
    *,
    microphone: bool = False,
    accessibility: bool = False,
    input_monitoring: bool = False,
) -> PermissionReport:
    return PermissionReport(
        microphone=microphone,
        accessibility=accessibility,
        input_monitoring=input_monitoring,
    )


def test_onboarding_flow_exposes_ordered_steps_and_completion_state() -> None:
    flow = OnboardingFlow()

    assert flow.steps == [
        "language",
        "hotkey",
        "microphone",
        "accessibility",
        "input_monitoring",
        "done",
    ]
    assert flow.current_step == "language"
    assert flow.is_complete is False

    flow.advance()
    assert flow.current_step == "hotkey"

    flow.advance()
    assert flow.current_step == "microphone"

    flow.advance()
    assert flow.current_step == "accessibility"

    flow.advance()
    assert flow.current_step == "input_monitoring"

    flow.advance()
    assert flow.current_step == "done"
    assert flow.is_complete is True

    flow.advance()
    assert flow.current_step == "done"


def test_choose_language_accepts_supported_codes_and_advances() -> None:
    flow = OnboardingFlow()

    flow.choose_language("ja")

    assert flow.selected_language == "ja"
    assert flow.current_step == "hotkey"


def test_confirm_hotkey_accepts_supported_keys_and_advances_to_permissions() -> None:
    flow = OnboardingFlow()
    flow.choose_language("ja")

    flow.confirm_hotkey("right_shift")

    assert flow.selected_hotkey == "right_shift"
    assert flow.current_step == "microphone"


def test_confirm_hotkey_rejects_unsupported_keys_without_advancing() -> None:
    flow = OnboardingFlow()
    flow.choose_language("ja")

    with pytest.raises(ValueError, match="Unsupported hotkey"):
        flow.confirm_hotkey("shift_right")

    assert flow.selected_hotkey is None
    assert flow.current_step == "hotkey"


def test_choose_language_rejects_unsupported_codes_without_advancing() -> None:
    flow = OnboardingFlow()

    with pytest.raises(ValueError, match="Unsupported language code"):
        flow.choose_language("fr")

    assert flow.selected_language is None
    assert flow.current_step == "language"


def test_refresh_advances_current_permission_step_when_permission_is_granted() -> None:
    flow = OnboardingFlow()
    flow.choose_language("en")
    flow.confirm_hotkey("right_cmd")

    flow.refresh(_report(microphone=False))
    assert flow.current_step == "microphone"

    flow.refresh(_report(microphone=True))
    assert flow.current_step == "accessibility"

    flow.refresh(_report(microphone=True, accessibility=True))
    assert flow.current_step == "input_monitoring"

    flow.refresh(_report(microphone=True, accessibility=True, input_monitoring=True))
    assert flow.current_step == "done"


def test_refresh_skips_consecutive_permission_steps_that_are_already_granted() -> None:
    flow = OnboardingFlow()
    flow.choose_language("zh")
    flow.confirm_hotkey("left_ctrl")

    flow.refresh(_report(microphone=True, accessibility=True, input_monitoring=False))

    assert flow.current_step == "input_monitoring"
    assert flow.is_complete is False

    flow.refresh(_report(microphone=True, accessibility=True, input_monitoring=True))

    assert flow.current_step == "done"
    assert flow.is_complete is True


def test_start_skips_granted_permission_steps_after_language_selection() -> None:
    flow = OnboardingFlow()
    flow.choose_language("en")
    flow.confirm_hotkey("right_cmd")

    flow.start(_report(microphone=True, accessibility=False, input_monitoring=True))

    assert flow.current_step == "accessibility"


def test_start_does_not_skip_language_selection() -> None:
    flow = OnboardingFlow()

    flow.start(_report(microphone=True, accessibility=True, input_monitoring=True))

    assert flow.current_step == "language"
    assert flow.is_complete is False


def test_start_skips_language_but_waits_for_hotkey_confirmation() -> None:
    flow = OnboardingFlow()

    flow.start(
        _report(microphone=True, accessibility=True, input_monitoring=True),
        language_already_selected=True,
        hotkey_already_confirmed=False,
    )

    assert flow.current_step == "hotkey"
    assert flow.is_complete is False


def test_start_skips_language_and_hotkey_when_both_were_already_confirmed() -> None:
    flow = OnboardingFlow()

    flow.start(
        _report(microphone=True, accessibility=True, input_monitoring=True),
        language_already_selected=True,
        hotkey_already_confirmed=True,
    )

    assert flow.current_step == "done"
    assert flow.is_complete is True


def test_start_keeps_language_when_language_was_not_already_selected() -> None:
    flow = OnboardingFlow()

    flow.start(
        _report(microphone=True, accessibility=True, input_monitoring=True),
        language_already_selected=False,
    )

    assert flow.current_step == "language"
    assert flow.is_complete is False


def test_language_selection_state_round_trips_through_persisted_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    state_path = tmp_path / "Library" / "Application Support" / "ptarmigan-flow" / "state.json"
    monkeypatch.setattr(
        onboarding_flow_module,
        "onboarding_state_path",
        lambda: state_path,
        raising=False,
    )

    assert onboarding_flow_module.language_was_selected() is False

    onboarding_flow_module.mark_language_selected()

    assert json.loads(state_path.read_text(encoding="utf-8")) == {"language_selected": True}
    assert onboarding_flow_module.language_was_selected() is True


def test_hotkey_confirmation_state_round_trips_through_persisted_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    state_path = tmp_path / "Library" / "Application Support" / "ptarmigan-flow" / "state.json"
    monkeypatch.setattr(
        onboarding_flow_module,
        "onboarding_state_path",
        lambda: state_path,
        raising=False,
    )

    assert onboarding_flow_module.hotkey_was_confirmed() is False

    onboarding_flow_module.mark_language_selected()
    onboarding_flow_module.mark_hotkey_confirmed()

    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "language_selected": True,
        "hotkey_confirmed": True,
    }
    assert onboarding_flow_module.language_was_selected() is True
    assert onboarding_flow_module.hotkey_was_confirmed() is True


def test_language_was_selected_returns_false_for_malformed_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    state_path = tmp_path / "onboarding_state.json"
    state_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(onboarding_flow_module, "onboarding_state_path", lambda: state_path)

    assert onboarding_flow_module.language_was_selected() is False


def test_hotkey_was_confirmed_returns_false_for_malformed_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    state_path = tmp_path / "onboarding_state.json"
    state_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(onboarding_flow_module, "onboarding_state_path", lambda: state_path)

    assert onboarding_flow_module.hotkey_was_confirmed() is False
