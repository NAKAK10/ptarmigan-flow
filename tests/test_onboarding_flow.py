from __future__ import annotations

import pytest

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

    assert flow.steps == ["language", "microphone", "accessibility", "input_monitoring", "done"]
    assert flow.current_step == "language"
    assert flow.is_complete is False

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
    assert flow.current_step == "microphone"


def test_choose_language_rejects_unsupported_codes_without_advancing() -> None:
    flow = OnboardingFlow()

    with pytest.raises(ValueError, match="Unsupported language code"):
        flow.choose_language("fr")

    assert flow.selected_language is None
    assert flow.current_step == "language"


def test_refresh_advances_current_permission_step_when_permission_is_granted() -> None:
    flow = OnboardingFlow()
    flow.choose_language("en")

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

    flow.refresh(_report(microphone=True, accessibility=True, input_monitoring=False))

    assert flow.current_step == "input_monitoring"
    assert flow.is_complete is False

    flow.refresh(_report(microphone=True, accessibility=True, input_monitoring=True))

    assert flow.current_step == "done"
    assert flow.is_complete is True


def test_start_skips_granted_permission_steps_after_language_selection() -> None:
    flow = OnboardingFlow()
    flow.choose_language("en")

    flow.start(_report(microphone=True, accessibility=False, input_monitoring=True))

    assert flow.current_step == "accessibility"


def test_start_does_not_skip_language_selection() -> None:
    flow = OnboardingFlow()

    flow.start(_report(microphone=True, accessibility=True, input_monitoring=True))

    assert flow.current_step == "language"
    assert flow.is_complete is False
