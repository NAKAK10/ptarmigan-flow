"""Pure helpers that describe incomplete setup for the macOS status-bar menu."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ptarmigan_flow.onboarding_flow import OnboardingFlow
from ptarmigan_flow.permissions import PermissionReport

SETUP_ITEMS: tuple[str, ...] = (
    "language",
    "hotkey",
    "microphone",
    "accessibility",
    "input_monitoring",
)
_PERMISSION_ITEMS: tuple[str, ...] = ("microphone", "accessibility", "input_monitoring")

STATUS_BUTTON_TITLE = "Pt"
STATUS_BUTTON_SETUP_REQUIRED_TITLE = "Pt ⚠︎"


def missing_setup_items(
    current_step: str,
    report: PermissionReport | None,
) -> list[str]:
    """Return the setup item keys that still block voice input, in onboarding order.

    ``current_step`` is the onboarding step. Language and hotkey are missing
    while onboarding has not moved past them. Permissions come from ``report``
    when known; without a report, every permission step not yet passed counts
    as missing.
    """
    steps = OnboardingFlow.steps
    step_index = steps.index(current_step) if current_step in steps else 0
    missing: list[str] = []
    for item in SETUP_ITEMS:
        if item in _PERMISSION_ITEMS and report is not None:
            if not bool(getattr(report, item)):
                missing.append(item)
        elif step_index <= steps.index(item):
            missing.append(item)
    return missing


def setup_required_label(missing: list[str], strings: Mapping[str, str]) -> str:
    """Format the status row, e.g. ``セットアップが必要です（入力監視）``."""
    labels = [strings[f"setup_item_{item}"] for item in missing]
    return strings["setup_required_menu"].format(
        items=strings["setup_item_separator"].join(labels)
    )


@dataclass(frozen=True)
class StatusMenuState:
    status_title: str
    button_title: str
    show_open_setup: bool


def status_menu_state(
    missing: list[str],
    *,
    daemon_running: bool,
    strings: Mapping[str, str],
) -> StatusMenuState:
    """Decide the status row, status button title and setup item visibility."""
    if daemon_running:
        return StatusMenuState(
            status_title=strings["dictation_running_menu"],
            button_title=STATUS_BUTTON_TITLE,
            show_open_setup=False,
        )
    if missing:
        return StatusMenuState(
            status_title=setup_required_label(missing, strings),
            button_title=STATUS_BUTTON_SETUP_REQUIRED_TITLE,
            show_open_setup=True,
        )
    return StatusMenuState(
        status_title=strings["dictation_stopped_menu"],
        button_title=STATUS_BUTTON_TITLE,
        show_open_setup=False,
    )


__all__ = [
    "SETUP_ITEMS",
    "STATUS_BUTTON_SETUP_REQUIRED_TITLE",
    "STATUS_BUTTON_TITLE",
    "StatusMenuState",
    "missing_setup_items",
    "setup_required_label",
    "status_menu_state",
]
