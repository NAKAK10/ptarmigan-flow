from __future__ import annotations

import pytest

from ptarmigan_flow import onboarding_strings
from ptarmigan_flow.permissions import PermissionReport
from ptarmigan_flow.setup_status import (
    SETUP_ITEMS,
    missing_permission_items,
    missing_setup_items,
    setup_required_label,
    status_menu_state,
)


def _report(*, microphone=True, accessibility=True, input_monitoring=True) -> PermissionReport:
    return PermissionReport(
        microphone=microphone,
        accessibility=accessibility,
        input_monitoring=input_monitoring,
    )


def test_missing_items_without_report_lists_every_step_not_yet_passed() -> None:
    assert missing_setup_items("language", None) == list(SETUP_ITEMS)
    assert missing_setup_items("hotkey", None) == [
        "hotkey",
        "microphone",
        "accessibility",
        "input_monitoring",
    ]
    assert missing_setup_items("accessibility", None) == ["accessibility", "input_monitoring"]
    assert missing_setup_items("done", None) == []


def test_missing_items_uses_permission_report_when_known() -> None:
    report = _report(microphone=True, accessibility=False, input_monitoring=False)

    assert missing_setup_items("language", report) == [
        "language",
        "hotkey",
        "accessibility",
        "input_monitoring",
    ]
    assert missing_setup_items("input_monitoring", _report(input_monitoring=False)) == [
        "input_monitoring"
    ]


def test_missing_items_after_onboarding_reports_revoked_permissions() -> None:
    assert missing_setup_items("done", _report()) == []
    assert missing_setup_items("done", _report(microphone=False)) == ["microphone"]


def test_missing_permission_items_lists_ungranted_permissions_in_order() -> None:
    assert missing_permission_items(_report()) == []
    assert missing_permission_items(_report(microphone=False, input_monitoring=False)) == [
        "microphone",
        "input_monitoring",
    ]


def test_missing_items_treats_unknown_step_as_start() -> None:
    assert missing_setup_items("bogus", None) == list(SETUP_ITEMS)


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("ja", "セットアップが必要です（入力監視）"),
        ("en", "Setup required (Input Monitoring)"),
        ("zh", "需要完成初始设置（输入监控）"),
    ],
)
def test_setup_required_label_single_item(language: str, expected: str) -> None:
    strings = onboarding_strings.strings_for(language)

    assert setup_required_label(["input_monitoring"], strings) == expected


def test_setup_required_label_joins_items_with_localized_separator() -> None:
    ja = onboarding_strings.strings_for("ja")
    en = onboarding_strings.strings_for("en")

    assert setup_required_label(["microphone", "input_monitoring"], ja) == (
        "セットアップが必要です（マイク、入力監視）"
    )
    assert setup_required_label(["language", "hotkey"], en) == (
        "Setup required (Language, Hotkey)"
    )


def test_status_menu_state_setup_required_while_stopped() -> None:
    strings = onboarding_strings.strings_for("ja")

    state = status_menu_state(["input_monitoring"], daemon_running=False, strings=strings)

    assert state.status_title == "セットアップが必要です（入力監視）"
    assert state.button_title == "⚠︎"
    assert state.show_open_setup is True


def test_status_menu_state_plain_stopped_when_nothing_missing() -> None:
    strings = onboarding_strings.strings_for("ja")

    state = status_menu_state([], daemon_running=False, strings=strings)

    assert state.status_title == "音声入力 停止中"
    assert state.button_title == ""
    assert state.show_open_setup is False


def test_status_menu_state_running_wins_over_stale_missing_items() -> None:
    strings = onboarding_strings.strings_for("en")

    state = status_menu_state(["microphone"], daemon_running=True, strings=strings)

    assert state.status_title == "Voice input running"
    assert state.button_title == ""
    assert state.show_open_setup is False
