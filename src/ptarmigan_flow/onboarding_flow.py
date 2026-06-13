"""Pure onboarding step state machine for the macOS app."""

from __future__ import annotations

from ptarmigan_flow.permissions import PermissionReport

SUPPORTED_LANGUAGES = frozenset({"en", "ja", "zh"})


class OnboardingFlow:
    """Track one-step-at-a-time onboarding progress."""

    steps = ["language", "microphone", "accessibility", "input_monitoring", "done"]
    _permission_step_attrs = {
        "microphone": "microphone",
        "accessibility": "accessibility",
        "input_monitoring": "input_monitoring",
    }

    def __init__(self) -> None:
        self._step_index = 0
        self.selected_language: str | None = None

    @property
    def current_step(self) -> str:
        return self.steps[self._step_index]

    @property
    def is_complete(self) -> bool:
        return self.current_step == "done"

    def advance(self) -> None:
        if self._step_index < len(self.steps) - 1:
            self._step_index += 1

    def start(self, report: PermissionReport | None = None) -> None:
        if report is not None:
            self.refresh(report)

    def refresh(self, report: PermissionReport) -> None:
        while self.current_step in self._permission_step_attrs:
            permission_attr = self._permission_step_attrs[self.current_step]
            if not bool(getattr(report, permission_attr)):
                return
            self.advance()

    def choose_language(self, code: str) -> None:
        normalized = code.strip().lower()
        if normalized not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {code}")
        self.selected_language = normalized
        if self.current_step == "language":
            self.advance()


__all__ = ["OnboardingFlow", "SUPPORTED_LANGUAGES"]
